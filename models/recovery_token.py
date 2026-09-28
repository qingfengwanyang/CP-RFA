#!/usr/bin/env python3
"""Unified recovery token API: v2 (legacy) and v3 (anon-bound)."""

import torch

from models.recovery_sidecar import SidecarData
from models.recovery_token_v2 import (
    MAGIC as MAGIC_V2,
    is_v2_token,
    pack_recovery_token_v2,
    unpack_recovery_token_v2,
)
from models.recovery_token_v3 import (
    MAGIC as MAGIC_V3,
    is_v3_token,
    pack_recovery_token_v3,
    unpack_recovery_token_v3,
)


class AnonBindMismatchError(ValueError):
    """Anonymous crop digest does not match token binding."""


class TokenFormatError(ValueError):
    """Malformed token string or failed integrity check."""


def token_magic(token: str) -> str:
    return token.strip().split('|')[0] if token else ''


def is_recovery_token(token: str) -> bool:
    return is_v2_token(token) or is_v3_token(token)


def pack_recovery_token(image_id: str, z_enc: torch.Tensor, sidecar: SidecarData,
                        password: torch.Tensor, anon_bind_digest: bytes = None,
                        version: str = 'v3') -> str:
    if version == 'v3':
        if anon_bind_digest is None:
            raise ValueError('v3 token requires anon_bind_digest')
        return pack_recovery_token_v3(image_id, z_enc, sidecar, password, anon_bind_digest)
    if version == 'v2':
        return pack_recovery_token_v2(image_id, z_enc, sidecar, password)
    raise ValueError('unknown token version: %s' % version)


def unpack_recovery_token(token: str, password: torch.Tensor):
    """
    返回 (image_id, z_enc, sidecar, bind_digest)。
    v2 的 bind_digest 为 None。
    """
    if is_v3_token(token):
        return unpack_recovery_token_v3(token, password)
    if is_v2_token(token):
        image_id, z_enc, sidecar = unpack_recovery_token_v2(token, password)
        return image_id, z_enc, sidecar, None
    raise TokenFormatError('unsupported token magic: %s' % token[:32])


def tamper_token_crc(token: str) -> str:
    """Flip last CRC hex char for integrity attack."""
    parts = token.strip().split('|')
    if len(parts) != 4:
        return token
    crc = parts[3]
    if not crc:
        return token
    last = crc[-1]
    repl = '0' if last != '0' else '1'
    parts[3] = crc[:-1] + repl
    return '|'.join(parts)


def tamper_token_cipher_byte(token: str, byte_index: int = 0) -> str:
    """Flip one byte in the base64 ciphertext payload."""
    import base64
    parts = token.strip().split('|')
    if len(parts) != 4:
        return token
    b64 = parts[2]
    pad = '=' * ((4 - len(b64) % 4) % 4)
    raw = bytearray(base64.urlsafe_b64decode(b64 + pad))
    if not raw:
        return token
    idx = byte_index % len(raw)
    raw[idx] ^= 0x01
    parts[2] = base64.urlsafe_b64encode(bytes(raw)).decode('ascii').rstrip('=')
    return '|'.join(parts)
