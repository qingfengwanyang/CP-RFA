#!/usr/bin/env python3
"""
Recovery Token v3：v2 payload + anon-image binding digest.

格式：CPRFA-1|<image_id>|<base64 ciphertext>|<CRC16>
明文：SHA256(anon_crop) || int16[512] z_enc || sidecar_bytes
兼容旧线格式 FIT-CHAOS-3。
"""

import base64
import re
import zlib

import torch

from models.anon_bind import BIND_DIGEST_LEN
from models.recovery_payload_cipher import PayloadCipher
from models.recovery_sidecar import SidecarData, sidecar_from_bytes, sidecar_to_bytes
from models.recovery_token_v2 import (
    MAGIC as MAGIC_V2,
    _DIM,
    _crc16,
    _dequantize_z,
    _quantize_z,
    unpack_recovery_token_v2,
)

MAGIC = 'CPRFA-1'
MAGIC_ALIASES = (MAGIC, 'FIT-CHAOS-3')
_CIPHER = PayloadCipher()


def set_payload_cipher(cipher: PayloadCipher) -> None:
    """Share the pipeline HSIO module so token XOR uses the same keystream."""
    global _CIPHER
    _CIPHER = cipher


def _pack_plain_v3(bind_digest: bytes, z_enc: torch.Tensor, sidecar: SidecarData) -> bytes:
    if len(bind_digest) != BIND_DIGEST_LEN:
        raise ValueError('bind digest must be %d bytes' % BIND_DIGEST_LEN)
    return bind_digest + _quantize_z(z_enc) + sidecar_to_bytes(sidecar)


def _unpack_plain_v3(data: bytes):
    if len(data) < BIND_DIGEST_LEN + _DIM * 2:
        raise ValueError('v3 plaintext too short')
    bind_digest = data[:BIND_DIGEST_LEN]
    z_enc = _dequantize_z(data[BIND_DIGEST_LEN:BIND_DIGEST_LEN + _DIM * 2])
    sidecar = sidecar_from_bytes(data[BIND_DIGEST_LEN + _DIM * 2:])
    return bind_digest, z_enc, sidecar


def pack_recovery_token_v3(image_id: str, z_enc: torch.Tensor, sidecar: SidecarData,
                           password: torch.Tensor, anon_bind_digest: bytes) -> str:
    image_id = re.sub(r'[|\s]+', '_', str(image_id).strip())[:120]
    plain = _pack_plain_v3(anon_bind_digest, z_enc, sidecar)
    if password.dim() == 1:
        password = password.unsqueeze(0)
    cipher = _CIPHER.encrypt(plain, password.to(z_enc.device))
    b64 = base64.urlsafe_b64encode(cipher).decode('ascii').rstrip('=')
    crc = _crc16(cipher)
    return '%s|%s|%s|%s' % (MAGIC, image_id, b64, crc)


def unpack_recovery_token_v3(token: str, password: torch.Tensor):
    """返回 (image_id, z_enc, sidecar, bind_digest)。"""
    parts = token.strip().split('|')
    if len(parts) != 4 or parts[0] not in MAGIC_ALIASES:
        raise ValueError('invalid AnonBind token: %s' % token[:48])
    image_id, b64, crc = parts[1], parts[2], parts[3]
    pad = '=' * ((4 - len(b64) % 4) % 4)
    cipher = base64.urlsafe_b64decode(b64 + pad)
    if _crc16(cipher) != crc.upper():
        raise ValueError('CRC mismatch for image_id=%s' % image_id)
    if password.dim() == 1:
        password = password.unsqueeze(0)
    plain = _CIPHER.decrypt(cipher, password)
    bind_digest, z_enc, sidecar = _unpack_plain_v3(plain)
    return image_id, z_enc, sidecar, bind_digest


def is_v3_token(token: str) -> bool:
    head = token.strip().split('|', 1)[0]
    return head in MAGIC_ALIASES
