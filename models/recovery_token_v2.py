#!/usr/bin/env python3
"""
Recovery Token v2：z_enc + 加密 sidecar（浅层 stats + 残差 + RGB）。

格式：FIT-CHAOS-2|<image_id>|<base64 ciphertext>|<CRC16>
明文：int16[512] z_enc || sidecar_bytes
密文：PayloadCipher XOR(password)
"""

import base64
import re
import struct
import zlib

import torch

from models.recovery_payload_cipher import PayloadCipher
from models.recovery_sidecar import SidecarData, sidecar_from_bytes, sidecar_to_bytes

MAGIC = 'FIT-CHAOS-2'
_DIM = 512
_CIPHER = PayloadCipher()


def set_payload_cipher(cipher: PayloadCipher) -> None:
    global _CIPHER
    _CIPHER = cipher


def _crc16(data: bytes) -> str:
    return '%04X' % (zlib.crc32(data) & 0xFFFF)


def _quantize_z(z_enc: torch.Tensor) -> bytes:
    if z_enc.dim() == 2:
        z_enc = z_enc[0]
    arr = z_enc.detach().float().cpu().numpy().astype('float32')
    arr = arr.clip(-1.5, 1.5)
    return (arr * 10000.0).astype('int16').tobytes()


def _dequantize_z(payload: bytes) -> torch.Tensor:
    import numpy as np
    q = np.frombuffer(payload, dtype='int16')
    if q.size != _DIM:
        raise ValueError('z dim %d != %d' % (q.size, _DIM))
    return torch.from_numpy(q.astype('float32') / 10000.0)


def _pack_plain(z_enc: torch.Tensor, sidecar: SidecarData) -> bytes:
    return _quantize_z(z_enc) + sidecar_to_bytes(sidecar)


def _unpack_plain(data: bytes):
    z_bytes = data[:_DIM * 2]
    side = sidecar_from_bytes(data[_DIM * 2:])
    return _dequantize_z(z_bytes), side


def pack_recovery_token_v2(image_id: str, z_enc: torch.Tensor,
                           sidecar: SidecarData, password: torch.Tensor) -> str:
    image_id = re.sub(r'[|\s]+', '_', str(image_id).strip())[:120]
    plain = _pack_plain(z_enc, sidecar)
    if password.dim() == 1:
        password = password.unsqueeze(0)
    cipher = _CIPHER.encrypt(plain, password.to(z_enc.device))
    b64 = base64.urlsafe_b64encode(cipher).decode('ascii').rstrip('=')
    crc = _crc16(cipher)
    return '%s|%s|%s|%s' % (MAGIC, image_id, b64, crc)


def unpack_recovery_token_v2(token: str, password: torch.Tensor):
    """返回 (image_id, z_enc, sidecar)。"""
    parts = token.strip().split('|')
    if len(parts) != 4 or parts[0] != MAGIC:
        raise ValueError('invalid v2 token: %s' % token[:48])
    image_id, b64, crc = parts[1], parts[2], parts[3]
    pad = '=' * ((4 - len(b64) % 4) % 4)
    cipher = base64.urlsafe_b64decode(b64 + pad)
    if _crc16(cipher) != crc.upper():
        raise ValueError('CRC mismatch for image_id=%s' % image_id)
    if password.dim() == 1:
        password = password.unsqueeze(0)
    plain = _CIPHER.decrypt(cipher, password)
    z_enc, sidecar = _unpack_plain(plain)
    return image_id, z_enc, sidecar


def is_v2_token(token: str) -> bool:
    return token.strip().startswith(MAGIC + '|')
