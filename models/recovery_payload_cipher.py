#!/usr/bin/env python3
"""密码驱动的混沌流 XOR，用于 sidecar payload 加解密（非 L2 身份向量）。"""

import numpy as np
import torch

from models.format_preserving_encryptor import FormatPreservingEncryptor


class PayloadCipher:
    """与 HSIO 同源的 Logistic 混沌流；错密码 → 解密乱码。"""

    def __init__(self, fpe=None, init_seed=0):
        self._fpe = fpe if fpe is not None else FormatPreservingEncryptor(
            init_seed=init_seed)
        for p in self._fpe.parameters():
            p.requires_grad = False

    def _keystream(self, password: torch.Tensor, length: int) -> bytes:
        if password.dim() == 1:
            password = password.unsqueeze(0)
        device = password.device
        fpe = self._fpe.to(device)
        with torch.no_grad():
            seed = fpe.seed_proj(password).mean(dim=-1, keepdim=True)
            seed = (seed.tanh() + 1.0) * 0.499 + 0.001
            n_float = (length + 3) // 4
            chaos = fpe._logistic_map(seed, max(n_float, 4))
            flat = chaos.reshape(-1)[:length]
            arr = (flat * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        if arr.size < length:
            rep = int(np.ceil(length / max(arr.size, 1)))
            arr = np.tile(arr, rep)[:length]
        return arr.tobytes()

    def encrypt(self, plain: bytes, password: torch.Tensor) -> bytes:
        ks = self._keystream(password, len(plain))
        p = np.frombuffer(plain, dtype=np.uint8)
        k = np.frombuffer(ks, dtype=np.uint8)
        return bytes((p ^ k).astype(np.uint8))

    def decrypt(self, cipher: bytes, password: torch.Tensor) -> bytes:
        return self.encrypt(cipher, password)
