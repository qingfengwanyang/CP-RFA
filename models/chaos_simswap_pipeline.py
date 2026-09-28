#!/usr/bin/env python3
"""
ChaosSimSwap 生产管线：FPE 身份 + SimSwap 匿名 + Token v2 sidecar 恢复。

注册（需原图，一次性）：
  z_enc = FPE(z_orig, p)
  anon  = SimSwap(Mapper(z_enc,p), orig)
  rec0  = SimSwap(decrypt(z_enc,p), anon)
  token = pack_v2(z_enc, sidecar(stats, residual, rgb), p)

恢复（仅需 anon + token + password）：
  z_dec, sidecar = decrypt(token, p)
  rec0 = SimSwap(z_dec, anon)
  rec  = rec0 + sidecar_residual + RGB align
"""

from __future__ import annotations

import dataclasses
import os.path as osp
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch

from models.chaos_simswap_mapper import PasswordLatentMapper
from models.format_preserving_encryptor import FormatPreservingEncryptor
from models.recovery_sidecar import SidecarData, apply_sidecar, build_sidecar
from models.anon_bind import DEFAULT_CROP_SIZE as BIND_CROP_SIZE
from models.anon_bind import anon_crop_digest, verify_anon_bind
from models.recovery_payload_cipher import PayloadCipher
from models.recovery_token import (
    AnonBindMismatchError,
    TokenFormatError,
    is_recovery_token,
    pack_recovery_token,
    unpack_recovery_token,
)
from models.recovery_token_v2 import set_payload_cipher as set_v2_cipher
from models.recovery_token_v3 import set_payload_cipher as set_v3_cipher


def bgr_crop_to_tensor(bgr, device, crop_size=224):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image
    from torchvision import transforms
    pil = Image.fromarray(rgb)
    if pil.size != (crop_size, crop_size):
        pil = pil.resize((crop_size, crop_size), Image.BICUBIC)
    return transforms.ToTensor()(pil).unsqueeze(0).to(device)


def tensor_to_bgr_crop(t):
    t = t.detach().clamp(0, 1)[0].cpu()
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


@dataclasses.dataclass
class ChaosSimSwapConfig:
    simswap_root: str = ''
    mapper_ckpt: str = 'checkpoints/mapper_best.pth.tar'
    crop_size: int = 256
    password_dim: int = 16
    sidecar_color_weight: float = 0.35
    use_mask: bool = True
    token_version: str = 'v3'
    enforce_anon_bind: bool = True
    swap_backend: str = 'hyperswap'
    hyperswap_model: str = ''
    insightface_root: str = ''
    hyperswap_swap_weight: float = 1.0
    use_simswap_shallow_stats: bool = False
    fpe_num_rounds: int = 4


@dataclasses.dataclass
class RegisterResult:
    slug: str
    token: str
    anon_crop_bgr: np.ndarray
    rec0_crop_bgr: np.ndarray
    rec_crop_bgr: np.ndarray
    z_orig: torch.Tensor
    z_enc: torch.Tensor
    id_anon: float
    id_rec0: float
    id_rec: float
    decrypt_mae: float


@dataclasses.dataclass
class RecoverResult:
    rec0_crop_bgr: np.ndarray
    rec_crop_bgr: np.ndarray
    z_dec: torch.Tensor
    id_rec0: float
    id_rec: float


class ChaosSimSwapPipeline:
    """对齐 crop 上的完整 ChaosSimSwap 协议（v4 sidecar）。"""

    def __init__(self, config: ChaosSimSwapConfig, device=None, engine=None):
        self.cfg = config
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._engine = engine
        self.mapper = PasswordLatentMapper().to(self.device)
        self.fpe = FormatPreservingEncryptor(
            identity_dim=512, password_dim=config.password_dim,
            num_rounds=int(getattr(config, 'fpe_num_rounds', 4) or 4),
            init_seed=0,
        ).to(self.device)
        self._load_mapper_ckpt(config.mapper_ckpt)
        cipher = PayloadCipher(self.fpe)
        set_v2_cipher(cipher)
        set_v3_cipher(cipher)

    def _get_engine(self):
        if self._engine is None:
            from scripts.swap_engine import create_swap_engine
            kwargs = dict(
                simswap_root=self.cfg.simswap_root,
                crop_size=self.cfg.crop_size,
                use_mask=self.cfg.use_mask,
                swap_weight=self.cfg.hyperswap_swap_weight,
                use_simswap_shallow_stats=self.cfg.use_simswap_shallow_stats,
            )
            if self.cfg.hyperswap_model:
                kwargs['hyperswap_model'] = self.cfg.hyperswap_model
            if self.cfg.insightface_root:
                kwargs['insightface_root'] = self.cfg.insightface_root
            self._engine = create_swap_engine(
                self.cfg.swap_backend,
                **kwargs,
            )
            self._engine.load_finetune_ckpt(self.cfg.mapper_ckpt)
        return self._engine

    def _load_mapper_ckpt(self, ckpt_path):
        if not ckpt_path or not osp.isfile(ckpt_path):
            print('[CP-RFA] WARN: no mapper ckpt: %s' % ckpt_path)
            return
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'mapper' in ckpt:
            self.mapper.load_state_dict(ckpt['mapper'])
        if 'fpe' in ckpt:
            self.fpe.load_state_dict(ckpt['fpe'])
            for p in self.fpe.parameters():
                p.requires_grad = False
        self.mapper.eval()
        print('[CP-RFA] loaded mapper from %s' % ckpt_path)

    @torch.no_grad()
    def _latent_pair(self, engine, crop_bgr, z_a, z_b):
        za = engine.latent_from_crop_bgr(crop_bgr if z_a is None else z_a)
        if z_b is None:
            return za
        zb = z_b if z_b.dim() == 2 else z_b.unsqueeze(0)
        return torch.cosine_similarity(za, zb).item()

    @torch.no_grad()
    def register_from_path(
        self,
        orig_path: str,
        password: torch.Tensor,
        slug: str,
    ) -> RegisterResult:
        """注册：原图路径 + 密码 → anon、恢复图、token。"""
        engine = self._get_engine()
        orig_path = osp.abspath(orig_path)
        orig_crop, _, _ = engine.extract_align_crop(orig_path)
        z_orig = engine.latent_from_crop_bgr(orig_crop)
        enc_stats = engine.extract_shallow_stats(orig_crop)

        if password.dim() == 1:
            password = password.unsqueeze(0)

        z_enc, _ = self.fpe.encrypt(z_orig, password)
        z_swap = self.mapper(z_enc, password)
        anon_crop = engine.swap_latent_on_crop_bgr(z_swap, orig_crop)
        z_dec = self.fpe.decrypt(z_enc, password=password)
        rec0_crop = engine.swap_latent_on_crop_bgr(z_dec, anon_crop)

        orig_t = bgr_crop_to_tensor(orig_crop, self.device, self.cfg.crop_size)
        rec0_t = bgr_crop_to_tensor(rec0_crop, self.device, self.cfg.crop_size)
        sidecar = build_sidecar(orig_t, rec0_t, enc_stats)
        bind_digest = anon_crop_digest(anon_crop, BIND_CROP_SIZE)
        token = pack_recovery_token(
            slug, z_enc[0], sidecar, password[0],
            anon_bind_digest=bind_digest,
            version=self.cfg.token_version,
        )

        rec_t = apply_sidecar(
            rec0_t, sidecar,
            out_size=self.cfg.crop_size,
            color_weight=self.cfg.sidecar_color_weight,
        )
        rec_crop = tensor_to_bgr_crop(rec_t)

        id_anon = torch.cosine_similarity(
            z_orig, engine.latent_from_crop_bgr(anon_crop)).item()
        id_rec0 = torch.cosine_similarity(
            z_orig, engine.latent_from_crop_bgr(rec0_crop)).item()
        id_rec = torch.cosine_similarity(
            z_orig, engine.latent_from_crop_bgr(rec_crop)).item()
        dec_err = (z_dec - z_orig).abs().mean().item()

        return RegisterResult(
            slug=slug,
            token=token,
            anon_crop_bgr=anon_crop,
            rec0_crop_bgr=rec0_crop,
            rec_crop_bgr=rec_crop,
            z_orig=z_orig,
            z_enc=z_enc,
            id_anon=id_anon,
            id_rec0=id_rec0,
            id_rec=id_rec,
            decrypt_mae=dec_err,
        )

    @torch.no_grad()
    def recover_from_token(
        self,
        anon_crop_bgr: np.ndarray,
        token: str,
        password: torch.Tensor,
        z_orig_ref: Optional[torch.Tensor] = None,
        enforce_anon_bind: Optional[bool] = None,
    ) -> RecoverResult:
        """恢复：仅需 anon crop + token + password（无原图）。"""
        if not is_recovery_token(token):
            raise ValueError('CP-RFA requires a CPRFA-1 / FIT-CHAOS-3 AnonBind token')
        engine = self._get_engine()
        if password.dim() == 1:
            password = password.unsqueeze(0)

        check_bind = self.cfg.enforce_anon_bind if enforce_anon_bind is None else enforce_anon_bind
        try:
            _, z_enc_q, sidecar, bind_digest = unpack_recovery_token(token, password[0])
        except ValueError as e:
            raise TokenFormatError(str(e)) from e

        if bind_digest is not None and check_bind:
            if not verify_anon_bind(anon_crop_bgr, bind_digest, BIND_CROP_SIZE):
                raise AnonBindMismatchError('anonymous crop does not match token binding digest')
        z_enc_q = z_enc_q.to(self.device).unsqueeze(0)
        z_dec = self.fpe.decrypt(z_enc_q, password=password)

        rec0_crop = engine.swap_latent_on_crop_bgr(z_dec, anon_crop_bgr)
        rec0_t = bgr_crop_to_tensor(rec0_crop, self.device, self.cfg.crop_size)
        rec_t = apply_sidecar(
            rec0_t, sidecar,
            out_size=self.cfg.crop_size,
            color_weight=self.cfg.sidecar_color_weight,
        )
        rec_crop = tensor_to_bgr_crop(rec_t)

        if z_orig_ref is not None:
            id_rec0 = torch.cosine_similarity(
                z_orig_ref, engine.latent_from_crop_bgr(rec0_crop)).item()
            id_rec = torch.cosine_similarity(
                z_orig_ref, engine.latent_from_crop_bgr(rec_crop)).item()
        else:
            id_rec0 = id_rec = 0.0

        return RecoverResult(
            rec0_crop_bgr=rec0_crop,
            rec_crop_bgr=rec_crop,
            z_dec=z_dec,
            id_rec0=id_rec0,
            id_rec=id_rec,
        )

    @torch.no_grad()
    def recover_from_paths(
        self,
        anon_path: str,
        token: str,
        password: torch.Tensor,
        z_orig_ref: Optional[torch.Tensor] = None,
    ) -> RecoverResult:
        engine = self._get_engine()
        anon_crop, _, _ = engine.extract_align_crop(anon_path)
        return self.recover_from_token(anon_crop, token, password, z_orig_ref)

    @torch.no_grad()
    def verify_token_roundtrip(
        self,
        token: str,
        password: torch.Tensor,
        orig_path: str,
        anon_crop_bgr: np.ndarray,
        z_orig: torch.Tensor,
    ) -> Tuple[RecoverResult, float]:
        """解密 token 并恢复；返回结果与 decrypt MAE。"""
        if password.dim() == 1:
            password = password.unsqueeze(0)
        _, z_enc_q, _, _ = unpack_recovery_token(token, password[0])
        z_enc_q = z_enc_q.to(self.device).unsqueeze(0)
        z_dec = self.fpe.decrypt(z_enc_q, password=password)
        dec_err = (z_dec - z_orig).abs().mean().item()
        rec = self.recover_from_token(anon_crop_bgr, token, password, z_orig)
        return rec, dec_err


# Paper-facing aliases (method name CP-RFA).
CPRFAConfig = ChaosSimSwapConfig
CPRFAPipeline = ChaosSimSwapPipeline
