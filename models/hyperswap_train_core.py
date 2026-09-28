#!/usr/bin/env python3
"""HyperSwap + buffalo_l 训练封装（mapper 微调，ONNX swap 无梯度）。"""

from __future__ import annotations

import os
import os.path as osp
import sys
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class HyperSwapTrainCore(nn.Module):
    """CASIA 归一化张量 → buffalo_l latent + HyperSwap ONNX 换脸（推理无梯度）。"""

    def __init__(
        self,
        hyperswap_model: str,
        insightface_root: Optional[str] = None,
        simswap_root: Optional[str] = None,
        crop_size: int = 256,
        train_image_size: int = 128,
    ):
        super().__init__()
        from scripts.hyperswap_engine import HyperSwapEngine

        self.train_image_size = train_image_size
        self.crop_size = crop_size
        self._engine = HyperSwapEngine(
            model_path=hyperswap_model,
            insightface_root=insightface_root,
            crop_size=crop_size,
            simswap_root=simswap_root,
            use_simswap_shallow_stats=False,
        )
        self.device = self._engine.device

    def trainable_parameters(self):
        return []

    def _normed_to_bgr(self, img_normed: torch.Tensor, out_size: int) -> np.ndarray:
        """[-1,1] CHW → BGR uint8 out_size×out_size。"""
        x = img_normed.detach().float().cpu()
        x = (x + 1.0) * 0.5
        x = x.clamp(0, 1).permute(1, 2, 0).numpy()
        rgb = (x * 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if bgr.shape[0] != out_size or bgr.shape[1] != out_size:
            bgr = cv2.resize(bgr, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        return bgr

    def latent_from_normed(self, img_normed: torch.Tensor) -> torch.Tensor:
        """单张 [-1,1] CHW → (1,512) buffalo_l embedding。"""
        crop = self._normed_to_bgr(img_normed, 112)
        emb = self._engine._embedding_from_crop_bgr(crop)
        return self._engine._latent_tensor_from_np(emb)

    def latent_from_normed_batch(self, img_normed: torch.Tensor) -> torch.Tensor:
        """BCHW [-1,1] → (B,512)。"""
        latents = []
        for i in range(img_normed.size(0)):
            latents.append(self.latent_from_normed(img_normed[i]))
        return torch.cat(latents, dim=0)

    def swap_normed(self, img_normed: torch.Tensor, latent_id: torch.Tensor) -> np.ndarray:
        """单张换脸，返回 BGR crop。"""
        crop_bgr = self._normed_to_bgr(img_normed, self.crop_size)
        if latent_id.dim() == 1:
            latent_id = latent_id.unsqueeze(0)
        return self._engine.swap_latent_on_crop_bgr(latent_id, crop_bgr)

    def swap_normed_batch(
        self,
        img_normed: torch.Tensor,
        latent_ids: torch.Tensor,
    ) -> List[np.ndarray]:
        """批量换脸（逐张 ONNX）。"""
        outs = []
        for i in range(img_normed.size(0)):
            z = latent_ids[i]
            if z.dim() == 2:
                z = z.squeeze(0)
            outs.append(self.swap_normed(img_normed[i], z))
        return outs

    def latent_from_bgr_crops(self, crops_bgr: List[np.ndarray]) -> torch.Tensor:
        latents = []
        for crop in crops_bgr:
            emb = self._engine._embedding_from_crop_bgr(crop)
            latents.append(self._engine._latent_tensor_from_np(emb))
        return torch.cat(latents, dim=0)
