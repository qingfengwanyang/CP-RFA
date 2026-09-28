#!/usr/bin/env python3
"""HyperSwap 256 推理封装（ONNX + InsightFace buffalo_l ArcFace）。"""

from __future__ import annotations

import os
import os.path as osp
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _resolve_torch_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    try:
        probe = torch.zeros(1, device='cuda')
        probe += 1
        torch.cuda.synchronize()
        return torch.device('cuda')
    except RuntimeError:
        return torch.device('cpu')


def _l2_normalize_np(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32).reshape(-1)
    n = np.linalg.norm(vec)
    if n < 1e-8:
        return vec
    return vec / n


class HyperSwapEngine:
    """
    与 SimSwapEngine 对齐的换脸接口：
      - latent: buffalo_l ArcFace 512-d（L2 归一化）
      - swap: G(target_crop_256, source_embedding)
    HyperSwap 直接使用 embedding_norm，无需 InSwapper 的 emap 变换。
    """

    MODEL_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    MODEL_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    NATIVE_SIZE = 256

    def __init__(
        self,
        model_path: str,
        insightface_root: Optional[str] = None,
        crop_size: int = 256,
        det_thresh: float = 0.2,
        swap_weight: float = 1.0,
        simswap_root: Optional[str] = None,
        use_simswap_shallow_stats: bool = True,
        providers: Optional[list] = None,
    ):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                'HyperSwap 需要 onnxruntime，请执行: pip install onnxruntime-gpu 或 onnxruntime'
            ) from e
        try:
            import insightface
            from insightface.utils import face_align
        except ImportError as e:
            raise ImportError(
                'HyperSwap 需要 insightface>=0.7，请执行: pip install insightface'
            ) from e

        self.crop_size = crop_size
        self.det_thresh = det_thresh
        self.swap_weight = float(np.clip(swap_weight, 0.0, 1.0))
        self.device = _resolve_torch_device()
        self._face_align = face_align

        if not osp.isfile(model_path):
            raise FileNotFoundError(
                '未找到 HyperSwap 权重: %s\n'
                '请按 docs/HYPERSWAP_SETUP.md 下载 hyperswap_1a_256.onnx' % model_path
            )

        if providers is None:
            avail = ort.get_available_providers()
            providers = []
            if 'CUDAExecutionProvider' in avail:
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
        self.session = ort.InferenceSession(model_path, providers=providers)
        self._input_names = [i.name for i in self.session.get_inputs()]
        self._output_names = [o.name for o in self.session.get_outputs()]
        if len(self._input_names) < 2:
            raise RuntimeError('HyperSwap ONNX 输入异常: %s' % self._input_names)

        if insightface_root is None:
            insightface_root = osp.join(ROOT, 'checkpoints', 'insightface')
        self.insightface_root = osp.abspath(insightface_root)
        os.makedirs(osp.join(self.insightface_root, 'models'), exist_ok=True)

        from insightface.app import FaceAnalysis

        # root 指向含 models/ 子目录的父路径（如 checkpoints/insightface）
        self.app = FaceAnalysis(
            name='buffalo_l',
            root=self.insightface_root,
            allowed_modules=['detection', 'recognition'],
        )
        ctx_id = 0 if self.device.type == 'cuda' else -1
        self.app.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=(640, 640))
        self.rec_model = self.app.models['recognition']

        self._stats_engine = None
        self._use_simswap_shallow_stats = use_simswap_shallow_stats
        if use_simswap_shallow_stats and simswap_root:
            self._simswap_root = osp.abspath(simswap_root)
        else:
            self._simswap_root = None

        print('[HyperSwapEngine] model=%s inputs=%s outputs=%s crop=%d' % (
            osp.basename(model_path), self._input_names, self._output_names, crop_size))

    def _read_bgr(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError('cannot read image (corrupt/missing): %s' % path)
        h, w = img.shape[:2]
        if max(h, w) < 512:
            scale = 512.0 / max(h, w)
            img = cv2.resize(
                img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        return img

    def _largest_face(self, faces):
        if not faces:
            return None
        return max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

    def _align_crop(self, bgr: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
        faces = self.app.get(bgr)
        face = self._largest_face(faces)
        if face is None:
            raise RuntimeError('face detection failed (try larger input or lower det_thresh)')
        crop, M = self._face_align.norm_crop2(bgr, face.kps, image_size=size)
        return crop, M

    def _resize_crop(self, crop_bgr: np.ndarray, size: int) -> np.ndarray:
        if crop_bgr.shape[0] == size and crop_bgr.shape[1] == size:
            return crop_bgr
        return cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_LINEAR)

    def _embedding_from_crop_bgr(self, crop_bgr: np.ndarray) -> np.ndarray:
        """buffalo_l 512-d normed embedding, shape (1, 512) float32 numpy."""
        feat = self.rec_model.get_feat(crop_bgr)
        if isinstance(feat, list):
            feat = feat[0]
        feat = np.asarray(feat, dtype=np.float32).reshape(-1)
        feat = _l2_normalize_np(feat)
        return feat.reshape(1, -1)

    def _balance_embedding(
        self,
        source_emb: np.ndarray,
        target_emb: np.ndarray,
    ) -> np.ndarray:
        """与 FaceFusion 一致：swap_weight=1 → 纯 source 身份。"""
        w = float(np.interp(self.swap_weight, [0.0, 1.0], [0.35, -0.35]))
        target_emb = _l2_normalize_np(target_emb.reshape(-1)).reshape(1, -1)
        source_emb = source_emb.reshape(1, -1).astype(np.float32)
        out = source_emb * (1.0 - w) + target_emb * w
        return _l2_normalize_np(out).reshape(1, -1)

    def _preprocess_target(self, crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.MODEL_MEAN) / self.MODEL_STD
        rgb = rgb.transpose(2, 0, 1)[None, ...]
        return rgb.astype(np.float32)

    def _postprocess_output(self, tensor: np.ndarray) -> np.ndarray:
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.shape[0] == 3:
            img = tensor.transpose(1, 2, 0)
        else:
            img = tensor
        img = img.astype(np.float32) * self.MODEL_STD + self.MODEL_MEAN
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _resolve_source_target_names(self) -> Tuple[str, str]:
        names = set(self._input_names)
        if 'source' in names and 'target' in names:
            return 'source', 'target'
        # fallback: 第一个为 target 图像，第二个为 source 向量（FaceFusion 顺序）
        return self._input_names[1], self._input_names[0]

    def _forward_swap(self, source_emb: np.ndarray, target_crop_bgr: np.ndarray) -> np.ndarray:
        target_crop_bgr = self._resize_crop(target_crop_bgr, self.NATIVE_SIZE)
        target_t = self._preprocess_target(target_crop_bgr)
        source_name, target_name = self._resolve_source_target_names()
        feed = {
            source_name: source_emb.astype(np.float32),
            target_name: target_t,
        }
        out = self.session.run(self._output_names, feed)[0]
        swapped = self._postprocess_output(out)
        return self._resize_crop(swapped, self.crop_size)

    def _latent_tensor_from_np(self, emb_np: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(emb_np.reshape(1, -1).astype(np.float32))
        return F.normalize(t, p=2, dim=1).to(self.device)

    def _latent_np_from_tensor(self, latent_id: torch.Tensor) -> np.ndarray:
        if latent_id.dim() == 1:
            latent_id = latent_id.unsqueeze(0)
        z = latent_id.detach().float().cpu().numpy().reshape(1, -1)
        return _l2_normalize_np(z).reshape(1, -1)

    def _get_stats_engine(self):
        if not self._use_simswap_shallow_stats or not self._simswap_root:
            return None
        if self._stats_engine is None:
            from scripts.simswap_engine import SimSwapEngine
            self._stats_engine = SimSwapEngine(
                self._simswap_root,
                crop_size=224,
                use_mask=False,
            )
        return self._stats_engine

    # ── SimSwapEngine 兼容 API ──

    def extract_align_crop(self, image_path: str):
        img_whole = self._read_bgr(osp.abspath(image_path))
        crop, M = self._align_crop(img_whole, self.crop_size)
        return crop, M, img_whole

    def extract_latent(self, source_path: str) -> torch.Tensor:
        crop, _, _ = self.extract_align_crop(source_path)
        emb = self._embedding_from_crop_bgr(crop)
        return self._latent_tensor_from_np(emb)

    def latent_from_crop_bgr(self, crop_bgr: np.ndarray) -> torch.Tensor:
        crop_bgr = self._resize_crop(crop_bgr, self.crop_size)
        emb = self._embedding_from_crop_bgr(crop_bgr)
        return self._latent_tensor_from_np(emb)

    def swap_latent_on_crop_bgr(self, latent_id: torch.Tensor, crop_bgr: np.ndarray) -> np.ndarray:
        crop_bgr = self._resize_crop(crop_bgr, self.crop_size)
        source_emb = self._latent_np_from_tensor(latent_id)
        target_emb = self._embedding_from_crop_bgr(crop_bgr)
        source_emb = self._balance_embedding(source_emb, target_emb)
        return self._forward_swap(source_emb, crop_bgr)

    def extract_shallow_stats(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """
        Sidecar 兼容：默认仍用 SimSwap 浅层统计（需 simswap_root）。
        未配置时退化为多尺度 RGB mean/std（896 维），仅供 smoke test。
        """
        stats_eng = self._get_stats_engine()
        if stats_eng is not None:
            crop224 = self._resize_crop(crop_bgr, 224)
            return stats_eng.extract_shallow_stats(crop224)

        parts = []
        rgb = cv2.cvtColor(self._resize_crop(crop_bgr, self.crop_size), cv2.COLOR_BGR2RGB)
        for size in (256, 128, 64):
            small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            for c in range(3):
                ch = small[:, :, c]
                parts.append(ch.mean())
                parts.append(max(ch.std(), 1e-6))
        vec = np.array(parts, dtype=np.float32)
        if vec.size < 896:
            vec = np.pad(vec, (0, 896 - vec.size))
        else:
            vec = vec[:896]
        print('[HyperSwapEngine] WARN: 使用 RGB 占位 shallow stats，正式实验请配置 simswap_root')
        return torch.from_numpy(vec)

    def load_finetune_ckpt(self, ckpt_path: str) -> None:
        """HyperSwap 阶段暂无 netG 微调权重；保留接口以兼容现有管线。"""
        if ckpt_path and osp.isfile(ckpt_path):
            print('[HyperSwapEngine] finetune ckpt ignored (hyperswap ONNX): %s' % ckpt_path)

    def swap(self, source_path: str, target_path: str, out_path: Optional[str] = None) -> np.ndarray:
        z = self.extract_latent(source_path)
        crop, _, _ = self.extract_align_crop(target_path)
        out = self.swap_latent_on_crop_bgr(z, crop)
        if out_path:
            os.makedirs(osp.dirname(out_path) or '.', exist_ok=True)
            cv2.imwrite(out_path, out)
        return out

    def swap_with_latent(
        self,
        latent_id: torch.Tensor,
        target_path: str,
        out_path: Optional[str] = None,
    ) -> np.ndarray:
        crop, _, _ = self.extract_align_crop(target_path)
        out = self.swap_latent_on_crop_bgr(latent_id, crop)
        if out_path:
            os.makedirs(osp.dirname(out_path) or '.', exist_ok=True)
            cv2.imwrite(out_path, out)
        return out
