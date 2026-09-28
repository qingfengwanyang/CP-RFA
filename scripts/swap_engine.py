#!/usr/bin/env python3
"""统一换脸引擎工厂：simswap | hyperswap。"""

from __future__ import annotations

import os
import os.path as osp
from typing import Literal, Optional, Union

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))

SwapBackend = Literal['simswap', 'hyperswap']


def default_hyperswap_model_path() -> str:
    return osp.join(
        ROOT,
        'checkpoints',
        'hyperswap',
        os.environ.get('HYPERSWAP_MODEL', 'hyperswap_1a_256.onnx'),
    )


def create_swap_engine(
    backend: SwapBackend = 'simswap',
    *,
    simswap_root: Optional[str] = None,
    hyperswap_model: Optional[str] = None,
    insightface_root: Optional[str] = None,
    crop_size: int = 224,
    use_mask: bool = True,
    swap_weight: float = 1.0,
    use_simswap_shallow_stats: bool = True,
):
    """
    创建与 ChaosSimSwapPipeline 兼容的换脸引擎。

    backend='hyperswap' 时建议 crop_size=256。
    """
    if backend == 'simswap':
        from scripts.simswap_engine import SimSwapEngine
        if simswap_root is None:
            simswap_root = os.environ.get('SIMSWAP_ROOT', '')
        if not simswap_root:
            raise ValueError('Set SIMSWAP_ROOT for backend=simswap')
        return SimSwapEngine(
            simswap_root,
            crop_size=crop_size,
            use_mask=use_mask,
        )

    if backend == 'hyperswap':
        from scripts.hyperswap_engine import HyperSwapEngine
        if hyperswap_model is None:
            hyperswap_model = default_hyperswap_model_path()
        if insightface_root is None:
            insightface_root = os.environ.get(
                'INSIGHTFACE_ROOT',
                osp.join(ROOT, 'checkpoints', 'insightface'),
            )
        if simswap_root is None:
            simswap_root = os.environ.get('SIMSWAP_ROOT', '')
        native_crop = crop_size if crop_size >= 256 else 256
        return HyperSwapEngine(
            model_path=hyperswap_model,
            insightface_root=insightface_root,
            crop_size=native_crop,
            swap_weight=swap_weight,
            simswap_root=simswap_root if use_simswap_shallow_stats else None,
            use_simswap_shallow_stats=use_simswap_shallow_stats,
        )

    raise ValueError('unknown swap backend: %s' % backend)
