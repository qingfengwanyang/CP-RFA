#!/usr/bin/env python3
"""HyperSwap 换脸结果 embedding 代理：z_swap -> 预测 anon ArcFace embedding。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwapEmbeddingSurrogate(nn.Module):
    """MLP 近似 f(z_swap) ≈ z_a（buffalo_l 512-d，L2 归一化）。"""

    def __init__(self, dim=512, hidden=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, z_swap):
        if z_swap.dim() == 1:
            z_swap = z_swap.unsqueeze(0)
        return F.normalize(self.net(z_swap), p=2, dim=1)
