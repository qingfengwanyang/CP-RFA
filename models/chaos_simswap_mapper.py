#!/usr/bin/env python3
"""密码条件身份映射：强化不同 password → 不同 SimSwap latent。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PasswordLatentMapper(nn.Module):
    """z_swap = normalize(z_enc + scale * MLP([z_enc, password]))"""

    def __init__(self, identity_dim=512, password_dim=16, init_scale=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(identity_dim + password_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, identity_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, z, password):
        if password.dim() == 1:
            password = password.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        delta = self.net(torch.cat([z, password], dim=1))
        return F.normalize(z + self.scale * torch.tanh(delta), p=2, dim=1)
