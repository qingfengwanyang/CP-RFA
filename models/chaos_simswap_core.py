#!/usr/bin/env python3
"""可微 SimSwap G + ArcFace（用于 ChaosSimSwap 微调）。"""

import os
import os.path as osp
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimSwapTrainCore(nn.Module):
    def __init__(self, simswap_root, unfreeze_g_blocks=3):
        super().__init__()
        self.simswap_root = osp.abspath(simswap_root)
        self._load_modules(unfreeze_g_blocks)

    def _load_modules(self, unfreeze_g_blocks):
        saved = {k: sys.modules[k] for k in list(sys.modules)
                 if k == 'models' or k.startswith('models.')}
        for k in saved:
            del sys.modules[k]
        if self.simswap_root not in sys.path:
            sys.path.insert(0, self.simswap_root)

        from models.fs_networks import Generator_Adain_Upsample
        from models.fs_model import SpecificNorm

        self.netG = Generator_Adain_Upsample(
            input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False)
        g_path = osp.join(self.simswap_root, 'checkpoints/people/latest_net_G.pth')
        self.netG.load_state_dict(torch.load(g_path, map_location='cpu'), strict=True)

        arc_path = osp.join(self.simswap_root, 'arcface_model/arcface_checkpoint.tar')
        self.netArc = torch.load(arc_path, map_location='cpu')
        self.spNorm = SpecificNorm()

        sys.modules.update(saved)

        self.netArc.eval()
        for p in self.netArc.parameters():
            p.requires_grad = False

        for p in self.netG.parameters():
            p.requires_grad = False
        blocks = list(self.netG.BottleNeck.children())
        for block in blocks[-unfreeze_g_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

    def img_att_from_normed(self, img_normed):
        """[-1,1] BCHW → [0,1] 224 SimSwap 输入。"""
        x = F.interpolate(img_normed, size=(224, 224), mode='bilinear', align_corners=False)
        return (x + 1.0) * 0.5

    def latent_from_att(self, img_att):
        """img_att [0,1] → ArcFace 512-d。"""
        img_down = F.interpolate(img_att, size=(112, 112), mode='bilinear', align_corners=False)
        img_down = self.spNorm(img_down)
        z = self.netArc(img_down)
        return F.normalize(z, p=2, dim=1)

    def swap(self, img_att, latent_id):
        if latent_id.dim() == 1:
            latent_id = latent_id.unsqueeze(0)
        latent_id = F.normalize(latent_id.float(), p=2, dim=1)
        return self.netG(img_att, latent_id)

    def trainable_parameters(self):
        return [p for p in self.netG.parameters() if p.requires_grad]
