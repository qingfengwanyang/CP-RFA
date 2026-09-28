#!/usr/bin/env python3
"""Register and recover one face with CP-RFA (HyperSwap backend)."""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import cv2
import torch

ROOT = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, ROOT)

from models.chaos_simswap_pipeline import CPRFAConfig, CPRFAPipeline
from utils.password import generate_code


def parse_args():
    p = argparse.ArgumentParser(description='CP-RFA demo: anonymize and recover one face')
    p.add_argument('--image', required=True, help='Input image with one visible face')
    p.add_argument('--out', default='runs/demo', help='Output directory')
    p.add_argument(
        '--mapper_ckpt',
        default=osp.join(ROOT, 'checkpoints/mapper_best.pth.tar'),
    )
    p.add_argument(
        '--hyperswap_model',
        default=os.environ.get(
            'HYPERSWAP_MODEL',
            osp.join(ROOT, 'checkpoints/hyperswap/hyperswap_1a_256.onnx'),
        ),
    )
    p.add_argument(
        '--insightface_root',
        default=os.environ.get(
            'INSIGHTFACE_ROOT',
            osp.join(ROOT, 'checkpoints/insightface'),
        ),
    )
    p.add_argument('--seed', type=int, default=0, help='Password RNG seed')
    return p.parse_args()


def main():
    args = parse_args()
    if not osp.isfile(args.hyperswap_model):
        raise FileNotFoundError(
            'HyperSwap ONNX not found: %s\nSee README.md (Weights).' % args.hyperswap_model
        )

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    cfg = CPRFAConfig(
        mapper_ckpt=args.mapper_ckpt,
        swap_backend='hyperswap',
        hyperswap_model=args.hyperswap_model,
        insightface_root=args.insightface_root,
        crop_size=256,
        use_simswap_shallow_stats=False,
        token_version='v3',
    )
    pipe = CPRFAPipeline(cfg, device=device)

    password, _, _, _ = generate_code(
        passwd_length=16, batch_size=1, device=device,
        inv=False, use_minus_one=True, gen_random_WR=False,
    )

    reg = pipe.register_from_path(args.image, password, slug='demo')
    rec = pipe.recover_from_token(
        reg.anon_crop_bgr, reg.token, password, z_orig_ref=reg.z_orig)

    cv2.imwrite(osp.join(args.out, 'anon.png'), reg.anon_crop_bgr)
    cv2.imwrite(osp.join(args.out, 'rec0.png'), rec.rec0_crop_bgr)
    cv2.imwrite(osp.join(args.out, 'rec.png'), rec.rec_crop_bgr)
    with open(osp.join(args.out, 'recovery.token'), 'w') as f:
        f.write(reg.token)

    print('token: %s' % reg.token.split('|', 1)[0])
    print('anon ID (lower is more anonymous): %.4f' % reg.id_anon)
    print('authorized rec0 ID (mapper bypassed): %.4f' % rec.id_rec0)
    print('authorized rec ID (Sidecar, higher is better): %.4f' % rec.id_rec)
    print('HSIO decrypt MAE: %.2e' % reg.decrypt_mae)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
