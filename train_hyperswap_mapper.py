#!/usr/bin/env python3
"""
HyperSwap mapper v3：代理网络 + 图像级匿名监督 + LFW 验证选 best epoch。

HyperSwap ONNX 不可微 → 在线训练 SwapEmbeddingSurrogate(z_swap)≈z_a，
对 surrogate 输出施加 loss_away_img / loss_div_img（梯度经 surrogate 回 mapper）。
"""

import json
import os
import os.path as osp
import subprocess
import sys

import torch
torch.backends.cudnn.enabled = False
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import datasets
from models.hyperswap_train_core import HyperSwapTrainCore
from models.chaos_simswap_mapper import PasswordLatentMapper
from models.format_preserving_encryptor import FormatPreservingEncryptor
from models.swap_embedding_surrogate import SwapEmbeddingSurrogate
from utils.password import generate_code

ROOT = osp.dirname(osp.abspath(__file__))
PY = sys.executable


class TrainConfig:
    data_root = os.environ.get('CHAOS_DATA_ROOT', '')
    hyperswap_model = os.environ.get(
        'HYPERSWAP_MODEL',
        osp.join(ROOT, 'checkpoints/hyperswap/hyperswap_1a_256.onnx'))
    insightface_root = os.environ.get(
        'INSIGHTFACE_ROOT',
        osp.join(ROOT, 'checkpoints/insightface'))
    simswap_root = os.environ.get('SIMSWAP_ROOT', '')
    init_mapper_ckpt = os.environ.get('HYPERSWAP_INIT_MAPPER', '')

    batch_size = 4
    workers = 4
    image_size = 128
    password_dim = 16

    epochs = 5
    max_steps_per_epoch = 1000
    lr_mapper = 5e-5
    lr_surrogate = 2e-3
    sur_steps_per_batch = 2

    lambda_align = 0.0
    lambda_away = 6.0
    lambda_cross = 8.0
    lambda_pwd_div = 18.0
    lambda_away_img = 12.0
    lambda_div_img = 15.0
    pwd_div_margin = 0.15

    ckpt_dir = 'checkpoints/chaos_hyperswap_mapper_v3'


def build_loader(cfg):
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    args = type('obj', (object,), {
        'evaluate': False, 'during_training': True,
        'data_root': cfg.data_root, 'test_size': cfg.batch_size,
    })()
    ds = datasets.CASIA(train=True, transform=transform, args=args)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                      num_workers=cfg.workers, pin_memory=True)


def _shuffle_negatives(z_orig, labels):
    B = z_orig.size(0)
    perm = torch.arange(B, device=z_orig.device)
    for _ in range(8):
        perm = perm[torch.randperm(B, device=z_orig.device)]
        if (labels[perm] != labels).any():
            break
    z_neg = z_orig[perm]
    same = labels == labels[perm]
    if same.any():
        z_neg = z_neg.clone()
        z_neg[same] = z_orig[(perm[same] + 1) % B]
    return z_neg


def _update_surrogate(surrogate, opt_sur, z_swap1, z_swap2, z_a1, z_a2, steps):
    surrogate.train()
    z_all = torch.cat([z_swap1, z_swap2], dim=0).detach()
    t_all = torch.cat([z_a1, z_a2], dim=0).detach()
    for _ in range(steps):
        pred = surrogate(z_all)
        loss = (1.0 - F.cosine_similarity(pred, t_all, dim=1)).mean()
        opt_sur.zero_grad()
        loss.backward()
        opt_sur.step()
    return loss.item()


def train_epoch(core, mapper, surrogate, fpe, loader, opt_m, opt_s, cfg, epoch, device):
    mapper.train()
    stats = {
        'loss': [], 'away': [], 'away_img': [], 'div': [], 'div_img': [], 'sur': [],
    }
    pbar = tqdm(loader, desc='hsv3-ep%d' % (epoch + 1), total=cfg.max_steps_per_epoch)

    for step, (img, labels, _, _) in enumerate(pbar):
        if step >= cfg.max_steps_per_epoch:
            break
        img = img.to(device)
        labels = labels.to(device)
        B = img.size(0)

        with torch.no_grad():
            z_orig = core.latent_from_normed_batch(img)

        z_p1, _, z_p2, _ = generate_code(
            cfg.password_dim, B, device, inv=False,
            use_minus_one='half', gen_random_WR=False)

        z_enc1, _ = fpe.encrypt(z_orig, z_p1)
        z_enc2, _ = fpe.encrypt(z_orig, z_p2)
        z_swap1 = mapper(z_enc1, z_p1)
        z_swap2 = mapper(z_enc2, z_p2)

        with torch.no_grad():
            anon1_list = core.swap_normed_batch(img, z_swap1)
            anon2_list = core.swap_normed_batch(img, z_swap2)
            z_a1 = core.latent_from_bgr_crops(anon1_list)
            z_a2 = core.latent_from_bgr_crops(anon2_list)

        sur_loss = _update_surrogate(
            surrogate, opt_s, z_swap1, z_swap2, z_a1, z_a2, cfg.sur_steps_per_batch)

        pred1 = surrogate(z_swap1)
        pred2 = surrogate(z_swap2)

        loss_align = torch.tensor(0.0, device=device)
        if cfg.lambda_align > 0:
            loss_align = (
                (1 - F.cosine_similarity(z_swap1, z_a1, dim=1)).mean()
                + (1 - F.cosine_similarity(z_swap2, z_a2, dim=1)).mean()
            ) * cfg.lambda_align

        loss_away = (
            F.cosine_similarity(z_swap1, z_orig, dim=1).mean()
            + F.cosine_similarity(z_swap2, z_orig, dim=1).mean()
        ) * cfg.lambda_away

        z_neg = _shuffle_negatives(z_orig, labels)
        loss_cross = (
            (1 - F.cosine_similarity(z_swap1, z_neg, dim=1)).mean()
            + (1 - F.cosine_similarity(z_swap2, z_neg, dim=1)).mean()
        ) * cfg.lambda_cross

        cos_cross = F.cosine_similarity(z_swap1, z_swap2, dim=1)
        loss_div = F.relu(cos_cross - cfg.pwd_div_margin).mean() * cfg.lambda_pwd_div

        loss_away_img = (
            F.cosine_similarity(pred1, z_orig, dim=1).mean()
            + F.cosine_similarity(pred2, z_orig, dim=1).mean()
        ) * cfg.lambda_away_img

        cos_pred = F.cosine_similarity(pred1, pred2, dim=1)
        loss_div_img = F.relu(cos_pred - cfg.pwd_div_margin).mean() * cfg.lambda_div_img

        loss = (loss_align + loss_away + loss_cross + loss_div
                + loss_away_img + loss_div_img)
        opt_m.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mapper.parameters(), 5.0)
        opt_m.step()

        away_img = (
            F.cosine_similarity(z_a1, z_orig, dim=1).mean()
            + F.cosine_similarity(z_a2, z_orig, dim=1).mean()
        ) * 0.5
        stats['loss'].append(loss.item())
        stats['away'].append(loss_away.item() / cfg.lambda_away)
        stats['away_img'].append(away_img.item())
        stats['div'].append(cos_cross.mean().item())
        stats['div_img'].append(cos_pred.mean().item())
        stats['sur'].append(sur_loss)
        pbar.set_postfix({
            'L': '%.2f' % loss.item(),
            'aimg': '%.3f' % away_img.item(),
            'pdiv': '%.3f' % cos_pred.mean().item(),
            'sur': '%.3f' % sur_loss,
        })

    return {k: sum(v) / max(len(v), 1) for k, v in stats.items()}


def save_ckpt(mapper, cfg, epoch, stats, tag=None):
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    name = tag or ('checkpoint_ep%d.pth.tar' % (epoch + 1))
    path = osp.join(cfg.ckpt_dir, name)
    torch.save({
        'epoch': epoch,
        'mapper': mapper.state_dict(),
        'stats': stats,
        'variant': 'chaos_hyperswap_mapper_v3',
        'swap_backend': 'hyperswap',
        'identity_encoder': 'buffalo_l',
    }, path)
    print('  saved %s' % path)
    return path


def eval_lfw_score(ckpt_path, cfg):
    cmd = [
        PY, osp.join(ROOT, 'scripts', 'mapper_lfw_score.py'),
        '--ckpt', ckpt_path,
        '--hyperswap_model', cfg.hyperswap_model,
        '--insightface_root', cfg.insightface_root,
        '--simswap_root', cfg.simswap_root,
        '--data_root', cfg.data_root,
    ]
    env = os.environ.copy()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or '')[-800:]
        print('  LFW eval failed:', err)
        return None
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)


def _release_core(core):
    """释放 HyperSwap ONNX 占用的 GPU 显存，供 LFW 子进程使用。"""
    if core is not None:
        eng = getattr(core, '_engine', None)
        if eng is not None:
            if hasattr(eng, 'session'):
                del eng.session
            if hasattr(eng, 'app'):
                del eng.app
        del core
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_core(cfg, device):
    return HyperSwapTrainCore(
        hyperswap_model=cfg.hyperswap_model,
        insightface_root=cfg.insightface_root,
        simswap_root=cfg.simswap_root,
        train_image_size=cfg.image_size,
    )


def main():
    cfg = TrainConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    core = _build_core(cfg, device)
    mapper = PasswordLatentMapper().to(device)
    if cfg.init_mapper_ckpt and osp.isfile(cfg.init_mapper_ckpt):
        ckpt = torch.load(cfg.init_mapper_ckpt, map_location='cpu')
        if 'mapper' in ckpt:
            mapper.load_state_dict(ckpt['mapper'], strict=True)
            print('[train] warm-start mapper from %s' % cfg.init_mapper_ckpt)

    surrogate = SwapEmbeddingSurrogate().to(device)
    fpe = FormatPreservingEncryptor(
        identity_dim=512, password_dim=16, num_rounds=4).to(device)
    for p in fpe.parameters():
        p.requires_grad = False

    opt_m = torch.optim.Adam(mapper.parameters(), lr=cfg.lr_mapper, betas=(0.5, 0.999))
    opt_s = torch.optim.Adam(surrogate.parameters(), lr=cfg.lr_surrogate, betas=(0.5, 0.999))
    loader = build_loader(cfg)

    print('HyperSwap mapper v3: device=%s epochs=%d steps/ep=%d' % (
        device, cfg.epochs, cfg.max_steps_per_epoch))

    best_score = 1e9
    best_path = None
    eval_log = []

    for ep in range(cfg.epochs):
        stats = train_epoch(core, mapper, surrogate, fpe, loader, opt_m, opt_s, cfg, ep, device)
        print('ep%d train stats: %s' % (ep + 1, stats))
        ckpt_path = save_ckpt(mapper, cfg, ep, stats)

        print('  running LFW validation...')
        _release_core(core)
        metrics = eval_lfw_score(ckpt_path, cfg)
        core = _build_core(cfg, device)
        if metrics:
            print('  LFW: id_anon=%.4f pwd_cross=%.4f id_rec=%.4f score=%.4f' % (
                metrics['id_anon'], metrics['pwd_cross'],
                metrics['id_rec'], metrics['score']))
            eval_log.append({'epoch': ep + 1, 'ckpt': ckpt_path, **metrics})
            if metrics['score'] < best_score:
                best_score = metrics['score']
                best_path = save_ckpt(
                    mapper, cfg, ep, stats, tag='checkpoint_best.pth.tar')
                with open(osp.join(cfg.ckpt_dir, 'best_lfw.json'), 'w') as f:
                    json.dump(metrics, f, indent=2)

    with open(osp.join(cfg.ckpt_dir, 'eval_log.json'), 'w') as f:
        json.dump(eval_log, f, indent=2)

    print('\n=== done ===')
    print('best LFW score=%.4f ckpt=%s' % (best_score, best_path))
    print('eval log: %s' % osp.join(cfg.ckpt_dir, 'eval_log.json'))


if __name__ == '__main__':
    main()
