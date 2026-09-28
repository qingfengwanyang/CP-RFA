#!/usr/bin/env python3
"""恢复 sidecar：浅层编码器统计量 + 下采样残差 + RGB 色彩对齐。"""

import dataclasses
import struct

import numpy as np
import torch
import torch.nn.functional as F

RES_GRID = 64
STAT_LAYERS = (64, 128, 256)
HEADER_FMT = '<4sHHI'  # magic, stat_dim, res_grid, res_scale_bits
HEADER_MAGIC = b'CS2\x00'
RGB_DIM = 6


@dataclasses.dataclass
class SidecarData:
    stats: torch.Tensor
    rgb_mu: torch.Tensor
    rgb_sig: torch.Tensor
    res_scale: float
    res_q: np.ndarray


def rgb_stats_from_tensor(img_t):
    """img_t: (1,3,H,W) [0,1] → mu/sig (3,)"""
    mu = img_t.mean(dim=[2, 3])[0]
    sig = img_t.std(dim=[2, 3])[0].clamp(min=1e-6)
    return mu, sig


def rgb_adain_to_ref(img_t, ref_mu, ref_sig):
    mu = img_t.mean(dim=[2, 3], keepdim=True)
    sig = img_t.std(dim=[2, 3], keepdim=True).clamp(min=1e-6)
    ref_mu = ref_mu.view(1, -1, 1, 1).to(img_t.device, img_t.dtype)
    ref_sig = ref_sig.view(1, -1, 1, 1).to(img_t.device, img_t.dtype)
    out = (img_t - mu) / sig * ref_sig + ref_mu
    return out.clamp(0.0, 1.0)


def encode_residual(orig_t, rec0_t, grid=RES_GRID):
    res = orig_t - rec0_t
    res_ds = F.interpolate(
        res, size=(grid, grid), mode='bilinear', align_corners=False)
    scale = float(res_ds.abs().max().item()) + 1e-6
    q = (res_ds / scale * 127.0).round().clamp(-128, 127).to(torch.int8)
    return q[0].cpu().numpy(), scale


def decode_residual_to_full(rec0_t, res_q, scale, grid=RES_GRID, out_size=224):
    res_ds = torch.from_numpy(res_q.astype(np.float32)).to(
        rec0_t.device, rec0_t.dtype)
    res_ds = res_ds / 127.0 * scale
    res_ds = res_ds.unsqueeze(0)
    res_up = F.interpolate(
        res_ds, size=(out_size, out_size), mode='bilinear', align_corners=False)
    return (rec0_t + res_up).clamp(0.0, 1.0)


def build_sidecar(orig_t, rec0_t, enc_stats=None, grid=RES_GRID):
    """orig_t/rec0_t: (1,3,224,224) [0,1]；enc_stats: (stat_dim,) tensor。"""
    rgb_mu, rgb_sig = rgb_stats_from_tensor(orig_t)
    res_q, scale = encode_residual(orig_t, rec0_t, grid=grid)
    if enc_stats is None:
        enc_stats = torch.zeros(sum(STAT_LAYERS) * 2)
    return SidecarData(
        stats=enc_stats.detach().float().cpu(),
        rgb_mu=rgb_mu.detach().float().cpu(),
        rgb_sig=rgb_sig.detach().float().cpu(),
        res_scale=scale,
        res_q=res_q,
    )


def _sidecar_grid(sidecar: SidecarData) -> int:
    return int(sidecar.res_q.shape[-1])


def apply_sidecar(rec0_t, sidecar: SidecarData, out_size=224, color_weight=0.35, grid=None):
    grid = grid if grid is not None else _sidecar_grid(sidecar)
    rec = decode_residual_to_full(
        rec0_t, sidecar.res_q, sidecar.res_scale,
        grid=grid, out_size=out_size)
    rec = rgb_adain_to_ref(rec, sidecar.rgb_mu, sidecar.rgb_sig)
    if color_weight < 1.0:
        base = decode_residual_to_full(
            rec0_t, sidecar.res_q, sidecar.res_scale,
            grid=grid, out_size=out_size)
        rec = (color_weight * rec + (1.0 - color_weight) * base).clamp(0.0, 1.0)
    return rec


def sidecar_to_bytes(sidecar: SidecarData) -> bytes:
    stat = sidecar.stats.numpy().astype(np.float16)
    rgb = torch.cat([sidecar.rgb_mu, sidecar.rgb_sig]).numpy().astype(np.float16)
    res_flat = sidecar.res_q.astype(np.int8).reshape(-1)
    grid = _sidecar_grid(sidecar)
    header = struct.pack(
        HEADER_FMT,
        HEADER_MAGIC,
        stat.size,
        grid,
        struct.unpack('I', struct.pack('f', sidecar.res_scale))[0],
    )
    return header + stat.tobytes() + rgb.tobytes() + res_flat.tobytes()


def sidecar_from_bytes(data: bytes) -> SidecarData:
    hsize = struct.calcsize(HEADER_FMT)
    magic, stat_dim, grid, scale_bits = struct.unpack(HEADER_FMT, data[:hsize])
    if magic != HEADER_MAGIC:
        raise ValueError('invalid sidecar magic')
    scale = struct.unpack('f', struct.pack('I', scale_bits))[0]
    off = hsize
    stat = np.frombuffer(data[off:off + stat_dim * 2], dtype=np.float16)
    off += stat_dim * 2
    rgb = np.frombuffer(data[off:off + RGB_DIM * 2], dtype=np.float16)
    off += RGB_DIM * 2
    res_n = 3 * grid * grid
    res_q = np.frombuffer(data[off:off + res_n], dtype=np.int8).reshape(3, grid, grid)
    return SidecarData(
        stats=torch.from_numpy(stat.astype(np.float32)),
        rgb_mu=torch.from_numpy(rgb[:3].astype(np.float32)),
        rgb_sig=torch.from_numpy(rgb[3:].astype(np.float32)),
        res_scale=float(scale),
        res_q=res_q.copy(),
    )
