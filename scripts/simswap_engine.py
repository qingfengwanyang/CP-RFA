#!/usr/bin/env python3
"""SimSwap 推理封装（需在 SimSwap 根目录相对路径下初始化）。"""

import os
import os.path as osp
import sys

import cv2
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)


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


class SimSwapEngine:
    """整图换脸：source 身份 → target 姿态/背景。"""

    def __init__(self, simswap_root, crop_size=224, use_mask=True, name='people'):
        self.root = osp.abspath(simswap_root)
        self.crop_size = crop_size
        self.use_mask = use_mask
        self._orig_cwd = os.getcwd()

        saved = {k: sys.modules[k] for k in list(sys.modules)
                 if k == 'models' or k.startswith('models.')}
        for k in saved:
            del sys.modules[k]
        if self.root not in sys.path:
            sys.path.insert(0, self.root)
        os.chdir(self.root)

        from options.test_options import TestOptions
        from models.models import create_model
        from insightface_func.face_detect_crop_single import Face_detect_crop
        from parsing_model.model import BiSeNet
        from util.reverse2original import reverse2wholeimage
        from util.add_watermark import watermark_image
        from util.norm import SpecificNorm

        argv = [
            'simswap_engine',
            '--name', name,
            '--Arc_path', 'arcface_model/arcface_checkpoint.tar',
            '--crop_size', str(crop_size),
        ]
        if use_mask:
            argv.append('--use_mask')
        sys.argv = argv
        opt = TestOptions().parse()
        if crop_size == 512:
            opt.which_epoch = 550000
            opt.name = '512'
            mode = 'ffhq'
        else:
            mode = 'None'

        self.model = create_model(opt)
        self.model.eval()
        self.device = _resolve_torch_device()
        if hasattr(self.model, 'netG'):
            self.model.netG.to(self.device)
        if hasattr(self.model, 'netArc'):
            self.model.netArc.to(self.device)
        self.reverse2wholeimage = reverse2wholeimage
        self.sp_norm = SpecificNorm()
        self.logoclass = watermark_image('./simswaplogo/simswaplogo.png')

        self.app = Face_detect_crop(name='antelope', root='./insightface_func/models')
        self.app.prepare(ctx_id=0, det_thresh=0.2, det_size=(640, 640), mode=mode)
        # Newer insightface scrfd detect() dropped `threshold`; SimSwap still passes it.
        det_model = self.app.det_model
        det_size = self.app.det_size
        _orig_detect = det_model.detect

        def _detect_compat(img, threshold=None, max_num=0, metric='default',
                           input_size=None, **kwargs):
            return _orig_detect(
                img,
                input_size=input_size or det_size,
                max_num=max_num,
                metric=metric,
            )

        det_model.detect = _detect_compat

        self.transformer_arc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.parsing_net = None
        if use_mask:
            net = BiSeNet(n_classes=19)
            net.to(self.device)
            pth = osp.join(self.root, 'parsing_model/checkpoint/79999_iter.pth')
            net.load_state_dict(torch.load(pth, map_location='cpu'))
            net.eval()
            self.parsing_net = net

        os.chdir(self._orig_cwd)
        sys.modules.update(saved)

    def _read_bgr(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError('cannot read image: %s' % path)
        h, w = img.shape[:2]
        if max(h, w) < 512:
            scale = 512.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        return img

    def _get_face(self, bgr):
        out = self.app.get(bgr, self.crop_size)
        if out is None:
            raise RuntimeError('face detection failed (try lower det_thresh or larger input)')
        return out

    def _run_swap_context(self):
        saved = {k: sys.modules[k] for k in list(sys.modules)
                 if k == 'models' or k.startswith('models.')}
        for k in saved:
            del sys.modules[k]
        os.chdir(self.root)
        if self.root not in sys.path:
            sys.path.insert(0, self.root)
        return saved

    def _restore_swap_context(self, saved):
        os.chdir(self._orig_cwd)
        sys.modules.update(saved)

    def _latent_from_align_crop(self, align_crop_bgr):
        """ArcFace 512-d latent from BGR aligned crop."""
        img_pil = Image.fromarray(cv2.cvtColor(align_crop_bgr, cv2.COLOR_BGR2RGB))
        img_id = self.transformer_arc(img_pil).view(1, 3, self.crop_size, self.crop_size).to(self.device)
        img_id_down = F.interpolate(img_id, size=(112, 112))
        latent_id = self.model.netArc(img_id_down)
        return F.normalize(latent_id, p=2, dim=1)

    def extract_align_crop(self, image_path):
        """整图路径 → (align_crop_bgr, affine_mat, whole_bgr)。"""
        saved = self._run_swap_context()
        try:
            img_whole = self._read_bgr(osp.abspath(image_path))
            align_list, mat_list = self._get_face(img_whole)
            return align_list[0], mat_list[0], img_whole
        finally:
            self._restore_swap_context(saved)

    def extract_latent(self, source_path):
        """从图像提取 ArcFace 身份向量 (1, 512)，已 L2 归一化。"""
        saved = self._run_swap_context()
        try:
            with torch.no_grad():
                img_whole = self._read_bgr(osp.abspath(source_path))
                align_crop, _ = self._get_face(img_whole)
                return self._latent_from_align_crop(align_crop[0])
        finally:
            self._restore_swap_context(saved)

    def _normalize_swap_output(self, out):
        """fs_model eval 返回 (B,3,H,W)；train 返回 [losses, img_fake]。"""
        if isinstance(out, (list, tuple)):
            out = out[-1]
        if not torch.is_tensor(out):
            out = torch.as_tensor(out)
        return out

    def _tensor_to_bgr_crop(self, swap_tensor):
        swap_tensor = self._normalize_swap_output(swap_tensor)
        if swap_tensor.dim() == 4:
            swap_tensor = swap_tensor[0]
        out = swap_tensor.permute(1, 2, 0).detach().cpu().numpy()
        out = (out * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    def swap_latent_on_crop_bgr(self, latent_id, crop_bgr):
        """在已对齐的 224 BGR crop 上做身份 swap，返回 swap 后 BGR crop。"""
        saved = self._run_swap_context()
        try:
            with torch.no_grad():
                if latent_id.dim() == 1:
                    latent_id = latent_id.unsqueeze(0)
                latent_id = F.normalize(latent_id.float().to(self.device), p=2, dim=1)
                b_t = _totensor(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))[None, ...].to(self.device)
                swap = self.model(None, b_t, latent_id, None, True)
                return self._tensor_to_bgr_crop(swap)
        finally:
            self._restore_swap_context(saved)

    def latent_from_crop_bgr(self, crop_bgr):
        """从对齐 crop BGR 提取 ArcFace latent (1,512)。"""
        saved = self._run_swap_context()
        try:
            with torch.no_grad():
                return self._latent_from_align_crop(crop_bgr)
        finally:
            self._restore_swap_context(saved)

    def extract_shallow_stats(self, crop_bgr):
        """SimSwap 编码器浅层 channel mean/std → (896,) float32 CPU。"""
        saved = self._run_swap_context()
        try:
            with torch.no_grad():
                b_t = _totensor(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))[None, ...].to(self.device)
                g = self.model.netG
                skip1 = g.first_layer(b_t)
                skip2 = g.down1(skip1)
                skip3 = g.down2(skip2)
                parts = []
                for feat in (skip1, skip2, skip3):
                    parts.append(feat.mean(dim=[2, 3]).squeeze(0))
                    parts.append(feat.std(dim=[2, 3]).squeeze(0).clamp(min=1e-6))
                return torch.cat(parts).float().cpu()
        finally:
            self._restore_swap_context(saved)

    def _swap_latent_to_array(self, latent_id, target_path):
        """latent_id (1,512) + target 整图 → BGR 结果。"""
        saved = self._run_swap_context()
        try:
            with torch.no_grad():
                if latent_id.dim() == 1:
                    latent_id = latent_id.unsqueeze(0)
                latent_id = F.normalize(latent_id.float().to(self.device), p=2, dim=1)

                img_b_whole = self._read_bgr(osp.abspath(target_path))
                align_list, mat_list = self._get_face(img_b_whole)
                swap_results = []
                crop_tensors = []
                for b_crop in align_list:
                    b_t = _totensor(cv2.cvtColor(b_crop, cv2.COLOR_BGR2RGB))[None, ...].to(self.device)
                    swap = self._normalize_swap_output(
                        self.model(None, b_t, latent_id, None, True))
                    if swap.dim() == 4:
                        swap = swap[0]
                    swap_results.append(swap)
                    crop_tensors.append(b_t)

                out_path = osp.join(self.root, 'output/_simswap_temp_out.jpg')
                os.makedirs(osp.dirname(out_path), exist_ok=True)
                self.reverse2wholeimage(
                    crop_tensors, swap_results, mat_list, self.crop_size, img_b_whole,
                    self.logoclass, out_path, True,
                    pasring_model=self.parsing_net, use_mask=self.use_mask, norm=self.sp_norm)

                result = cv2.imread(out_path)
        finally:
            self._restore_swap_context(saved)
        if result is None:
            raise RuntimeError('SimSwap latent swap failed for target %s' % target_path)
        return result

    def _swap_to_array(self, source_path, target_path):
        latent_id = self.extract_latent(source_path)
        return self._swap_latent_to_array(latent_id, target_path)

    def swap(self, source_path, target_path, out_path=None):
        """source=身份来源图, target=保留姿态/背景的目标图。返回 BGR uint8。"""
        source_path = osp.abspath(source_path)
        target_path = osp.abspath(target_path)
        if out_path:
            out_path = osp.abspath(out_path)
        bgr = self._swap_to_array(source_path, target_path)
        if out_path:
            os.makedirs(osp.dirname(out_path) or '.', exist_ok=True)
            cv2.imwrite(out_path, bgr)
        return bgr

    def swap_with_latent(self, latent_id, target_path, out_path=None):
        """身份向量 + target 姿态/背景 → 换脸结果。latent_id: (512,) or (1,512) ArcFace。"""
        target_path = osp.abspath(target_path)
        if out_path:
            out_path = osp.abspath(out_path)
        bgr = self._swap_latent_to_array(latent_id, target_path)
        if out_path:
            os.makedirs(osp.dirname(out_path) or '.', exist_ok=True)
            cv2.imwrite(out_path, bgr)
        return bgr

    def load_finetune_ckpt(self, ckpt_path):
        """加载 ChaosSimSwap 微调权重（netG 部分块 + 可选 mapper 在外部）。"""
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'netG_partial' not in ckpt:
            return
        saved = self._run_swap_context()
        try:
            self.model.netG.load_state_dict(ckpt['netG_partial'], strict=False)
            print('[SimSwapEngine] loaded finetune netG from %s' % ckpt_path)
        finally:
            self._restore_swap_context(saved)
