"""从路径列表 + landmarks JSON 加载图像（外部测试集）。"""

import json
import os.path as osp

import torch
import torch.utils.data as data
from PIL import Image

__all__ = ['ImageList']


class ImageList(data.Dataset):
    def __init__(self, train, transform, args):
        self.transform = transform
        self.image_root = getattr(args, 'image_root', '') or ''
        with open(args.val_list) as f:
            self.paths = [line.strip() for line in f if line.strip()]
        lm_path = getattr(args, 'landmarks_json', None)
        if not lm_path:
            raise ValueError('ImageList 需要 args.landmarks_json')
        with open(lm_path) as f:
            lm_data = json.load(f)
        self.landmarks = {}
        for p in self.paths:
            key = p.replace('\\', '/')
            if key not in lm_data:
                alt = osp.basename(key)
                if alt not in lm_data:
                    raise KeyError('missing landmarks for %s' % p)
                key = alt
            self.landmarks[p] = lm_data[key]

    def __len__(self):
        return len(self.paths)

    def _resolve(self, rel):
        if osp.isabs(rel):
            return rel
        if self.image_root:
            return osp.join(self.image_root, rel)
        return rel

    def __getitem__(self, index):
        rel = self.paths[index]
        img_path = self._resolve(rel)
        pts = self.landmarks[rel]
        landmarks = torch.tensor(pts, dtype=torch.float32)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, 0, landmarks, img_path

    def __repr__(self):
        return 'ImageList(n=%d, root=%s)' % (len(self), self.image_root)
