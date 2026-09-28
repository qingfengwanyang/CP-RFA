#!/usr/bin/env python3
"""
V6 损失函数模块 - 严格约束版

核心设计（相对V5的改进）：
1. 冻结身份提取器(SphereFace)：eval模式 + requires_grad=False，防止反向洗脑
2. 背景L1约束：只对背景区域计算，严格保留背景
3. 属性约束(Attribute Loss)：VGG特征约束表情/姿态一致性
4. 身份Loss：匿名身份逼近加密身份（FPE加密后的目标）
5. GAN损失：全局 + 局部判别器

V5失败教训：
- gamma被训练到0.013，AdaIN注入被关闭
- V6已通过"硬替换"消除了gamma偷懒可能
- 但仍需严格约束防止其他偷懒行为

作者: FIT改进 V6
日期: 2026-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGGFeatureExtractor(nn.Module):
    """
    VGG特征提取器 - 用于属性约束
    
    提取多层特征，约束匿名图像与原图在属性（表情、姿态）上的一致性
    防止生成"畸形脸"或"面无表情的标准脸"
    """
    
    def __init__(self, layers=['conv1_2', 'conv2_2', 'conv3_3'], pretrained=True):
        super().__init__()
        
        vgg = models.vgg19(pretrained=pretrained).features
        
        self.layer_mapping = {
            'conv1_2': 3,
            'conv2_2': 8,
            'conv3_3': 17,
            'conv4_3': 26,
            'conv5_3': 35
        }
        
        self.layers = layers
        self.model = vgg
        
        # 冻结VGG参数！
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        参数:
            x: (B, 3, H, W) 输入图像，范围[-1, 1]
        
        返回:
            features: dict of {layer_name: feature_tensor}
        """
        # 确保模型在正确设备上
        if x.is_cuda and not next(self.model.parameters()).is_cuda:
            self.model = self.model.to(x.device)
        
        # VGG期望输入范围[0, 1]
        x = (x + 1) / 2.0
        x = x[:, [2, 1, 0], :, :]  # RGB -> BGR
        x = x * 255.0
        
        # VGG标准化
        mean = torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1).to(x.device)
        x = x - mean
        
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            for layer_name, idx in self.layer_mapping.items():
                if int(name) == idx and layer_name in self.layers:
                    features[layer_name] = x
        
        return features


class AttributeLoss(nn.Module):
    """
    属性约束损失（V6.1版 - 步骤三+四修复）
    
    关键修复：
    1. 只使用浅层VGG特征（conv1_2, conv2_2），不用深层！
       深层特征包含长相信息，会引发身份泄露
    2. 使用 .mean() 而非 .sum()，避免Loss量级爆炸
    3. 动态权重缩放，强行对齐到与IdentityLoss同量级
    """
    
    def __init__(self, layers=None, lambda_attr=1.0, target_scale=10.0):
        """
        参数:
            layers: VGG层列表，只使用浅层！默认['conv1_2', 'conv2_2']
            lambda_attr: 基础权重
            target_scale: 目标量级（与IdentityLoss对齐，约10~50）
        """
        super().__init__()
        
        # 步骤四：只用浅层VGG特征！
        # conv1_2: 纹理/边缘（安全）
        # conv2_2: 局部模式/颜色（安全）
        # conv3_3+: 包含长相信息（危险！会泄露身份）
        if layers is None:
            layers = ['conv1_2', 'conv2_2']
        
        self.vgg = VGGFeatureExtractor(layers=layers)
        self.lambda_attr = lambda_attr
        self.target_scale = target_scale
        self._running_scale = 1.0  # 动态缩放系数
        self._scale_initialized = False
    
    def forward(self, anon_img, orig_img, face_mask):
        """
        V7.6修改：attr_loss不再乘face_mask！
        之前face_mask让attr约束只对面部区域生效，但这和identity_loss冲突：
        - identity_loss要改脸 → attr_loss不让脸变 → 矛盾！
        - 结果：生成器学会了加紫边伪影（face_mask边缘处约束最弱）
        
        现在改为：整体图像浅层VGG特征匹配，不过分约束face区域
        浅层VGG（conv1_2, conv2_2）主要捕获纹理/颜色，不会泄露身份
        """
        # 确保VGG在正确设备上
        if anon_img.is_cuda:
            self.vgg = self.vgg.to(anon_img.device)
        
        # 提取特征
        anon_features = self.vgg(anon_img)
        orig_features = self.vgg(orig_img)
        
        loss = 0.0
        num_layers = 0
        for i, layer_name in enumerate(anon_features.keys()):
            anon_feat = anon_features[layer_name]
            orig_feat = orig_features[layer_name]
            
            # 整体特征匹配（不乘face_mask）
            # 浅层VGG只捕获纹理/颜色，不会约束身份
            diff = (anon_feat - orig_feat).pow(2).mean()
            loss += diff
            num_layers += 1
        
        if num_layers > 0:
            loss = loss / num_layers
        
        # 动态缩放到目标量级（限制缩放范围，防止过度压缩）
        if not self._scale_initialized:
            raw_val = loss.item()
            if raw_val > 0:
                raw_scale = self.target_scale / (raw_val + 1e-8)
                # 限制缩放因子在[0.01, 10.0]范围
                self._running_scale = max(0.01, min(10.0, raw_scale))
                self._scale_initialized = True
        
        loss = loss * self._running_scale * self.lambda_attr
        
        return loss


class IdentityLoss(nn.Module):
    """
    身份改变损失（V6版）
    
    关键设计：
    1. 匿名身份应接近FPE加密后的目标身份
    2. 匿名身份应远离原始身份
    3. 阈值惩罚：身份相似度超过阈值时额外惩罚
    
    注意：计算identity_loss的FR网络必须冻结！
    """
    
    def __init__(self, lambda_identity=10.0, identity_threshold=0.3):
        super().__init__()
        
        self.lambda_identity = lambda_identity
        self.identity_threshold = identity_threshold
    
    def forward(self, anon_identity, target_identity, orig_identity):
        """
        参数:
            anon_identity: (B, 512) 匿名图像提取的身份
            target_identity: (B, 512) FPE加密后的目标身份
            orig_identity: (B, 512) 原始身份
        
        返回:
            loss_identity: 身份损失
            loss_threshold: 阈值惩罚
            identity_sim: 原始-匿名身份相似度（监控用）
        """
        # ===== V8.4 identity_loss：推离原始 + 余弦推向加密方向（只管方向不管幅值）=====
        # V8.3问题：MSE推向加密 → 同时约束方向+幅值 → 加密幅值不在流形 → 扭曲
        # V8.4修正：
        # 1. 推离原始身份（余弦，15x） → 保证匿名性
        # 2. 推向加密方向（余弦，5x） → 只约束方向不约束幅值！
        #    - 不同密码 → 不同加密身份 → 不同方向 → 密码敏感性
        #    - 不管幅值 → GAN负责把输出拉回流形 → 不扭曲
        # 3. GAN（5x） → 保真，确保像真脸
                
        # 1. 推离原始身份
        cos_sim_orig = F.cosine_similarity(anon_identity, orig_identity)
        loss_away = cos_sim_orig.mean()  # 最小化 → 远离原始
                
        # 2. 推向加密方向（余弦：只管方向，不管幅值）
        # 负号：最小化 → 最大化cos(anon, enc) → 往加密方向走
        cos_sim_enc = F.cosine_similarity(anon_identity, target_identity)
        loss_toward = -cos_sim_enc.mean()
                
        # 3. 阈值惩罚
        threshold_loss = F.relu(cos_sim_orig - self.identity_threshold).mean()
                
        total_loss = loss_away * self.lambda_identity + loss_toward * 5.0
        threshold_loss = threshold_loss * self.lambda_identity
                        
        return total_loss, threshold_loss, cos_sim_orig.mean()


class V6Loss(nn.Module):
    """
    V6损失函数 - 严格约束版
    
    结构：
    1. loss_bg: 背景L1损失（只对背景区域）
    2. loss_attr: 属性损失（VGG约束人脸区域属性一致性）
    3. loss_identity: 身份改变损失
    4. loss_GAN_global: 全局GAN损失
    5. loss_GAN_local: 局部GAN损失（人脸区域）
    
    关键改进（相对V5）：
    - 冻结FR：严格eval + requires_grad=False
    - 属性Loss使用L2（更严格）
    - 损失权重调整：背景L1权重降低，身份权重提高
    """
    
    def __init__(self, lambda_bg=3.0, lambda_attr=2.0, lambda_identity=15.0,
                 lambda_GAN_global=1.0, lambda_GAN_local=1.0, identity_threshold=0.3):
        super().__init__()
        
        # 损失权重
        # 相对V5的调整：
        # - lambda_bg: 5.0→3.0（降低，因为硬替换已保证背景保留）
        # - lambda_identity: 10.0→15.0（提高，强化身份改变）
        self.lambda_bg = lambda_bg
        self.lambda_attr = lambda_attr
        self.lambda_identity = lambda_identity
        self.lambda_GAN_global = lambda_GAN_global
        self.lambda_GAN_local = lambda_GAN_local
        self.identity_threshold = identity_threshold
        
        # 损失模块
        self.attr_loss = AttributeLoss(lambda_attr=lambda_attr)
        self.identity_loss_fn = IdentityLoss(
            lambda_identity=lambda_identity,
            identity_threshold=identity_threshold
        )
        self.l1_loss = nn.L1Loss()
        
        # 损失名称
        self.loss_names = ['bg', 'attr', 'identity', 'threshold', 'GAN_global', 'GAN_local']
    
    def background_loss(self, anon_img, orig_img, bg_mask):
        """背景L1损失：只对背景区域计算"""
        return self.l1_loss(anon_img * bg_mask, orig_img * bg_mask) * self.lambda_bg
    
    def attribute_loss(self, anon_img, orig_img, face_mask):
        """属性损失：VGG约束人脸区域"""
        return self.attr_loss(anon_img, orig_img, face_mask)
    
    def identity_loss(self, anon_identity, target_identity, orig_identity):
        """身份损失"""
        return self.identity_loss_fn(anon_identity, target_identity, orig_identity)
    
    def forward(self, anon_img, orig_img, face_mask, bg_mask,
                anon_identity, target_identity, orig_identity,
                discriminator_global=None, discriminator_local=None):
        """
        计算所有损失
        
        返回:
            losses: dict of losses
            total_loss: 总损失
        """
        losses = {}
        
        # 1. 背景L1损失
        losses['bg'] = self.background_loss(anon_img, orig_img, bg_mask)
        
        # 2. 属性损失
        losses['attr'] = self.attribute_loss(anon_img, orig_img, face_mask)
        
        # 3. 身份损失
        losses['identity'], losses['threshold'], losses['identity_sim'] = \
            self.identity_loss(anon_identity, target_identity, orig_identity)
        
        # 4. 全局GAN损失
        if discriminator_global is not None:
            fake_out = discriminator_global(anon_img)
            losses['GAN_global'] = F.mse_loss(fake_out, torch.ones_like(fake_out)) * self.lambda_GAN_global
        else:
            losses['GAN_global'] = torch.tensor(0.0, device=anon_img.device)
        
        # 5. 局部GAN损失
        if discriminator_local is not None:
            fake_local = anon_img * face_mask
            fake_out_local = discriminator_local(fake_local)
            losses['GAN_local'] = F.mse_loss(fake_out_local, torch.ones_like(fake_out_local)) * self.lambda_GAN_local
        else:
            losses['GAN_local'] = torch.tensor(0.0, device=anon_img.device)
        
        # 总损失
        total_loss = losses['bg'] + losses['attr'] + losses['identity'] + \
                     losses['threshold'] + losses['GAN_global'] + losses['GAN_local']
        
        return losses, total_loss


# 测试代码
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    
    print("="*60)
    print("测试 V6Loss 模块")
    print("="*60)
    
    # 创建损失模块
    loss_fn = V6Loss()
    
    # 测试输入
    B = 4
    anon_img = torch.randn(B, 3, 128, 128)
    orig_img = torch.randn(B, 3, 128, 128)
    face_mask = torch.rand(B, 1, 128, 128)
    bg_mask = 1.0 - face_mask
    
    anon_identity = torch.randn(B, 512)
    target_identity = torch.randn(B, 512)
    orig_identity = torch.randn(B, 512)
    
    # 计算损失
    losses, total = loss_fn(
        anon_img, orig_img, face_mask, bg_mask,
        anon_identity, target_identity, orig_identity
    )
    
    print("\n损失统计:")
    for name, value in losses.items():
        if isinstance(value, torch.Tensor):
            print("  loss_%s: %.4f" % (name, value.item()))
    
    print("\n总损失: %.4f" % total.item())
    
    # 验证属性损失（关键！）
    print("\n验证属性损失:")
    # 相同图像 → 属性损失应为0
    same_losses, _ = loss_fn(
        orig_img, orig_img, face_mask, bg_mask,
        anon_identity, target_identity, orig_identity
    )
    print("  相同图像的属性损失: %.6f" % same_losses['attr'].item())
    
    if same_losses['attr'].item() < 0.01:
        print("  ✅ 属性损失正确（相同图像≈0）")
    
    print("\n" + "="*60)
    print("V6Loss 模块测试完成")
    print("="*60)
