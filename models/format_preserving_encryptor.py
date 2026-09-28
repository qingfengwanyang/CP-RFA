#!/usr/bin/env python3
"""
保格式混沌流密码器 V8 (Format-Preserving Chaotic Encryptor)

核心设计原则：
1. 所有操作严格保L2范数 → 加密身份仍在SphereFace特征空间合理范围内
2. 只用离散操作：置乱(argsort) + 符号翻转({1,-1}掩膜)
3. Logistic映射生成确定性混沌序列 → 同密码同序列 → 100%可逆
4. 加微小抖动打破argsort平局，确保跨平台确定性

V7的问题回顾：
- XOR/SHIFT等操作破坏L2范数 → 加密身份偏离人脸流形 → 生成器扭曲
- 本模块只做置乱+翻转，范数严格不变 → 加密身份仍是"合法脸"

数学证明：
- 置乱：||P(x)||₂ = ||x||₂（只是重排，不改变元素值）
- 翻转：||x ⊙ m||₂ = ||x||₂（当m∈{1,-1}时，m²=1）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FormatPreservingEncryptor(nn.Module):
    """
    保格式混沌流密码器
    
    操作：
    - 置乱：用argsort打乱512维向量排列顺序
    - 翻转：用{1,-1}掩膜逐元素相乘
    
    两种操作都严格保L2范数，加密后的身份仍在SphereFace特征空间的合理范围内。
    """
    
    def __init__(self, identity_dim=512, password_dim=16, num_rounds=4,
                 init_seed=0):
        """
        参数:
            identity_dim: 身份特征维度 (512)
            password_dim: 密码维度 (16)
            num_rounds: 加密轮数 (4轮足够: 2^4=16种操作组合)
            init_seed: 冻结 seed_proj 的确定性初始化（跨进程可复现）
        """
        super().__init__()
        self.identity_dim = identity_dim
        self.password_dim = password_dim
        self.num_rounds = num_rounds
        self.init_seed = int(init_seed)
        
        # 密码 → Logistic映射初始种子
        # 用简单线性层将密码映射到(0,1)区间作为Logistic初值
        self.seed_proj = nn.Sequential(
            nn.Linear(password_dim, 64),
            nn.Tanh(),  # 输出[-1,1]
        )
        # 独立 Generator，避免被 mapper / 全局 RNG 打乱
        g = torch.Generator(device='cpu')
        g.manual_seed(self.init_seed)
        for m in self.seed_proj:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.5, generator=g)
                nn.init.zeros_(m.bias)
        # 冻结参数！密码→种子的映射必须是确定性的
        for p in self.seed_proj.parameters():
            p.requires_grad = False
    
    def _logistic_map(self, seed, length):
        """
        确定性Logistic映射生成混沌序列
        
        x_{n+1} = μ * x_n * (1 - x_n), μ=4.0 (满混沌)
        
        参数:
            seed: (B, 1) 初始值，范围(0,1)
            length: 生成序列长度
        
        返回:
            sequence: (B, length) 混沌序列，范围(0,1)
        """
        mu = 4.0
        B = seed.size(0)
        device = seed.device
        dtype = seed.dtype
        
        # 用float64计算以保证数值精度
        x = (seed * 0.5 + 0.5).double()  # map to (0,1), use float64
        x = x.clamp(1e-10, 1.0 - 1e-10)  # 避免边界值
        
        sequence = []
        for _ in range(length):
            x = mu * x * (1.0 - x)
            x = x.clamp(1e-10, 1.0 - 1e-10)
            sequence.append(x.float())  # 转回float32存储
        
        return torch.cat(sequence, dim=1)  # (B, length)
    
    def _generate_chaos_for_round(self, password, round_idx):
        """
        为指定轮次生成混沌序列
        
        每轮需要：
        - 512个值用于置乱 (argsort的key)
        - 512个值用于决定是否翻转
        
        参数:
            password: (B, password_dim)
            round_idx: 轮次索引
        
        返回:
            perm_chaos: (B, identity_dim) 用于argsort的混沌key
            flip_chaos: (B, identity_dim) 用于决定翻转的混沌key
        """
        B = password.size(0)
        device = password.device
        
        # 每轮用不同的偏移量避免序列重复
        seed = self.seed_proj(password)  # (B, 64)
        # 用round_idx对应的维度作为该轮的种子
        round_seed = seed[:, round_idx % 64:round_idx % 64 + 1]  # (B, 1)
        
        # 生成本轮需要的混沌序列 (identity_dim * 2个值)
        chaos = self._logistic_map(round_seed, self.identity_dim * 2)
        perm_chaos = chaos[:, :self.identity_dim]
        flip_chaos = chaos[:, self.identity_dim:]
        
        # 加微小抖动打破argsort平局（关键！确保跨平台确定性）
        arange = torch.arange(self.identity_dim, device=device, dtype=perm_chaos.dtype)
        perm_chaos = perm_chaos + 1e-7 * arange.unsqueeze(0)
        
        return perm_chaos, flip_chaos
    
    def _apply_permute(self, feat, perm_chaos):
        """
        置乱：根据混沌key重排向量
        
        严格保范数：||P(x)||₂ = ||x||₂
        
        参数:
            feat: (B, D) 输入向量
            perm_chaos: (B, D) 用于argsort的混沌key
        
        返回:
            result: (B, D) 置乱后的向量
            perm_idx: (B, D) 置换索引（解密用）
        """
        perm_idx = torch.argsort(perm_chaos, dim=1)
        result = torch.gather(feat, 1, perm_idx)
        return result, perm_idx
    
    def _apply_permute_inv(self, feat, perm_idx):
        """
        置乱逆操作
        
        参数:
            feat: (B, D) 置乱后的向量
            perm_idx: (B, D) 加密时的置换索引
        
        返回:
            result: (B, D) 还原后的向量
        """
        B, D = feat.shape
        inv_idx = torch.zeros_like(perm_idx)
        arange = torch.arange(D, device=feat.device).unsqueeze(0).expand(B, -1)
        inv_idx.scatter_(1, perm_idx, arange)
        result = torch.gather(feat, 1, inv_idx)
        return result
    
    def _apply_flip(self, feat, flip_chaos):
        """
        符号翻转：混沌key>0的位置乘-1
        
        严格保范数：||x ⊙ m||₂ = ||x||₂（当m∈{1,-1}）
        
        参数:
            feat: (B, D) 输入向量
            flip_chaos: (B, D) 用于决定翻转的混沌key
        
        返回:
            result: (B, D) 翻转后的向量
            flip_mask: (B, D) 翻转掩膜（解密用）
        """
        flip_mask = torch.where(flip_chaos > 0.5, 
                                torch.ones_like(feat) * -1.0, 
                                torch.ones_like(feat))
        result = feat * flip_mask
        return result, flip_mask
    
    def _apply_flip_inv(self, feat, flip_mask):
        """
        翻转逆操作：再乘一次同样的掩膜（(-1)²=1）
        """
        return feat * flip_mask
    
    def encrypt(self, identity_feat, password):
        """
        加密身份特征
        
        每轮随机选择置乱或翻转，共num_rounds轮。
        所有操作严格保L2范数。
        
        参数:
            identity_feat: (B, 512) 原始身份特征
            password: (B, 16) 密码
        
        返回:
            encrypted: (B, 512) 加密后的身份特征
            info: dict 加密信息（用于解密）
        """
        B = identity_feat.size(0)
        current = identity_feat.clone()
        
        round_infos = []
        
        for r in range(self.num_rounds):
            perm_chaos, flip_chaos = self._generate_chaos_for_round(password, r)
            
            # 交替执行：偶数轮置乱，奇数轮翻转
            # 这样保证置乱和翻转交替出现，最大化混淆
            ri = {}
            if r % 2 == 0:
                # 置乱轮
                current, perm_idx = self._apply_permute(current, perm_chaos)
                ri['op'] = 'permute'
                ri['perm_idx'] = perm_idx
            else:
                # 翻转轮
                current, flip_mask = self._apply_flip(current, flip_chaos)
                ri['op'] = 'flip'
                ri['flip_mask'] = flip_mask
            
            round_infos.append(ri)
        
        return current, {'round_infos': round_infos}
    
    def decrypt(self, encrypted_feat, password=None, info=None):
        """
        解密身份特征（100%数学无损）
        
        逆序执行加密的逆操作。
        同密码 → 同混沌序列 → 同操作 → 完美还原。
        
        参数:
            encrypted_feat: (B, 512) 加密的身份特征
            password: (B, 16) 密码（用于重建操作序列）
            info: dict 加密信息（如果有，可跳过混沌生成）
        
        返回:
            decrypted: (B, 512) 解密后的身份特征
        """
        if info is None:
            if password is None:
                raise ValueError("需要password或info来解密")
            # 用密码重建加密信息
            _, info = self.encrypt(torch.zeros_like(encrypted_feat), password)
        
        current = encrypted_feat.clone()
        round_infos = info['round_infos']
        
        # 逆序执行逆操作
        for r in range(self.num_rounds - 1, -1, -1):
            ri = round_infos[r]
            if ri['op'] == 'permute':
                current = self._apply_permute_inv(current, ri['perm_idx'])
            elif ri['op'] == 'flip':
                current = self._apply_flip_inv(current, ri['flip_mask'])
        
        return current
    
    def forward(self, identity_feat, password):
        """加密接口"""
        return self.encrypt(identity_feat, password)
    
    def get_operation_sequence(self, password):
        """获取操作序列（调试用）"""
        with torch.no_grad():
            seq = []
            for r in range(self.num_rounds):
                if r % 2 == 0:
                    seq.append('PERMUTE')
                else:
                    seq.append('FLIP')
            return [seq]  # 所有样本操作序列相同（交替模式）


# 独立测试
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    
    print("=" * 60)
    print("测试 保格式混沌流密码器 V8")
    print("=" * 60)
    
    encryptor = FormatPreservingEncryptor(identity_dim=512, password_dim=16, num_rounds=4)
    
    # 测试数据
    B = 4
    identity = F.normalize(torch.randn(B, 512), p=2, dim=1) * 22.0
    password = torch.randn(B, 16)
    
    # 1. 加密
    encrypted, info = encryptor.encrypt(identity, password)
    
    # 2. 验证L2范数保留
    orig_norm = identity.norm(p=2, dim=1)
    enc_norm = encrypted.norm(p=2, dim=1)
    norm_diff = (orig_norm - enc_norm).abs().max().item()
    
    print(f"\n1. L2范数保留验证:")
    print(f"   原始范数: {orig_norm.mean().item():.6f}")
    print(f"   加密范数: {enc_norm.mean().item():.6f}")
    print(f"   最大范数差异: {norm_diff:.10f}")
    print(f"   {'✅' if norm_diff < 1e-6 else '❌'} L2范数保留")
    
    # 3. 验证身份改变
    cos_sim = F.cosine_similarity(identity, encrypted).mean().item()
    print(f"\n2. 身份改变验证:")
    print(f"   加密前后余弦相似度: {cos_sim:.4f}")
    print(f"   {'✅' if cos_sim < 0.3 else '⚠️'} 身份改变")
    
    # 4. 验证可逆性（方式1：用info解密）
    decrypted1 = encryptor.decrypt(encrypted, info=info)
    error1 = (identity - decrypted1).abs().max().item()
    print(f"\n3. 可逆性验证(info解密):")
    print(f"   最大误差: {error1:.10f}")
    print(f"   {'✅' if error1 < 1e-5 else '❌'} 无损可逆")
    
    # 5. 验证可逆性（方式2：仅用密码解密）
    decrypted2 = encryptor.decrypt(encrypted, password=password)
    error2 = (identity - decrypted2).abs().max().item()
    print(f"\n4. 可逆性验证(密码解密):")
    print(f"   最大误差: {error2:.10f}")
    print(f"   {'✅' if error2 < 1e-5 else '❌'} 无损可逆")
    
    # 6. 安全性：错误密码无法解密
    wrong_pwd = torch.randn(B, 16)
    wrong_decrypted = encryptor.decrypt(encrypted, password=wrong_pwd)
    wrong_sim = F.cosine_similarity(identity, wrong_decrypted).mean().item()
    print(f"\n5. 安全性验证:")
    print(f"   错误密码解密相似度: {wrong_sim:.4f}")
    print(f"   {'✅' if wrong_sim < 0.5 else '❌'} 安全性")
    
    # 7. 不同密码 → 不同加密结果
    password2 = torch.randn(B, 16)
    encrypted2, _ = encryptor.encrypt(identity, password2)
    cross_sim = F.cosine_similarity(encrypted, encrypted2).mean().item()
    print(f"\n6. 密码差异性验证:")
    print(f"   不同密码加密结果相似度: {cross_sim:.4f}")
    print(f"   {'✅' if cross_sim < 0.3 else '⚠️'} 密码差异性")
    
    # 8. 操作序列
    seq = encryptor.get_operation_sequence(password)
    print(f"\n7. 操作序列: {' -> '.join(seq[0])}")
    
    print("\n" + "=" * 60)
    print("保格式混沌流密码器测试完成")
    print("=" * 60)
