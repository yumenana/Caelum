"""
PPBUNet v1.4 — Palette-Painter-Brush U-Net for Anime Super-Resolution (x4)
===========================================================================

4x super-resolution network for anime illustration restoration.

Architecture — Palette . Painter . Brush

  Named after the three stages of an artist's workflow:
    [Palette]  Extract the global color scheme from low-frequency features
    [Painter]  Reconstruct structure via attention-driven U-Net decoding
    [Brush]    Refine fine geometry at full resolution and upsample to HR

  v1.4 变更 (vs v1.3):
    - 删除 AnimeCommitteeRefiner (无功能分化驱动力, 梯度死重)
    - CMW 从 Decoder L0 之后移入 Decoder L0 中间 (前半 HAT → CMW → 后半 HAT)
    - 新增 CreviceAuxHead: 训练时在 CMW 输出处直接施加拓扑 loss, 解决梯度淹没
    - mi_loss (InfoNCE) 已接入训练, MIM 门控不再是死代码

  Pipeline:

      Input
        |
      RepSRBlock ------------------------------------------- x_shallow
        |                                                         |
        +---> ParallelOAM (0/90/45/135 deg bypass) -----+         |
        |                                               |         |
      Encoder: L0 (RCAB x2) -d2- L1 (RCAB x2) -d2       |         |
        |                                               |         |
      +- Bottleneck ----------------------------------+ |         |
      | FreqRouter -- DC ----+                        | |         |  * Palette
      |            '- AC --> PSMamba x N --+          | |         |
      |                                   |           | |         |
      +-----------------------------------------------+ |         |
        |                                               |         |
      +- Decoder -------------------------------------+ |         |
      | MIM+RMA (skip_l1) --> HAT Decoder L1          | |         |  * Painter
      | MIM+RMA (skip_l0) --> HAT L0 前半              | |         |
      | CMW(feat, feat_ac, feat_dc) — 虫洞色彩注入      | |         |  * Palette
      |   [CreviceAuxHead — 训练辅助监督]               | |         |
      | HAT L0 后半 → dec_l0_conv                      | |         |
      +-----------------------------------------------+ |         |
        |                                               |         |
      TopologyGuidedDCN (OAM-guided, local only) <------+         |  * Brush
        |                                                         |
      BaseAnchoredDetailInjector <-- x_shallow -------------------+
        |
      SATUpsampler_v2 --> SR Output (4x)

核心模块:
  ★ RepSRBlock: 结构重参数化 (训练多分支, 推理折叠为单 Conv)
  ★ ParallelOAM: 0°/90°/45°/135° 绝对方向基底, 消灭曼哈顿阶梯锯齿
  ★ FrequencyRouter: 频率解耦 → DC 色彩调色盘 + AC PS-Mamba 拓扑追踪
  ★ ConnectedManifoldWormhole: 双边流形虫洞 (嵌入 Decoder L0 中间, 直接辅助监督)
  ★ CreviceAuxHead: CMW 直接拓扑监督 (训练时启用, 推理零开销)
  ★ MIM + RMA: 互信息提纯跳跃连接 (InfoNCE 已接入) + 超球面流形对齐融合
  ★ HAT: 混合注意力 Transformer (窗口自注意力 + OCAB)
  ★ TopologyGuidedDCN: OAM 方向感知 + 拓扑引导局部可变形卷积 (3~5px 几何精修)
  ★ BaseAnchoredDetailInjector: 恒等锚定细节注入 (ControlNet Zero-Init)
  ★ SATUpsampler_v2: 奇异点感知拓扑上采样 (各向异性动态滤波 + 隐式多边形渲染)


作者: YumeNana
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

try:
    from .ps_mamba import PSMambaBlock
    from .hat import ResidualHybridAttentionGroup
except ImportError:
    from ps_mamba import PSMambaBlock
    from hat import ResidualHybridAttentionGroup


# ======================================================================
# Basic Modules (基础模块)
# ======================================================================


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation Channel Attention (SE 通道注意力).

    全局平均池化 → MLP 瓶颈 → Sigmoid 通道缩放,
    让网络自适应地对不同特征通道赋予不同重要性。

    参数:
        dim:       通道数
        reduction: MLP 压缩比
    """

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        mid = max(dim // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        w = self.pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1, 1)
        return x * w


class RCAB(nn.Module):
    """Residual Channel Attention Block (残差通道注意力块).

    在残差块基础上增加通道注意力, 自适应强调纹理/边缘等高频通道。
    x + CA(Conv3×3(LeakyReLU(Conv3×3(x))))

    参数:
        dim:       通道数
        reduction: SE 压缩比
    """

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )
        self.ca = ChannelAttention(dim, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ca(self.body(x))


class RepSRBlock(nn.Module):
    """Structural Re-parameterization Block (结构重参数化块).

    训练时 3×3 / 1×1 / Identity 三分支并行提供隐式正则化,
    推理时 switch_to_deploy() 折叠为单个 3×3 Conv, 零精度损失。
    无 BN 设计避免小 batch 统计量噪声。

    参数:
        in_channels:  输入通道数
        out_channels: 输出通道数
        deploy:       是否为推理模式
    """

    def __init__(self, in_channels: int, out_channels: int, deploy: bool = False):
        super().__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels,
                                         kernel_size=3, padding=1, bias=True)
        else:
            self.rbr_dense = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)
            self.has_identity = (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.rbr_reparam(x)
        out = self.rbr_dense(x) + self.rbr_1x1(x)
        if self.has_identity:
            out = out + x
        return out

    def switch_to_deploy(self):
        """将多分支折叠为单个 3×3 卷积。"""
        if self.deploy:
            return

        w_dense = self.rbr_dense.weight.data
        b_dense = self.rbr_dense.bias.data
        w_1x1 = F.pad(self.rbr_1x1.weight.data, (1, 1, 1, 1))
        b_1x1 = self.rbr_1x1.bias.data

        fused_kernel = w_dense + w_1x1
        fused_bias = b_dense + b_1x1

        if self.has_identity:
            w_id = torch.zeros_like(w_dense)
            for i in range(self.in_channels):
                w_id[i, i, 1, 1] = 1.0
            fused_kernel = fused_kernel + w_id

        self.rbr_reparam = nn.Conv2d(
            self.in_channels, self.out_channels,
            kernel_size=3, padding=1, bias=True,
        ).to(fused_kernel.device)
        self.rbr_reparam.weight.data = fused_kernel
        self.rbr_reparam.bias.data = fused_bias

        for attr in ('rbr_dense', 'rbr_1x1'):
            if hasattr(self, attr):
                delattr(self, attr)
        self.deploy = True


# ======================================================================
# Orientation-Aware Bypass (方向感知旁路)
# ======================================================================


class ParallelOAM(nn.Module):
    """Parallel Orientation-Aware Module (平行方向感知模块).

    在最高分辨率提取 0°/90°/45°/135° 四个绝对方向基底,
    对角线核通过物理掩码 (torch.eye) 约束感受野严格沿对角方向,
    彻底消灭正交基底线性组合导致的曼哈顿阶梯锯齿。

    作为独立旁路运行, 不经过下采样路径, 避免混叠摧毁方向锐度。
    输出纯几何方向特征 (非残差), 在 Decoder L0 后融合回主干。

    参数:
        dim: 通道数
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        self.conv_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), padding=(0, 3),
                                groups=dim, bias=False)
        self.conv_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), padding=(3, 0),
                                groups=dim, bias=False)
        self.conv_d45 = nn.Conv2d(dim, dim, kernel_size=7, padding=3,
                                  groups=dim, bias=False)
        self.conv_d135 = nn.Conv2d(dim, dim, kernel_size=7, padding=3,
                                   groups=dim, bias=False)

        diag_mask = torch.eye(7, dtype=torch.float32)
        anti_diag_mask = torch.fliplr(diag_mask)
        self.register_buffer(
            'mask_d45',
            diag_mask.view(1, 1, 7, 7).expand(dim, 1, 7, 7).contiguous(),
        )
        self.register_buffer(
            'mask_d135',
            anti_diag_mask.view(1, 1, 7, 7).expand(dim, 1, 7, 7).contiguous(),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        mid = max(dim // 2, 4)
        self.attention = nn.Sequential(
            nn.Linear(dim * 4, mid, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(mid, dim * 4, bias=False),
        )
        self.project = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        feat_h = self.conv_h(x)
        feat_v = self.conv_v(x)

        weight_d45 = self.conv_d45.weight * self.mask_d45
        weight_d135 = self.conv_d135.weight * self.mask_d135
        feat_d45 = F.conv2d(x, weight_d45, padding=3, groups=self.dim)
        feat_d135 = F.conv2d(x, weight_d135, padding=3, groups=self.dim)

        feats_cat = torch.cat([feat_h, feat_v, feat_d45, feat_d135], dim=1)
        context = self.pool(feats_cat.pow(2)).view(B, -1)

        attn = self.attention(context).view(B, 4, C, 1, 1)
        attn = F.softmax(attn, dim=1)

        feat_geom = (
            attn[:, 0] * feat_h +
            attn[:, 1] * feat_v +
            attn[:, 2] * feat_d45 +
            attn[:, 3] * feat_d135
        )

        return self.project(feat_geom)


# ======================================================================
# Frequency Decoupling & Chromaticity Palette (频率解耦与色彩调色盘)
# ======================================================================


class FrequencyRouter(nn.Module):
    """Explicit Frequency Decoupler via Spatial Low-Pass Filtering (空间域频率解耦器).

    将混合特征剥离为低频直流分量 (DC, 色块均值场) 和高频交流分量 (AC, 线条拓扑)。
    深度可分离卷积初始化为标准平均滤波核, 作为可微低通滤波器。

    参数:
        dim:         通道数
        kernel_size: 低通滤波核大小
    """

    def __init__(self, dim: int, kernel_size: int = 5):
        super().__init__()
        self.lpf = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                             padding=kernel_size // 2, groups=dim, bias=False)
        nn.init.constant_(self.lpf.weight, 1.0 / (kernel_size ** 2))

    def forward(self, x: torch.Tensor) -> tuple:
        feat_dc = self.lpf(x)
        feat_ac = x - feat_dc
        return feat_dc, feat_ac


class ConnectedManifoldWormhole(nn.Module):
    """Connected Manifold Wormhole — 双边流形虫洞 (连通域交叉注意力).

    ■ 第一性原理 ■
    K-Slot 离散色卡对赛璐璐有效, 但对现代插画的柔渐变/厚涂/光晕是过度简化。
    128×128 patch 内不存在完整语义, 但存在局部"连通域" — 连续的色彩流形。

    机制: 双边交叉注意力 (Bilateral Cross-Attention)
    - Key/Value (颜料库): 来自 Bottleneck。feat_ac+坐标 作为 Key, feat_dc 作为 Value
    - Query (画笔): 来自 Decoder L0。feat_hr+坐标 作为 Query

    自适应行为:
    - 渐变区: AC 特征平滑 → 注意力由坐标主导 → 等效双线性插值 → 渐变完美保留
    - 发尖奇点: AC 特征剧变 → 语义项一票否决近邻背景 → 全局检索发根 → 虫洞跃迁

    安全设计:
    - 独立分支融合 (方案 B): 特征与坐标在 dict_dim 空间加法融合,
      避免坐标淹没 (2ch vs 256ch), 确保坐标维度有等维表达能力
    - gamma 微小正数初始化 (1e-4): 打破双零死锁, 训练初期近似恒等透传
    - F.scaled_dot_product_attention: FlashAttention 优化显存

    参数:
        query_dim: 高分辨率特征通道数 (dim_l0)
        key_dim:   AC 特征通道数 (dim_bn)
        val_dim:   DC 特征通道数 (dim_bn)
        dict_dim:  注意力内部维度
    """

    def __init__(self, query_dim: int, key_dim: int, val_dim: int, dict_dim: int = 64):
        super().__init__()
        self.dict_dim = dict_dim

        # 方案 B: 独立分支加法融合, 消除坐标淹没
        self.q_feat = nn.Conv2d(query_dim, dict_dim, 1, bias=False)
        self.q_coord = nn.Conv2d(2, dict_dim, 1, bias=False)

        self.k_feat = nn.Conv2d(key_dim, dict_dim, 1, bias=False)
        self.k_coord = nn.Conv2d(2, dict_dim, 1, bias=False)

        self.v_proj = nn.Conv2d(val_dim, dict_dim, 1, bias=False)

        self.out_proj = nn.Conv2d(dict_dim, query_dim, 1)
        # 注意: 不零初始化 out_proj!
        # gamma=0 + out_proj=0 会导致双零梯度死锁:
        #   gamma 梯度 ∝ out_proj(hint) = 0, out_proj 梯度 ∝ gamma = 0
        #   两个参数互相等待, 永远无法启动学习。
        # 保持 Kaiming 默认初始化, 配合小 gamma 保证训练稳定。

        # Post-Attention LayerNorm (解耦幅度与方向)
        # 标准 Transformer 做法。不做此操作时, feat_dc 幅度漂移 (无 BN/LN 约束)
        # 会透传到 color_hint, 迫使 gamma 被动压缩以补偿—— gamma 无法自由增长。
        # LayerNorm 后 gamma 可以真实反映 CMW 贡献强度。
        self.color_norm = nn.LayerNorm(dict_dim)

        # 小正数初始化 (非零!) — 打破双零死锁
        self.gamma = nn.Parameter(torch.ones(1, query_dim, 1, 1) * 1e-4)

    @staticmethod
    def _generate_coords(B: int, H: int, W: int,
                         device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """生成归一化到 [-1, 1] 的绝对物理坐标。"""
        y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([gx, gy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

    def forward(self, feat_hr: torch.Tensor,
                feat_ac: torch.Tensor, feat_dc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_hr: (B, C_q, H, W) — 高分辨率解码器特征 (Query)
            feat_ac: (B, C_k, h, w) — Bottleneck AC 特征 (Key 语义)
            feat_dc: (B, C_v, h, w) — Bottleneck DC 特征 (Value 色彩)

        Returns:
            (B, C_q, H, W) — 色彩注入后的特征
        """
        B, C_q, H, W = feat_hr.shape
        _, _, h, w = feat_ac.shape

        # 1. 物理坐标 (独立分支)
        coords_hr = self._generate_coords(B, H, W, feat_hr.device, feat_hr.dtype)
        coords_lr = self._generate_coords(B, h, w, feat_ac.device, feat_ac.dtype)

        # 2. Query = feat_proj + coord_proj (独立投影后加法融合)
        Q = self.q_feat(feat_hr) + self.q_coord(coords_hr)  # (B, dict_dim, H, W)
        Q = Q.flatten(2).transpose(1, 2)                     # (B, H*W, dict_dim)

        # 3. Key = feat_proj + coord_proj
        K = self.k_feat(feat_ac) + self.k_coord(coords_lr)   # (B, dict_dim, h, w)
        K = K.flatten(2).transpose(1, 2)                     # (B, h*w, dict_dim)

        # 4. Value = DC 色彩特征
        V = self.v_proj(feat_dc).flatten(2).transpose(1, 2)  # (B, h*w, dict_dim)

        # 5. Scaled Dot-Product Attention (FlashAttention 优化)
        color_hint = F.scaled_dot_product_attention(Q, K, V)  # (B, H*W, dict_dim)

        # 6. Post-Attention LayerNorm (解耦 V 幅度漂移)
        # 无此操作时: feat_dc 幅度随训练增长 (norm-free 网络) → color_hint 爆炸
        # → gamma 被迫缩小补偿, gamma 不再代表 CMW 贡献强度
        # 有此操作: color_hint 方向保留, 幅度标准化; gamma 可以自由增长
        color_hint = self.color_norm(color_hint)               # (B, H*W, dict_dim)
        color_hint = color_hint.transpose(1, 2).reshape(B, self.dict_dim, H, W)

        # 7. 投影并恒等注入
        out = feat_hr + self.gamma * self.out_proj(color_hint)

        if self.training:
            with torch.no_grad():
                hint_mag = color_hint.abs().mean().item()
                gamma_abs = self.gamma.abs().mean().item()
                self.last_color_hint_mag = hint_mag
                self.last_gamma_abs = gamma_abs
                self.last_cmw_eff = gamma_abs * hint_mag  # 真实有效注入量

        return out


# Skip Connection Modules (跳跃连接模块)
# ======================================================================


class MIMFeatureFilter(nn.Module):
    """Mutual Information Maximization Feature Filter (互信息特征提纯器).

    利用 InfoNCE 对比学习最大化 Encoder/Decoder 特征的互信息下界,
    学习通道与空间的动态门控, 过滤跳跃连接中的高熵压缩伪影。
    采用 Sobel 方差感知掩码, 仅从高频结构区域采样正负对, 保护平坦区。
    门控透明启动 (bias=3.0, sigmoid≈0.95), 训练初期近乎不过滤。
    InfoNCE 仅训练时计算, 推理时零额外开销。

    参数:
        enc_dim:     Encoder 特征通道数
        dec_dim:     Decoder 参考特征通道数
        hidden_dim:  投影维度 (默认 enc_dim // 2)
        temperature: InfoNCE 温度系数
        num_samples: 空间随机采样点数
    """

    def __init__(self, enc_dim: int, dec_dim: int, hidden_dim: int = None,
                 temperature: float = 0.07, num_samples: int = 1024):
        super().__init__()
        self.temperature = temperature
        self.num_samples = num_samples
        hidden_dim = hidden_dim or max(enc_dim // 2, 16)

        self.proj_enc = nn.Sequential(
            nn.Conv2d(enc_dim, hidden_dim, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
        )
        self.proj_dec = nn.Sequential(
            nn.Conv2d(dec_dim, hidden_dim, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
        )

        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(hidden_dim, enc_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.channel_fc[0].weight)
        nn.init.constant_(self.channel_fc[0].bias, 3.0)

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=7, padding=3, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.spatial_conv[0].weight)
        nn.init.constant_(self.spatial_conv[0].bias, 3.0)

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32,
        )
        self.register_buffer('_sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('_sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, feat_enc: torch.Tensor, feat_dec: torch.Tensor):
        """返回 (提纯后特征, InfoNCE 损失)。"""
        if feat_dec.shape[2:] != feat_enc.shape[2:]:
            feat_dec = F.interpolate(
                feat_dec, size=feat_enc.shape[2:],
                mode='bilinear', align_corners=False,
            )

        B, _, H, W = feat_enc.shape

        raw_proj_enc = self.proj_enc(feat_enc)
        z_enc = F.normalize(raw_proj_enc, dim=1)
        z_dec = F.normalize(self.proj_dec(feat_dec), dim=1)

        mi_loss = torch.zeros(1, device=feat_enc.device, dtype=feat_enc.dtype)
        if self.training:
            N = H * W
            num_s = min(self.num_samples, N)

            z_e = z_enc.reshape(B, -1, N)
            z_d = z_dec.reshape(B, -1, N)

            with torch.no_grad():
                energy = raw_proj_enc.detach().pow(2).sum(dim=1, keepdim=True)
                gx = F.conv2d(energy, self._sobel_x, padding=1)
                gy = F.conv2d(energy, self._sobel_y, padding=1)
                grad_mag = (gx.pow(2) + gy.pow(2)).sqrt()
                avg_grad = grad_mag.mean(dim=0).view(-1)
                tau = avg_grad.mean() + 0.5 * avg_grad.std()
                struct_idx = (avg_grad > tau).nonzero(as_tuple=True)[0]

            actual_num_s = min(num_s, struct_idx.numel())
            if actual_num_s >= 16:
                perm = torch.randperm(struct_idx.numel(), device=feat_enc.device)[:actual_num_s]
                idx = struct_idx[perm]

                z_e = z_e[:, :, idx]
                z_d = z_d[:, :, idx]

                logits = torch.bmm(z_e.transpose(1, 2), z_d) / self.temperature
                labels = torch.arange(actual_num_s, device=logits.device)
                labels = labels.unsqueeze(0).expand(B, -1)
                mi_loss = F.cross_entropy(
                    logits.reshape(-1, actual_num_s), labels.reshape(-1),
                )

        shared = z_enc * z_dec
        c_gate = self.channel_fc(self.channel_pool(shared))
        s_gate = self.spatial_conv(shared)

        return feat_enc * c_gate * s_gate, mi_loss


class RiemannianManifoldAlignment(nn.Module):
    """Riemannian Manifold Alignment on Hypersphere (超球面黎曼流形对齐融合).

    将 Encoder 高频线稿与 Decoder 低频色彩特征投影至超球面 S^{n-1},
    通过黎曼对数/指数映射在局部欧氏切空间完成无损对齐, 避免流形坍缩。

    径-向分解 (Radius-Direction Decomposition):
      方向 d = f/||f|| ∈ S^{n-1} 用于语义对齐,
      幅值 r = ||f|| ∈ R+ 保留绝对色彩强度,
      二者解耦处理后重组, 避免归一化摧毁色彩信息。

    参数:
        enc_dim: Encoder 特征维度
        dec_dim: Decoder 特征维度
        out_dim: 输出维度 (默认 = enc_dim)
        eps:     数值稳定性常数
    """

    def __init__(self, enc_dim: int, dec_dim: int, out_dim: int = None,
                 eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        out_dim = out_dim or enc_dim

        self.proj_enc = nn.Conv2d(enc_dim, out_dim, kernel_size=1, bias=False)
        self.proj_dec = nn.Conv2d(dec_dim, out_dim, kernel_size=1, bias=False)

        self.tangent_fusion = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
        )

        mid_mag = max(out_dim // 4, 4)
        self.mag_gate = nn.Sequential(
            nn.Conv2d(out_dim * 2, mid_mag, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_mag, out_dim, kernel_size=1),
        )
        nn.init.zeros_(self.mag_gate[-1].weight)
        nn.init.zeros_(self.mag_gate[-1].bias)

    def log_map(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """对数映射 log_p(q): 将 q 投影到 p 的切空间 T_p(S^{n-1})。

        使用 atan2 替代 acos 消除 inner→±1 时的梯度爆炸 (acos 导数在 ±1 处趋于无穷)。
        atan2(y, x) 在全域梯度有界, 天然适配 AMP float16。
        """
        inner = torch.sum(p * q, dim=1, keepdim=True)
        v = q - inner * p
        v_norm = torch.norm(v, p=2, dim=1, keepdim=True).clamp(min=self.eps)
        theta = torch.atan2(v_norm, inner)
        return (theta / v_norm) * v

    def exp_map(self, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """指数映射 exp_p(v): 将切向量 v 沿测地线映射回超球面。

        直接 clamp v_norm 替代 torch.where 分支 (torch.where 不阻断梯度流,
        被屏蔽分支的 inf/nan 梯度仍会反向传播导致 AMP 下权重污染)。
        v_norm→0 时: cos(ε)·p + sin(ε)/ε·v ≈ p + v (Taylor 一阶), 数学等价。
        """
        v_norm = torch.norm(v, p=2, dim=1, keepdim=True).clamp(min=self.eps)
        return torch.cos(v_norm) * p + (torch.sin(v_norm) / v_norm) * v

    def forward(self, feat_enc: torch.Tensor,
                feat_dec: torch.Tensor) -> torch.Tensor:
        if feat_dec.shape[2:] != feat_enc.shape[2:]:
            feat_dec = F.interpolate(
                feat_dec, size=feat_enc.shape[2:],
                mode='bilinear', align_corners=False,
            )

        f_enc = self.proj_enc(feat_enc)
        f_dec = self.proj_dec(feat_dec)

        r_enc = torch.norm(f_enc, p=2, dim=1, keepdim=True).clamp(min=self.eps)
        r_dec = torch.norm(f_dec, p=2, dim=1, keepdim=True).clamp(min=self.eps)
        d_enc = f_enc / r_enc
        d_dec = f_dec / r_dec

        tangent_enc = self.log_map(d_dec, d_enc)
        tangent_fused = self.tangent_fusion(
            torch.cat([tangent_enc, d_dec], dim=1),
        )
        d_fused = self.exp_map(d_dec, tangent_fused)

        w = torch.sigmoid(self.mag_gate(torch.cat([f_enc, f_dec], dim=1)))
        r_fused = w * r_enc + (1.0 - w) * r_dec

        return d_fused * r_fused


# ======================================================================
# Decoder Modules (解码器模块)
# ======================================================================


class DensePathExtractor(nn.Module):
    """
    轻量级残差密集块 (RDB) — 提取拓扑路径上下文。
    通过密集连接同时持有局部端点与宏观走向的视野。
    """
    def __init__(self, dim: int, growth_rate: int = 16):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(dim, growth_rate, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(dim + growth_rate, growth_rate, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(dim + 2 * growth_rate, growth_rate, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        # 特征压缩融合
        self.fuse = nn.Conv2d(dim + 3 * growth_rate, dim, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat([x, x1], dim=1))
        x3 = self.conv3(torch.cat([x, x1, x2], dim=1))
        out = self.fuse(torch.cat([x, x1, x2, x3], dim=1))
        return out + x # 残差直连

class TopologyGuidedDCN(nn.Module):
    """Topology-Guided Local Deformable Convolution
    (拓扑引导局部可变形卷积: OAM 方向感知 + Brush 几何精修).

    ■ 第一性原理 ■
    长距离色彩搬运已由 GlobalColorDictionary + HardRoutingRenderer 接管
    (全局注意力内积, 梯度全局连通, 无 grid_sample 插值污染)。

    TopologyGuidedDCN 现在专注于唯一职责:
    局部 3~5px 几何精修 — 角点曲率校准、发尖末端锐化。

    流水线:
      geom_prior → compress(dim→8) → geom_code (方向码)
      x → corner_prior → topo_map (拓扑热图: 边缘/角点/发尖)
      [x, topo_map, geom_code] → DensePathExtractor → path_feat

      path_feat → local_head → local_offset+mask → deform_conv2d
      feat = x + γ · DCN(x)

    参数:
        dim:      主干通道数
        geom_dim: OAM 方向先验通道数 (默认 = dim)
    """

    def __init__(self, dim: int, geom_dim: int | None = None):
        super().__init__()
        self.dim = dim
        geom_dim = geom_dim or dim

        # === OAM 方向先验压缩 ===
        geom_code_ch = max(dim // 8, 4)
        self.geom_compress = nn.Sequential(
            nn.Conv2d(geom_dim, geom_code_ch, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.geom_code_ch = geom_code_ch

        # === 共享拓扑骨干 ===
        self.corner_prior = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid(),
        )

        # path_ch = x(dim) + topo_map(1) + geom_code(geom_code_ch)
        path_ch = dim + 1 + geom_code_ch
        self.path_abstractor = DensePathExtractor(path_ch)

        # === 局部 DCN (Brush — 3~5px 几何精修) ===
        self.local_head = nn.Sequential(
            nn.Conv2d(path_ch, dim // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 2, 27, 3, 1, 1),  # 18 offset + 9 mask
        )
        nn.init.zeros_(self.local_head[-1].weight)
        nn.init.zeros_(self.local_head[-1].bias)

        self.local_dcn_weight = nn.Parameter(torch.empty(dim, dim, 3, 3))
        self.local_dcn_bias = nn.Parameter(torch.zeros(dim))
        nn.init.kaiming_uniform_(self.local_dcn_weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(dim * 9)
        nn.init.uniform_(self.local_dcn_bias, -bound, bound)

        self.local_gamma = nn.Parameter(torch.ones(1, dim, 1, 1) * 1e-2)

    def forward(self, x: torch.Tensor, geom_prior: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:           (B, C, H, W) — 解码器特征
            geom_prior:  (B, C, H, W) — ParallelOAM 方向先验

        Returns:
            feat_refined: (B, C, H, W) — 局部几何精修后的特征
        """
        # === 拓扑骨干 ===
        topo_map = self.corner_prior(x)
        geom_code = self.geom_compress(geom_prior)
        path_input = torch.cat([x, topo_map, geom_code], dim=1)
        path_feat = self.path_abstractor(path_input)

        # === 局部 DCN ===
        local_om = self.local_head(path_feat)
        local_offset = local_om[:, :18]
        local_mask = torch.sigmoid(local_om[:, 18:])

        local_dcn = deform_conv2d(
            input=x, offset=local_offset, weight=self.local_dcn_weight,
            bias=self.local_dcn_bias, stride=1, padding=1, mask=local_mask,
        )
        feat = x + self.local_gamma * local_dcn

        if self.training:
            with torch.no_grad():
                self.last_local_offset_mag = local_offset.abs().mean().item()
                self.last_topo_map_mean = topo_map.mean().item()

        return feat




class BaseAnchoredDetailInjector(nn.Module):
    """
    基础锚定细节注入器 (Base-Anchored Detail Injector - BADI).

    ■ 第一性原理重构 (跳出零和博弈) ■
    彻底抛弃 Out = a*Deep + (1-a)*Shallow 的零和门控机制，根除深层特征的"梯度断头台"效应。

    采用坚不可摧的"恒等锚定 (Identity-Anchored)"法则：
    1. 承重墙直通: 深层纯净特征 (feat_deep) 永远不乘以任何掩码，直接参与加法，
       保证其永远受到 100% 的重建损失约束，绝无可能发生数值爆炸。
    2. 跨层细节萃取: 联合深浅特征，提取出一个包含高频边缘的"修补残差"。
    3. 基底主导门控: 由深层纯净语义自己决定，在什么地方 (线稿) 引入残差，
       在什么地方 (平坦噪点区) 将残差门控归零。
    """
    def __init__(self, dim: int):
        super().__init__()

        self.detail_extractor = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1, bias=True)
        )

        nn.init.zeros_(self.detail_extractor[-1].weight)
        nn.init.zeros_(self.detail_extractor[-1].bias)

        nn.init.zeros_(self.spatial_gate[-1].weight)
        nn.init.constant_(self.spatial_gate[-1].bias, -3.0)

    def forward(self, feat_deep: torch.Tensor, feat_shallow: torch.Tensor) -> torch.Tensor:
        context = torch.cat([feat_deep, feat_shallow], dim=1)
        detail_residual = self.detail_extractor(context)

        gate = torch.sigmoid(self.spatial_gate(feat_deep))

        latent_merged = feat_deep + gate * detail_residual

        if self.training:
            self.last_gate = gate
            with torch.no_grad():
                self.last_gate_mean = gate.mean().item()
                self.last_detail_mag = detail_residual.abs().mean().item()

        return latent_merged



# ======================================================================
# Upsampler (上采样器)
# ======================================================================


class ImplicitPolygonInjector(nn.Module):
    """Implicit Polygon Injector — 隐式多边形奇点渲染器.

    ■ 第一性原理 ■
    PDE 对流公式 Δ=v·∇I 在 C⁰ 奇点 (V 字发尖) 处梯度湮灭 (∇I=0),
    导致无论网络预测多大的速度场, 对流项恒为 0 — 方程在最该发力处自杀。

    破局: 放弃连续域导数, 改用离散域几何切割。
    数学定理: ReLU 两层 MLP = 多超平面交集 = 分段线性多边形。
    给网络亚像素相对坐标 (dx, dy), 让 MLP 直接在 S×S 高分辨率网格里
    "画" 出锐利的多边形夹角, 无需依赖梯度。

    安全设计:
    - 门控 bias=-3.0 → sigmoid≈0.047, 初始近乎静默, 不干扰平滑流形
    - MLP 末层零初始化 → 训练初期输出为 0, 基础滤波完整透传

    参数:
        dim:   特征通道数
        scale: 上采样因子
    """

    def __init__(self, dim: int, scale: int = 4):
        super().__init__()
        self.scale = scale

        # 门控: 只在奇点/高频拓扑处激活, 绝不干扰平滑流形
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 4, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate[-2].bias, -3.0)

        # 隐式多边形渲染 MLP (特征 dim + 坐标 2)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim + 2, dim, 1, bias=True),
            nn.ReLU(inplace=True),  # 核心: ReLU 切割出 C⁰ 锐利边界
            nn.Conv2d(dim, dim, 1, bias=True),
        )
        # 零初始化: 训练初期完全不干预基础滤波
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # 预计算固定的亚像素相对坐标网格
        coords = torch.linspace(-1 + 1 / scale, 1 - 1 / scale, scale)
        y, x = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer('subpixel_grid', torch.stack([x, y], dim=0))  # (2, S, S)

    def forward(self, feat_lr: torch.Tensor) -> torch.Tensor:
        """隐式多边形几何残差.

        Args:
            feat_lr: (B, C, H, W) — 低分辨率特征

        Returns:
            (B, C, H*S, W*S) — 亚像素级多边形几何残差
        """
        B, C, H, W = feat_lr.shape
        S = self.scale

        # 1. 奇点门控 → 展开到 HR 空间
        gate_lr = self.gate(feat_lr)  # (B, 1, H, W)
        gate_hr = gate_lr.repeat_interleave(S, dim=2).repeat_interleave(S, dim=3)

        # 2. LR 特征展开到 HR 空间 (Nearest, 4×4 块内特征一致)
        feat_hr = feat_lr.repeat_interleave(S, dim=2).repeat_interleave(S, dim=3)

        # 3. 铺设亚像素相对坐标
        grid_hr = self.subpixel_grid.view(1, 2, S, S).repeat(1, 1, H, W).expand(B, -1, -1, -1)

        # 4. 拼接特征与坐标 → MLP 渲染多边形
        x = torch.cat([feat_hr, grid_hr], dim=1)
        residual_hr = self.mlp(x)

        # 5. 门控注入: 仅在发尖/边缘处释放多边形残差
        out = residual_hr * gate_hr

        if self.training:
            with torch.no_grad():
                self.last_gate_mean = gate_lr.mean().item()
                self.last_residual_mag = residual_hr.abs().mean().item()

        return out


class SATUpsampler_v2(nn.Module):
    """Singularity-Aware Topological Upsampler v2 (奇异点感知拓扑上采样器 v2).

    v1 的 PDE 对流注入 (Δ=v·∇I) 在 C⁰ 奇点处梯度湮灭, 已被根除。

    1. 各向异性动态滤波 (保留, 处理 C^∞ 平滑流形):
       为每个 LR 像素预测 S² 个独立且空间异构的 K×K 滤波核。
       Softmax 归一化确保滤波核权重和为 1, 绝对保留直流(DC)色彩信息。

    2. 隐式多边形注入 (新增, 处理 C⁰ 奇点):
       ReLU MLP + 亚像素坐标直接 "画" 出锐利多边形夹角,
       绕过梯度依赖, 在 ∇I=0 的奇点处仍能精确渲染。

    参数:
        dim:           通道数
        scale:         上采样因子
        filter_kernel: 动态滤波核大小
    """

    def __init__(self, dim: int, scale: int = 4, filter_kernel: int = 5):
        super().__init__()
        self.scale = scale
        self.dim = dim
        self.k = filter_kernel
        self.n_phases = scale ** 2

        # === 模块一: 各向异性动态滤波核生成器 (处理 C^∞ 平滑流形) ===
        self.filter_gen = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, self.n_phases * (self.k ** 2), kernel_size=1, bias=False),
        )

        # === 模块二: 隐式多边形注入器 (处理 C⁰ 奇点) ===
        self.polygon_injector = ImplicitPolygonInjector(dim, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        S = self.scale
        K = self.k

        # === 1. 各向异性流形滤波 (重构拓扑基底) ===
        filters = self.filter_gen(x)
        filters = filters.view(B, self.n_phases, K ** 2, H, W)
        # 空间 Softmax: 归一化权重和为 1, 绝对保留 DC 色彩
        filters = F.softmax(filters, dim=2)

        # 提取 x 的局部 KxK 邻域张量
        pad = K // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        x_unfold = F.unfold(x_pad, kernel_size=K)
        x_unfold = x_unfold.view(B, C, K ** 2, H, W)

        # einsum: 通道共享滤波核, 保证跨通道颜色一致性
        out_base = torch.einsum('bckhw, bskhw -> bcshw', x_unfold, filters)
        out_base = out_base.reshape(B, C * self.n_phases, H, W)
        out_base = F.pixel_shuffle(out_base, S)

        # === 2. 隐式多边形残差注入 (替代 PDE 对流) ===
        polygon_residual = self.polygon_injector(x)

        return out_base + polygon_residual


class AdvancedUpsampler(nn.Module):
    """SAT-based Upsampler v2 (SAT 上采样器 v2).

    封装 SATUpsampler_v2 + PReLU + 尾部 Conv 输出 RGB。

    参数:
        dim:          特征维度
        out_channels: 输出通道数
        scale:        上采样因子
    """

    def __init__(self, dim: int, out_channels: int = 3, scale: int = 4):
        super().__init__()
        self.scale = scale

        if scale == 1:
            self.up = nn.Identity()
        elif scale in (2, 4):
            self.up = nn.Sequential(
                SATUpsampler_v2(dim, scale=scale, filter_kernel=5),
                nn.PReLU(dim),
            )
        else:
            raise ValueError(f"Unsupported scale factor: {scale}")

        self.tail = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.up(x))


class CreviceAuxHead(nn.Module):
    """CMW 辅助监督头 — 训练时在 CMW 输出处直接施加拓扑 loss.

    ■ 设计目的 ■
    CMW 的 gamma 在 v1.3 中始终 ≈ 0, 500ep 不激活。
    根因: 拓扑 loss (LRT, Crevice) 的梯度经 40+ 层反传后被 L1/DC 的均匀梯度淹没,
    CMW 从未收到有效的驱动信号。

    解决方案: 在 CMW 输出处直接施加辅助监督 (deep supervision),
    拓扑 loss 的梯度只需穿过 1 层 Conv 即可到达 CMW。

    极简结构: 1×1 投影 + PixelShuffle 上采样到 HR → 3ch RGB。
    推理时不使用, 零额外开销。

    参数:
        dim:   特征通道数
        scale: 上采样因子
    """

    def __init__(self, dim: int, scale: int = 4):
        super().__init__()
        self.scale = scale
        self.proj = nn.Conv2d(dim, 3 * scale ** 2, 1, bias=True)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(F.pixel_shuffle(self.proj(feat), self.scale))


# ======================================================================
# PPBUNet — Main Model (主模型)
# ======================================================================


class PPBUNet(nn.Module):
    """Palette-Painter-Brush U-Net for Anime Super-Resolution.

    训练时通过 model.mi_loss 获取 MIM 正则化损失 (建议 λ=0.01)。
    推理前调用 switch_to_deploy() 折叠 RepSRBlock。
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 64,
        scale: int = 4,
        num_ps_mamba_blocks: int = 4,
        num_hat_groups: int = 4,
        num_hat_blocks_per_group: int = 6,
        window_size: int = 8,
        num_heads: int = 4,
        ssm_d_state: int = 16,
        num_color_slots: int = 32,
        dict_dim: int = 64,
        split_levels: tuple = (1, 2, 4),
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.scale = scale
        self._pad_divisor = 16

        dim_l0 = dim
        dim_l1 = dim * 2
        dim_bn = dim * 4

        bn_depth = max(num_ps_mamba_blocks, 2)
        dec_depth = max(num_hat_groups, 2)
        dec_blocks = max(num_hat_blocks_per_group, 2)

        dec_l0_pre_depth = dec_depth // 2
        dec_l0_post_depth = dec_depth - dec_l0_pre_depth

        print(f"[PPBUNet] v1.4 — Palette-Painter-Brush U-Net (Optimized)")
        print(f"[PPBUNet] 维度: L0={dim_l0}, L1={dim_l1}, BN={dim_bn}")
        print(f"[PPBUNet] 深度: Enc=2+2, BN(AC)={bn_depth}, Dec={dec_depth}+{dec_depth}")
        print(f"[PPBUNet] 上采样: SATUpsampler_v2 {scale}x (各向异性滤波 + 隐式多边形)")
        print(f"[PPBUNet] 旁路: ParallelOAM (0°/90°/45°/135°)")
        print(f"[PPBUNet] 瓶颈: FreqRouter → DC/AC → PSMamba×{bn_depth}")
        print(f"[PPBUNet] 解码: RMA + HAT(heads={num_heads}, ws={window_size}, blocks={dec_blocks}) ×{dec_depth}")
        print(f"[PPBUNet] CMW: 嵌入 Decoder L0 中间 (前{dec_l0_pre_depth}+后{dec_l0_post_depth}组 HAT, dict_dim={dict_dim})")
        print(f"[PPBUNet] 辅助: CreviceAuxHead (训练时直接监督 CMW, 推理零开销)")
        print(f"[PPBUNet] 精修: TopologyGuidedDCN (OAM local)")
        print(f"[PPBUNet] 融合: BaseAnchoredDetailInjector (Identity-Anchored)")
        print(f"[PPBUNet] 跳连: MIM (InfoNCE) + RMA")

        # === Shallow ===
        self.shallow = nn.Sequential(
            RepSRBlock(in_channels, dim_l0, deploy=False),
            nn.LeakyReLU(0.2, inplace=True),
            RepSRBlock(dim_l0, dim_l0, deploy=False),
        )

        # === Parallel OAM Bypass ===
        self.parallel_oam = ParallelOAM(dim_l0)

        # === Encoder L0 ===
        self.enc_l0 = nn.Sequential(*[RCAB(dim_l0) for _ in range(2)])
        self.down_0 = nn.Sequential(
            nn.Conv2d(dim_l0, dim_l1, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # === Encoder L1 ===
        self.enc_l1 = nn.Sequential(*[RCAB(dim_l1) for _ in range(2)])
        self.down_1 = nn.Sequential(
            nn.Conv2d(dim_l1, dim_bn, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # === Bottleneck ===
        self.freq_router = FrequencyRouter(dim_bn, kernel_size=5)
        self.cmw = ConnectedManifoldWormhole(
            query_dim=dim_l0, key_dim=dim_bn, val_dim=dim_bn, dict_dim=dict_dim,
        )
        self.bottleneck_ac = nn.Sequential(
            *[PSMambaBlock(dim_bn, d_state=ssm_d_state, split_levels=split_levels)
              for _ in range(bn_depth)]
        )
        self.bn_conv = nn.Conv2d(dim_bn, dim_bn, 3, 1, 1)

        # === MIM Feature Filters ===
        self.mim_l1 = MIMFeatureFilter(enc_dim=dim_l1, dec_dim=dim_bn)
        self.mim_l0 = MIMFeatureFilter(enc_dim=dim_l0, dec_dim=dim_l1)

        # === Decoder L1 ===
        self.fuse_1 = RiemannianManifoldAlignment(enc_dim=dim_l1, dec_dim=dim_bn, out_dim=dim_l1)
        self.dec_l1 = nn.Sequential(
            *[ResidualHybridAttentionGroup(
                dim_l1, num_heads * 2,
                num_blocks=max(dec_blocks // 2, 1), window_size=window_size,
              ) for _ in range(dec_depth)]
        )
        self.dec_l1_conv = nn.Conv2d(dim_l1, dim_l1, 3, 1, 1)

        # === Decoder L0 (split for mid-decoder CMW insertion) ===
        self.fuse_0 = RiemannianManifoldAlignment(enc_dim=dim_l0, dec_dim=dim_l1, out_dim=dim_l0)
        self.dec_l0_pre = nn.Sequential(
            *[ResidualHybridAttentionGroup(
                dim_l0, num_heads,
                num_blocks=dec_blocks, window_size=window_size,
              ) for _ in range(dec_l0_pre_depth)]
        )
        self.dec_l0_post = nn.Sequential(
            *[ResidualHybridAttentionGroup(
                dim_l0, num_heads,
                num_blocks=dec_blocks, window_size=window_size,
              ) for _ in range(dec_l0_post_depth)]
        )
        self.dec_l0_conv = nn.Conv2d(dim_l0, dim_l0, 3, 1, 1)

        # === Topology-Guided DCN (OAM 方向感知 + 局部几何精修) ===
        self.topo_dcn = TopologyGuidedDCN(dim_l0, geom_dim=dim_l0)

        # === Connected Manifold Wormhole (嵌入 Decoder L0 中间) ===
        # CMW 已在 Bottleneck 段实例化 (self.cmw), 此处标注其架构位置

        # === Crevice Auxiliary Supervision Head (训练时直接监督 CMW, 推理零开销) ===
        self.aux_head = CreviceAuxHead(dim_l0, scale)

        # === Base-Anchored Detail Injector ===
        self.badi = BaseAnchoredDetailInjector(dim=dim_l0)

        # === Upsampler ===
        self.upsampler = AdvancedUpsampler(dim_l0, out_channels, scale)

    def _pad(self, x: torch.Tensor):
        """反射填充, 使 H、W 被 _pad_divisor 整除。"""
        _, _, H, W = x.shape
        d = self._pad_divisor
        pad_h = (d - H % d) % d
        pad_w = (d - W % d) % d
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, H, W

    def switch_to_deploy(self):
        """折叠所有 RepSRBlock 为单 Conv。"""
        for m in self.modules():
            if isinstance(m, RepSRBlock):
                m.switch_to_deploy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in, orig_H, orig_W = self._pad(x)
        x_shallow = self.shallow(x_in)

        geom_prior = self.parallel_oam(x_shallow)

        skip_l0 = self.enc_l0(x_shallow)
        feat = self.down_0(skip_l0)
        skip_l1 = self.enc_l1(feat)
        feat = self.down_1(skip_l1)

        bn_in = feat
        feat_dc, feat_ac = self.freq_router(feat)
        feat_ac = self.bottleneck_ac(feat_ac)
        feat = self.bn_conv(feat_ac) + bn_in

        skip_l1, mi_loss_l1 = self.mim_l1(skip_l1, feat)

        feat = self.fuse_1(feat_enc=skip_l1, feat_dec=feat)
        dec1_in = feat
        feat = self.dec_l1(feat)
        feat = self.dec_l1_conv(feat) + dec1_in

        skip_l0, mi_loss_l0 = self.mim_l0(skip_l0, feat)
        self.mi_loss = mi_loss_l0 + mi_loss_l1

        feat = self.fuse_0(feat_enc=skip_l0, feat_dec=feat)
        dec0_in = feat
        feat = self.dec_l0_pre(feat)
        feat = self.cmw(feat, feat_ac, feat_dc)

        # Auxiliary crevice supervision: 拓扑 loss 直达 CMW (训练时)
        if self.training:
            aux = self.aux_head(feat)
            self.aux_sr = aux[:, :, :orig_H * self.scale, :orig_W * self.scale]

        feat = self.dec_l0_post(feat)
        feat = self.dec_l0_conv(feat) + dec0_in

        feat = self.topo_dcn(feat, geom_prior)
        latent_merged = self.badi(feat_deep=feat, feat_shallow=x_shallow)

        out = self.upsampler(latent_merged)
        return out[:, :, :orig_H * self.scale, :orig_W * self.scale]


# ======================================================================
# Self-Test (自检)
# ======================================================================
if __name__ == "__main__":
    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    cfg = dict(
        in_channels=3, out_channels=3, dim=64, scale=4,
        num_ps_mamba_blocks=4, num_hat_groups=4,
        num_hat_blocks_per_group=6, window_size=8, num_heads=4,
        ssm_d_state=16, num_color_slots=32, dict_dim=64, split_levels=(1, 2, 4),
        use_checkpoint=True,
    )

    model = PPBUNet(**cfg)
    print("=" * 60)
    print("  PPBUNet v1.4 — Functional Validation")
    print("=" * 60)
    print(f"  Trainable params : {count_params(model):,} ({count_params(model)/1e6:.2f}M)")
    print(f"  Upscale factor   : {cfg['scale']}x")
    print(f"  Pad divisor      : {model._pad_divisor}")
    print()

    model.eval()
    for name, shape in [("aligned 32x32", (1, 3, 32, 32)),
                         ("unaligned 47x63", (1, 3, 47, 63)),
                         ("batch=2 64x64", (2, 3, 64, 64))]:
        x = torch.randn(*shape)
        with torch.no_grad():
            y = model(x)
        s = cfg['scale']
        exp = (shape[0], 3, shape[2] * s, shape[3] * s)
        status = "OK" if tuple(y.shape) == exp else "FAIL"
        print(f"  [{status}] {name}: {tuple(x.shape)} -> {tuple(y.shape)}")

    print()
    print("  --- Mutual-Information Maximisation (MIM) Loss ---")
    model.train()
    x_mi = torch.randn(1, 3, 32, 32)
    y_mi = model(x_mi)
    s = cfg['scale']
    exp_mi = (1, 3, 32 * s, 32 * s)
    status_mi = "OK" if tuple(y_mi.shape) == exp_mi else "FAIL"
    print(f"  [{status_mi}] Train mode : {tuple(x_mi.shape)} -> {tuple(y_mi.shape)}")
    print(f"  InfoNCE loss     : {model.mi_loss.item():.6f}")

    print()
    print("  --- CreviceAuxHead Diagnostics ---")
    aux_sr = model.aux_sr
    print(f"  aux_sr shape     : {tuple(aux_sr.shape)}  (expect (1, 3, {32*s}, {32*s}))")
    print(f"  aux_sr range     : [{aux_sr.min().item():.4f}, {aux_sr.max().item():.4f}]  (expect [0, 1])")

    print()
    print("  --- TopologyGuidedDCN Diagnostics ---")
    td = model.topo_dcn
    print(f"  local_offset_mag   : {td.last_local_offset_mag:.6f}  (expect ≈ 0 at init)")
    print(f"  topo_map_mean      : {td.last_topo_map_mean:.6f}")

    print()
    print("  --- ConnectedManifoldWormhole Diagnostics ---")
    cmw = model.cmw
    print(f"  color_hint_mag   : {cmw.last_color_hint_mag:.6f}  (wormhole output magnitude)")
    print(f"  gamma_abs        : {cmw.last_gamma_abs:.6f}  (expect 1e-4 at init)")

    model.eval()
    print()
    print("  --- Structural Re-parameterisation (RepSRBlock Fusion) ---")
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y_train = model(x)

    model.switch_to_deploy()

    with torch.no_grad():
        y_deploy = model(x)

    diff = (y_train - y_deploy).abs().max().item()
    print(f"  Max abs error    : |train - deploy| = {diff:.6e} "
          f"{'OK' if diff < 1e-4 else 'FAIL'}")
    print(f"  Post-fusion params: {count_params(model):,} ({count_params(model)/1e6:.2f}M)")

    print()
    print("  Usage: config['architecture'] = 'ppbunet_v1'")
    print("=" * 60)
