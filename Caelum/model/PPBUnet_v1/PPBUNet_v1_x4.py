"""
PPBUNet v1.0 — Palette-Painter-Brush U-Net for Anime Super-Resolution (x4)
===========================================================================

4x super-resolution network for anime illustration restoration.

Architecture — Palette . Painter . Brush

  Named after the three stages of an artist's workflow:
    [Palette]  Extract the global color scheme from low-frequency features
    [Painter]  Reconstruct structure via attention-driven U-Net decoding
    [Brush]    Refine fine geometry at full resolution and upsample to HR

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
      | FreqRouter -- DC --> PaletteExtractor --> pal | |         |  * Palette
      |            '- AC --> PSMamba x N              | |         |
      +-----------------------------------------------+ |         |
        |                                               |         |
      +- Decoder -------------------------------------+ |         |
      | MIM+RMA (skip_l1) --> HAT Decoder L1          | |         |  * Painter
      | MIM+RMA (skip_l0) --> HAT Decoder L0          | |         |
      +-----------------------------------------------+ |         |
        |                                               |         |
      GeometryFusion <-- ParallelOAM -------------------+         |
      CornerAwareDCN                                              |  * Brush
      PaletteModulation <-- pal                                   |
        |                                                         |
      BaseAnchoredDetailInjector <-- x_shallow --------------------+
        |
      AMADSUpsampler --> SR Output (4x)

核心模块:
  ★ RepSRBlock: 结构重参数化 (训练多分支, 推理折叠为单 Conv)
  ★ ParallelOAM: 0°/90°/45°/135° 绝对方向基底, 消灭曼哈顿阶梯锯齿
  ★ FrequencyRouter: 频率解耦 → DC 色彩调色盘 + AC PS-Mamba 拓扑追踪
  ★ MIM + RMA: 互信息提纯跳跃连接 + 超球面流形对齐融合
  ★ HAT: 混合注意力 Transformer (窗口自注意力 + OCAB)
  ★ CornerAwareDCN: 角点感知可变形卷积, 二阶曲率精修
  ★ BaseAnchoredDetailInjector: 恒等锚定细节注入 (ControlNet Zero-Init)
  ★ PaletteModulation: 全局色彩渲染
  ★ AMADSUpsampler: 各向异性流形感知动态上采样 (坐标偏移 + grid_sample)


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


class ChromaticityPaletteExtractor(nn.Module):
    """Chromaticity Palette Extractor via Soft K-Means (色彩调色盘提取器).

    从低频直流分量中通过 Soft K-Means 聚类提取全局色彩原型,
    归一化保证尺度不变性, 不受推理阶段分辨率变化影响。

    参数:
        dim:        通道数
        num_colors: 色彩原型数量
    """

    def __init__(self, dim: int, num_colors: int = 16):
        super().__init__()
        self.num_colors = num_colors
        self.prototypes = nn.Parameter(torch.randn(1, num_colors, dim) * 0.02)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, feat_dc: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat_dc.shape
        x_flat = feat_dc.view(B, C, H * W).transpose(1, 2)

        x_norm = F.normalize(x_flat, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)

        sim = torch.matmul(x_norm, proto_norm.transpose(1, 2)) / self.temperature
        attn = F.softmax(sim, dim=-1)

        attn_t = attn.transpose(1, 2)
        palette_sum = torch.matmul(attn_t, x_flat)
        attn_weights_sum = attn_t.sum(dim=-1, keepdim=True)
        palette_normalized = palette_sum / (attn_weights_sum + 1e-6)

        return palette_normalized.mean(dim=1)


# ======================================================================
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

        z_enc = F.normalize(self.proj_enc(feat_enc), dim=1)
        z_dec = F.normalize(self.proj_dec(feat_dec), dim=1)

        mi_loss = torch.zeros(1, device=feat_enc.device, dtype=feat_enc.dtype)
        if self.training:
            N = H * W
            num_s = min(self.num_samples, N)

            z_e = z_enc.reshape(B, -1, N)
            z_d = z_dec.reshape(B, -1, N)

            with torch.no_grad():
                energy = z_enc.detach().pow(2).sum(dim=1, keepdim=True)
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
                 eps: float = 1e-5):
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
        """对数映射 log_p(q): 将 q 投影到 p 的切空间 T_p(S^{n-1})。"""
        inner = torch.sum(p * q, dim=1, keepdim=True)
        v = q - inner * p

        safe_threshold = 1.0 - 1e-3
        near_aligned = (inner > safe_threshold)

        inner_clamped = inner.clamp(-1.0 + self.eps, 1.0 - self.eps)
        theta = torch.acos(inner_clamped)
        v_norm = torch.norm(v, p=2, dim=1, keepdim=True).clamp(min=self.eps)

        scale = torch.where(near_aligned, torch.ones_like(theta), theta / v_norm)
        return scale * v

    def exp_map(self, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """指数映射 exp_p(v): 将切向量 v 沿测地线映射回超球面。"""
        v_norm = torch.norm(v, p=2, dim=1, keepdim=True)

        near_zero = (v_norm < 1e-3)
        v_norm_safe = v_norm.clamp(min=self.eps)

        full_result = torch.cos(v_norm) * p + torch.sin(v_norm) * (v / v_norm_safe)
        taylor_result = p + v

        return torch.where(near_zero, taylor_result, full_result)

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


class PaletteModulation(nn.Module):
    """Palette Modulation via LayerScale (色彩调制器).

    将全局调色盘色彩向量投影到特征空间, 以 LayerScale (γ=1e-4)
    加性融合回重构特征, 无 InstanceNorm 保留绝对色彩信息。

    参数:
        feat_dim:    空间特征维度
        palette_dim: 调色盘维度
    """

    def __init__(self, feat_dim: int, palette_dim: int):
        super().__init__()
        self.palette_proj = nn.Sequential(
            nn.Linear(palette_dim, feat_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 3, 1, 1),
        )
        self.gamma = nn.Parameter(torch.ones(1, feat_dim, 1, 1) * 1e-4)

    def forward(self, feat: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        pal = self.palette_proj(palette)
        pal_map = pal.view(B, C, 1, 1).expand(B, C, H, W)
        fused = self.fuse(torch.cat([feat, pal_map], dim=1))
        return feat + self.gamma * fused


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

class CornerAwareDCN(nn.Module):
    """
    角点感知可变形卷积 — 路径抽象版。
    将 RDB 内嵌于偏移量生成器之前，赋予网络对“线条走向”的深刻理解，
    指导 DCN 精确变形，杜绝发尖粘连。
    """
    def __init__(self, dim: int, init_gamma: float = 1e-2):
        super().__init__()
        self.dim = dim

        self.corner_prior = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid(),
        )

        self.path_abstractor = DensePathExtractor(dim + 1)
        
        self.offset_gen = nn.Sequential(
            self.path_abstractor,
            nn.Conv2d(dim + 1, dim // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 2, 27, 3, 1, 1),
        )
        nn.init.constant_(self.offset_gen[-1].weight, 0.0)
        nn.init.constant_(self.offset_gen[-1].bias, 0.0)

        self.dcn_weight = nn.Parameter(torch.empty(dim, dim, 3, 3))
        self.dcn_bias = nn.Parameter(torch.zeros(dim))
        nn.init.kaiming_uniform_(self.dcn_weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(dim * 9)
        nn.init.uniform_(self.dcn_bias, -bound, bound)

        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1) * init_gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c_prior = self.corner_prior(x)
        offset_mask = self.offset_gen(torch.cat([x, c_prior], dim=1))
        offset = offset_mask[:, :18, :, :]
        mask = torch.sigmoid(offset_mask[:, 18:, :, :])

        dcn_out = deform_conv2d(
            input=x, offset=offset, weight=self.dcn_weight,
            bias=self.dcn_bias, stride=1, padding=1, mask=mask,
        )
        return x + self.gamma * dcn_out


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


class AMADSUpsampler(nn.Module):
    """Anisotropic Manifold-Aware Dynamic Sampler (各向异性流形感知动态采样器).

    从"预测滤波核"升维至"预测流形坐标偏移" (DySample 机制):
    全卷积密集几何感知 + grid_sample 双线性插值, 天生空间连续,
    从数学底层免疫阶梯伪影。门控 α(x) 调制偏移幅度: 平坦区纯双线性,
    边缘处释放各向异性偏移。相位域零和高频残差补偿插值高频衰减。

    参数:
        dim:   通道数
        scale: 上采样因子
    """

    def __init__(self, dim: int, scale: int = 4):
        super().__init__()
        self.scale = scale
        self.dim = dim
        self.n_phases = scale ** 2

        self.geometry_extractor = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, 32, kernel_size=1, bias=False),
        )

        self.offset_generator = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(64, self.n_phases * 2, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.offset_generator[-1].weight)

        self.edge_gate = nn.Sequential(
            nn.Conv2d(dim, 16, kernel_size=3, padding=1, bias=False),
            nn.Mish(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.detail_injector = nn.Sequential(
            nn.Conv2d(dim, dim * self.n_phases, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
        )
        nn.init.zeros_(self.detail_injector[0].weight)
        nn.init.zeros_(self.detail_injector[0].bias)

        self.max_neg_lobe = 0.25

        phase_offsets = self._generate_phase_ticks(scale)
        self.register_buffer('phase_offsets', phase_offsets)

    def _generate_phase_ticks(self, scale: int) -> torch.Tensor:
        step = 1.0 / scale
        start = (step - 1.0) / 2.0
        ticks = torch.linspace(start, -start, scale)
        y, x = torch.meshgrid(ticks, ticks, indexing='ij')
        return torch.stack([x.flatten(), y.flatten()], dim=1).view(1, scale ** 2, 2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        s = self.scale

        geom_feat = self.geometry_extractor(x)
        alpha = self.edge_gate(x)

        raw_offsets = self.offset_generator(geom_feat).view(B, self.n_phases, 2, H, W)
        offsets = raw_offsets * alpha.unsqueeze(1)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=x.dtype, device=x.device),
            torch.arange(W, dtype=x.dtype, device=x.device),
            indexing='ij',
        )
        grid_base = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).unsqueeze(1)
        grid_sampled = grid_base + self.phase_offsets + offsets

        grid_sampled[:, :, 0, :, :] = 2.0 * (grid_sampled[:, :, 0, :, :] + 0.5) / W - 1.0
        grid_sampled[:, :, 1, :, :] = 2.0 * (grid_sampled[:, :, 1, :, :] + 0.5) / H - 1.0

        grid_sampled = grid_sampled.transpose(1, 2).reshape(B, 2 * self.n_phases, H, W)
        grid_sampled = F.pixel_shuffle(grid_sampled, s)
        grid_sampled = grid_sampled.permute(0, 2, 3, 1)

        out_base = F.grid_sample(
            x, grid_sampled, mode='bilinear',
            padding_mode='border', align_corners=False,
        )

        detail_logits = self.detail_injector(x)
        detail_logits = detail_logits.view(B, self.dim, self.n_phases, H, W)
        detail_logits = detail_logits - detail_logits.mean(dim=2, keepdim=True)
        detail_logits = detail_logits.view(B, self.dim * self.n_phases, H, W)
        detail_expanded = F.pixel_shuffle(detail_logits, s)

        alpha_up = F.interpolate(alpha, scale_factor=float(s), mode='nearest')
        return out_base + alpha_up * (self.max_neg_lobe * detail_expanded)


class AdvancedUpsampler(nn.Module):
    """AMADS-based Upsampler (AMADS 上采样器).

    封装 AMADSUpsampler + PReLU + 尾部 Conv 输出 RGB。

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
        elif scale == 2:
            self.up = nn.Sequential(
                AMADSUpsampler(dim, scale=2),
                nn.PReLU(dim),
            )
        elif scale == 4:
            self.up = nn.Sequential(
                AMADSUpsampler(dim, scale=4),
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
        num_palette_colors: int = 16,
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

        print(f"[PPBUNet] v1.0 — Palette-Painter-Brush U-Net")
        print(f"[PPBUNet] 维度: L0={dim_l0}, L1={dim_l1}, BN={dim_bn}")
        print(f"[PPBUNet] 深度: Enc=2+2, BN(AC)={bn_depth}, Dec={dec_depth}+{dec_depth}")
        print(f"[PPBUNet] 上采样: AMADSUpsampler {scale}x")
        print(f"[PPBUNet] 旁路: ParallelOAM (0°/90°/45°/135°)")
        print(f"[PPBUNet] 瓶颈: FreqRouter → DC→Palette({num_palette_colors}) / AC→PSMamba×{bn_depth}")
        print(f"[PPBUNet] 解码: RMA + HAT(heads={num_heads}, ws={window_size}, blocks={dec_blocks}) ×{dec_depth}")
        print(f"[PPBUNet] 精修: CornerAwareDCN + PaletteModulation")
        print(f"[PPBUNet] 融合: BaseAnchoredDetailInjector (Identity-Anchored)")
        print(f"[PPBUNet] 跳连: MIM + RMA")

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
        self.palette_extractor = ChromaticityPaletteExtractor(dim_bn, num_palette_colors)
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

        # === Decoder L0 ===
        self.fuse_0 = RiemannianManifoldAlignment(enc_dim=dim_l0, dec_dim=dim_l1, out_dim=dim_l0)
        self.dec_l0 = nn.Sequential(
            *[ResidualHybridAttentionGroup(
                dim_l0, num_heads,
                num_blocks=dec_blocks, window_size=window_size,
              ) for _ in range(dec_depth)]
        )
        self.dec_l0_conv = nn.Conv2d(dim_l0, dim_l0, 3, 1, 1)

        # === Geometry Fusion ===
        self.geom_fusion = nn.Sequential(
            nn.Conv2d(dim_l0 * 2, dim_l0, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_l0, dim_l0, kernel_size=3, padding=1, bias=False),
        )
        self.geom_gamma = nn.Parameter(torch.ones(1, dim_l0, 1, 1) * 1e-2)

        # === Corner-Aware DCN ===
        self.corner_dcn = CornerAwareDCN(dim_l0, init_gamma=1e-2)

        # === Palette Modulation ===
        self.color_render = PaletteModulation(feat_dim=dim_l0, palette_dim=dim_bn)

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
        palette = self.palette_extractor(feat_dc)
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
        feat = self.dec_l0(feat)
        feat = self.dec_l0_conv(feat) + dec0_in

        geom_fused = self.geom_fusion(torch.cat([feat, geom_prior], dim=1))
        feat = feat + self.geom_gamma * geom_fused

        feat = self.corner_dcn(feat)
        rendered_feat = self.color_render(feat=feat, palette=palette)
        latent_merged = self.badi(feat_deep=rendered_feat, feat_shallow=x_shallow)

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
        ssm_d_state=16, num_palette_colors=16, split_levels=(1, 2, 4),
        use_checkpoint=True,
    )

    model = PPBUNet(**cfg)
    print("=" * 60)
    print("  PPBUNet v1.0 — Functional Validation")
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
