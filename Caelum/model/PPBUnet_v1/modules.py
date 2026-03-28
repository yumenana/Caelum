"""
Supporting Modules
=============================================================

  ChromaticityPaletteExtractor  - Color prototype extraction (palette)
  NormalizedEdgeAttention       - Edge-aware feature modulation (NEA)
  MambaTransformerBridge        - PS-Mamba to HAT transition layer
  AdvancedUpsampler             - INR + 7x7 DWConv output stage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChromaticityPaletteExtractor(nn.Module):
    """Chromaticity Palette Extractor (色彩原型调色盘提取器).

    从深层特征中通过可学习软聚类 (余弦相似度) 提取纯色彩原型,
    忽略局部压缩伪影, 输出全局色彩向量。

    Args:
        dim:        Feature dimension.
        num_colors: Number of learnable color prototypes.
    """

    def __init__(self, dim: int, num_colors: int = 16):
        super().__init__()
        self.num_colors = num_colors

        self.prototypes = nn.Parameter(
            torch.randn(1, num_colors, dim) * 0.02
        )
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.out_proj = nn.Sequential(
            nn.Linear(num_colors * dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        keys = F.normalize(self.key_proj(x_flat), dim=-1)
        protos = F.normalize(self.prototypes.expand(B, -1, -1), dim=-1)

        sim = torch.bmm(keys, protos.transpose(1, 2))
        tau = self.temperature.abs().clamp(min=0.1)
        assignment = F.softmax(sim / tau, dim=-1)

        values = self.value_proj(x_flat)
        palette_colors = torch.bmm(assignment.transpose(1, 2), values)

        return self.out_proj(palette_colors.reshape(B, -1))


class NormalizedEdgeAttention(nn.Module):
    """Normalized Edge Attention (归一化边缘注意力).

    多尺度边缘检测 + 实例归一化门控, 区分真实线稿结构与压缩伪影。

    Args:
        dim: Feature dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.edge_3x3 = nn.Conv2d(dim, dim, 3, 1, 1,
                                   groups=dim, bias=False)
        self.edge_5x5 = nn.Conv2d(dim, dim, 5, 1, 2,
                                   groups=dim, bias=False)
        self.refine = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
        )
        self.norm = nn.InstanceNorm2d(dim, affine=True)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge_fine = self.edge_3x3(x)
        edge_coarse = self.edge_5x5(x)
        edges = self.refine(torch.cat([edge_fine, edge_coarse], dim=1))
        edges = self.norm(edges)
        return x + self.scale * (x * self.gate(edges))


class MambaTransformerBridge(nn.Module):
    """Mamba-Transformer Bridge Layer (Mamba-Transformer 桥接层).

    PS-Mamba (全局语义) 与 HAT (局部重建) 之间的轻量过渡层,
    大核深度卷积空间精化 + 通道重标定。

    Args:
        dim: Feature dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(dim, dim, 7, 1, 3, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.spatial(x)
        w = self.channel_gate(x).unsqueeze(-1).unsqueeze(-1)
        x = x * w
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + residual


class AdvancedUpsampler(nn.Module):
    """INR-based Upsampler (隐式神经表示上采样器).

    双线性插值特征空间连续展开 + 亚像素相位坐标 → 轻量解码器输出 RGB。
    7x7 DWConv 提供 ~4x4 LR 等效感受野, 平坦区域天然结构性平滑。

    Args:
        dim:          Input feature dimension.
        out_channels: Output image channels (3 for RGB).
        scale:        Upsampling factor (1, 2, or 4).
    """

    def __init__(self, dim: int, out_channels: int = 3, scale: int = 2):
        super().__init__()
        self.scale = scale

        if scale == 1:
            self.decode = nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(dim, out_channels, 3, 1, 1),
            )
            return

        if scale not in (2, 4):
            raise ValueError(f"Unsupported scale factor: {scale}")

        self.decode = nn.Sequential(
            nn.Conv2d(dim + 2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 7, 1, 3, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, out_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 1:
            return self.decode(x)

        B, C, H, W = x.shape
        feat = F.interpolate(
            x, scale_factor=self.scale,
            mode='bilinear', align_corners=False,
        )

        phase = self._make_phase(H, W, x.device)
        return self.decode(
            torch.cat([feat, phase.expand(B, -1, -1, -1)], dim=1),
        )

    @torch.no_grad()
    def _make_phase(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Generate sub-pixel phase coordinate grid."""
        s = self.scale

        def _phase_1d(n_lr: int) -> torch.Tensor:
            hr = torch.arange(n_lr * s, device=device, dtype=torch.float32)
            lr = (hr + 0.5) / s - 0.5
            return lr - lr.floor()

        py, px = torch.meshgrid(
            _phase_1d(H), _phase_1d(W), indexing='ij',
        )
        return torch.stack([py, px], dim=0).unsqueeze(0)
