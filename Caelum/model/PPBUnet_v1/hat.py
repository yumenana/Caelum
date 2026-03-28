"""
HAT: Hybrid Attention Transformer for Anime Illustration Reconstruction
========================================================================

Combines window-based self-attention with channel attention for pixel-level
reconstruction. The key addition is the Overlapping Cross-Attention Block
(OCAB), which prevents "window grid artifacts" in flat anime color regions.

Components:
  - WindowAttention:  Multi-head self-attention within fixed windows,
                      with learnable relative position bias.
  - ChannelAttention: Squeeze-and-Excitation channel recalibration.
  - HybridAttentionBlock (HAB): Window attn + channel attn + MLP,
                      alternating regular/shifted window partitions.
  - OverlappingCrossAttention (OCAB): Queries from non-overlapping windows,
                      keys/values from overlapping patches.
  - ResidualHybridAttentionGroup (RHAG): HAB stack + OCAB tail + residual.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition feature map into non-overlapping windows.

    Args:
        x: (B, H, W, C)
    Returns:
        (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B,
               H // window_size, window_size,
               W // window_size, window_size,
               C)
    return (x.permute(0, 1, 3, 2, 4, 5)
             .contiguous()
             .view(-1, window_size, window_size, C))


def window_reverse(windows: torch.Tensor,
                   window_size: int, H: int, W: int) -> torch.Tensor:
    """Inverse of window_partition.

    Args:
        windows: (num_windows*B, window_size, window_size, C)
    Returns:
        (B, H, W, C)
    """
    nH, nW = H // window_size, W // window_size
    B = windows.shape[0] // (nH * nW)
    x = windows.view(B, nH, nW, window_size, window_size, -1)
    return (x.permute(0, 1, 3, 2, 4, 5)
             .contiguous()
             .view(B, H, W, -1))


# ---------------------------------------------------------------------------
# Window-based Multi-head Self-Attention
# ---------------------------------------------------------------------------
class WindowAttention(nn.Module):
    """Window-based Multi-head Self-Attention with Relative Position Bias
    (W-MSA / SW-MSA).

    Args:
        dim:         Feature dimension.
        window_size: Attention window size (single side).
        num_heads:   Number of attention heads.
        qkv_bias:    Add learnable bias to QKV projection.
    """

    def __init__(self, dim: int, window_size: int,
                 num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1),
                        num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            indexing="ij",
        ))
        coords_flat = torch.flatten(coords, 1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", rel.sum(-1))

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:    (nW*B, N, C) where N = window_size^2
            mask: (nW, N, N) or None
        Returns:
            (nW*B, N, C)
        """
        B_, N, C = x.shape
        head_dim = C // self.num_heads

        qkv = (self.qkv(x)
               .reshape(B_, N, 3, self.num_heads, head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.view(B_ // nW, nW, self.num_heads, N, N)
                    + mask.unsqueeze(1).unsqueeze(0))
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        return (attn @ v).transpose(1, 2).reshape(B_, N, C)


# ---------------------------------------------------------------------------
# Channel Attention Block
# ---------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation Channel Attention with Conv body."""

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        mid = max(dim // reduction, 4)
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.body(x)
        w = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * w + residual


# ---------------------------------------------------------------------------
# Overlapping Cross-Attention Block (OCAB)
# ---------------------------------------------------------------------------
class OverlappingCrossAttention(nn.Module):
    """Overlapping Cross-Attention Block (OCAB).

    Queries from non-overlapping windows, keys/values from overlapping
    patches — crosses window boundaries for seamless flat-color rendering.

    Args:
        dim:           Feature dimension.
        num_heads:     Number of attention heads.
        window_size:   Non-overlapping window size.
        overlap_ratio: Extra context ratio per side (relative to window_size).
    """

    def __init__(self, dim: int, num_heads: int,
                 window_size: int = 8, overlap_ratio: float = 0.5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        overlap = int(window_size * overlap_ratio)
        self.overlap_size = window_size + 2 * overlap
        self.scale = (dim // num_heads) ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.unfold = nn.Unfold(
            kernel_size=self.overlap_size,
            stride=window_size,
            padding=overlap,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)  H, W must be divisible by window_size.
        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape
        residual = x
        ws = self.window_size
        os_ = self.overlap_size
        nh = self.num_heads
        hd = C // nh

        x_nhwc = self.norm(x.permute(0, 2, 3, 1))
        x_bchw = x_nhwc.permute(0, 3, 1, 2)

        q_win = window_partition(x_nhwc, ws)
        nW = q_win.shape[0] // B
        q = self.q_proj(q_win.reshape(-1, ws * ws, C))

        kv_unfold = self.unfold(x_bchw)
        kv_unfold = (kv_unfold
                     .view(B, C, os_ * os_, nW)
                     .permute(0, 3, 2, 1)
                     .reshape(B * nW, os_ * os_, C))
        kv = self.kv_proj(kv_unfold)
        k, v = kv.chunk(2, dim=-1)

        q = q.view(-1, ws * ws, nh, hd).transpose(1, 2)
        k = k.view(-1, os_ * os_, nh, hd).transpose(1, 2)
        v = v.view(-1, os_ * os_, nh, hd).transpose(1, 2)

        attn = F.softmax((q * self.scale) @ k.transpose(-2, -1), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(-1, ws * ws, C)
        out = self.out_proj(out)

        out = window_reverse(
            out.view(-1, ws, ws, C), ws, H, W,
        ).permute(0, 3, 1, 2)

        return out + residual


# ---------------------------------------------------------------------------
# Hybrid Attention Block (HAB)
# ---------------------------------------------------------------------------
class HybridAttentionBlock(nn.Module):
    """Hybrid Attention Block (HAB).

    Window self-attention + channel attention + MLP, alternating
    regular and shifted window partitions (Swin-style).

    Args:
        dim:         Feature dimension.
        num_heads:   Number of attention heads.
        window_size: Window side length.
        shift_size:  Shift amount (0 = regular, window_size//2 = shifted).
    """

    def __init__(self, dim: int, num_heads: int,
                 window_size: int = 8, shift_size: int = 0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        self.channel_attn = ChannelAttention(dim)
        self.conv_gate = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def _create_shift_mask(self, H: int, W: int,
                           device: torch.device) -> torch.Tensor | None:
        """Create Swin shifted-window attention mask."""
        if self.shift_size == 0:
            return None

        ws = self.window_size
        ss = self.shift_size
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_win = window_partition(img_mask, ws)
        mask_win = mask_win.view(-1, ws * ws)
        attn_mask = mask_win.unsqueeze(1) - mask_win.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
            attn_mask == 0, 0.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = self.window_size

        shortcut = x
        x_nhwc = self.norm1(x.permute(0, 2, 3, 1))

        if self.shift_size > 0:
            x_nhwc = torch.roll(
                x_nhwc, (-self.shift_size, -self.shift_size), (1, 2),
            )

        x_win = window_partition(x_nhwc, ws).view(-1, ws * ws, C)
        mask = self._create_shift_mask(H, W, x.device)
        attn_out = self.attn(x_win, mask).view(-1, ws, ws, C)
        x_nhwc = window_reverse(attn_out, ws, H, W)

        if self.shift_size > 0:
            x_nhwc = torch.roll(
                x_nhwc, (self.shift_size, self.shift_size), (1, 2),
            )

        x = shortcut + x_nhwc.permute(0, 3, 1, 2)
        x = x + self.conv_gate(x)
        x = self.channel_attn(x)

        shortcut = x
        x_nhwc = self.mlp(self.norm2(x.permute(0, 2, 3, 1)))
        x = shortcut + x_nhwc.permute(0, 3, 1, 2)

        return x


# ---------------------------------------------------------------------------
# Residual Hybrid Attention Group (RHAG)
# ---------------------------------------------------------------------------
class ResidualHybridAttentionGroup(nn.Module):
    """Residual Hybrid Attention Group (RHAG).

    HAB stack with alternating regular/shifted windows + OCAB tail
    for window-boundary repair + 3x3 Conv + residual.

    Args:
        dim:         Feature dimension.
        num_heads:   Number of attention heads.
        num_blocks:  Number of HABs in the group.
        window_size: Window side length.
    """

    def __init__(self, dim: int, num_heads: int,
                 num_blocks: int = 6, window_size: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            shift = 0 if (i % 2 == 0) else window_size // 2
            self.blocks.append(
                HybridAttentionBlock(dim, num_heads, window_size, shift)
            )

        self.ocab = OverlappingCrossAttention(dim, num_heads, window_size)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for blk in self.blocks:
            x = blk(x)
        x = self.ocab(x)
        x = self.conv(x)
        return x + residual
