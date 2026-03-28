"""
PS-Mamba (Super-Resolution Specialised Edition)
============================================================================

Original PS-Mamba performs progressive-split cross-directional SSM scanning
to capture anime line-art topology without flattening 2-D spatial structure.

This SR-specialised revision applies four surgical modifications that
eliminate information-destroying mechanisms harmful to high-frequency
super-resolution reconstruction:

  Surgery 1 — Abolish multiplicative z-gate
      Original: y = y * F.silu(z)   (kills high-freq when z -> 0)
      Revised : single-path projection, no z branch at all.

  Surgery 2 — Direct output projection
      SSM scan result is projected straight to d_model without any
      dynamic weight modulation.

  Surgery 3 — Pure linear multi-scale fusion
      Original: Conv1x1 + GELU  (non-linearity breaks phase superposition)
      Revised : Conv1x1 only, no bias, no activation.

  Surgery 4 — Absolute residual highway + SimpleGate FFN
      out = mamba(x) + x  (Mamba only predicts a small correction)
      FFN uses half-channel multiplicative gate (x1 * x2) instead of GELU,
      preserving high-frequency fidelity.

Complexity: O(N) per scan direction (vs O(N^2) for attention).

Note: Sequential scan loop uses pure PyTorch for Windows compatibility.
On Linux, replace _selective_scan with mamba_ssm.selective_scan_fn
(CUDA kernel) for ~10x speedup, or wrap with torch.compile() on
PyTorch 2.0+.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Selective State Space Model (S6) -- SR-specialised, de-gated
# ---------------------------------------------------------------------------
class SelectiveSSM(nn.Module):
    """SR-specialised Selective SSM.

    Compared to the standard Mamba SSM:
      - Single-path input projection (no z-gate branch).
      - Direct output: no y = y * F.silu(z) high-frequency crusher.
      - D skip-connection preserved for identity-mapping initialisation.

    State-space recurrence:
        h(t) = A_bar * h(t-1) + B_bar * x(t)
        y(t) = C * h(t) + D * x(t)

    Args:
        d_model:  Input / output feature dimension.
        d_state:  SSM hidden state dimension N.
        d_conv:   Causal 1-D convolution kernel size.
        expand:   Internal projection expansion factor.
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = max(1, math.ceil(d_model / 16))

        # Surgery 1: single-path projection (was d_inner * 2 for z-gate)
        self.in_proj = nn.Linear(d_model, self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * d_state, bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S4D-Real initialisation
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(
            torch.log(A.unsqueeze(0).expand(self.d_inner, -1))
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    # ---- Sequential scan (pure PyTorch, no CUDA dependency) ----
    def _selective_scan(self, x, delta, A, B, C):
        """Sequential selective scan.

        Args:
            x:     (batch, length, d_inner)
            delta: (batch, length, d_inner)
            A:     (d_inner, d_state)
            B:     (batch, length, d_state)
            C:     (batch, length, d_state)
        Returns:
            y:     (batch, length, d_inner)
        """
        bsz, L, D = x.shape
        N = A.shape[1]

        dA = torch.exp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )
        dBx = (
            delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)
        )

        h = x.new_zeros(bsz, D, N)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dBx[:, t]
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)
            ys.append(y_t)

        return torch.stack(ys, dim=1)

    # ---- Forward ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, length, d_model)
        Returns:
            (batch, length, d_model)
        """
        _, L, _ = x.shape

        # Single-path projection (no chunk, no z-gate)
        x_val = self.in_proj(x)

        # Causal 1-D depth-wise convolution for local context
        x_val = self.conv1d(
            x_val.transpose(1, 2)
        )[:, :, :L].transpose(1, 2)
        x_val = F.silu(x_val)

        # Input-dependent SSM parameter generation
        ssm_params = self.x_proj(x_val)
        dt_raw, B_param, C_param = ssm_params.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1,
        )
        delta = F.softplus(self.dt_proj(dt_raw))
        A = -torch.exp(self.A_log)

        # Core recurrent scan
        y = self._selective_scan(x_val, delta, A, B_param, C_param)
        y = y + x_val * self.D.unsqueeze(0).unsqueeze(0)

        # Surgery 2: direct output projection (no y = y * F.silu(z))
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# Cross-directional 2-D scan
# ---------------------------------------------------------------------------
class CrossDirectionalScan(nn.Module):
    """Scans a 2-D feature map along horizontal and vertical axes with
    independent SSMs, then fuses via linear projection.

    Captures spatial dependencies in both directions without destroying
    the 2-D topology critical for anime line-art.
    """

    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.ssm_h = SelectiveSSM(dim, d_state)
        self.ssm_v = SelectiveSSM(dim, d_state)
        self.fusion = nn.Linear(dim * 2, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Horizontal scan: each row as a sequence
        x_h = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        y_h = self.ssm_h(x_h).reshape(B, H, W, C)

        # Vertical scan: each column as a sequence
        x_v = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        y_v = self.ssm_v(x_v).reshape(B, W, H, C).permute(0, 2, 1, 3)

        # Linear fusion
        y = self.fusion(torch.cat([y_h, y_v], dim=-1))
        return y.permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Progressive-Split Mamba Block (SR-specialised)
# ---------------------------------------------------------------------------
class PSMambaBlock(nn.Module):
    """Progressive-Split Mamba block for SR high-frequency topology tracking.

    Multi-scale feature extraction via geometric partitioning:
      Level 1 (1x1): Global context  -- full feature map scan
      Level 2 (2x2): Regional context -- quadrant scans
      Level 3 (4x4): Local context    -- patch scans

    SR-specific modifications vs. original:
      - Surgery 3: Pure linear fusion (no GELU) to preserve phase superposition
      - Surgery 4: Absolute residual highway + SimpleGate FFN

    Args:
        dim:          Feature dimension (channels).
        d_state:      SSM hidden state dimension.
        split_levels: Tuple of partition counts per axis, e.g. (1, 2, 4).
    """

    def __init__(self, dim: int, d_state: int = 16,
                 split_levels: tuple = (1, 2, 4)):
        super().__init__()
        self.split_levels = split_levels

        self.scanners = nn.ModuleList([
            CrossDirectionalScan(dim, d_state) for _ in split_levels
        ])

        # Surgery 3: pure linear fusion -- no GELU, no bias
        self.fusion = nn.Conv2d(
            dim * len(split_levels), dim, 1, bias=False,
        )

        # SimpleGate FFN (half-channel multiplicative gate, no activation)
        self.norm = nn.LayerNorm(dim)
        self.ffn_proj1 = nn.Linear(dim, dim * 2)
        self.ffn_proj2 = nn.Linear(dim, dim)

    # ------------------------------------------------------------------
    @staticmethod
    def _split_scan_merge(
        x: torch.Tensor,
        scanner: CrossDirectionalScan,
        num_parts: int,
    ) -> torch.Tensor:
        """Split -> scan -> merge for a single partition level."""
        B, C, H, W = x.shape
        if num_parts == 1:
            return scanner(x)

        pH, pW = H // num_parts, W // num_parts

        x = (x
             .reshape(B, C, num_parts, pH, num_parts, pW)
             .permute(0, 2, 4, 1, 3, 5)
             .reshape(B * num_parts * num_parts, C, pH, pW))

        y = scanner(x)

        y = (y
             .reshape(B, num_parts, num_parts, C, pH, pW)
             .permute(0, 3, 1, 4, 2, 5)
             .reshape(B, C, H, W))
        return y

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) -- H, W must be divisible by max(split_levels).
        Returns:
            (B, C, H, W)
        """
        residual = x

        # Multi-scale progressive-split scan
        multi_scale = [
            self._split_scan_merge(x, scanner, n)
            for scanner, n in zip(self.scanners, self.split_levels)
        ]

        # Surgery 3: pure linear weighted fusion
        x_mamba = self.fusion(torch.cat(multi_scale, dim=1))

        # Surgery 4: absolute residual highway
        x = x_mamba + residual

        # SimpleGate FFN (pre-norm)
        shortcut = x
        x = x.permute(0, 2, 3, 1)                  # (B, H, W, C)
        x = self.norm(x)
        x1, x2 = self.ffn_proj1(x).chunk(2, dim=-1)
        x = x1 * x2                                 # SimpleGate: no activation
        x = self.ffn_proj2(x)
        x = x.permute(0, 3, 1, 2) + shortcut

        return x
