"""
Loss Functions for Anime Super-Resolution Training
===================================================

PPBUNet 训练用自包含损失函数库。

12 个子损失覆盖像素、色彩、频域、空间、感知、对抗六个维度,
按 2 阶段渐进式启用, 由 CaelumLossV2 统一调度。

各损失的职责划分:
  L1 + Flat     像素对不对 / 色块稳不稳    (空间域像素锚定)
  OKLCH + CG    颜色准不准 / 色彩溢不溢    (感知色彩空间)
  Crevice       夹缝色偏修不修              (形态学 + OKLCH)
  Histogram     分布偏不偏                  (边缘掩码不对称直方图)
  Gibbs         能量超没超                  (频域单侧天花板)
  STGV          平坦区纯不纯                (形态学硬掩码 + Charbonnier)
  SmoothGradH   渐变顺不顺                  (结构张量带通 + Hessian)
  Angular       线条直不直                  (Farid 梯度角距离)
  TurningPoint  拐弯尖不尖                  (结构张量角点响应 + 密度衰减空间NMS)
  Perceptual    语义对不对                  (Danbooru ConvNeXt 余弦距离)
  Decoupled D   结构真不真                  (解耦对抗, 纹理免罚)

Components:

  Utility:
    RGBToOklab                        sRGB -> Oklab perceptual color space
    compute_edge_mask                  Local variance -> soft edge mask

  Pixel:
    FlatRegionAwareLoss                Flat-region weighted L1

  Color:
    OklchColorLoss                     OKLCH chroma + hue cosine loss
    ChromaGradientLoss                 Oklab a/b chroma gradient alignment
    CreviceColorLoss                   Morphological crevice detection + OKLCH
    MaskedAsymmetricHistogramLoss      Masked asymmetric histogram

  Spatial / Frequency:
    GibbsRingingPenaltySWT             Haar SWT one-sided overshoot penalty
    StrictFlatTGVLoss                  Strict flat-region TGV (Charbonnier)
    SmoothGradientHessianLoss          Structure-tensor guided Hessian penalty
    AngularFluencyLoss                 Farid 7x7 gradient angular distance
    TopologicalSingularityLoss          Structure tensor corner + density-decayed spatial NMS

  Gate Regularization:
    GateTolerancePenalty               Masked soft-gating hinge + cosine annealing

  Perceptual:
    AnimePerceptualLossV2              Danbooru ConvNeXt cosine perceptual

  GAN:
    UNetDiscriminatorSN                U-Net D + spectral normalization
    DecoupledUNetDiscriminatorSN       Structure-texture decoupled D (内置 GuidedFilter)
    DecoupledGANLoss                   Asymmetric adversarial loss

  Combined:
    CaelumLossV2                       2-phase progressive (12 sub-losses)

作者: YumeNana
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

try:
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file as safetensors_load
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False


# ======================================================================
# Utility (基础工具)
# ======================================================================


class RGBToOklab(nn.Module):
    """sRGB to Oklab Differentiable Conversion (sRGB → Oklab 可微分转换).

    sRGB → Linear RGB (gamma) → LMS (M1) → LMS^(1/3) → Oklab (M2)。
    输出: L ∈ [0,1] 感知亮度, a/b ∈ ~[-0.4,0.4] 绿红/蓝黄轴。
    强制 float32: float16 下 pow(1/3) 反向梯度在 x≈0 处溢出。
    """

    def __init__(self):
        super().__init__()
        m1 = torch.tensor([
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ], dtype=torch.float32)
        self.register_buffer('m1', m1.unsqueeze(2).unsqueeze(3))

        m2 = torch.tensor([
            [ 0.2104542553,  0.7936177850, -0.0040720468],
            [ 1.9779984951, -2.4285922050,  0.4505937099],
            [ 0.0259040371,  0.7827717662, -0.8086757660],
        ], dtype=torch.float32)
        self.register_buffer('m2', m2.unsqueeze(2).unsqueeze(3))

    @staticmethod
    def _srgb_to_linear(srgb: torch.Tensor) -> torch.Tensor:
        srgb = srgb.clamp(0, 1)
        return torch.where(
            srgb <= 0.04045,
            srgb / 12.92,
            ((srgb + 0.055) / 1.055) ** 2.4
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type=rgb.device.type, enabled=False):
            rgb = rgb.float()
            linear = self._srgb_to_linear(rgb)
            lms = F.conv2d(linear, self.m1).clamp(min=1e-10).pow(1.0 / 3.0)
            return F.conv2d(lms, self.m2)


def compute_edge_mask(img: torch.Tensor, patch_size: int = 5) -> torch.Tensor:
    """Local variance -> soft edge mask. Flat->0, edge->1."""
    padding = patch_size // 2
    local_mean = F.avg_pool2d(img, patch_size, stride=1, padding=padding)
    local_var = (F.avg_pool2d(img ** 2, patch_size, stride=1, padding=padding)
                 - local_mean ** 2).clamp(min=0)
    local_var = local_var.max(dim=1, keepdim=True)[0]
    return (local_var / (3.0 / 255.0)).clamp(0, 1)


# ======================================================================
# Pixel Loss (像素损失)
# ======================================================================


class FlatRegionAwareLoss(nn.Module):
    """Flat-Region Aware Weighted L1 (平坦区域感知加权 L1).

    基于 GT 局部方差检测纯色区域, 放大其 L1 权重,
    防止 60-80% 平坦像素的微小误差被边缘像素梯度淹没。
    与 StrictFlatTGV 互补: TGV 约束梯度→0, 本损失锚定像素值→GT。

    Args:
        patch_size:    局部方差计算窗口
        flat_weight:   平坦区 L1 权重
        detail_weight: 细节区 L1 权重
    """

    def __init__(self, patch_size: int = 5, flat_weight: float = 10.0,
                 detail_weight: float = 1.0):
        super().__init__()
        self.patch_size = patch_size
        self.flat_weight = flat_weight
        self.detail_weight = detail_weight
        self.padding = patch_size // 2

    def _compute_local_variance(self, img: torch.Tensor) -> torch.Tensor:
        local_mean = F.avg_pool2d(img, self.patch_size, stride=1,
                                  padding=self.padding)
        local_mean_sq = F.avg_pool2d(img ** 2, self.patch_size, stride=1,
                                     padding=self.padding)
        local_var = (local_mean_sq - local_mean ** 2).clamp(min=0)
        return local_var.max(dim=1, keepdim=True)[0]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        local_var = self._compute_local_variance(target)
        flat_mask = (local_var < 1.0 / 255.0).float()
        weight_map = flat_mask * self.flat_weight + (1 - flat_mask) * self.detail_weight
        return (torch.abs(pred - target) * weight_map).mean()


# ======================================================================
# Color Losses (色彩损失)
# ======================================================================


class OklchColorLoss(nn.Module):
    """OKLCH Perceptual Color Loss (OKLCH 感知色彩损失).

    色度 L1 + 色相余弦联合约束, atan2-free。

    Args:
        alpha: 色度 L1 权重
        beta:  色相余弦权重
        eps:   低饱和度保护阈值
    """

    def __init__(self, alpha: float = 1.0, beta: float = 2.0, eps: float = 0.01):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.rgb_to_oklab = RGBToOklab()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_lab = self.rgb_to_oklab(pred)
        target_lab = self.rgb_to_oklab(target)

        p_a, p_b = pred_lab[:, 1:2], pred_lab[:, 2:3]
        t_a, t_b = target_lab[:, 1:2], target_lab[:, 2:3]

        C_pred = (p_a ** 2 + p_b ** 2 + 1e-12).sqrt()
        C_gt = (t_a ** 2 + t_b ** 2 + 1e-12).sqrt()

        chroma_loss = torch.abs(C_pred - C_gt).mean()

        dot_ab = p_a * t_a + p_b * t_b
        cos_hue_diff = (dot_ab / (C_pred * C_gt + 1e-12)).clamp(-1, 1)
        hue_loss = (C_gt.clamp(min=self.eps) * (1.0 - cos_hue_diff)).mean()

        return self.alpha * chroma_loss + self.beta * hue_loss


class ChromaGradientLoss(nn.Module):
    """Oklab Chroma Gradient Alignment Loss (Oklab 色度梯度对齐损失).

    Sobel 梯度直接约束色度边缘的位置和强度与 GT 对齐, 抗色彩溢出。

    Args:
        a_weight: Oklab a 轴权重
        b_weight: Oklab b 轴权重
    """

    def __init__(self, a_weight: float = 1.0, b_weight: float = 1.0):
        super().__init__()
        self.rgb_to_oklab = RGBToOklab()
        self.a_weight = a_weight
        self.b_weight = b_weight

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def _get_gradient(self, tensor: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(tensor, self.sobel_x, padding=1)
        gy = F.conv2d(tensor, self.sobel_y, padding=1)
        return torch.abs(gx) + torch.abs(gy)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_lab = self.rgb_to_oklab(pred)
        target_lab = self.rgb_to_oklab(target)

        loss_a = torch.abs(
            self._get_gradient(pred_lab[:, 1:2]) - self._get_gradient(target_lab[:, 1:2])
        ).mean()
        loss_b = torch.abs(
            self._get_gradient(pred_lab[:, 2:3]) - self._get_gradient(target_lab[:, 2:3])
        ).mean()
        return loss_a * self.a_weight + loss_b * self.b_weight


class CreviceColorLoss(nn.Module):
    """Crevice Region Color Recovery Loss (夹缝区域色彩恢复损失).

    形态学闭运算检测双线夹缝 + OKLCH 色度/色相恢复。
    窄色带被描边夹持时, JPEG 4:2:0 色度子采样导致色相严重偏移。

    Args:
        closing_kernel:  闭运算核大小
        edge_threshold:  边缘检测阈值
        post_dilation:   后处理膨胀量
        eps:             低饱和度保护阈值
    """

    def __init__(self, closing_kernel: int = 11,
                 edge_threshold: float = 0.5,
                 post_dilation: int = 2, eps: float = 0.01):
        super().__init__()
        self.closing_kernel = closing_kernel
        self.closing_padding = closing_kernel // 2
        self.edge_threshold = edge_threshold
        self.post_dilation = post_dilation
        self.eps = eps
        self.rgb_to_oklab = RGBToOklab()

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def _get_omni_edge(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        img_flat = img.reshape(B * C, 1, H, W)
        gx = F.conv2d(img_flat, self.sobel_x, padding=1)
        gy = F.conv2d(img_flat, self.sobel_y, padding=1)
        grad = (torch.abs(gx) + torch.abs(gy)).reshape(B, C, H, W)
        return grad.max(dim=1, keepdim=True)[0]

    def _compute_crevice_mask(self, target: torch.Tensor) -> torch.Tensor:
        edge_soft = (self._get_omni_edge(target) / self.edge_threshold).clamp(0, 1)
        ck, cp = self.closing_kernel, self.closing_padding
        dilated = F.max_pool2d(edge_soft, ck, stride=1, padding=cp)
        closed = -F.max_pool2d(-dilated, ck, stride=1, padding=cp)
        mask = F.relu(closed - edge_soft)
        if self.post_dilation > 0:
            dk = 2 * self.post_dilation + 1
            mask = F.max_pool2d(mask, dk, stride=1, padding=self.post_dilation)
        return mask

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        crevice_mask = self._compute_crevice_mask(target)
        mask_sum = crevice_mask.sum().clamp(min=1.0)

        pred_lab = self.rgb_to_oklab(pred)
        target_lab = self.rgb_to_oklab(target)
        p_a, p_b = pred_lab[:, 1:2], pred_lab[:, 2:3]
        t_a, t_b = target_lab[:, 1:2], target_lab[:, 2:3]

        C_pred = (p_a ** 2 + p_b ** 2 + 1e-12).sqrt()
        C_gt = (t_a ** 2 + t_b ** 2 + 1e-12).sqrt()

        chroma_loss = (torch.abs(C_pred - C_gt) * crevice_mask).sum() / mask_sum

        dot_ab = p_a * t_a + p_b * t_b
        cos_hue_diff = (dot_ab / (C_pred * C_gt + 1e-12)).clamp(-1, 1)
        hue_loss = (C_gt.clamp(min=self.eps)
                    * (1.0 - cos_hue_diff)
                    * crevice_mask).sum() / mask_sum

        return chroma_loss + 2.0 * hue_loss


class MaskedAsymmetricHistogramLoss(nn.Module):
    """Masked Asymmetric Histogram Loss (掩码不对称直方图损失).

    仅在边缘膨胀区域提取软直方图, 防止全局平坦色块稀释局部异色梯度。
    GT 经高斯+双线性凸包降采样模拟退化色彩混合。
    不对称散度: 重罚"无中生有"杂色 (pred > gt), 轻罚"未能恢复"细节。

    Args:
        num_bins:       直方图量化箱数
        sigma:          软直方图高斯核宽
        weight_extra:   幻觉杂色惩罚权重 (重罚)
        weight_missing: 丢失色彩惩罚权重 (轻罚)
        mask_dilation:  边缘掩码膨胀范围
    """

    def __init__(self, num_bins: int = 64, sigma: float = 0.02,
                 weight_extra: float = 5.0, weight_missing: float = 1.0,
                 mask_dilation: int = 8):
        super().__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        self.weight_extra = weight_extra
        self.weight_missing = weight_missing
        self.mask_dilation = mask_dilation

        self.register_buffer('bin_centers', torch.linspace(0, 1, num_bins))

        sobel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel.T.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

        grid = torch.arange(-1., 2.)
        gaussian = torch.exp(-grid ** 2 / 2.0)
        g_kernel = torch.outer(gaussian, gaussian)
        g_kernel = g_kernel / g_kernel.sum()
        self.register_buffer('g_kernel', g_kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def _get_dilated_edge_mask(self, x: torch.Tensor) -> torch.Tensor:
        x_pad = F.pad(x, (1, 1, 1, 1), mode='reflect')
        gx = F.conv2d(x_pad, self.sobel_x, groups=3)
        gy = F.conv2d(x_pad, self.sobel_y, groups=3)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8).max(dim=1, keepdim=True)[0]
        core_edge = (grad_mag > 0.05).float()
        kernel_size = self.mask_dilation * 2 + 1
        dilated_mask = F.max_pool2d(core_edge, kernel_size=kernel_size,
                                    stride=1, padding=self.mask_dilation)
        return dilated_mask

    def _masked_soft_histogram(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.reshape(B, C, N, 1)
        mask_flat = mask.reshape(B, 1, N, 1)
        bins = self.bin_centers.view(1, 1, 1, -1)
        weights = torch.exp(-((x_flat - bins) ** 2) / (2 * self.sigma ** 2))
        hist_unnorm = (weights * mask_flat).sum(dim=2)
        valid_pixels = mask_flat.sum(dim=2) + 1e-8
        return hist_unnorm / valid_pixels

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hr_mask = self._get_dilated_edge_mask(target)
            t_blur = F.conv2d(F.pad(target, (1, 1, 1, 1), mode='reflect'),
                              self.g_kernel, groups=3)
            t_down = F.interpolate(t_blur, scale_factor=0.5, mode='bilinear',
                                   align_corners=False)

        p_down = F.interpolate(pred, scale_factor=0.5, mode='bilinear',
                               align_corners=False)
        mask_down = F.interpolate(hr_mask, scale_factor=0.5, mode='nearest')

        hist_pred = self._masked_soft_histogram(p_down.clamp(0, 1), mask_down)
        hist_gt = self._masked_soft_histogram(t_down.clamp(0, 1), mask_down)

        diff = hist_pred - hist_gt
        loss_extra = F.relu(diff).mean()
        loss_missing = F.relu(-diff).mean()

        return self.weight_extra * loss_extra + self.weight_missing * loss_missing


# ======================================================================
# GAN Base (GAN 基类)
# ======================================================================


class UNetDiscriminatorSN(nn.Module):
    """U-Net Discriminator with Spectral Normalization (谱归一化 U-Net 判别器).

    编码器 3→nf→2nf→4nf→8nf, 解码器 8nf→4nf→2nf→nf, 输出逐像素 logit。

    Args:
        num_feat:        基础通道数
        skip_connection: 是否启用 skip connection
    """

    def __init__(self, num_feat: int = 64, skip_connection: bool = True):
        super().__init__()
        self.skip_connection = skip_connection
        nf = num_feat
        sn = spectral_norm

        self.conv0 = nn.Conv2d(3, nf, 3, 1, 1)
        self.conv1 = sn(nn.Conv2d(nf, nf * 2, 4, 2, 1))
        self.conv2 = sn(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1))
        self.conv3 = sn(nn.Conv2d(nf * 4, nf * 8, 4, 2, 1))

        self.conv4 = sn(nn.Conv2d(nf * 8, nf * 4, 3, 1, 1))
        self.conv5 = sn(nn.Conv2d(nf * 4, nf * 2, 3, 1, 1))
        self.conv6 = sn(nn.Conv2d(nf * 2, nf, 3, 1, 1))

        self.conv7 = sn(nn.Conv2d(nf, nf, 3, 1, 1))
        self.conv8 = sn(nn.Conv2d(nf, nf, 3, 1, 1))
        self.conv9 = nn.Conv2d(nf, 1, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.lrelu(self.conv0(x))
        e1 = self.lrelu(self.conv1(e0))
        e2 = self.lrelu(self.conv2(e1))
        e3 = self.lrelu(self.conv3(e2))

        d0 = self.lrelu(self.conv4(
            F.interpolate(e3, size=e2.shape[2:], mode='bilinear', align_corners=False)))
        if self.skip_connection:
            d0 = d0 + e2
        d1 = self.lrelu(self.conv5(
            F.interpolate(d0, size=e1.shape[2:], mode='bilinear', align_corners=False)))
        if self.skip_connection:
            d1 = d1 + e1
        d2 = self.lrelu(self.conv6(
            F.interpolate(d1, size=e0.shape[2:], mode='bilinear', align_corners=False)))
        if self.skip_connection:
            d2 = d2 + e0

        return self.conv9(self.lrelu(self.conv8(self.lrelu(self.conv7(d2)))))


# ======================================================================
# Perceptual Loss (感知损失)
# ======================================================================


class AnimePerceptualLossV2(nn.Module):
    """Anime Perceptual Loss v2 — Cosine Manifold Distance
    (二次元域内感知损失 v2 — 余弦流形距离).

    使用 Danbooru ConvNeXt 提取特征, 以通道维余弦距离替代 L1,
    度量特征方向而非绝对幅值, 避免强制网络幻觉不可重建的纹理。
    GT 幅值门控过滤平涂区域噪声方向, 聚焦边缘/线条区域。

    Args:
        layer_weights: 特征层权重 (默认 stage0=1.0, stage1=0.5)
    """

    HF_REPO_ID = 'SmilingWolf/wd-convnext-tagger-v3'

    def __init__(self, layer_weights: dict = None):
        super().__init__()

        if not _HAS_TIMM:
            raise ImportError(
                "AnimePerceptualLossV2 需要 timm, huggingface_hub, safetensors。"
                "请运行: pip install timm huggingface_hub safetensors"
            )

        if layer_weights is None:
            layer_weights = {'stage0': 1.0, 'stage1': 0.5}
        self.layer_weights = layer_weights

        self.backbone = timm.create_model(
            'convnext_base', pretrained=False, features_only=True,
            out_indices=[0, 1, 2, 3], act_layer='gelu_tanh',
        )
        self._load_danbooru_weights()

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self._stage_to_idx = {
            'stage0': 0, 'stage1': 1, 'stage2': 2, 'stage3': 3,
        }
        self.register_buffer('mean',
            torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std',
            torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    def _load_danbooru_weights(self):
        weights_path = hf_hub_download(self.HF_REPO_ID, 'model.safetensors')
        raw_sd = safetensors_load(weights_path)
        mapped_sd = {}
        for key, value in raw_sd.items():
            if key.startswith('head.'):
                continue
            mapped_sd[key.replace('stages.', 'stages_').replace('stem.', 'stem_')] = value
        result = self.backbone.load_state_dict(mapped_sd, strict=True)
        assert not result.missing_keys and not result.unexpected_keys

    def _extract_features(self, x: torch.Tensor) -> dict:
        x = (x - self.mean) / self.std
        all_feats = self.backbone(x)
        return {name: all_feats[self._stage_to_idx[name]]
                for name in self.layer_weights}

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_feats = self._extract_features(pred)
        with torch.no_grad():
            target_feats = self._extract_features(target)

        loss = torch.tensor(0.0, device=pred.device)
        for name, w in self.layer_weights.items():
            B, C, H, W = pred_feats[name].shape
            p_flat = pred_feats[name].reshape(B, C, -1)
            t_flat = target_feats[name].detach().reshape(B, C, -1)

            p_norm = F.normalize(p_flat, p=2, dim=1)
            t_norm = F.normalize(t_flat, p=2, dim=1)

            cos_sim = (p_norm * t_norm).sum(dim=1)
            cos_dist = 1.0 - cos_sim

            magnitude = t_flat.norm(p=2, dim=1)
            mag_weight = magnitude / (magnitude.mean(dim=-1, keepdim=True) + 1e-8)

            loss = loss + w * (cos_dist * mag_weight).mean()

        return loss

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self


# ======================================================================
# Spatial & Frequency Regularization (空间/频域正则)
# ======================================================================


class GibbsRingingPenaltySWT(nn.Module):
    """Multi-scale Stationary Wavelet Asymmetric Ringing Penalty
    (多尺度稳态小波不对称振铃惩罚).

    单侧频域约束: 仅惩罚高频过冲 (|pred| > |target|), 不干预正常锐化。
    Haar SWT (非抽取, 平移不变) 三子带 (HL/LH/HH) 覆盖水平/垂直/对角。
    À Trous 多尺度 (dilation=1,2,4) 捕获 1-4px 级振铃。
    """

    def __init__(self):
        super().__init__()

        hl = torch.tensor([[-1, -1], [ 1,  1]], dtype=torch.float32) / 4.0
        lh = torch.tensor([[-1,  1], [-1,  1]], dtype=torch.float32) / 4.0
        hh = torch.tensor([[ 1, -1], [-1,  1]], dtype=torch.float32) / 4.0

        self.register_buffer('hl', hl.view(1, 1, 2, 2).repeat(3, 1, 1, 1))
        self.register_buffer('lh', lh.view(1, 1, 2, 2).repeat(3, 1, 1, 1))
        self.register_buffer('hh', hh.view(1, 1, 2, 2).repeat(3, 1, 1, 1))

        self.dilations = [1, 2, 4]

    def _get_high_freqs(self, x: torch.Tensor, dilation: int = 1):
        pad = dilation
        x_pad = F.pad(x, (0, pad, 0, pad), mode='reflect')
        return (
            F.conv2d(x_pad, self.hl, groups=3, dilation=dilation),
            F.conv2d(x_pad, self.lh, groups=3, dilation=dilation),
            F.conv2d(x_pad, self.hh, groups=3, dilation=dilation),
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for d in self.dilations:
            p_h, p_v, p_d = self._get_high_freqs(pred, dilation=d)
            with torch.no_grad():
                t_h, t_v, t_d = self._get_high_freqs(target, dilation=d)

            loss = loss + (
                F.relu(torch.abs(p_h) - torch.abs(t_h)).mean()
                + F.relu(torch.abs(p_v) - torch.abs(t_v)).mean()
                + F.relu(torch.abs(p_d) - torch.abs(t_d)).mean()
            )

        return loss / len(self.dilations)


class StrictFlatTGVLoss(nn.Module):
    """Strict Flat-region Total Generalized Variation
    (严格平坦区总广义变分).

    形态学硬掩码隔离纯平涂区, Charbonnier 惩罚 (非 L1) 避免 Adam 极限环震荡,
    一阶+二阶导数联合约束梯度→0, 彻底根治平坦区纹波。
    与 FlatRegionAwareLoss 互补: TGV 约束梯度=0, Flat 锚定像素值=GT。

    Args:
        alpha1:          一阶变分权重
        alpha2:          二阶变分权重
        flat_threshold:  绝对平坦梯度幅值上限
        safe_margin:     形态学隔离带 (像素)
        charbonnier_eps: Charbonnier 极小值
    """

    def __init__(self, alpha1: float = 1.0, alpha2: float = 2.0,
                 flat_threshold: float = 0.01, safe_margin: int = 3,
                 charbonnier_eps: float = 1e-4):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.flat_threshold = flat_threshold
        self.safe_margin = safe_margin
        self.eps_sq = charbonnier_eps ** 2

        kernel_dx = torch.tensor([[[[0, 0, 0], [0, -1, 1], [0, 0, 0]]]], dtype=torch.float32)
        kernel_dy = torch.tensor([[[[0, 0, 0], [0, -1, 0], [0, 1, 0]]]], dtype=torch.float32)
        kernel_dxx = torch.tensor([[[[0, 0, 0], [1, -2, 1], [0, 0, 0]]]], dtype=torch.float32)
        kernel_dyy = torch.tensor([[[[0, 1, 0], [0, -2, 0], [0, 1, 0]]]], dtype=torch.float32)
        kernel_dxy = torch.tensor([[[[0, 0, 0], [0, -1, 1], [0, 1, -1]]]], dtype=torch.float32)

        self.register_buffer('kx', kernel_dx)
        self.register_buffer('ky', kernel_dy)
        self.register_buffer('kxx', kernel_dxx)
        self.register_buffer('kyy', kernel_dyy)
        self.register_buffer('kxy', kernel_dxy)

    def _charbonnier(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x ** 2 + self.eps_sq)

    def _compute_grads_direct(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C, 1, H, W)
        x_pad = F.pad(x_flat, (1, 1, 1, 1), mode='reflect')
        gx = F.conv2d(x_pad, self.kx)
        gy = F.conv2d(x_pad, self.ky)
        gxx = F.conv2d(x_pad, self.kxx)
        gyy = F.conv2d(x_pad, self.kyy)
        gxy = F.conv2d(x_pad, self.kxy)
        return gx, gy, gxx, gyy, gxy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            t_gx, t_gy, _, _, _ = self._compute_grads_direct(target)
            grad_mag = torch.sqrt(t_gx ** 2 + t_gy ** 2 + 1e-12)

            B, C, H, W = target.shape
            grad_mag_rgb = grad_mag.reshape(B, C, H, W).max(dim=1, keepdim=True)[0]

            kernel_size = 2 * self.safe_margin + 1
            padding = self.safe_margin
            safe_grad_mag = F.max_pool2d(grad_mag_rgb, kernel_size=kernel_size,
                                         stride=1, padding=padding)

            strict_flat_mask = (safe_grad_mag < self.flat_threshold).float()
            strict_flat_mask = strict_flat_mask.expand(-1, C, -1, -1).reshape(B * C, 1, H, W)

        p_gx, p_gy, p_gxx, p_gyy, p_gxy = self._compute_grads_direct(pred)

        mask_sum = strict_flat_mask.sum().clamp(min=1.0)

        loss_1st = (strict_flat_mask * (self._charbonnier(p_gx) + self._charbonnier(p_gy))).sum() / mask_sum
        loss_2nd = (strict_flat_mask * (self._charbonnier(p_gxx) + self._charbonnier(p_gyy)
                    + 2 * self._charbonnier(p_gxy))).sum() / mask_sum

        return self.alpha1 * loss_1st + self.alpha2 * loss_2nd


class AngularFluencyLoss(nn.Module):
    """Angular Fluency Loss — Edge Gradient Direction Alignment
    (线条流畅性梯度方向损失).

    约束每个边缘像素的梯度方向与 GT 一致, 直接消除超分锯齿。
    使用 Farid 7×7 旋转等变微分算子 (任意角度精度一致),
    余弦角距离 (连续有界, 无 atan2 ±π 奇点)。
    GT 幅值 sigmoid 掩码: 边缘→1 平坦→0, 与 StrictFlatTGV 领域零重叠。

    Args:
        threshold: GT 梯度幅值激活阈值
        eps:       数值稳定性极小值
    """

    def __init__(self, threshold: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.threshold = threshold
        self.eps = eps

        p = torch.tensor([0.004711, 0.069321, 0.245410, 0.361117,
                          0.245410, 0.069321, 0.004711], dtype=torch.float32)
        d = torch.tensor([-0.018708, -0.125376, -0.193091, 0.000000,
                           0.193091,  0.125376,  0.018708], dtype=torch.float32)

        kernel_x = torch.outer(p, d)
        kernel_y = torch.outer(d, p)

        self.register_buffer('kx', kernel_x.view(1, 1, 7, 7).repeat(3, 1, 1, 1))
        self.register_buffer('ky', kernel_y.view(1, 1, 7, 7).repeat(3, 1, 1, 1))

    def _get_gradients(self, x: torch.Tensor):
        x_pad = F.pad(x, (3, 3, 3, 3), mode='reflect')
        gx = F.conv2d(x_pad, self.kx, groups=3)
        gy = F.conv2d(x_pad, self.ky, groups=3)
        return gx, gy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p_gx, p_gy = self._get_gradients(pred)
        with torch.no_grad():
            t_gx, t_gy = self._get_gradients(target)
            t_mag = torch.sqrt(t_gx ** 2 + t_gy ** 2 + self.eps)
            mask = torch.sigmoid((t_mag - self.threshold) * 500.0)
            t_dir_x = t_gx / t_mag
            t_dir_y = t_gy / t_mag

        p_mag = torch.sqrt(p_gx ** 2 + p_gy ** 2 + self.eps)
        p_dir_x = p_gx / p_mag
        p_dir_y = p_gy / p_mag

        cos_sim = p_dir_x * t_dir_x + p_dir_y * t_dir_y
        angular_dist = 1.0 - cos_sim

        return (angular_dist * mask).mean()


class TopologicalSingularityLoss(nn.Module):
    """Topological Singularity Loss — Density-Decayed Spatial NMS
    (拓扑奇异点感知损失 — 密度衰减版).

    基于"峰值能量占比 (Peak-to-Energy Ratio)"机制, 根除"密度坍塌悖论"。
    通过计算结构张量角点响应 (C) 的局部空间密度 (E), 施加指数级衰减。
    - 孤立 "V", "Y": 密度极低, 保留 100% 超级惩罚, 逼迫网络雕刻锋利针尖。
    - 交叉 "X", "+": 密度中等, 保留中度惩罚, 维持拓扑连通。
    - 混乱 "#", "$": 密度极高, 指数衰减彻底剥夺惩罚权重,
      释放梯度压力, 允许网络在密集网格处执行平滑降噪, 防止整体"摆烂"。

    数学核心:
      E = AvgPool_{K×K}(C_gt)           局部角点空间密度
      S = exp(-decay_factor × E)        指数级密度衰减因子
      W = 1 + β × C_gt × S             密度调制后的动态权重

    Args:
        beta:           奇异点 L1 放大基准倍数
        gamma:          弯曲能量回归权重
        density_kernel: 计算局部密度的窗口大小 (对应 ~3px 半径)
        decay_factor:   指数衰减陡峭度 (越大对密集区容忍度越低)
        eps:            数值安全常数
    """
    def __init__(self, beta: float = 10.0, gamma: float = 2.0,
                 density_kernel: int = 7, decay_factor: float = 15.0,
                 eps: float = 1e-6):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.decay_factor = decay_factor
        self.eps = eps

        self.density_kernel = density_kernel
        self.density_padding = density_kernel // 2

        p = torch.tensor([0.004711, 0.069321, 0.245410, 0.361117,
                          0.245410, 0.069321, 0.004711], dtype=torch.float32)
        d = torch.tensor([-0.018708, -0.125376, -0.193091, 0.000000,
                           0.193091,  0.125376,  0.018708], dtype=torch.float32)
        self.register_buffer('kx', torch.outer(p, d).view(1, 1, 7, 7))
        self.register_buffer('ky', torch.outer(d, p).view(1, 1, 7, 7))

        grid = torch.arange(3, dtype=torch.float32) - 1
        gaussian_1d = torch.exp(-grid ** 2 / (2.0 * 0.5 ** 2))
        g_kernel = torch.outer(gaussian_1d, gaussian_1d)
        g_kernel = g_kernel / g_kernel.sum()
        self.register_buffer('g_kernel', g_kernel.view(1, 1, 3, 3))

    def _corner_map(self, x: torch.Tensor) -> torch.Tensor:
        """提取亚像素级角点响应图 C ∈ [0, 1]."""
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        gray_pad = F.pad(gray, (3, 3, 3, 3), mode='reflect')
        Ix = F.conv2d(gray_pad, self.kx)
        Iy = F.conv2d(gray_pad, self.ky)

        Ixx, Iyy, Ixy = Ix * Ix, Iy * Iy, Ix * Iy

        pad_g = (1, 1, 1, 1)
        Sxx = F.conv2d(F.pad(Ixx, pad_g, mode='reflect'), self.g_kernel)
        Syy = F.conv2d(F.pad(Iyy, pad_g, mode='reflect'), self.g_kernel)
        Sxy = F.conv2d(F.pad(Ixy, pad_g, mode='reflect'), self.g_kernel)

        det_S = Sxx * Syy - Sxy * Sxy
        trace_S = Sxx + Syy
        C = (4.0 * det_S) / (trace_S ** 2 + self.eps)
        return torch.clamp(C, 0.0, 1.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C_pred = self._corner_map(pred)
        with torch.no_grad():
            C_gt = self._corner_map(target)

            E_gt = F.avg_pool2d(C_gt, kernel_size=self.density_kernel,
                                stride=1, padding=self.density_padding)

            S_decay = torch.exp(-self.decay_factor * E_gt)

            W_gt = 1.0 + self.beta * C_gt * S_decay

        loss_weighted_l1 = (W_gt * torch.abs(pred - target)).mean()

        loss_bending = (S_decay * (C_pred - C_gt) ** 2).mean()

        return loss_weighted_l1 + self.gamma * loss_bending


TurningPointLoss = TopologicalSingularityLoss


class SmoothGradientHessianLoss(nn.Module):
    """Smooth Gradient Region Hessian Penalty Loss
    (平滑渐变区 Hessian 惩罚损失 — 结构张量引导).

    针对插画中大面积的色彩渐变区 (如天空、肤色阴影)。
    这些区域一阶导数不为零 (有色彩过渡), 但二阶导数 (Hessian) 应该绝对为零 (匀速过渡)。
    利用结构张量 (Structure Tensor) 的特征值相干性 (Coherence) 与能量带通滤波,
    精准锁定"方向一致且能量适中"的渐变坡面, 惩罚其 Hessian 范数,
    彻底消除超分导致的色彩断层 (Color Banding) 和微小波纹。
    与 StrictFlatTGV 互补: TGV 管绝对平坦 (一阶导=0), 本损失管平滑渐变 (二阶导=0)。

    Args:
        tau_low:  能量下界 (过滤绝对平坦区, 交由 StrictFlatTGV 处理)
        tau_high: 能量上界 (过滤锐利边缘/线稿, 交由 AngularFluency 处理)
        k1, k2:   Sigmoid 软阈值陡峭度
        epsilon:  防除零/NaN 安全常数
    """

    def __init__(self, tau_low: float = 0.001, tau_high: float = 0.05,
                 k1: float = 100.0, k2: float = 100.0, epsilon: float = 1e-6):
        super().__init__()
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.k1 = k1
        self.k2 = k2
        self.epsilon = epsilon

        scharr_x = torch.tensor([[-3., 0., 3.],
                                 [-10., 0., 10.],
                                 [-3., 0., 3.]], dtype=torch.float32) / 32.0
        scharr_y = scharr_x.t()

        d_xx = torch.tensor([[0., 0., 0.],
                             [1., -2., 1.],
                             [0., 0., 0.]], dtype=torch.float32)
        d_yy = d_xx.t()
        d_xy = torch.tensor([[1., 0., -1.],
                             [0., 0., 0.],
                             [-1., 0., 1.]], dtype=torch.float32) / 4.0

        gaussian = torch.tensor([[1., 2., 1.],
                                 [2., 4., 2.],
                                 [1., 2., 1.]], dtype=torch.float32) / 16.0

        self.register_buffer('w_x', scharr_x.view(1, 1, 3, 3))
        self.register_buffer('w_y', scharr_y.view(1, 1, 3, 3))
        self.register_buffer('w_xx', d_xx.view(1, 1, 3, 3))
        self.register_buffer('w_yy', d_yy.view(1, 1, 3, 3))
        self.register_buffer('w_xy', d_xy.view(1, 1, 3, 3))
        self.register_buffer('w_g', gaussian.view(1, 1, 3, 3))

    def _apply_conv(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.reshape(b * c, 1, h, w)
        x_pad = F.pad(x_flat, (1, 1, 1, 1), mode='reflect')
        out = F.conv2d(x_pad, weight)
        return out.view(b, c, h, w)

    def _compute_mask(self, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            smooth_target = self._apply_conv(target, self.w_g)

            I_x = self._apply_conv(smooth_target, self.w_x)
            I_y = self._apply_conv(smooth_target, self.w_y)

            S_xx = self._apply_conv(I_x ** 2, self.w_g)
            S_yy = self._apply_conv(I_y ** 2, self.w_g)
            S_xy = self._apply_conv(I_x * I_y, self.w_g)

            trace = S_xx + S_yy
            det = S_xx * S_yy - S_xy ** 2

            discriminant = torch.sqrt(
                torch.clamp(trace ** 2 - 4 * det, min=0.0) + self.epsilon)

            lambda_1 = (trace + discriminant) / 2.0
            lambda_2 = (trace - discriminant) / 2.0

            coherence = ((lambda_1 - lambda_2)
                         / (lambda_1 + lambda_2 + self.epsilon)) ** 2

            energy = trace
            mask_low = torch.sigmoid(self.k1 * (energy - self.tau_low))
            mask_high = torch.sigmoid(self.k2 * (self.tau_high - energy))

            mask = mask_low * mask_high * coherence
            spatial_mask = mask.mean(dim=1, keepdim=True)

        return spatial_mask

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = self._compute_mask(target)

        pred_xx = self._apply_conv(pred, self.w_xx)
        pred_yy = self._apply_conv(pred, self.w_yy)
        pred_xy = self._apply_conv(pred, self.w_xy)

        hessian_norm = torch.sqrt(
            pred_xx ** 2 + 2 * pred_xy ** 2 + pred_yy ** 2 + self.epsilon)

        masked_hessian = mask * hessian_norm
        valid_area = mask.sum().clamp(min=self.epsilon)
        return masked_hessian.sum() / valid_area


class GateTolerancePenalty(nn.Module):
    """Masked Soft-Gating Penalty with Time Annealing
    (基于平坦掩膜的铰链门控惩罚 + 余弦时间退火).

    在 GT 平坦区域中, 当 BADI 门控值超过容忍阈值时施加单侧惩罚,
    防止门控惰性导致浅层噪声泄漏至深层特征。
    仅惩罚超出 tolerance 的部分 (铰链损失), 边缘区域不受约束。
    训练前 cutoff_ratio 阶段生效, 之后余弦退火至 0 防止梯度断崖。

    公式: Loss = w(t) × Mean( ReLU(alpha ⊙ flat_mask - tolerance) )
    其中 w(t) = cosine_anneal(progress, cutoff_ratio, decay_width)

    Args:
        tolerance:      平坦区门控容忍上限 (默认 0.15)
        cutoff_ratio:   训练进度截断点 (默认 0.7)
        decay_width:    截断前余弦衰减窗口宽度 (默认 0.1)
        flat_threshold: 局部方差平坦判定阈值 (默认 1/255)
        patch_size:     局部方差计算窗口 (默认 5)
    """

    def __init__(self, tolerance: float = 0.15, cutoff_ratio: float = 0.7,
                 decay_width: float = 0.1, flat_threshold: float = 1.0 / 255.0,
                 patch_size: int = 5):
        super().__init__()
        self.tolerance = tolerance
        self.cutoff_ratio = cutoff_ratio
        self.decay_width = decay_width
        self.flat_threshold = flat_threshold
        self.patch_size = patch_size
        self.padding = patch_size // 2
        self._progress = 0.0

    def set_progress(self, progress: float):
        """Set current training progress in [0, 1]."""
        self._progress = max(0.0, min(1.0, progress))

    def _get_annealing_weight(self) -> float:
        """Cosine annealing: 1.0 → decay → 0.0 at cutoff."""
        p = self._progress
        c = self.cutoff_ratio
        d = self.decay_width
        if p >= c:
            return 0.0
        decay_start = c - d
        if p <= decay_start:
            return 1.0
        t = (p - decay_start) / d
        return 0.5 * (1.0 + math.cos(math.pi * t))

    def _compute_flat_mask(self, target: torch.Tensor) -> torch.Tensor:
        """GT 局部方差→平坦区硬掩膜 (1=平坦, 0=边缘)."""
        local_mean = F.avg_pool2d(target, self.patch_size, stride=1,
                                  padding=self.padding)
        local_mean_sq = F.avg_pool2d(target ** 2, self.patch_size, stride=1,
                                     padding=self.padding)
        local_var = (local_mean_sq - local_mean ** 2).clamp(min=0)
        local_var = local_var.max(dim=1, keepdim=True)[0]
        return (local_var < self.flat_threshold).float()

    def forward(self, gate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args:
            gate:   BADI 空间门控 (B, 1, H_lr, W_lr), 值域 [0, 1]
            target: HR GT 图像 (B, 3, H_hr, W_hr)
        """
        weight = self._get_annealing_weight()
        if weight <= 0:
            return torch.zeros(1, device=gate.device, dtype=gate.dtype)

        with torch.no_grad():
            flat_mask = self._compute_flat_mask(target)
            if flat_mask.shape[2:] != gate.shape[2:]:
                flat_mask = F.interpolate(
                    flat_mask, gate.shape[2:],
                    mode='bilinear', align_corners=False,
                )

        excess = F.relu(gate * flat_mask - self.tolerance)
        loss = excess.mean()
        return weight * loss


# ======================================================================
# GAN Components (GAN 组件)
# ======================================================================


class _GuidedFilter(nn.Module):
    """Differentiable Guided Filter (可微分引导滤波).

    a = Var(I)/(Var(I)+ε): 边缘 a≈1 保持, 纹理 a≈0 平滑。
    纯 avg_pool2d 实现, 无可学习参数。
    DecoupledUNetDiscriminatorSN 内部前端使用, 外部不直接调用。
    """

    def __init__(self, r: int = 3, eps: float = 0.04):
        super().__init__()
        self.r = r
        self.eps = eps

    def _box_filter(self, x: torch.Tensor) -> torch.Tensor:
        ks = 2 * self.r + 1
        return F.avg_pool2d(x, ks, stride=1, padding=self.r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_I = self._box_filter(x)
        var_I = (self._box_filter(x * x) - mean_I ** 2).clamp(min=0)
        a = var_I / (var_I + self.eps)
        b = mean_I - a * mean_I
        return self._box_filter(a) * x + self._box_filter(b)


class DecoupledUNetDiscriminatorSN(nn.Module):
    """Structure-Texture Decoupled U-Net Discriminator
    (结构-纹理双流解耦 U-Net 判别器).

    引导滤波前端将输入分解为结构 (保边平滑) 和纹理 (高频残差)。
    结构分支: 全功率 U-Net 逐像素拓扑审查。
    纹理分支: 轻量 AdaptiveAvgPool 全局统计, 不做像素级匹配。
    防止 D 利用不可重建的 GT 纹理获得无限区分力导致 G 幻觉伪影。

    Args:
        num_feat:        结构分支基础通道数
        skip_connection: 结构分支 skip connection
    """

    def __init__(self, num_feat: int = 64, skip_connection: bool = True):
        super().__init__()
        self.decoupler = _GuidedFilter(r=3, eps=0.04)
        self.struct_d = UNetDiscriminatorSN(num_feat, skip_connection)

        sn = spectral_norm
        self.texture_d = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(32, 64, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(64, 128, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            sn(nn.Linear(128, 1)),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """Returns (struct_logit, texture_logit)."""
        x_struct = self.decoupler(x)
        x_texture = x - x_struct
        struct_logit = self.struct_d(x_struct)
        texture_logit = self.texture_d(x_texture)
        return struct_logit, texture_logit


class DecoupledGANLoss(nn.Module):
    """Decoupled Adversarial Loss (双流解耦对抗损失).

    配合 DecoupledUNetDiscriminatorSN 使用。
    结构分支全功率 (×1.0), 纹理分支衰减 (×texture_tolerance)。

    Args:
        gan_type:          'vanilla' (BCE) 或 'lsgan' (MSE)
        texture_tolerance: 纹理分支权重衰减系数
    """

    def __init__(self, gan_type: str = 'vanilla',
                 texture_tolerance: float = 0.1):
        super().__init__()
        self.gan_type = gan_type
        self.texture_tolerance = texture_tolerance
        if gan_type == 'vanilla':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif gan_type == 'lsgan':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported GAN type: {gan_type}")

    def forward(self, preds: tuple, target_is_real: bool,
                weight_mask: torch.Tensor = None) -> dict:
        """Returns {'struct_adv': ..., 'texture_adv': ...}."""
        struct_pred, texture_pred = preds

        target_struct = torch.full_like(struct_pred,
                                        1.0 if target_is_real else 0.0)
        target_texture = torch.full_like(texture_pred,
                                         1.0 if target_is_real else 0.0)

        if weight_mask is not None:
            if self.gan_type == 'vanilla':
                pp = F.binary_cross_entropy_with_logits(
                    struct_pred, target_struct, reduction='none')
            else:
                pp = F.mse_loss(struct_pred, target_struct, reduction='none')
            struct_loss = (pp * weight_mask).sum() / weight_mask.sum().clamp(min=1.0)
        else:
            struct_loss = self.loss_fn(struct_pred, target_struct)

        texture_loss = self.loss_fn(texture_pred, target_texture) * self.texture_tolerance

        return {'struct_adv': struct_loss, 'texture_adv': texture_loss}


# ======================================================================
# Combined Loss (组合损失)
# ======================================================================


class CaelumLossV2(nn.Module):
    """Combined Loss with 2-Phase Progressive Training (2 阶段渐进组合损失).

    所有子损失输出原始未加权值, 权重乘法在 forward() 统一完成。
    self.weights 是公开 dict, 训练循环可随时修改。
    返回 dict: 'total' 为加权总和 (tensor), 其余为原始值 (.item() float)。

    Phase 1 (0% ~ phase_2_start): l1, flat, oklch, stgv, smooth_grad_hessian
    Phase 2 (phase_2_start ~ 100%): + chroma_grad, crevice, histogram,
        gibbs, angular, turning_point, perceptual

    Args:
        weights:            损失权重字典 (覆盖 DEFAULT_WEIGHTS)
        phase_2_start:      Phase 2 启用阈值 (训练进度)
        oklch_alpha/beta:   OKLCH 内部权重
        stgv_alpha1/alpha2: StrictFlatTGV 变分权重
        stgv_flat_threshold: 绝对平坦梯度幅值上限
        stgv_safe_margin:   形态学隔离带大小
        angular_threshold:  线条方向损失梯度幅值激活阈值
        tp_beta/gamma:      转折点 L1 放大倍数 / 弯曲能量权重
        perceptual_layers:  感知损失层配置
    """

    DEFAULT_WEIGHTS = {
        'l1':           1.0,
        'flat':         1.0,
        'oklch':        4.0,
        'stgv':         0.1,
        'chroma_grad':  2.0,
        'crevice':      4.0,
        'histogram':    2.0,
        'gibbs':        16.0,
        'smooth_grad_hessian': 2.0,
        'angular':      4.0,
        'turning_point': 1.0,
        'perceptual':   0.5,
    }

    def __init__(
        self,
        weights: dict = None,
        phase_2_start: float = 0.3,
        oklch_alpha: float = 1.0,
        oklch_beta: float = 3.0,
        stgv_alpha1: float = 1.0,
        stgv_alpha2: float = 2.0,
        stgv_flat_threshold: float = 0.01,
        stgv_safe_margin: int = 3,
        angular_threshold: float = 0.05,
        tp_beta: float = 10.0,
        tp_gamma: float = 2.0,
        perceptual_layers: dict = None,
    ):
        super().__init__()

        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights is not None:
            self.weights.update(weights)

        self.phase_2_start = phase_2_start
        self._progress = 0.0

        # === Phase 1 ===
        self.l1_loss = nn.L1Loss()
        self.flat_loss = FlatRegionAwareLoss()
        self.oklch_loss = OklchColorLoss(oklch_alpha, oklch_beta)
        self.stgv_loss = StrictFlatTGVLoss(stgv_alpha1, stgv_alpha2,
                                              stgv_flat_threshold, stgv_safe_margin)
        self.smooth_grad_hessian_loss = SmoothGradientHessianLoss()

        # === Phase 2 ===
        self.chroma_grad_loss = ChromaGradientLoss()
        self.crevice_loss = CreviceColorLoss()
        self.histogram_loss = MaskedAsymmetricHistogramLoss()
        self.gibbs_loss = GibbsRingingPenaltySWT()
        self.angular_loss = AngularFluencyLoss(angular_threshold)
        self.turning_point_loss = TopologicalSingularityLoss(tp_beta, tp_gamma)

        self.perceptual_loss = None
        if self.weights.get('perceptual', 0) > 0:
            self.perceptual_loss = AnimePerceptualLossV2(perceptual_layers)

    def set_progress(self, progress: float):
        self._progress = max(0.0, min(1.0, progress))

    def get_phase(self) -> int:
        return 2 if self._progress >= self.phase_2_start else 1

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> dict:
        w = self.weights
        phase = self.get_phase()
        zero = 0.0

        raw_l1 = self.l1_loss(pred, target)
        raw_flat = self.flat_loss(pred, target)
        raw_oklch = self.oklch_loss(pred, target)
        raw_stgv = self.stgv_loss(pred, target)
        raw_sgh = self.smooth_grad_hessian_loss(pred, target)

        total = (w['l1'] * raw_l1
                 + w['flat'] * raw_flat
                 + w['oklch'] * raw_oklch
                 + w['stgv'] * raw_stgv
                 + w['smooth_grad_hessian'] * raw_sgh)

        raw_cg = raw_crv = raw_hist = zero
        raw_gibbs = raw_angular = raw_tp = raw_perc = zero

        if phase >= 2:
            raw_cg = self.chroma_grad_loss(pred, target)
            raw_crv = self.crevice_loss(pred, target)
            raw_hist = self.histogram_loss(pred, target)
            raw_gibbs = self.gibbs_loss(pred, target)
            raw_angular = self.angular_loss(pred, target)
            raw_tp = self.turning_point_loss(pred, target)

            total = total + (w['chroma_grad'] * raw_cg
                             + w['crevice'] * raw_crv
                             + w['histogram'] * raw_hist
                             + w['gibbs'] * raw_gibbs
                             + w['angular'] * raw_angular
                             + w['turning_point'] * raw_tp)

            if self.perceptual_loss is not None:
                raw_perc = self.perceptual_loss(pred, target)
                total = total + w['perceptual'] * raw_perc

        def _v(x):
            return x.item() if isinstance(x, torch.Tensor) else x

        return {
            'total': total,
            'l1': _v(raw_l1),
            'flat': _v(raw_flat),
            'oklch': _v(raw_oklch),
            'stgv': _v(raw_stgv),
            'smooth_grad_hessian': _v(raw_sgh),
            'chroma_grad': _v(raw_cg),
            'crevice': _v(raw_crv),
            'histogram': _v(raw_hist),
            'gibbs': _v(raw_gibbs),
            'angular': _v(raw_angular),
            'turning_point': _v(raw_tp),
            'perceptual': _v(raw_perc),
        }


# ======================================================================
# Self-Test (自检)
# ======================================================================


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("  losses.py 验证")
    print("=" * 60)

    pred = torch.randn(2, 3, 64, 64, device=device).clamp(0, 1).requires_grad_(True)
    target = torch.randn(2, 3, 64, 64, device=device).clamp(0, 1)

    print("\n■ GibbsRingingPenaltySWT:")
    gibbs = GibbsRingingPenaltySWT().to(device)
    val = gibbs(pred, target)
    val.backward()
    print(f"  随机对:       loss={val.item():.6f}, grad={pred.grad.norm().item():.6f}")
    pred.grad.zero_()

    val_self = gibbs(target, target)
    print(f"  自身比较:     loss={val_self.item():.6f} (应=0)")

    sharpened = target + 0.1 * torch.randn_like(target)
    val_sharp = gibbs(sharpened.clamp(0, 1).requires_grad_(False), target)
    print(f"  添加噪声:     loss={val_sharp.item():.6f} (应>0)")

    blurred = F.avg_pool2d(F.pad(target, (1,1,1,1), mode='reflect'), 3, 1)
    val_blur = gibbs(blurred.requires_grad_(False), target)
    print(f"  模糊版本:     loss={val_blur.item():.6f} (应≈0)")

    print(f"  多尺度级别:   dilations={gibbs.dilations}")
    for d in gibbs.dilations:
        p_h, p_v, p_d = gibbs._get_high_freqs(pred, dilation=d)
        print(f"    dilation={d}: HL={p_h.shape}, 幅值={p_h.abs().mean().item():.6f}")

    print("\n■ StrictFlatTGVLoss:")
    atgv = StrictFlatTGVLoss().to(device)

    flat = torch.full((2, 3, 64, 64), 0.5, device=device)
    val_flat = atgv(flat, flat)
    print(f"  纯平坦自身:   loss={val_flat.item():.8f} (应≈0)")

    flat_target = torch.full((2, 3, 64, 64), 0.5, device=device)
    noisy_pred = (flat_target + 0.01 * torch.randn_like(flat_target)).requires_grad_(True)
    val_noisy = atgv(noisy_pred, flat_target)
    val_noisy.backward()
    print(f"  平坦+微噪声:  loss={val_noisy.item():.6f}, grad={noisy_pred.grad.norm().item():.6f}")

    grad_t = torch.linspace(0, 1, 64, device=device).view(1, 1, 1, 64).expand(2, 3, 64, 64).contiguous()
    noisy_grad = grad_t + 0.01 * torch.randn_like(grad_t)
    val_grad = atgv(noisy_grad, grad_t)
    print(f"  渐变区+噪声:  loss={val_grad.item():.8f} (应=0)")

    half_t = torch.full((2, 3, 64, 64), 0.5, device=device)
    half_t[:, :, :, 32:] = torch.linspace(0.2, 0.8, 32, device=device)
    noisy_half = half_t + 0.005 * torch.randn_like(half_t)
    val_half = atgv(noisy_half, half_t)
    print(f"  半平坦半渐变:  loss={val_half.item():.6f} (>0)")

    print("\n■ AngularFluencyLoss:")
    angular = AngularFluencyLoss().to(device)

    ang_pred = torch.randn(2, 3, 64, 64, device=device).clamp(0, 1).requires_grad_(True)
    val_ang = angular(ang_pred, target)
    val_ang.backward()
    print(f"  随机对:       loss={val_ang.item():.6f}, grad={ang_pred.grad.norm().item():.6f}")
    ang_pred.grad.zero_()

    val_ang_self = angular(target, target)
    print(f"  自身比较:     loss={val_ang_self.item():.6f} (应≈0)")

    edge_ang = torch.zeros(2, 3, 64, 64, device=device)
    edge_ang[:, :, :, 32:] = 1.0
    val_ang_edge = angular(edge_ang.clone(), edge_ang)
    print(f"  完美边缘:     loss={val_ang_edge.item():.6f} (应≈0)")

    flat_ang = torch.full((2, 3, 64, 64), 0.5, device=device)
    val_ang_flat = angular(flat_ang, flat_ang)
    print(f"  纯平坦:       loss={val_ang_flat.item():.6f} (应≈0)")

    print("\n■ SmoothGradientHessianLoss:")
    sgh = SmoothGradientHessianLoss().to(device)

    sgh_pred = torch.randn(2, 3, 64, 64, device=device).clamp(0, 1).requires_grad_(True)
    val_sgh = sgh(sgh_pred, target)
    val_sgh.backward()
    print(f"  随机对:       loss={val_sgh.item():.6f}, grad={sgh_pred.grad.norm().item():.6f}")
    sgh_pred.grad.zero_()

    flat_sgh = torch.full((2, 3, 64, 64), 0.5, device=device)
    val_sgh_flat = sgh(flat_sgh, flat_sgh)
    print(f"  纯平坦自身:   loss={val_sgh_flat.item():.8f} (应≈0, 平坦区被排除)")

    grad_sgh = torch.linspace(0, 1, 64, device=device).view(1, 1, 1, 64).expand(2, 3, 64, 64).contiguous()
    val_sgh_grad = sgh(grad_sgh, grad_sgh)
    print(f"  完美线性渐变: loss={val_sgh_grad.item():.8f} (应≈0, Hessian=0)")

    noisy_grad_sgh = grad_sgh + 0.02 * torch.sin(torch.linspace(0, 20 * 3.14159, 64, device=device)).view(1, 1, 1, 64)
    val_sgh_wavy = sgh(noisy_grad_sgh.clamp(0, 1), grad_sgh)
    print(f"  渐变+波纹:    loss={val_sgh_wavy.item():.6f} (应>0, 波纹被惩罚)")

    print("\n■ TopologicalSingularityLoss:")
    tp = TopologicalSingularityLoss().to(device)

    tp_pred = torch.randn(2, 3, 64, 64, device=device).clamp(0, 1).requires_grad_(True)
    val_tp = tp(tp_pred, target)
    val_tp.backward()
    print(f"  随机对:       loss={val_tp.item():.6f}, grad={tp_pred.grad.norm().item():.6f}")
    tp_pred.grad.zero_()

    val_tp_self = tp(target, target)
    print(f"  自身比较:     loss={val_tp_self.item():.6f} (应≈0)")

    flat_tp = torch.full((2, 3, 64, 64), 0.5, device=device)
    val_tp_flat = tp(flat_tp, flat_tp)
    print(f"  纯平坦:       loss={val_tp_flat.item():.6f} (应≈0)")

    cross = torch.zeros(2, 3, 64, 64, device=device)
    cross[:, :, 30:34, :] = 1.0
    cross[:, :, :, 30:34] = 1.0
    val_tp_corner = tp._corner_map(cross)
    print(f"  十字交叉 C_max: {val_tp_corner.max().item():.4f} (应>0)")

    with torch.no_grad():
        C_cross = tp._corner_map(cross)
        E_cross = F.avg_pool2d(C_cross, kernel_size=tp.density_kernel,
                               stride=1, padding=tp.density_padding)
        S_cross = torch.exp(-tp.decay_factor * E_cross)
        print(f"  十字交叉 S_decay 范围: [{S_cross.min().item():.4f}, {S_cross.max().item():.4f}]")

    iso_v = torch.zeros(2, 3, 64, 64, device=device)
    iso_v[:, :, 28:36, 30:34] = 1.0
    with torch.no_grad():
        C_iso = tp._corner_map(iso_v)
        E_iso = F.avg_pool2d(C_iso, kernel_size=tp.density_kernel,
                             stride=1, padding=tp.density_padding)
        S_iso = torch.exp(-tp.decay_factor * E_iso)
        print(f"  孤立线段 S_decay 范围: [{S_iso.min().item():.4f}, {S_iso.max().item():.4f}] (应接近1)")

    print("\n■ DecoupledUNetDiscriminatorSN:")
    disc = DecoupledUNetDiscriminatorSN(64).to(device)
    disc_input = torch.randn(2, 3, 64, 64, device=device).clamp(0, 1)
    s_logit, t_logit = disc(disc_input)
    d_params = sum(p.numel() for p in disc.parameters())
    print(f"  struct_logit:  {list(s_logit.shape)}")
    print(f"  texture_logit: {list(t_logit.shape)}")
    print(f"  参数:          {d_params:,d} ({d_params/1e6:.2f}M)")

    print("\n■ DecoupledGANLoss:")
    dgan = DecoupledGANLoss()
    g_loss = dgan((s_logit, t_logit), True)
    d_loss = dgan((s_logit, t_logit), False)
    print(f"  G(real):  struct={g_loss['struct_adv'].item():.4f}, "
          f"texture={g_loss['texture_adv'].item():.4f}")
    print(f"  D(fake):  struct={d_loss['struct_adv'].item():.4f}, "
          f"texture={d_loss['texture_adv'].item():.4f}")

    if _HAS_TIMM:
        print("\n■ AnimePerceptualLossV2:")
        perc = AnimePerceptualLossV2().to(device)
        pred2 = torch.randn(2, 3, 64, 64, device=device).clamp(0, 1).requires_grad_(True)

        val = perc(pred2, target)
        val.backward()
        print(f"  随机对:       loss={val.item():.6f}, grad={pred2.grad.norm().item():.6f}")

        val_self = perc(target, target)
        print(f"  自身比较:     loss={val_self.item():.6f} (应≈0)")
    else:
        print("\n■ AnimePerceptualLossV2: 跳过 (缺少 timm)")

    print("\n■ CaelumLossV2:")
    v2_weights = dict(CaelumLossV2.DEFAULT_WEIGHTS)
    v2_weights['perceptual'] = 0.0
    criterion = CaelumLossV2(weights=v2_weights).to(device)
    v2_pred = torch.randn(2, 3, 64, 64, device=device).clamp(0, 1).requires_grad_(True)
    v2_target = torch.randn(2, 3, 64, 64, device=device).clamp(0, 1)

    criterion.set_progress(0.0)
    out1 = criterion(v2_pred, v2_target)
    out1['total'].backward()
    print(f"  Phase 1: total={out1['total'].item():.4f} "
          f"l1={out1['l1']:.4f} flat={out1['flat']:.4f} "
          f"oklch={out1['oklch']:.4f} stgv={out1['stgv']:.4f} "
          f"sgh={out1['smooth_grad_hessian']:.4f}")
    print(f"           chroma_grad={out1['chroma_grad']} (应=0.0)")
    v2_pred.grad.zero_()

    criterion.set_progress(0.5)
    out2 = criterion(v2_pred, v2_target)
    out2['total'].backward()
    print(f"  Phase 2: total={out2['total'].item():.4f} "
          f"gibbs={out2['gibbs']:.4f} "
          f"angular={out2['angular']:.4f} "
          f"tp={out2['turning_point']:.4f} cg={out2['chroma_grad']:.4f}")
    v2_pred.grad.zero_()

    criterion.weights['l1'] = 0.0
    out3 = criterion(v2_pred, v2_target)
    print(f"  权重修改: l1 weight=0, raw l1={out3['l1']:.4f}")
    print(f"  默认权重: {CaelumLossV2.DEFAULT_WEIGHTS}")

    print("\n" + "=" * 60)
