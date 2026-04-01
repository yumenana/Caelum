"""
Dataset Pipeline for Anime Super-Resolution Training
=====================================================

PPBUNet 训练数据管线 (scale-aware v2)。
模拟"电子包浆", 从 HR 源图像实时生成 (LR, HR) 配对。
支持 scale_factor ∈ {1, 2, 4}:
  1× 纯去伪影 — 零模糊, 仅压缩链路
  2× 轻度超分 — 模糊核 ≤ 3×3, 概率减半
  4× 标准超分 — 模糊核 ≤ 5×5, 完整退化链路

Pipeline:
  HR image (>=480p) → random crop → augmentation → degradation → (LR, HR) pair

Degradation Modes:
  mode 0: Pure bicubic downscale (validation; 1× = identity)
  mode 1: Pre-blur + bicubic downscale (1× = identity)
  mode 2: Downscale + random JPEG/WebP (1× = compression only)
  mode 3: 3-stage high-order lv2 (medium — "social media repost")
  mode 4: 3-stage high-order lv3 (heavy — "digital patina")

第一性原理: blur_kernel ≤ scale_factor + 1
  下采样前模糊核不超出计算窗口, 保证退化是自然抗锯齿而非过度平滑。

Mixed Sampling (CaelumMixedDataset):
  mode1=10%, mode2=30%, mode3=35%, mode4=25%
  同一张源图在不同 epoch 经历不同退化, 网络同时学习处理全谱退化。

作者: YumeNana
"""

import io
import os
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms import functional as TF


# ======================================================================
# Degradation Utilities (退化工具函数)
# ======================================================================


def jpeg_quality_to_webp(jpeg_q: int) -> int:
    """JPEG quality → 感知等价 WebP quality。

    分段线性映射, 因为 WebP 在低质量端压缩效率远高于 JPEG,
    直接用相同数值会导致 WebP 退化程度远轻于 JPEG。
    """
    if jpeg_q <= 50:
        return max(1, int(5 + (jpeg_q - 10) * 0.75))
    elif jpeg_q <= 75:
        return int(35 + (jpeg_q - 50) * 1.0)
    else:
        return min(100, int(60 + (jpeg_q - 75) * 1.4))


def apply_jpeg_compression(img: Image.Image, quality: int) -> Image.Image:
    """内存中完整 JPEG 编解码循环。

    不落盘, 避免文件系统开销; 完整编解码保证退化真实性,
    仅做量化表查找无法重现 DCT 块效应和色度子采样。
    """
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def apply_webp_compression(img: Image.Image, quality: int) -> Image.Image:
    """内存中完整 WebP 编解码循环。

    WebP 有损模式使用 VP8 帧内预测 + 变换编码,
    产生与 JPEG 不同模式的色彩偏移和块效应, 增加退化多样性。
    """
    buf = io.BytesIO()
    img.save(buf, format='WEBP', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def apply_random_compression(img: Image.Image, jpeg_quality: int,
                             webp_ratio: float = 0.3) -> Image.Image:
    """随机选择 JPEG(70%) 或 WebP(30%) 压缩。

    现实中 JPEG 仍是主流, 但 WebP 在社交平台快速普及;
    70/30 比例反映当前互联网图片格式分布。
    """
    if random.random() < webp_ratio:
        return apply_webp_compression(img, jpeg_quality_to_webp(jpeg_quality))
    return apply_jpeg_compression(img, jpeg_quality)


def apply_pre_blur(img: Image.Image, max_sigma: float = 1.0) -> Image.Image:
    """降采样前预模糊 (scale-aware)。

    sigma ∈ [max_sigma/2, max_sigma], 确保模糊核不超出下采样计算窗口:
      4× → max_sigma=1.0 → GaussianBlur(σ≤1.0) → 有效核 ≤ 5×5 (∈ 4+1)
      2× → max_sigma=0.5 → GaussianBlur(σ≤0.5) → 有效核 ≤ 3×3 (∈ 2+1)
    连续随机 sigma 比离散选择提供更丰富的退化多样性。
    """
    sigma = random.uniform(max_sigma * 0.5, max_sigma)
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_stage_blur(img: Image.Image, max_sigma: float = 0.8) -> Image.Image:
    """退化阶段间模糊 (scale-aware)。

    sigma ∈ [max_sigma*0.3, max_sigma], 比 pre_blur 更轻。
    模拟平台转存时服务端图像处理管线的二次平滑。
    """
    sigma = random.uniform(max_sigma * 0.3, max_sigma)
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def break_dct_grid(img: Image.Image) -> tuple:
    """循环位移 1-7px 打破 JPEG 8×8 DCT 网格对齐。

    两次压缩间若块边界重合, 重复压缩近似幂等 (退化不足);
    随机偏移迫使第二次编码重新量化, 产生真实的多重压缩伪影。
    返回 (shifted_img, (dy, dx))。
    """
    arr = np.array(img)
    dx, dy = random.randint(1, 7), random.randint(1, 7)
    return Image.fromarray(np.roll(arr, (dy, dx), (0, 1))), (dy, dx)


def undo_dct_shift(img: Image.Image, dy: int, dx: int) -> Image.Image:
    """撤销 break_dct_grid 的循环位移, 恢复原始空间对齐。"""
    return Image.fromarray(np.roll(np.array(img), (-dy, -dx), (0, 1)))


def sample_first_stage_scale(scale_min: float = 0.6) -> float:
    """第一阶段缩放因子采样 (scale-aware)。

    scale_min 由下采样倍率决定:
      4× → scale_min=0.6 → 创作者可能中幅缩放上传
      2× → scale_min=0.8 → 缩放幅度更保守
      1× → scale_min=1.0 → 不缩放

    采样分布: 40%→scale_min | 30%→1.0 | 30%→U(scale_min, 1.0)
    """
    if scale_min >= 1.0:
        return 1.0
    r = random.random()
    if r < 0.40:
        return scale_min
    elif r < 0.70:
        return 1.0
    return random.uniform(scale_min, 1.0)


def generate_sinc_kernel(kernel_size: int, omega_c: float) -> torch.Tensor:
    """2D Sinc 低通核 + Hamming 窗。

    截断 sinc 函数的 Gibbs 现象产生振铃过冲,
    恰好模拟图像压缩/缩放中的振铃伪影。
    Hamming 窗控制旁瓣衰减速度。
    """
    assert kernel_size % 2 == 1
    half = kernel_size // 2
    n = torch.arange(-half, half + 1, dtype=torch.float32)
    n_safe = torch.where(n == 0, torch.ones_like(n), n)
    h = torch.sin(omega_c * n_safe) / (math.pi * n_safe)
    h[half] = omega_c / math.pi
    w = 0.54 + 0.46 * torch.cos(math.pi * n / half)
    h = h * w
    k2d = torch.outer(h, h)
    return k2d / k2d.sum()


def apply_sinc_filter(img: Image.Image, kernel: torch.Tensor) -> Image.Image:
    """Sinc 滤波 (reflect padding, CPU)。"""
    t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
    t = t.permute(2, 0, 1).unsqueeze(0)
    k = kernel.unsqueeze(0).unsqueeze(0).expand(3, -1, -1, -1)
    pad = kernel.shape[0] // 2
    out = F.conv2d(F.pad(t, [pad]*4, mode='reflect'), k, groups=3)
    out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1)
    return Image.fromarray((out.numpy() * 255).round().astype(np.uint8))


def apply_sinc_ringing(img: Image.Image, omega_range: tuple,
                       ks_choices: tuple) -> Image.Image:
    """施加 Sinc 振铃伪影 (scale-aware)。

    omega_range, ks_choices 由 SCALE_PROFILES 提供, 保证振铃强度与缩放倍率匹配:
      4× → ω 接近 π (轻度截频), 核 9-11
      2× → ω 更接近 π (极轻截频), 核 5-7
    """
    ks = random.choice(ks_choices)
    omega = random.uniform(*omega_range)
    return apply_sinc_filter(img, generate_sinc_kernel(ks, omega))


# ======================================================================
# Degradation Pipeline (退化管线)
# ======================================================================


MODE_NAMES = {
    0: '纯 bicubic (验证用; 1× = identity)',
    1: '纯放大 (预模糊 + bicubic; 1× = identity)',
    2: '轻度退化 (bicubic + 随机压缩; 1× = 仅压缩)',
    3: '中度退化 (3 阶段, lv2 — "社交媒体转存")',
    4: '重度退化 (3 阶段, lv3 — "电子包浆")',
}


class DegradationPipeline:
    """Scale-aware multi-level degradation generator.

    ■ 多倍率支持 (scale_factor ∈ {1, 2, 4}) ■
    ──────────────────────────────────────────
    第一性原理: 模糊核不超出下采样计算窗口。
      N× 下采样时, 每个 LR 像素聚合 N×N HR 区域,
      模糊核 ≤ (N+1)×(N+1) 是自然抗锯齿, 超出则为人工过度平滑。

      4× → blur σ≤1.0, 有效核 ≤ 5×5 (∈ 4+1)
      2× → blur σ≤0.5, 有效核 ≤ 3×3 (∈ 2+1)
      1× → 零模糊, 仅保留压缩伪影链路

    ■ 高阶退化流程 (mode 3/4, 均为 3 阶段) ■
    ──────────────────────────────────────────
      Stage 1 (创作者上传): [模糊] → [中间缩放] → 压缩
      Stage 2 (平台转存):   [Sinc振铃] → [模糊] → 下采样 → DCT偏移 → 压缩
      Stage 3 (终端获取):   [缩放截图] → DCT偏移 → 最终压缩

      1× 模式: 所有 [方括号] 操作跳过, 退化为纯多轮压缩 + DCT偏移,
      完美模拟 "原图被多次转存压缩" 的纯伪影退化。

    Args:
        mode:         退化模式 (0-4)
        scale_factor: 下采样倍率 (1, 2, 4)
    """

    INTERPOLATION_METHODS = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC]

    # ── Scale-dependent degradation profiles ──
    # blur_sigma: PIL GaussianBlur(radius=sigma), 有效核 ≈ (2*ceil(2σ)+1)²
    # 概率与强度随 scale_factor 缩减, 确保退化强度与信息损失匹配
    PROFILES = {
        4: dict(
            pre_blur_prob=0.30,   pre_blur_sigma=1.0,       # 5×5 ≤ 4+1 ✓
            stage_blur_prob=0.30, stage_blur_sigma=0.8,
            sinc_prob=0.15,
            sinc_omega_lv2=(3*math.pi/4, math.pi),          # 截止 top 25% freq
            sinc_omega_lv3=(math.pi/2, 3*math.pi/4),
            sinc_ks_small=(7, 9),  sinc_ks_large=(9, 11),
            stage1_q=(80, 95),  stage2_q=(60, 85),
            stage3_q_lv2=(50, 80), stage3_q_lv3=(20, 50),
            zoom_prob_lv2=0.15, zoom_prob_lv3=0.35,
            first_stage_scale_min=0.6,
        ),
        2: dict(
            pre_blur_prob=0.15,   pre_blur_sigma=0.5,       # 3×3 ≤ 2+1 ✓
            stage_blur_prob=0.15, stage_blur_sigma=0.4,
            sinc_prob=0.10,
            sinc_omega_lv2=(5*math.pi/6, math.pi),          # 截止 top ~17% freq
            sinc_omega_lv3=(2*math.pi/3, 5*math.pi/6),
            sinc_ks_small=(5, 7),  sinc_ks_large=(7, 9),
            stage1_q=(85, 95),  stage2_q=(70, 90),
            stage3_q_lv2=(60, 85), stage3_q_lv3=(30, 60),
            zoom_prob_lv2=0.10, zoom_prob_lv3=0.20,
            first_stage_scale_min=0.8,
        ),
        1: dict(
            pre_blur_prob=0.0,    pre_blur_sigma=0.0,       # 零模糊
            stage_blur_prob=0.0,  stage_blur_sigma=0.0,
            sinc_prob=0.0,                                    # 零振铃
            sinc_omega_lv2=(math.pi, math.pi),
            sinc_omega_lv3=(math.pi, math.pi),
            sinc_ks_small=(5, 7),  sinc_ks_large=(7, 9),
            stage1_q=(75, 95),  stage2_q=(50, 85),
            stage3_q_lv2=(40, 75), stage3_q_lv3=(15, 45),
            zoom_prob_lv2=0.0, zoom_prob_lv3=0.0,            # 零缩放截图
            first_stage_scale_min=1.0,                        # 无中间缩放
        ),
    }

    def __init__(self, mode: int = 1, scale_factor: int = 4):
        assert scale_factor in self.PROFILES, \
            f"scale_factor 必须为 {set(self.PROFILES.keys())}, 当前: {scale_factor}"
        self.mode = mode
        self.scale_factor = scale_factor
        self.p = self.PROFILES[scale_factor]

    @staticmethod
    def _random_downsample_interp() -> int:
        """50% Bicubic / 50% Bilinear, 模拟不同平台的缩放算法差异。"""
        return Image.BICUBIC if random.random() < 0.50 else Image.BILINEAR

    def __call__(self, hr_patch: Image.Image) -> Image.Image:
        if self.mode == 0:
            return self._degrade_pure(hr_patch, blur=False)
        elif self.mode == 1:
            return self._degrade_pure(hr_patch, blur=True)
        elif self.mode == 2:
            return self._degrade_light(hr_patch)
        elif self.mode in (3, 4):
            return self._degrade_high_order(hr_patch, level=self.mode - 1)
        raise ValueError(f"未知退化模式: {self.mode}")

    def _degrade_pure(self, hr: Image.Image, blur: bool) -> Image.Image:
        """Mode 0/1: 纯下采样, 可选预模糊。1× 时返回原图。"""
        if self.scale_factor <= 1:
            return hr
        w, h = hr.size
        lr_w, lr_h = w // self.scale_factor, h // self.scale_factor
        if blur and random.random() < 0.5:
            hr = apply_pre_blur(hr, self.p['pre_blur_sigma'])
        interp = self._random_downsample_interp() if blur else Image.BICUBIC
        return hr.resize((lr_w, lr_h), interp)

    def _degrade_light(self, hr: Image.Image) -> Image.Image:
        """Mode 2: 下采样 + 随机压缩。1× 时仅压缩。"""
        w, h = hr.size
        if self.scale_factor <= 1:
            # 1× 纯去伪影: 仅 1-2 次压缩, 无下采样
            n_passes = random.choices([1, 2], weights=[7, 3], k=1)[0]
            img = hr
            for _ in range(n_passes):
                img = apply_random_compression(img, random.randint(60, 95))
            return img
        lr_w, lr_h = w // self.scale_factor, h // self.scale_factor
        lr = hr.resize((lr_w, lr_h), self._random_downsample_interp())
        n_passes = random.choices([1, 2, 3], weights=[3, 5, 2], k=1)[0]
        for _ in range(n_passes):
            lr = apply_random_compression(lr, random.randint(50, 95))
        return lr

    def _degrade_high_order(self, hr: Image.Image, level: int) -> Image.Image:
        """Mode 3/4: 3 阶段高阶退化 (scale-aware).

        level=2 (mode 3): lv2 — "社交媒体转存"
        level=3 (mode 4): lv3 — "电子包浆"

        所有模糊/缩放操作受 PROFILES 门控:
          4× → 完整退化链路, σ≤1.0
          2× → 概率减半, σ≤0.5
          1× → 全部跳过, 退化为纯多轮压缩 + DCT 偏移
        """
        p = self.p
        w, h = hr.size
        sf = self.scale_factor
        lr_w = w // sf if sf > 1 else w
        lr_h = h // sf if sf > 1 else h
        img = hr

        # ═══ Stage 1: 创作者上传 ═══
        if random.random() < p['pre_blur_prob']:
            img = apply_pre_blur(img, p['pre_blur_sigma'])

        if sf > 1:
            scale = sample_first_stage_scale(p['first_stage_scale_min'])
            if scale < 1.0:
                img = img.resize(
                    (max(lr_w, int(w * scale)), max(lr_h, int(h * scale))),
                    self._random_downsample_interp(),
                )

        img = apply_random_compression(img, random.randint(*p['stage1_q']))

        # ═══ Stage 2: 平台转存 ═══
        if random.random() < p['sinc_prob']:
            omega = p['sinc_omega_lv2'] if level == 2 else p['sinc_omega_lv3']
            ks = p['sinc_ks_small'] if min(img.size) < 128 else p['sinc_ks_large']
            img = apply_sinc_ringing(img, omega, ks)

        if random.random() < p['stage_blur_prob']:
            img = apply_stage_blur(img, p['stage_blur_sigma'])

        if sf > 1:
            img = img.resize((lr_w, lr_h), self._random_downsample_interp())

        img, (dy2, dx2) = break_dct_grid(img)
        img = apply_random_compression(img, random.randint(*p['stage2_q']))
        img = undo_dct_shift(img, dy2, dx2)

        # ═══ Stage 3: 终端获取 ═══
        zoom_prob = p['zoom_prob_lv3'] if level == 3 else p['zoom_prob_lv2']
        if random.random() < zoom_prob and lr_w > 64:
            small = random.randint(max(32, lr_w // 2), lr_w - 1)
            img = img.resize((small, small), self._random_downsample_interp())
            if p['sinc_prob'] > 0:
                omega = p['sinc_omega_lv2'] if level == 2 else p['sinc_omega_lv3']
                img = apply_sinc_ringing(img, omega, p['sinc_ks_small'])
            img = img.resize((lr_w, lr_h), random.choice(self.INTERPOLATION_METHODS))

        q3 = p['stage3_q_lv2'] if level == 2 else p['stage3_q_lv3']
        img, (dy3, dx3) = break_dct_grid(img)
        img = apply_random_compression(img, random.randint(*q3))
        img = undo_dct_shift(img, dy3, dx3)

        return img


# ======================================================================
# Data Augmentation (数据增强)
# ======================================================================


def paired_random_augment(hr: Image.Image, lr: Image.Image):
    """配对几何增强: 水平翻转(50%) + 垂直翻转(50%) + 90°旋转(均匀)。

    8 种组合 = 8× 有效数据扩充。
    不做颜色增强, 因为色彩精度是损失函数显式优化的目标。
    """
    if random.random() > 0.5:
        hr, lr = TF.hflip(hr), TF.hflip(lr)
    if random.random() > 0.5:
        hr, lr = TF.vflip(hr), TF.vflip(lr)
    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        hr, lr = TF.rotate(hr, angle), TF.rotate(lr, angle)
    return hr, lr


# ======================================================================
# Datasets (数据集)
# ======================================================================


class AnimeSRDataset(Dataset):
    """Single-mode SR Dataset (单模式超分数据集).

    从 HR 目录在线生成 (LR, HR) 配对。
    流程: 加载 → 随机裁剪 HR patch → 退化→LR → 配对增强 → tensor [0,1]。
    支持混合分辨率: 低分辨率源自动缩小裁剪尺寸。

    Args:
        hr_dir:            HR 图像目录
        hr_patch_size:     HR 裁剪尺寸
        scale_factor:      下采样倍率
        degradation_mode:  退化模式 (0-4)
        augment:           是否启用几何增强
    """

    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

    def __init__(self, hr_dir: str, hr_patch_size: int = 512,
                 scale_factor: int = 4, degradation_mode: int = 1,
                 augment: bool = True):
        super().__init__()
        self.hr_patch_size = hr_patch_size
        self.scale_factor = scale_factor
        self.augment = augment
        self.degradation = DegradationPipeline(degradation_mode, scale_factor)
        self.to_tensor = transforms.ToTensor()
        self.min_lr_size = 32

        self.image_paths = sorted(set(
            str(p) for ext in self.SUPPORTED_EXTENSIONS
            for p in list(Path(hr_dir).rglob(f'*{ext}'))
                     + list(Path(hr_dir).rglob(f'*{ext.upper()}'))
        ))
        assert hr_patch_size % scale_factor == 0
        if not self.image_paths:
            raise FileNotFoundError(f"'{hr_dir}' 中无支持格式的图像")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))

        w, h = img.size
        hr_crop = min(self.hr_patch_size, min(w, h))
        hr_crop = (hr_crop // self.scale_factor) * self.scale_factor
        if hr_crop // self.scale_factor < self.min_lr_size:
            return self.__getitem__(random.randint(0, len(self) - 1))

        top = random.randint(0, h - hr_crop)
        left = random.randint(0, w - hr_crop)
        hr_patch = img.crop((left, top, left + hr_crop, top + hr_crop))
        lr_patch = self.degradation(hr_patch)

        if self.augment:
            hr_patch, lr_patch = paired_random_augment(hr_patch, lr_patch)

        return {
            'lr': self.to_tensor(lr_patch),
            'hr': self.to_tensor(hr_patch),
            'filename': os.path.basename(self.image_paths[idx]),
        }


class CaelumMixedDataset(Dataset):
    """Mixed Degradation Mode Dataset (混合退化模式数据集).

    按可配置概率采样多种退化模式, 让网络同时学习处理全谱退化。
    单一模式训练的网络只擅长该模式; 混合训练覆盖从"仅缩放"到"重度包浆"。
    同一张源图在不同 epoch 经历不同退化 → 巨大的等效数据增强。

    默认采样概率:
      mode 1 (纯放大):   10%  — 最简单, 少量即可
      mode 2 (轻度压缩): 30%  — 常见场景
      mode 3 (中度包浆): 35%  — 主力
      mode 4 (重度包浆): 25%  — 困难样本

    Args:
        hr_dir:        HR 图像目录
        hr_patch_size: HR 裁剪尺寸
        scale_factor:  下采样倍率
        augment:       是否启用几何增强
        mode_probs:    {mode: probability} 采样概率字典
    """

    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

    def __init__(self, hr_dir: str, hr_patch_size: int = 512,
                 scale_factor: int = 4, augment: bool = True,
                 mode_probs: dict = None):
        super().__init__()
        self.hr_patch_size = hr_patch_size
        self.scale_factor = scale_factor
        self.augment = augment
        self.to_tensor = transforms.ToTensor()
        self.min_lr_size = 32

        if mode_probs is None:
            mode_probs = {1: 0.10, 2: 0.30, 3: 0.35, 4: 0.25}
        self.modes = list(mode_probs.keys())
        self.probs = list(mode_probs.values())

        self.pipelines = {m: DegradationPipeline(m, scale_factor)
                          for m in self.modes}

        self.image_paths = sorted(set(
            str(p) for ext in self.SUPPORTED_EXTENSIONS
            for p in list(Path(hr_dir).rglob(f'*{ext}'))
                     + list(Path(hr_dir).rglob(f'*{ext.upper()}'))
        ))
        assert hr_patch_size % scale_factor == 0
        if not self.image_paths:
            raise FileNotFoundError(f"'{hr_dir}' 中无支持格式的图像")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))

        w, h = img.size
        hr_crop = min(self.hr_patch_size, min(w, h))
        hr_crop = (hr_crop // self.scale_factor) * self.scale_factor
        if hr_crop // self.scale_factor < self.min_lr_size:
            return self.__getitem__(random.randint(0, len(self) - 1))

        top = random.randint(0, h - hr_crop)
        left = random.randint(0, w - hr_crop)
        hr_patch = img.crop((left, top, left + hr_crop, top + hr_crop))

        mode = random.choices(self.modes, weights=self.probs, k=1)[0]
        lr_patch = self.pipelines[mode](hr_patch)

        if self.augment:
            hr_patch, lr_patch = paired_random_augment(hr_patch, lr_patch)

        return {
            'lr': self.to_tensor(lr_patch),
            'hr': self.to_tensor(hr_patch),
            'filename': os.path.basename(self.image_paths[idx]),
            'mode': mode,
        }


# ======================================================================
# Collate & DataLoader Factory (Collate 与 DataLoader 工厂)
# ======================================================================


def sr_collate_fn(batch: list) -> dict:
    """处理 batch 内不同 patch 尺寸: 中心裁剪到最小尺寸。

    无插值, 像素精确。≥720p 数据源下绝大多数 batch 内尺寸一致, 几乎不触发裁剪。
    """
    lr_sizes = [item['lr'].shape[-1] for item in batch]
    min_lr = min(lr_sizes)
    scale = batch[0]['hr'].shape[-1] // batch[0]['lr'].shape[-1]
    min_hr = min_lr * scale

    lr_list, hr_list, fnames = [], [], []
    for item in batch:
        lr, hr = item['lr'], item['hr']
        if lr.shape[-1] > min_lr or lr.shape[-2] > min_lr:
            lr = TF.center_crop(lr, [min_lr, min_lr])
        if hr.shape[-1] > min_hr or hr.shape[-2] > min_hr:
            hr = TF.center_crop(hr, [min_hr, min_hr])
        lr_list.append(lr)
        hr_list.append(hr)
        fnames.append(item['filename'])

    result = {
        'lr': torch.stack(lr_list),
        'hr': torch.stack(hr_list),
        'filename': fnames,
    }
    if 'mode' in batch[0]:
        result['mode'] = [item['mode'] for item in batch]
    return result


def create_train_dataloader(
    hr_dir: str,
    batch_size: int = 8,
    hr_patch_size: int = 512,
    scale_factor: int = 4,
    degradation_mode: int = None,
    mode_probs: dict = None,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """创建训练 DataLoader。

    degradation_mode=None (默认) → CaelumMixedDataset 混合模式;
    degradation_mode=整数 → AnimeSRDataset 单模式。
    """
    if degradation_mode is not None:
        dataset = AnimeSRDataset(hr_dir, hr_patch_size, scale_factor,
                                 degradation_mode, augment=True)
        desc = MODE_NAMES.get(degradation_mode, '未知')
        print(f"[Dataset] {len(dataset)} 张图像, 模式 {degradation_mode}: {desc}")
    else:
        dataset = CaelumMixedDataset(hr_dir, hr_patch_size, scale_factor,
                                      augment=True, mode_probs=mode_probs)
        print(f"[Dataset] {len(dataset)} 张图像, 混合退化模式")

    lr_size = hr_patch_size // scale_factor
    print(f"[Dataset] HR: {hr_patch_size}x{hr_patch_size} -> LR: {lr_size}x{lr_size}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=sr_collate_fn,
        persistent_workers=num_workers > 0,
    )


def create_val_dataloader(
    hr_dir: str,
    batch_size: int = 1,
    hr_patch_size: int = 512,
    scale_factor: int = 4,
    num_workers: int = 2,
) -> DataLoader:
    """验证 DataLoader: mode=0 纯 bicubic, 无增强, 不打乱。"""
    dataset = AnimeSRDataset(hr_dir, hr_patch_size, scale_factor,
                             degradation_mode=0, augment=False)
    print(f"[Val] {len(dataset)} 张图像")
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=sr_collate_fn,
    )


# ======================================================================
# Self-Test (自检)
# ======================================================================


def verify_dataset(hr_dir: str):
    """验证数据管线: 全倍率 × 全模式 + 混合模式 + DataLoader 迭代。"""
    print("=" * 70)
    print("  数据管线验证 (scale-aware)")
    print("=" * 70)

    for sf in [4, 2, 1]:
        print(f"\n{'─' * 40}")
        print(f"  scale_factor = {sf}×")
        print(f"{'─' * 40}")

        for mode in range(5):
            desc = MODE_NAMES.get(mode, '?')
            print(f"\n  ■ 模式 {mode}: {desc}")
            ds = AnimeSRDataset(hr_dir, 512, sf, mode, augment=(mode > 0))
            print(f"    扫描: {len(ds)} 张")
            s = ds[0]
            lr, hr = s['lr'], s['hr']
            print(f"    LR: {list(lr.shape)}, [{lr.min():.3f}, {lr.max():.3f}]")
            print(f"    HR: {list(hr.shape)}, [{hr.min():.3f}, {hr.max():.3f}]")
            if sf > 1:
                assert hr.shape[-1] == lr.shape[-1] * sf, \
                    f"尺寸不匹配: HR={hr.shape[-1]}, LR={lr.shape[-1]}, scale={sf}"
            else:
                assert hr.shape[-1] == lr.shape[-1], \
                    f"1× 应同尺寸: HR={hr.shape[-1]}, LR={lr.shape[-1]}"
            assert 0 <= lr.min() and lr.max() <= 1
            print(f"    ✓ 通过")

        print(f"\n  ■ 混合退化模式 ({sf}×):")
        mixed = CaelumMixedDataset(hr_dir, 512, sf, augment=True)
        modes_seen = set()
        for i in range(min(20, len(mixed))):
            s = mixed[i]
            modes_seen.add(s['mode'])
        print(f"    20 次采样覆盖模式: {sorted(modes_seen)}")

    print(f"\n■ DataLoader 迭代 (4×):")
    loader = create_train_dataloader(hr_dir, batch_size=4, num_workers=0)
    batch = next(iter(loader))
    print(f"  LR batch: {list(batch['lr'].shape)}")
    print(f"  HR batch: {list(batch['hr'].shape)}")
    print(f"  ✓ 正常")

    print("\n" + "=" * 70)
    print("  验证完成 ✓")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        verify_dataset(sys.argv[1])
    else:
        print("用法: python dataset.py <HR图像目录>")
