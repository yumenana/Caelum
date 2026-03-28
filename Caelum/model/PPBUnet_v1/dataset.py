"""
Dataset Pipeline for Anime Super-Resolution Training
=====================================================

PPBUNet 训练数据管线。
模拟"电子包浆"
从 HR 源图像实时生成 (LR, HR) 配对。

Pipeline:
  HR image (>=480p) -> random crop -> augmentation -> degradation -> (LR, HR) pair

Degradation Modes:
  mode 0: Pure bicubic downscale 4x (validation)
  mode 1: Pre-blur + bicubic downscale 4x (upscale only)
  mode 2: Bicubic downscale 4x + random 1-3x JPEG/WebP (light)
  mode 3: 2-stage high-order degradation lv2 (medium — "social media repost")
  mode 4: 3-stage high-order degradation lv3 (heavy — "digital patina")

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


def apply_pre_blur(img: Image.Image) -> Image.Image:
    """降采样前预模糊。

    模拟创作者上传前的抗锯齿或软件自动平滑;
    50% Gaussian(r=2) / 50% Gaussian(r=1)+Box 两种模式覆盖不同软件行为。
    """
    if random.random() < 0.5:
        return img.filter(ImageFilter.GaussianBlur(radius=2))
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    return img.filter(ImageFilter.BLUR)


def apply_stage_blur(img: Image.Image) -> Image.Image:
    """退化阶段间模糊。

    模拟平台转存时服务端图像处理管线的二次平滑;
    三种核 (Gaussian r=1 / Box / Gaussian r=2) 覆盖不同平台实现。
    """
    choice = random.randint(0, 2)
    if choice == 0:
        return img.filter(ImageFilter.GaussianBlur(radius=1))
    elif choice == 1:
        return img.filter(ImageFilter.BLUR)
    return img.filter(ImageFilter.GaussianBlur(radius=2))


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


def sample_first_stage_scale() -> float:
    """第一阶段缩放因子采样。

    50%→0.5 (常见的 2 倍缩放上传) |
    25%→1.0 (原尺寸上传) |
    25%→U(0.5,1.0) (任意缩放)。
    覆盖创作者上传时的各种缩放行为。
    """
    r = random.random()
    if r < 0.50:
        return 0.5
    elif r < 0.75:
        return 1.0
    return random.uniform(0.5, 1.0)


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


def apply_sinc_ringing(img: Image.Image, level: int) -> Image.Image:
    """施加 Sinc 振铃伪影。

    lv2 截止频率 ω∈[2π/3,π] → 轻度振铃 (仅最高频过冲);
    lv3 截止频率 ω∈[π/3,2π/3] → 重度振铃 (中频也受影响, 可见马赫带)。
    小图用 7/9 核避免超出图像尺寸, 大图用 11/13 核覆盖更宽振铃。
    """
    ks = random.choice([7, 9] if min(img.size) < 128 else [11, 13])
    if level == 2:
        omega = random.uniform(2 * math.pi / 3, math.pi)
    else:
        omega = random.uniform(math.pi / 3, 2 * math.pi / 3)
    return apply_sinc_filter(img, generate_sinc_kernel(ks, omega))


# ======================================================================
# Degradation Pipeline (退化管线)
# ======================================================================


MODE_NAMES = {
    0: '纯 bicubic (验证用)',
    1: '纯放大 (预模糊 + bicubic)',
    2: '轻度退化 (bicubic + 随机多次压缩)',
    3: '中度退化 (高阶 2 阶段, lv2)',
    4: '重度退化 (高阶 3 阶段, lv3)',
}


class DegradationPipeline:
    """Multi-level Degradation Generator (多级退化生成器).

    高阶退化 (mode 3/4) 模拟"电子包浆"全链路:
      Stage 1 (创作者上传): HR → 模糊 → 缩放 → 压缩 (q=75-95)
      Stage 2 (平台转存):   → DCT偏移 → 振铃 → 模糊 → 缩放 → 压缩 (q=50-80)
      Stage 3 (终端获取):   → DCT偏移 → 放大截图 → 最终压缩 (lv2:40-75 / lv3:10-40)

    每个阶段独立随机化, 组合爆炸式覆盖真实退化空间。

    Args:
        mode:         退化模式 (0-4)
        scale_factor: 下采样倍率
    """

    INTERPOLATION_METHODS = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC]

    def __init__(self, mode: int = 1, scale_factor: int = 4):
        self.mode = mode
        self.scale_factor = scale_factor

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
        """Mode 0/1: 纯下采样, 可选预模糊。"""
        w, h = hr.size
        lr_w, lr_h = w // self.scale_factor, h // self.scale_factor
        if blur and random.random() < 0.5:
            hr = apply_pre_blur(hr)
        interp = self._random_downsample_interp() if blur else Image.BICUBIC
        return hr.resize((lr_w, lr_h), interp)

    def _degrade_light(self, hr: Image.Image) -> Image.Image:
        """Mode 2: 下采样 + 随机 1-3 次压缩。

        多次压缩模拟图片经多个平台转存的累积退化。
        """
        w, h = hr.size
        lr_w, lr_h = w // self.scale_factor, h // self.scale_factor
        lr = hr.resize((lr_w, lr_h), self._random_downsample_interp())
        n_passes = random.choices([1, 2, 3], weights=[3, 5, 2], k=1)[0]
        for _ in range(n_passes):
            lr = apply_random_compression(lr, random.randint(50, 95))
        return lr

    def _degrade_high_order(self, hr: Image.Image, level: int) -> Image.Image:
        """Mode 3/4: 高阶退化, 模拟互联网传播链路。

        level=2 (mode 3): 2 阶段 — Stage1 创作者上传 + Stage2 平台转存。
        level=3 (mode 4): 3 阶段 — 额外 Stage3 终端获取 (重度包浆)。
        """
        w, h = hr.size
        lr_w, lr_h = w // self.scale_factor, h // self.scale_factor
        img = hr

        # === Stage 1: 创作者上传 ===
        if random.random() < 0.5:
            img = apply_pre_blur(img)
        scale = sample_first_stage_scale()
        if scale < 1.0:
            img = img.resize((max(lr_w, int(w * scale)),
                              max(lr_h, int(h * scale))),
                             self._random_downsample_interp())
        img = apply_random_compression(img, random.randint(75, 95))

        # === Stage 2: 平台转存 ===
        if random.random() < 0.3:
            img = apply_sinc_ringing(img, level)
        if random.random() < 0.5:
            img = apply_stage_blur(img)
        img = img.resize((lr_w, lr_h), self._random_downsample_interp())
        img, (dy2, dx2) = break_dct_grid(img)
        img = apply_random_compression(img, random.randint(50, 80))
        img = undo_dct_shift(img, dy2, dx2)

        # === Stage 3: 终端获取 (仅 mode 4 / level=3) ===
        if level == 3:
            if random.random() < 0.50 and lr_w > 64:
                small = random.randint(max(32, lr_w // 2), lr_w - 1)
                img = img.resize((small, small), self._random_downsample_interp())
                img = apply_sinc_ringing(img, level)
                img = img.resize((lr_w, lr_h), random.choice(self.INTERPOLATION_METHODS))

            img, (dy3, dx3) = break_dct_grid(img)
            img = apply_random_compression(img, random.randint(10, 40))
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
    """验证数据管线: 各退化模式 + 混合模式 + DataLoader 迭代。"""
    print("=" * 70)
    print("  数据管线验证")
    print("=" * 70)

    for mode in range(5):
        desc = MODE_NAMES.get(mode, '?')
        print(f"\n■ 模式 {mode}: {desc}")
        ds = AnimeSRDataset(hr_dir, 512, 4, mode, augment=(mode > 0))
        print(f"  扫描: {len(ds)} 张")
        s = ds[0]
        lr, hr = s['lr'], s['hr']
        print(f"  LR: {list(lr.shape)}, [{lr.min():.3f}, {lr.max():.3f}]")
        print(f"  HR: {list(hr.shape)}, [{hr.min():.3f}, {hr.max():.3f}]")
        assert hr.shape[-1] == lr.shape[-1] * 4
        assert 0 <= lr.min() and lr.max() <= 1
        print(f"  ✓ 通过")

    print(f"\n■ 混合退化模式:")
    mixed = CaelumMixedDataset(hr_dir, 512, 4, augment=True)
    modes_seen = set()
    for i in range(min(20, len(mixed))):
        s = mixed[i]
        modes_seen.add(s['mode'])
    print(f"  20 次采样覆盖模式: {sorted(modes_seen)}")

    print(f"\n■ DataLoader 迭代:")
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
