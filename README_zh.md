[English](README.md) | **中文**

<div align="center">

<img src="Caelum/assets/logo.png" width="240" alt="Caelum Logo"/>

# Caelum「澄空」

**既然我们共享同一片天空，那我希望它是清晰的。**

*Inspired by Porter Robinson — "Look At The Sky"*

<br/>

<!-- Badges -->
![Status](https://img.shields.io/badge/status-active%20development-brightgreen)
[![GitHub Stars](https://img.shields.io/github/stars/yumenana/Caelum?style=flat&logo=github)](https://github.com/yumenana/Caelum/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/yumenana/Caelum?style=flat&logo=github)](https://github.com/yumenana/Caelum/forks)
![Platform](https://img.shields.io/badge/platform-Windows-0078D4?logo=windows)
![Code License](https://img.shields.io/badge/code-AGPL--3.0-blue)
![Model License](https://img.shields.io/badge/model%20%26%20training-CC%20BY--NC--SA%204.0-lightgrey?logo=creativecommons)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
<!-- Release 发布后取消注释下方链接 -->
<!-- [![Release](https://img.shields.io/github/v/release/yumenana/Caelum)](https://github.com/yumenana/Caelum/releases/latest) -->
<!-- [![Downloads](https://img.shields.io/github/downloads/yumenana/Caelum/total)](https://github.com/yumenana/Caelum/releases) -->

</div>

---

## 什么是 Caelum？

动漫插画在互联网上传播时，往往会经历多次缩放、JPEG/WebP 压缩，最终以一种模糊、有损的"电子包浆"状态呈现在你面前。

**Caelum 的目标，是尽可能把它还原回来。**

这是一个专门针对现代互联网图像退化场景（Pixiv、X、Facebook 等平台传播的动漫插画）的 **×4 超分辨率重建网络**。它不是扩散模型，不是在"生成"，而是在尝试"还原"——尽可能地接近那张图本来的样子。

---

## 🔍 横向对比

Caelum 聚焦一个特定的、尚未被充分解决的问题：**还原经历了真实互联网多平台转存链路后的动漫插画**。与通用超分工具不同，Caelum 的训练管线专门模拟图像在 Pixiv、Twitter/X、Facebook、Discord 以及截图循环中的真实退化方式。

| | Caelum | [waifu2x](https://github.com/nagadomi/waifu2x) | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | [RealCUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) | [Anime4K](https://github.com/bloc97/Anime4K) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **适用内容** | 动漫插画 | 动漫/插画 | 通用/动漫 | 动漫 | 动漫（视频） |
| **退化模型** | 多平台互联网转存链路 | 噪声 + 模糊 | 真实世界通用 | 噪声 + 压缩 | — |
| **网络架构** | PPBUNet（U-Net + HAT 注意力 + 连通流形虫洞） | SwinUNet / CNN | RRDB | U-Net | GLSL 着色器 |
| **输出倍率** | ×4 | ×1/×2/×4 | ×2/×4 | ×2/×3/×4 | 可变 |
| **JPEG/WebP 去伪影** | ✅ 多阶段 | ✅ | ✅ | ✅ | — |
| **推理后端** | DirectML (ONNX) | NCNN / Vulkan | NCNN / CUDA | NCNN | OpenCL / Vulkan |
| **Windows GUI** | ✅ 原生 | 部分 | 部分 | 部分 | ✅ |
| **免费** | ✅ | ✅ | ✅ | ✅ | ✅ |

> ⚠️ Caelum 目前正在积极训练中。量化对比指标（PSNR/SSIM/LPIPS）将随第一个稳定 Release 一同发布。

---

## ✨ 特性

- 🎯 **场景专注** — 专门模拟现代社交/图站平台的真实退化链路（缩放 + JPEG/WebP 压缩），而不是通用退化
- 🏗️ **PPBUnet** — Palette-Painter-Brush U-Net for Anime Super-Resolution
- ⚡ **×4 超分** — 主攻 4 倍超分辨率重建
- 🪟 **Windows GUI** — 开箱即用的 `.exe` 应用，无需 Python 或命令行
- 🔓 **免费开源** — 永久免费；源代码以 AGPL-3.0 / CC BY-NC-SA 4.0 开放

---

## 🖼️ 效果展示

<div align="center">

![Caelum Demo](Caelum/assets/demo0.png)
![Caelum Demo](Caelum/assets/demo1.png)

</div>

| 退化输入 | [waifu2x](https://github.com/nagadomi/waifu2x) SwinUNet noise2 ×4 | **Caelum PPBUNet** CAR2 ×4 | Ground Truth |
|:---:|:---:|:---:|:---:|
| ![退化输入](Caelum/assets/compare/degradation.png) | ![waifu2x](Caelum/assets/compare/waifu2x_SwinUNet_noise2_x4.png) | ![Caelum](Caelum/assets/compare/PPBUnet_CAR2_x4.png) | ![GT](Caelum/assets/compare/GT.png) |

> 📝 对比图像由 Google Gemini 生成，角色为[东方Project](https://www.thpatch.net/wiki/Touhou_Patch_Center:Main_page)（ZUN）的博丽灵梦。东方Project 允许非商业性质的二次创作，本项目为非商业开源项目，不存在版权问题。

#### 真实网图重建（无 GT 参考）

> 以下两组退化输入为从互联网随手保存的动漫插画裁剪块，没有原图作为参考。对比只反映实际使用场景下重建结果的主观观感（细节保留、纯净度、伪影抑制等）。

| 退化输入（网图 crop） | [waifu2x](https://github.com/nagadomi/waifu2x) SwinUNet noise1 ×4 | **Caelum PPBUNet** CAR1 ×4 |
|:---:|:---:|:---:|
| <img src="Caelum/assets/compare1/web_src.png" width="240"> | <img src="Caelum/assets/compare1/waifu2x_SwinUNet_noise1_x4.png" width="240"> | <img src="Caelum/assets/compare1/PPBUnet_CAR1_x4.png" width="240"> |

| 退化输入（网图 crop） | [waifu2x](https://github.com/nagadomi/waifu2x) SwinUNet noise2 ×4 | **Caelum PPBUNet** CAR1 ×4 |
|:---:|:---:|:---:|
| <img src="Caelum/assets/compare2/web_src.png" width="240"> | <img src="Caelum/assets/compare2/waifu2x_SwinUNet_noise2_x4.png" width="240"> | <img src="Caelum/assets/compare2/PPBUnet_CAR1_x4.png" width="240"> |

| 退化输入（网图 crop） | [waifu2x](https://github.com/nagadomi/waifu2x) SwinUNet noise3 ×4 | **Caelum PPBUNet** CAR2 ×4 |
|:---:|:---:|:---:|
| <img src="Caelum/assets/compare3/web_src.png" width="240"> | <img src="Caelum/assets/compare3/waifu2x_SwinUNet_noise3_x4.png" width="240"> | <img src="Caelum/assets/compare3/PPBUnet_CAR2_x4.png" width="240"> |

---

## 🚀 快速开始

### 下载与使用

发布包分为两个部分，请从 [Releases](https://github.com/yumenana/Caelum/releases) 下载两者：

| 压缩包 | 内容 |
|--------|------|
| **`Caelum_vX.Y.Z.zip`** | GUI 应用程序 |
| **`Models_YYYYMMDD.zip`** | 模型权重（日期版本号；每次仅包含有更新的模型） |

解压后，将 `Models/` 文件夹放入应用程序目录下：

```
Caelum/        ← 应用目录
└── Models/    ← 将解压出的 Models 文件夹放在这里
```

然后运行 `Caelum.exe`。

### 系统要求

- **操作系统**: Windows 10 版本 2004（内部版本 19041）或更高 / Windows 11
- **架构**: x64 或 ARM64
- **.NET 运行时**: [.NET 10 桌面运行时](https://dotnet.microsoft.com/download/dotnet/10.0)
- **GPU（可选）**: 任意兼容 DirectX 12 的 GPU（NVIDIA / AMD / Intel）
  - 若无可用的 DX12 GPU，推理将自动回退至 CPU（速度较慢）

---

## 🏗️ 网络架构

> 当前版本：**PPBUNet v1.7**

2 级 U-Net 金字塔 + 平行几何旁路，三阶段工作流：**Palette** 提取全局配色 → **Painter** 重建全局结构 → **Brush** 几何精修与上采样。


```mermaid
flowchart TD
    In(["Input Image"]) --> SE["Shallow · RepSRBlock"]
    SE -->|"parallel bypass"| OAM["ParallelOAM<br/>0° / 90° / 45° / 135°"]
    SE --> L0

    subgraph Encoder["Encoder — 2-Level Pyramid"]
        L0["Level 0 · 64ch · RCAB×2"] --> L1["Level 1 · 128ch · RCAB×2"]
    end

    L1 --> FR
    subgraph BN["Bottleneck — Frequency Decoupled"]
        FR["FrequencyRouter<br/>DC / AC split"]
        FR -->|"AC"| PAI["HAT Bottleneck<br/>Window Attention + OCAB"]
        FR -->|"DC"| DCBR["DC Branch<br/>(Color prototype path)"]
    end

    L1 -.->|"skip"| MIM1["MIM Filter L1<br/>InfoNCE-purified"]
    PAI --> MIM1
    MIM1 --> SF1["SimpleSkipFusion L1<br/>concat → 1×1 → GroupNorm"]
    SF1 --> DL1["Decoder L1<br/>HAT Groups"]

    L0 -.->|"skip"| MIM0["MIM Filter L0"]
    DL1 --> MIM0
    MIM0 --> SF0["SimpleSkipFusion L0"]
    SF0 --> DL0P["Decoder L0 · Pre HAT"]

    DL0P --> CMW["ConnectedManifoldWormhole<br/>Bilateral manifold injection"]
    DCBR --> CMW
    CMW -.->|"training only"| AUX["CreviceAuxHead<br/>Aux topology supervision"]
    CMW --> DL0Q["Decoder L0 · Post HAT"]
    DL0Q --> DCONV["dec_l0_conv"]

    DCONV --> GUARD["Guard GroupNorm"]
    GUARD --> TDCN["TopologyGuidedDCN<br/>OAM-guided · local 3-5px"]
    OAM -->|"geometry prior"| TDCN

    TDCN --> UP["SATUpsampler v2<br/>Anisotropic dynamic filtering<br/>+ Implicit polygon rendering"]
    UP --> Out(["Output · 4H × 4W"])

classDef io fill:#4A90D9,stroke:#2C5F8A,color:#fff
classDef se fill:#5D6D7E,stroke:#2E4057,color:#fff
classDef bypass fill:#8E44AD,stroke:#5B2C6F,color:#fff
classDef enc fill:#2980B9,stroke:#1A4F7A,color:#fff
classDef btn fill:#E67E22,stroke:#9A5C12,color:#fff
classDef pal fill:#D4AC0D,stroke:#8B6E0A,color:#fff
classDef skip fill:#16A085,stroke:#0B6B5A,color:#fff
classDef dec fill:#2471A3,stroke:#154360,color:#fff
classDef brush fill:#C0392B,stroke:#7B241C,color:#fff
classDef up fill:#1ABC9C,stroke:#0E7560,color:#fff
classDef aux fill:#7F8C8D,stroke:#34495E,color:#fff,stroke-dasharray:5 5

class In,Out io
class SE se
class OAM bypass
class L0,L1 enc
class FR,PAI btn
class DCBR,CMW pal
class MIM0,MIM1,SF0,SF1 skip
class DL1,DL0P,DL0Q,DCONV,GUARD dec
class TDCN brush
class UP up
class AUX aux
```

### Architecture Stages

| Stage | Function |
|-------|----------|
| **Shallow (RepSRBlock)** | 输入转化为共享潜在表示；训练多分支重参数化，推理折叠为单 Conv（零额外开销） |
| **ParallelOAM Bypass** | 全分辨率平行旁路，0°/90°/45°/135° 绝对方向基底，扫除曼哈顿阶梯锅齿 |
| **Encoder (2-Level)** | RCAB×2 逾级下采样，逐层聚合多尺度空间上下文 |
| **FrequencyRouter** | 显式分离 DC（平坦色块）与 AC（边缘拓扑）双流 |
| **HAT Bottleneck** | 窗口自注意力 + OCAB；v1.5 起替代 PSMamba，训练加速 2–3×且完全并行 |
| **MIM + SimpleSkipFusion** | InfoNCE 提纯的跳连过滤，随后 concat → 1×1 → GroupNorm 融合，限制压缩伪影传递 |
| **Decoder (HAT)** | 两级 HAT 解码器；L0 被拆为 Pre / Post 两段，中间插入 CMW |
| **ConnectedManifoldWormhole (CMW)** | 双边流形虫洞：从 DC 分支拉取全局调色原型，注入解码 L0 中间点 |
| **CreviceAuxHead** | 训练专用辅助监督头，在 CMW 输出处直接施加拓扑 loss，推理零开销 |
| **Guard GroupNorm** | 在 TopologyGuidedDCN 前插入 GN，切断解码末端可能出现的 1e10 量级幅值传播 |
| **TopologyGuidedDCN** | 可变形卷积，ParallelOAM 几何先验驱动，专治发尖与〔≤4px〕夹缝 3–5px 几何精修 |
| **SATUpsampler v2** | 各向异性动态滤波 + 隐式多边形渲染，奇异点感知 ×4 上采样 |

详细设计推导与模块规格，参见 [`Caelum/model/PPBUnet_v1/ARCHITECTURE.md`](Caelum/model/PPBUnet_v1/ARCHITECTURE.md)。

### 退化模拟设计

动漫插画在互联网上传播时，经历的不是单次压缩，而是一条完整的**多平台转存链路**。训练数据管线 (`dataset.py`) 用 5 种退化模式在线模拟这条链路，每个 batch 实时生成 (LR, HR) 配对，无需预先存储退化图像。

#### 退化模式

| 模式 | 名称 | 场景 | 采样比例 |
|:---:|------|------|:---:|
| 0 | 纯 Bicubic | 纯数学下采样（验证集专用） | — |
| 1 | 预模糊 + Bicubic | 模拟抗锯齿上传 | 10% |
| 2 | 轻度压缩 | Bicubic ↓4× + 随机 1-3 次 JPEG/WebP | 30% |
| 3 | 中度包浆 | 三阶段高阶退化 lv2（社交平台转存） | 35% |
| 4 | 重度包浆 | 三阶段高阶退化 lv3（深度电子包浆） | 25% |

训练使用 `CaelumMixedDataset`，每张图片在不同 epoch 经历不同退化，实现爆炸式等效数据增强。

#### 三阶段高阶退化链路 (Mode 3/4)

```
HR 原图
  │
  ▼ Stage 1 — 创作者上传
  ├─ 50% 概率预模糊 (Gaussian r=2 / r=1+Box)
  ├─ 随机缩放 (50%→0.5× · 25%→1.0× · 25%→均匀采样)
  └─ JPEG/WebP 压缩 (q = 75–95)  [JPEG 70% / WebP 30%]
  │
  ▼ Stage 2 — 平台转存
  ├─ 30% 概率 Sinc 振铃 (Hamming 窗, lv2: ω∈[2π/3,π] / lv3: ω∈[π/3,2π/3])
  ├─ 50% 概率二次模糊
  ├─ 双线性/双三次随机缩放 → 目标 LR 尺寸
  ├─ DCT 网格偏移 1-7px (打破量化网格对齐，产生真实多重压缩块效应)
  └─ JPEG/WebP 压缩 (q = 50–80)
  │
  ▼ Stage 3 — 终端获取
  ├─ lv2: 25% / lv3: 50% 概率截图放大再压缩
  ├─ DCT 网格偏移 + 最终压缩 (lv2: q=40–75 / lv3: q=10–40)
  └─ 恢复坐标对齐
  │
  ▼ LR 输出
```

#### 关键技术细节

| 技术 | 实现 | 目的 |
|------|------|------|
| **JPEG/WebP 分段线性映射** | `jpeg_quality_to_webp()` | WebP 低质量端效率远高于 JPEG，等感知强度需差异化映射 |
| **DCT 网格偏移** | `break_dct_grid()` 循环位移 1-7px | 两次压缩块边界不重合，产生真实的重叠块效应 |
| **Sinc 振铃** | Hamming 窗截断 sinc + À Trous 多尺度 | 模拟下采样/重采样引入的振铃过冲 (Gibbs 现象) |
| **混合插值** | 50% Bicubic / 50% Bilinear | 覆盖不同平台缩放算法差异 |
| **几何增强** | 水平翻转 × 垂直翻转 × 90° 旋转 | 8 种组合，8× 等效数据扩充 |
| **不落盘压缩** | `io.BytesIO` 内存编解码 | 完整 DCT 编解码保证退化真实性，无文件系统开销 |

---

### 损失函数设计

`CaelumLossV2` 统一调度 **12 个子损失**，覆盖像素、色彩、频域、空间、感知、对抗六个维度，采用**两阶段渐进式启用**策略。

#### 两阶段渐进策略

```
训练进度
0%──────────────30%──────────────────────────100%
│      Phase 1        │          Phase 2           │
│  像素 + 色彩锚定     │  + 高频 + 感知 + 对抗       │
└─────────────────────┴────────────────────────────┘
```

Phase 1 先让网络收敛到正确的色彩和像素分布；Phase 2 再引入强约束，精修线条、频域、语义细节，避免初期梯度震荡。

#### 子损失一览

**Phase 1（全程生效）**

| 损失 | 权重 | 作用 |
|------|:---:|------|
| `L1` | 1.0 | 像素级绝对误差基准 |
| `AdaptiveDCAnchorLoss` | 1.0 | Scharr 梯度能量驱动指数衰减软权重，平坦区 L1 放大、纹理区渐退，彻底消除硬阈值方差不稳定 |
| `OklchColorLoss` | 5.0 | OKLCH 感知色彩空间：色度 L1 + 色相余弦联合约束，atan2-free |
| `StrictFlatTGVLoss` | 1.0 | 形态学硬掩码隔离平涂区，Charbonnier 惩罚一阶+二阶导数→0，根治平坦区纹波 |
| `SmoothGradientHessianLoss` | 1.5 | 结构张量引导的渐变区 Hessian 惩罚，消除色彩断层 (Color Banding) 和微小波纹 |

**Phase 2（训练进度 ≥ 30% 后加入）**

| 损失 | 权重 | 作用 |
|------|:---:|------|
| `ChromaGradientLoss` | 1.5 | Sobel 直接约束 Oklab a/b 色度梯度与 GT 对齐，抗色彩溢出 |
| `CreviceColorLoss` | 6.0 | 形态学闭运算检测描边夹缝，修复 JPEG 4:2:0 色度子采样导致的色相偏移 |
| `MaskedAsymmetricHistogramLoss` | 1.5 | 边缘膨胀区域软直方图，非对称散度重罚"无中生有"杂色（×5）、轻罚"未能恢复"细节（×1） |
| `GibbsRingingPenaltySWT` | 4.0 | Haar SWT 三子带（HL/LH/HH）× À Trous 多尺度（d=1,2,4），单侧惩罚高频过冲，不干预正常锐化 |
| `AngularFluencyLoss` | 5.0 | Farid 7×7 旋转等变算子计算梯度方向角距离，直接消除超分锯齿 |
| `MacroscopicTurningPointLoss` | 0.5 | 膨胀 Scharr (d=2) + 11×11 高斯宏观积分，对比度不变角点响应 C = 4·det(S)/trace(S)² |
| `LaplacianResonanceTopologyLoss` | 1.5 | 双尺度膨胀拉普拉斯共振 + R^75 余弦拓扑 + Charbonnier 强度，修复缝隙拓扑 |
| `AnimePerceptualLossV2` | 0.5 | Danbooru ConvNeXt 余弦流形距离（stage0+stage1），GT 幅值门控聚焦边缘区域 |

**GAN 组件（可选）**

| 组件 | 说明 |
|------|------|
| `DecoupledUNetDiscriminatorSN` | 引导滤波前端分解结构/纹理双流；结构分支全功率 U-Net，纹理分支轻量全局统计，防止 D 利用不可重建纹理逼迫 G 产生幻觉 |
| `DecoupledGANLoss` | 结构对抗权重 ×1.0，纹理对抗权重 ×0.1（`texture_tolerance`），谱归一化稳定训练 |

**MIM 辅助损失**

训练时通过 `model.mi_loss` 获取 InfoNCE 跳跃连接互信息损失，建议加权 λ=0.01：

```python
loss = criterion(pred, hr) + 0.01 * model.mi_loss
```

---

## 📊 实验结果

这个项目不以论文发表为目标，没有控制变量消融实验。

架构设计从理论上推导处于前沿水平，特别是在现代互联网图像退化还原这一特定场景下。

> *如果你用它处理了一张图，觉得效果好——那就可以了。*

---

## 📁 项目结构

```
Caelum/                            ← 仓库根目录
├── .github/
│   └── FUNDING.yml
├── Caelum/                        ← Python 项目目录
│   ├── assets/
│   │   ├── logo.png               # 项目 Logo
│   │   ├── demo0.png              # GUI 演示图
│   │   ├── demo1.png              # GUI 演示图
│   │   └── compare/               # 效果对比图
│   │       ├── degradation.png
│   │       ├── GT.png
│   │       ├── PPBUnet_CAR2_x4.png
│   │       └── waifu2x_SwinUNet_noise2_x4.png
│   └── model/
│       └── PPBUnet_v1/
│           ├── ARCHITECTURE.md       # 架构详细设计文档
│           ├── PPBUNet_v1_x4.py      # 主网络定义            [AGPL-3.0]
│           ├── modules.py            # 核心模块库            [AGPL-3.0]
│           ├── hat.py                # HAT Decoder          [AGPL-3.0]
│           ├── ps_mamba.py           # PS-Mamba SSM 模块    [AGPL-3.0]
│           ├── dataset.py            # 在线退化数据管线      [CC BY-NC-SA 4.0]
│           ├── losses.py             # 定制化损失函数体系    [CC BY-NC-SA 4.0]
│           └── train.py              # 训练脚本             [AGPL-3.0]
├── README.md
├── README_zh.md
├── LICENSE
└── LICENSE-AGPL-3.0
```

> 模型权重及打包应用（`*.onnx`、`Caelum.exe`）通过 [Releases](https://github.com/yumenana/Caelum/releases) 分发，适用 CC BY-NC-SA 4.0。


---

## 🗺️ Roadmap

#### ✅ 已完成

- [x] PPBUNet v1.7 架构设计（ParallelOAM · FrequencyRouter · MIM+SimpleSkipFusion · HAT Bottleneck · ConnectedManifoldWormhole · CreviceAuxHead · TopologyGuidedDCN · SATUpsampler v2）
- [x] 多平台退化模拟 Pipeline（5 模式 · 三阶段高阶链路 · 在线实时生成）
- [x] 定制化损失函数体系（CaelumLossV2 · 12 子损失 · 两阶段渐进策略）
- [x] 早期检查点效果验证

#### 🔄 进行中

- [x] 模型训练完成 → 最终效果展示更新
- [ ] 打包 exe + ONNX 导出 → Release 发布

#### 🔭 后续计划

- [ ] **攻坚毛发重建** — 发尖与描边夹缝区域的细节恢复是当前最大短板，计划针对性设计发丝感知损失与几何精修模块
- [ ] **去残差化架构探索** — 残差连接在抑制伪影上存在根本局限（有害输入信息难以被切断），探索完全摆脱 skip-add 残差的纯注意力 / Mamba 前向架构
- [ ] **新架构探索** — 在 PPBUNet 经验基础上，持续探索更高效、更有意思的动漫超分架构方向
- [ ] **扩充训练数据集** — 现有数据集规模和多样性仍有瓶颈，计划引入更大规模的动漫插画数据（Danbooru · Pixiv 等），同时研究数据清洗与质量筛选流程

---

## 🌌 Origin Story

很多年前，我只是想把喜欢的动漫插画放大到能当桌面壁纸。

[waifu2x](https://github.com/nagadomi/waifu2x) 让我第一次意识到：神经网络可以很好的做到这一点，而且效果在当时可以说是对传统插值放大的降维打击。这引发了我强烈的好奇心——*它是怎么做到的？*

为了找到答案，我开始学深度学习，做出了我的第一个超分网络 [Entropia](https://github.com/yumenana/Entropia)。后来因工作搁置了很长时间。

现在，在LLM的帮助下我回来了，站在过去的自己和所有前人的肩膀上。

Caelum 不是一篇论文，不是一项研究成果，它只是一个问题的延续——**让我喜欢的东西更清晰一些，有什么不对吗？**

我在这条路上受益于太多人的无偿贡献。

---


## ❤️ 支持这个项目

Caelum 永远免费。如果它帮助了你，你可以通过 Ko-fi 请我喝一杯咖啡——这会直接转化为更多的 GPU 时间。

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://Ko-fi.com/yumenana)

---

## 📄 许可证

本项目采用**双许可证**模式：

| 适用范围 | 许可证 | 核心约束 |
|----------|--------|----------|
| 网络架构与训练源代码<br/>（`PPBUNet_v1_x4.py`, `modules.py`, `hat.py`, `ps_mamba.py`, `train.py`） | [AGPL-3.0](LICENSE-AGPL-3.0) | 衍生作品必须开源（含网络服务），**允许**商业使用 |
| 退化/损失代码 + 模型与应用发布文件<br/>（`dataset.py`, `losses.py`, `*.onnx`, `Caelum.exe`） | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) | **禁止**商业用途，衍生必须同协议开源 |

如需将训练代码或模型权重用于商业用途，请联系作者获取独立商业授权。

---

## 📬 致谢

感谢 waifu2x 的作者。

感谢所有愿意让世界变得更好的人。

---

<div align="center">

*"Look at the sky — I'm still here."*

</div>