# PPBUNet v1.7 — Architecture & Design

> PPBUNet: **P**alette-**P**ainter-**B**rush **U-Net** for Anime Super-Resolution

本文档阐述 PPBUNet v1.7 的架构设计。

三阶段工作流以画家创作过程命名：
- **Palette** — 通过连通流形虚洞 (CMW) 双边交叉注意力路由全局色彩
- **Painter** — 通过注意力驱动的 U-Net 解码 (HAT) 重建结构
- **Brush**   — 在全分辨率下精修几何细节并上采样到 HR

> **版本演进概览** (详见 `docs/EpistemicTrace/`)
> - **v1.3** 确立 Palette-Painter-Brush 三阶段管线，引入 CMW 连通流形虫洞取代离散色卡
> - **v1.4** 删除 AnimeCommitteeRefiner；CMW 上移至 Decoder L0 中段；接入 MIM 训练信号 + CreviceAuxHead 深监督
> - **v1.5** 瓶颈层 PSMamba → HAT（消除 SSM 串行瓶颈，训练加速）
> - **v1.6** 权重爆炸根治：GroupNorm 守卫 + FP32 安全区间（FP16 NaN 结构性修复）
> - **v1.7** 全局 reflect padding（修复推理边界暗带）；RMA → SimpleSkipFusion；删除 BADI 浅层注入

---

## 目录

1. [信号模型：动漫插画的双态分布](#1-信号模型动漫插画的双态分布)
2. [宏观数据流](#2-宏观数据流)
3. [核心模块](#3-核心模块)
   - 3.1 [隐空间基底 (Latent Base)](#31-隐空间基底-latent-base)
   - 3.2 [结构重参数化块 (RepSRBlock)](#32-结构重参数化块-repsrblock)
   - 3.3 [平行几何方向流 (ParallelOAM)](#33-平行几何方向流-paralleloam)
   - 3.4 [显式频率路由 (FrequencyRouter)](#34-显式频率路由-frequencyrouter)
   - 3.5 [连通流形虫洞 (ConnectedManifoldWormhole)](#35-连通流形虫洞-connectedmanifoldwormhole)
   - 3.6 [互信息特征提纯 (MIMFeatureFilter)](#36-互信息特征提纯-mimfeaturefilter)
   - 3.7 [简化跳跃融合 (SimpleSkipFusion)](#37-简化跳跃融合-simpleskipfusion)
   - 3.8 [HAT 解码器与瓶颈](#38-hat-解码器与瓶颈)
   - 3.9 [拓扑引导可变形卷积 (TopologyGuidedDCN)](#39-拓扑引导可变形卷积-topologyguideddcn)
   - 3.10 [奇异点感知拓扑上采样 (SATUpsampler_v2)](#310-奇异点感知拓扑上采样-satupsampler_v2)
   - 3.11 [拓扑辅助监督头 (CreviceAuxHead)](#311-拓扑辅助监督头-creviceauxhead)
4. [训练动力学](#4-训练动力学)
5. [数值稳定性：FP16 NaN 根治 (v1.6)](#5-数值稳定性fp16-nan-根治-v16)
6. [开发法则 (AI Handoff Rules)](#6-开发法则-ai-handoff-rules)

---

## 1. 信号模型：动漫插画的双态分布

动漫插画属于人造渲染信号，呈现极端的 **双态分布（Dual-State Distribution）**：

| 信号态 | 特征 | 物理含义 |
|--------|------|----------|
| **DC（低频色块）** | 局部方差极低，大面积平滑 | 全局绝对色度与光影氛围 |
| **AC（高频线条）** | 空间梯度剧烈跳变 | 拓扑结构与几何轮廓 |

退化痛点集中在 DC / AC 交界处：JPEG 块效应、下采样振铃、色度溢出。若用统一算子同时处理两种互斥信号，必然产生特征干涉——颜色变脏，线条断裂。

因此架构的出发点是：**必须在网络内部显式解耦 DC / AC 双流，分别处理后再融合。**

---

## 2. 宏观数据流

2 级 U-Net 金字塔 + 平行几何方向旁路，对照 `PPBUNet.forward()` 阅读：

```text
                          ┌────────── Latent Base (x_shallow) ─────────────────────┐
                          │                                                        │
[Input RGB] ─► reflect ──►│                                                        │
              pad         │                                                        │
              RepSR ×2 ──►│                                                        │
              (shallow)   │                                                        │
                          │──┬── Enc L0 (RCAB ×2, 64ch) ──► ↓2× (reflect)          │
                          │  │       │skip_l0                                      │
                          │  │  Enc L1 (RCAB ×2, 128ch) ──► ↓2× (reflect)          │
                          │  │       │skip_l1                                      │
                          │  │  ┌────┴─── Bottleneck (256ch) ─[FP32]──┐            │
                          │  │  │                                     │            │
                          │  │  │ FrequencyRouter (DW-LPF k=5)        │            │
                          │  │  │    │               │                │            │
                          │  │  │  Feat_DC         Feat_AC            │            │
                          │  │  │  (→CMW Value)    HAT ×N (并行)       │            │
                          │  │  │    │               │                │            │
                          │  │  │    │       bn_conv(feat_ac)+bn_in   │            │
                          │  │  └────│───────────────┘                │            │
                          │  │       │ feat_dc / feat_ac → CMW        │            │
                          │  │  MIM(skip_l1, feat)         [FP32]     │            │
                          │  │  SimpleSkipFusion(skip_l1) +GroupNorm  │  ★ Painter │
                          │  │  HAT Decoder L1 (128ch)                │            │
                          │  │       │                                │            │
                          │  │  MIM(skip_l0, feat)         [FP32]     │            │
                          │  │  SimpleSkipFusion(skip_l0) +GroupNorm  │            │
                          │  │       │                                │            │
                          │  │  HAT Dec L0_pre (64ch)      [FP32]     │            │
                          │  │       │                                │            │
                          │  │  CMW(feat, feat_ac, feat_dc) ◄─────────│ ★ Palette  │
                          │  │       │  └─[CreviceAuxHead 深监督/训练]  │            │
                          │  │  HAT Dec L0_post (64ch)                │            │
                          │  │       │                                │            │
  (ParallelOAM)           │  │  dec_l0_conv + 残差                    │            │
                          │  │       │                                │            │
                          │  │  guard_norm (GroupNorm 守卫)            │            │
                          │  │       ▼                                │            │
                          │  └► TopologyGuidedDCN ◄──────────────────│─ geom_prior │
                          │     (OAM 方向码 + 局部 DCN, γ=1e-2)        │  ★ Brush   │
                          │        │                                  │            │
                          │     AdvancedUpsampler                     │            │
                          │      ├ SATUpsampler_v2 (4×)               │            │
                          │      │  ├ 各向异性动态滤波 (Softmax)         │            │
                          │      │  └ ImplicitPolygonInjector          │            │
                          │      └ tail Conv → [Output RGB] ◄─────────┘
                          └────────────────────────────────────────────────────────┘
```

**两条独立数据通道在 Decoder L0 后交汇：**

| 通道 | 路径 | 物理含义 |
|------|------|----------|
| **低频语义主干** | Shallow → Enc → Bottleneck → Dec | 全局语义结构、色彩语境、跨区域拓扑 |
| **高频几何旁路** | Shallow → ParallelOAM (H×W) | 0°/90°/45°/135° 绝对方向线条锐度 |

**关键设计决策：**

| 决策 | 代码位置 | 原因 |
|------|----------|------|
| CMW 嵌入 Decoder L0 中段 | `dec_l0_pre → cmw → dec_l0_post` | 缩短梯度路径（约 40+ 层 → 20 层），配合 CreviceAuxHead 深监督驱动 CMW |
| CMW 双边交叉注意力取代离散色卡 | `cmw(feat, feat_ac, feat_dc)` | 以"连通色彩流形"为路由单元，自适应渐变/厚涂/光晕，无 grid_sample 插值污染 |
| ParallelOAM 平行旁路 | 与 `enc_l0` 并行 | 高频方向信息绕过下采样，避免 Shannon-Nyquist 混叠 |
| 物理掩码对角核 | `mask_d45 = eye(7)` | Z² 网格上正交基底无法表示斜线，掩码核沿对角线连续 |
| MIM + SimpleSkipFusion 替代裸 concat | `mim_l0/l1` + `fuse_0/1` | 互信息提纯过滤伪影 + concat→Conv1×1→GroupNorm 稳定融合 |
| 因果链: Dec→CMW→guard→DCN→Up | `forward()` 末段 | 结构重建 → 色彩虫洞路由 → 幅值守卫 → 局部几何精修 → 上采样 |
| DC 仅作 CMW Value，AC 仅走 HAT | `freq_router` 双分支 | 频率解耦保证信息正交性 |
| 全链 reflect padding | 所有 `k>1` 空间 Conv | 修复推理边界 zero-pad 暗带（训练 patch 从未见过真实零边界）|
| 删除 BADI 浅层注入 | — | 退化 SR 下浅层旁路注入 LR 域伪影致 DC 偏色（gate 单调降至 ≈0.025 自证）|

---

## 3. 核心模块

### 3.1 隐空间基底 (Latent Base)

| | 传统做法 | PPBUNet v1.0 | PPBUNet v1.7 (当前) |
|---|---------|---------|----------|
| **公式** | `out = network(x) + bicubic(x)` | `out = upsampler(feat_unet + x_shallow)` | `out = AdvancedUpsampler(guard_norm(dec(... CMW ...)) → topo_dcn)` |
| **问题** | 网络被迫生成"反向噪声"抵消 bicubic 振铃，导致懒惰收敛 | 简单加法不加区分地注入退化伪影（浅层 BADI 注入在退化 SR 下有害，已删除）| CMW 虫洞从瓶颈直接路由色彩，GroupNorm 守卫稳定幅值，奇异点感知上采样保护线条拓扑 |

> **注**：v1.7 删除了向 HR 末端注入 `x_shallow` 的 BADI 旁路。退化 SR 场景下，未经 Encoder 过滤的浅层特征携带 LR 域 JPEG/blur/noise 伪影，注入末端会造成 DC 偏色（实测大面积平坦色块 128→126）。Encoder-filtered skip（MIM + SimpleSkipFusion）已足够传递多尺度浅层信息。

### 3.2 结构重参数化块 (RepSRBlock)

训练时三分支并行（3×3 + 1×1 + Identity）激荡非线性方差，打破初始化死锁。推理时 `switch_to_deploy()` 无损折叠为单 3×3 Conv，零精度损失、零推理负担。

> 🚨 **禁止在 RepSR 中使用 BatchNorm！**
> 超分训练使用极小 batch size（如 batch=2），BN 会抹杀绝对色度（DC 分量），导致全局偏色和梯度爆炸。RepSR 是纯线性 Conv 加性融合，不含任何归一化层。

### 3.3 平行几何方向流 (ParallelOAM)

4 个绝对方向基底，彻底覆盖 0°/90°/45°/135°：

| 方向 | 核类型 | 机制 |
|------|--------|------|
| 0° (水平) | 1×7 DW Conv | 天然各向异性核 |
| 90° (垂直) | 7×1 DW Conv | 天然各向异性核 |
| 45° (主对角) | 7×7 DW Conv × `eye(7)` 掩码 | 仅主对角线有效 |
| 135° (反对角) | 7×7 DW Conv × `fliplr(eye(7))` 掩码 | 仅反对角线有效 |

对角掩码通过 `register_buffer` 注册，`forward()` 时 `weight * mask` 保证被遮蔽位置梯度为零。

4 方向分支经能量池化 (x² → GAP) → MLP → Softmax 归一化，逐通道竞争式路由，输出纯几何方向特征。

> 🚨 **方向感知模块禁止串联进 Encoder！**
> - Z² 网格上正交基底 (1×7, 7×1) 的线性组合表示斜线必然产生曼哈顿阶梯
> - 高频方向信息经 stride=2 下采样后混叠不可逆
> - 必须走平行旁路，在 Decoder 末端同分辨率融合

### 3.4 显式频率路由 (FrequencyRouter)

在 1/4 下采样最深处（256ch），初始化为均值滤波器的 `kernel=5` 深度可分离卷积分离出 $Feat_{DC}$，减法得到 $Feat_{AC}$。

频率解耦后两路分流：
- $Feat_{AC}$（高频线条拓扑）→ 瓶颈 HAT 处理，避免被平滑色块的无用信息分散算力
- $Feat_{DC}$（低频色块均值）→ 作为 CMW 的 Value（颜料库色彩源）

> 🚨 **遵守频率解耦法则**：$Feat_{DC}$ 只作 CMW 色彩源，$Feat_{AC}$ 只交 HAT。两者在瓶颈处保持正交，禁止随意 concat。

### 3.5 连通流形虫洞 (ConnectedManifoldWormhole)

**这是 v1.3 的核心架构突破，取代了早期的 ChromaticityPaletteExtractor / SpatialSemanticPalette / 离散色卡 (K-Slot) 等一系列失败方案。**

**第一性原理：** 离散色卡 (K-Slot + One-Hot) 对赛璐璐有效，但对现代插画的柔渐变 / 厚涂 / 光晕是过度简化，会产生 banding 伪影。128×128 patch 内不存在完整的"头发"/"皮肤"语义，但一定存在局部**连通色彩流形**——一段连续的渐变发丝、一块连续的阴影区。正确的路由单元是"连通域"，而非"语义类别"。

**机制 — 双边交叉注意力 (Bilateral Cross-Attention)：**

| 角色 | 来源 | 构成 |
|------|------|------|
| **Query（画笔）** | Decoder L0 (`dec_l0_pre` 输出) | `q_feat(feat_hr) + q_coord(coords)` |
| **Key（颜料库索引）** | Bottleneck $Feat_{AC}$ | `k_feat(feat_ac) + k_coord(coords)` |
| **Value（颜料库色彩）** | Bottleneck $Feat_{DC}$ | `v_proj(feat_dc)` |

经 `F.scaled_dot_product_attention`（FlashAttention 优化）路由后，Post-Attention LayerNorm 解耦幅值漂移，再以恒等注入：`out = feat_hr + γ · out_proj(color_hint)`。

**数学自适应行为：**
- **渐变区**: AC 特征平滑 → 注意力由坐标项主导 → 等效局部双线性插值 → 渐变完美保留
- **发尖奇点**: AC 特征剧变 → 语义项一票否决近邻背景 → 全局检索发根色 → "虫洞跃迁"

**关键安全设计：**
- **独立分支坐标融合（方案 B）**: 特征与坐标在 `dict_dim` 空间独立投影后加法，避免 2ch 坐标被 256ch 特征淹没
- **打破双零死锁**: `out_proj` 保持 Kaiming 默认初始化（**不**零初始化），`gamma` 初始化为 `1e-4`（非 0）。若两者同时为零，则 `gamma 梯度 ∝ out_proj(hint)=0`、`out_proj 梯度 ∝ gamma=0`，永久互锁
- **Post-Attention LayerNorm**: norm-free 网络中 `feat_dc` 幅值随训练增长会透传到 color_hint，迫使 gamma 被动压缩；LayerNorm 后 gamma 才能真实反映 CMW 贡献强度

**架构位置：** 嵌入 Decoder L0 中段（`dec_l0_pre → cmw → dec_l0_post`），配合 CreviceAuxHead 深监督，把梯度路径从 40+ 层缩短至约 20 层（参见 3.11）。

**监控接口：** `model.cmw.last_gamma_abs`、`last_color_hint_mag`、`last_cmw_eff`（真实有效注入量 = gamma × hint_mag）。

### 3.6 互信息特征提纯 (MIMFeatureFilter)

**vs 传统 U-Net 跳跃连接**：Encoder 特征携带大量高熵压缩伪影（JPEG 块效应、色带），直接 concat 会将伪影注入 Decoder。

MIMFeatureFilter 利用 InfoNCE 对比学习最大化 Encoder/Decoder 特征的互信息下界，学习动态门控过滤伪影：

- **通道门控**: GAP → FC → Sigmoid，按通道筛选有用特征
- **空间门控**: 7×7 Conv → Sigmoid，按空间位置筛选
- **Sobel 结构采样**: 仅从高频区域采样正负对，保护平坦区
- **透明启动**: bias=3.0 → sigmoid≈0.95，训练初期近乎不过滤
- **零推理开销**: InfoNCE 仅训练时计算
- **FP32 强制**: `proj_enc` 在 FP16 下对高能量特征可溢出至 inf，`F.normalize(inf)=NaN` 污染全网；整个 MIM 包入 `autocast(enabled=False)`

训练时通过 `model.mi_loss`（= `mi_loss_l0 + mi_loss_l1`）获取辅助损失（建议 λ=0.01）。**v1.4 起 mi_loss 已正式接入训练**（此前为死代码）。

### 3.7 简化跳跃融合 (SimpleSkipFusion)

**v1.7 结构简化：取代早期的 RiemannianManifoldAlignment (RMA)。**

RMA 的径-向分解 + 切空间黎曼对数/指数映射数学上优雅，但实践中暴露三个问题：
1. `r_fused` 无界增长 → `fuse_1.proj_dec` absmax 可达 7.4e5，需额外 GroupNorm 兜底
2. `log_map`/`exp_map` 中的 `torch.norm` 在 FP16 下极易溢出，逼迫全程 FP32 安全区
3. 过拟合表明容量过剩——简化反而有助泛化

**实现：** `concat[enc, dec] → Conv1×1 → GroupNorm(8)`。所有成熟 SR 架构（Real-ESRGAN / SwinIR / HAT）都用 concat+Conv 完成 skip 融合，这里额外加 GroupNorm 继承原 RMA out_norm 的幅值控制（同时是 v1.6 权重爆炸守卫的一环，参见第 5 节）。

### 3.8 HAT 解码器与瓶颈

**v1.5 起，瓶颈层 PSMamba 被 HAT 替换。** PSMamba（SSM 串行扫描）曾占 ~50% 参数，但运行在 128×128（1/4 分辨率，2px 发尖已降至 0.5px，低于 Nyquist 极限），SSM 的长距离追踪在此无从发挥，且串行计算让 GPU 并行利用率极低。HAT 窗口注意力在该分辨率提供等效全局关联，且完全可并行。

每级解码器/瓶颈堆叠 RHAG (Residual Hybrid Attention Group)，每组内含：
- **HybridAttentionBlock (HAB)**: 窗口自注意力 + 通道注意力 + MLP，交替常规/移位窗口
- **OverlappingCrossAttention (OCAB)**: 跨窗口边界注意力，消灭平涂区域的窗口网格伪影

| 层级 | 通道 | 注意力头 | 分组数 |
|------|------|---------|--------|
| Bottleneck AC | 256 | num_heads × 2 | dec_depth × bn_blocks |
| Dec L1 | 128 | num_heads × 2 | dec_depth |
| Dec L0 (pre/post) | 64 | num_heads | dec_l0_pre + dec_l0_post |

> 🚨 **`_pad_divisor=32`**：瓶颈 HAT 的 window_size=8 需在 1/4 分辨率整除（8×4=32），故输入反射填充至 32 的倍数。
> 🚨 **HAT block 的 `conv_gate` 旁路绕过 LayerNorm**，是权重爆炸的结构性盲点（参见第 5 节），瓶颈/Decoder 全程包入 FP32 安全区间。

### 3.9 拓扑引导可变形卷积 (TopologyGuidedDCN)

放置位置: **guard_norm 守卫之后、上采样器之前**，取代早期的 CornerAwareDCN。

**职责收窄：** 长距离色彩搬运已交由 CMW（全局注意力，梯度全局连通，无 grid_sample 污染），TopologyGuidedDCN 现在专注唯一职责——**局部 3~5px 几何精修**（角点曲率校准、发尖末端锐化）。

**流水线：**
1. `geom_compress`: OAM 方向先验 (`geom_prior`) → 压缩为方向码 (geom_code)
2. `corner_prior`: DW Conv → 单通道 Sigmoid 拓扑热图（边缘/角点/发尖）
3. `DensePathExtractor` (RDB 密集残差块): 联合 `[x, topo_map, geom_code]` 提取线条路径上下文
4. `local_head`: 路径特征 → 18 个坐标偏移 + 9 个注意力掩码
5. `deform_conv2d`: DCNv2 在变形网格上执行 3×3 卷积，采样点收缩包裹尖端

安全集成：零初始化 offset（第 0 步等价标准 Conv）+ 残差 LayerScale (γ=1e-2)。OAM 方向特征被 DCN 直接消费，无需重复学习方向。

**监控接口：** `model.topo_dcn.last_local_offset_mag`、`last_topo_map_mean`。

### 3.10 奇异点感知拓扑上采样 (SATUpsampler_v2)

由 `AdvancedUpsampler` 封装（`SATUpsampler_v2 → PReLU → tail Conv → RGB`）。相比 v1，**彻底根除了 PDE 对流注入**——`Δ=v·∇I` 在 C⁰ 奇点（V 字发尖）处梯度湮灭 (∇I=0)，导致方程在最该发力处自杀。

**模块一：各向异性动态滤波核（处理 C^∞ 平滑流形）**

对每个 LR 像素独立预测 $S^2$ 个空间异构的 K×K 滤波核，einsum 执行内容感知卷积后 PixelShuffle 展开：

$$\text{out}[b,c,s] = \sum_{k=1}^{K^2} K[b,c,k] \cdot \text{x\_unfold}[b,c,k,s],\quad K=\text{Softmax}_k(\text{filter\_gen}(f))$$

- `filter_gen`: DW 3×3 → GELU → 3×3 → GELU → 1×1 投影至 $S^2\cdot K^2$
- **Softmax 在 $k$ 维度归一化**：每个滤波核权重和恒为 1，绝对保留 DC 色彩，消除棋盘格漂移
- 通道共享滤波核，保证跨通道颜色一致性

**模块二：隐式多边形注入器 (ImplicitPolygonInjector)（处理 C⁰ 奇点）**

放弃连续域导数，改用离散域几何切割。数学定理：**ReLU 两层 MLP = 多超平面交集 = 分段线性多边形**，天然可表达 C⁰ 奇点。给 MLP 亚像素相对坐标 (dx, dy)，让它直接在 $S\times S$ 高分辨率网格内"画"出锐利的多边形夹角，无需依赖梯度。

- **门控**: bias=-3.0 → sigmoid≈0.047，初始近乎静默，只在奇点/高频拓扑处激活
- **MLP 末层零初始化**: 训练初期输出为 0，基础滤波完整透传
- 输出 `out = out_base + polygon_residual`

**监控接口：** `model.upsampler.up[0].polygon_injector.last_gate_mean`、`last_residual_mag`。

### 3.11 拓扑辅助监督头 (CreviceAuxHead)

**设计目的：** CMW 的 gamma 在 v1.3 曾长期 ≈ 0、500ep 不激活。根因是拓扑 loss (LRT, Crevice) 的梯度经 40+ 层反传后被 L1/DC 的均匀梯度淹没，CMW 从未收到有效驱动信号。

**解决方案：** 在 CMW 输出处直接施加辅助监督 (deep supervision)，拓扑 loss 的梯度只需穿过 1 层 Conv 即可到达 CMW。

**极简结构：** `Conv1×1 → PixelShuffle → Sigmoid` 直接输出 HR 域 3ch RGB（`model.aux_sr`）。仅训练时计算，推理零开销。训练循环对 `aux_sr` 施加 L1 + LRT 监督即可。

> 🚨 **删除的模块**：相比早期版本，v1.4~v1.7 已移除 `AnimeCommitteeRefiner`（功能同质化，梯度死重）、`PaletteModulation`、`BaseAnchoredDetailInjector` (BADI，退化 SR 下浅层注入有害)。这些类已从代码中删除，请勿在新代码中引用。

---

## 4. 训练动力学

隐空间基底训练前 10 个 Epoch 属于混沌建构期。以下规则不可违反：

**MIM 辅助损失**：训练时 `model.mi_loss` 返回 InfoNCE 损失（v1.4 起已接入 `g_total`），建议 λ=0.01。推理时自动为零。

**CMW 深监督**：训练时对 `model.aux_sr`（CreviceAuxHead 输出）施加 L1 + LRT 拓扑监督，绕过 40+ 层梯度衰减直达 CMW。

**3 阶段渐进损失 (CaelumLossV2)**：

| 阶段 | 进度区间 | 启用损失 |
|------|---------|---------|
| Phase 1 | 0% ~ 15% | L1(退火), DC(退火), OKLCH, STGV, SGH, Crevice, LRT |
| Phase 1.5 | 15% ~ 30%（余弦渐入）| + ChromaGrad, Gibbs, Angular, TurningPoint |
| Phase 2 | 30% ~ 100% | + Perceptual, Histogram(默认 0) |

**损失权重表 (CaelumLossV2.DEFAULT_WEIGHTS)**：

| 键 | 类 | 监管内容 | 默认权重 |
|-----|-----|---------|--------|
| `l1` | 像素 | L1 像素重建（前 20% 全量后余弦退火至 30%）| 1.0 |
| `dc` | DC 锁定 | Scharr 能量软权重 + Charbonnier（退火至 40%）| 1.0 |
| `oklch` | 感知色彩 | OKLab 圆柱坐标色差 | 5.0 |
| `stgv` | 全变分 | 严格平坦-TV + 二阶平滑（H7 强化）| 3.0 |
| `smooth_grad_hessian` | 平滑度 | Hessian 曲度约束 | 1.5 |
| `crevice` | 拓扑 | 夹缝与 ZNCC 高频（Phase 1 提前启用）| 6.0 |
| `lrt` | 拓扑 | 拉普拉斯共振拓扑 + Charbonnier（Phase 1）| 0.2 |
| `chroma_grad` | 色彩梯度 | 色度梯度对齐 | 1.5 |
| `gibbs` | Gibbs | SWT 振铃级差惩罚 | 4.0 |
| `angular` | 角度 | 梯度方向平滑度 | 5.0 |
| `turning_point` | 转折点 | 宏观曲率（张量角点响应）| 1.0 |
| `histogram` | 色彩分布 | 差分直方图匹配（默认关闭）| 0.0 |
| `perceptual` | 感知 | VGG 层级 + 频谱感知（Phase 2）| 0.5 |

**可选辅助损失：**
- `CommitteeOrthogonalityLoss` / `GateTolerancePenalty` — 随 AnimeCommitteeRefiner / BADI 删除而停用，模块已不在主网络中（保留在 `losses.py` 仅供历史参考）
- `DecoupledGANLoss` — 可选 GAN 分支（`lambda_gan`），默认不启用，初期收敛禁用（参见 Law 5）

**优化器配置**：

| 配置项 | 值 | 原因 |
|--------|---|------|
| 优化器 | AdamW | — |
| weight_decay | 0（建议 1e-4）| config 当前为 0；v1.6 诊断确认无 Norm 网络下 wd=0 会致权重单调膨胀，建议开启 1e-4（参见第 5 节）|
| clip_grad_norm | 1.0 | 驯服初期多分支梯度洪流 |
| AMP | FP16 + GradScaler | 高危路径已包入 FP32 安全区间（参见第 5 节）|

**收敛排查优先级**：

1. 检查 MIM 辅助损失 (`mi_loss`) 是否从 Epoch 0 生效
2. 检查 CMW 深监督 (`aux_sr`) 是否接入总损失
3. 检查是否有误引入的 BN / IN 层（主干禁止，幅值累积节点用 GroupNorm）
4. 检查 RepSR 是否意外引入归一化
5. 监控 absmax：> 1e4 警觉，> 1e6 必须干预（权重爆炸早期信号）
6. 不要用 GAN 损失"提速"初期收敛

---

## 5. 数值稳定性：FP16 NaN 根治 (v1.6)

v1.6 之前，模型在 epoch ~490 起间歇性全网 NaN。根因诊断（详见 `docs/EpistemicTrace/FP16_NaN_Debug_Report.md`）确认：**这不是单纯的 FP16 精度问题，而是权重爆炸 (weight explosion)。**

**根因链：**
1. `weight_decay=0` + Adam → 权重范数无约束、单调增大
2. Decoder 路径大量 Conv 无 Norm → 输出幅值 = `Σ w·x` 随权重正反馈放大
3. FP16 上限 65504 被率先击穿（FP32 下 absmax 实测达 1e10，只是炸得更晚）

**结构性修复（三道防线）：**

| 防线 | 措施 | 代码位置 |
|------|------|----------|
| ① 幅值守卫 | RMA/SimpleSkipFusion 出口 + topo_dcn 前插 `GroupNorm(8, affine=True)` | `fuse_0/1.out_norm`, `guard_norm_pre_topo` |
| ② 治本约束 | 开启 `weight_decay=1e-4`（config 仍需手动设置）| 训练配置 |
| ③ FP32 安全区间 | 高危路径包入 `autocast(enabled=False)` + `.clamp(±65504)` | `forward()` 区间 0/A/B/C/D |

**FP32 安全区间覆盖：**
- **区间 0**: `freq_router → bottleneck_ac → bn_conv`（瓶颈 HAT 反向梯度承接下游 absmax≈2e5）
- **区间 A**: `mim_l1 → fuse_1 → dec_l1 → dec_l1_conv`（fuse_1.proj_dec absmax≈2.3e5）
- **区间 B**: `mim_l0 → fuse_0`（fuse_0 mag_gate absmax≈3.0e5）
- **区间 D**: `dec_l0_pre → cmw → aux_head → dec_l0_post`（dec_l0_pre.conv_gate 首炸点）
- **区间 C**: `dec_l0_conv + 残差`（残差累加风险）

**修复效果：** `badi`/`topo_dcn` absmax 从 ~1e10 降至 ~30（降 9 个数量级），FP16 NaN count 从全炸归零。

**血泪教训（务必牢记）：**
- **FP32 无 NaN ≠ 安全**：absmax 是更敏感的早期预警，> 1e4 警觉，> 1e6 必须干预
- **forward-only 测试不够**：backward 梯度溢出更隐蔽，`isfinite(loss)` 安全阀拦不住（前向值正常、梯度才是杀手）。必须用完整 backward 诊断 + `instant_state_at_first_nan.pth` 保存瞬时权重
- **`weight_decay=0` 在无 Norm 网络中是定时炸弹**
- **HAT block 的 `conv_gate` 旁路绕过 LayerNorm**，是幅值爆炸的结构性传导通道
- **FP16 epsilon 安全阈值统一 ≥ 1e-4**：任何 `sqrt(eps²)` 必须保证结果 ≥ 1e-4，否则 FTZ 归零后 `sqrt(0)` 的梯度为 ∞

---

## 6. 开发法则 (AI Handoff Rules)

> **To the Future AI Assistant:**
> 接手分析或扩展此代码时必须强制遵守以下法则。

### Law 1 — NO BATCHNORM IN SR

禁止在主干网络引入 `BatchNorm`、`InstanceNorm` 或任何抹杀均值/方差的归一化。色彩的绝对准确性高于一切。
**例外**：幅值累积节点（RMA/SkipFusion 出口、Decoder 守卫）必须用 `GroupNorm`（对 batch size 不敏感、affine 保留学习能力）切断权重爆炸正反馈（参见第 5 节）。

### Law 2 — SCALE INVARIANCE IS LAW

禁止使用硬编码空间维度除法：

```python
# ❌ 禁止
x = x / (H * W)
x = x.mean(dim=(2, 3))

# ✅ 使用加权归一化
x = weighted_sum / (weights.sum() + eps)
```

保证 32×32 和 4096×4096 输入产生严格相等的数学响应。

### Law 3 — RESPECT THE FREQUENCY DECOUPLING

- $Feat_{DC}$ 仅作 CMW 的 Value（色彩源）
- $Feat_{AC}$ 仅交由瓶颈 HAT 处理，并作 CMW 的 Key（语义索引）
- 禁止随意 concat 未解耦的低维特征

### Law 4 — DE-GATED HIGH-FREQUENCY PATH

HAT 或任何 Transformer 变体严禁在信号主干输出路径使用乘性门控 `y = y * silu(z)` 或破坏相位叠加的激活。高频特征必须保留加性残差直路：`out = block(x) + x`。

### Law 5 — CHECK GRADIENT FLOW BEFORE OPTIMIZING

遇到损失下降缓慢时排查优先级：

1. MIM 辅助损失是否生效
2. 是否有新引入的归一化层
3. 不要建议引入 GAN 损失来提速初期收敛

### Law 6 — HIGH-FREQUENCY GEOMETRY MUST BYPASS DOWNSAMPLING

高频几何模块禁止串联进 Encoder 主干。必须使用平行旁路绕过所有下采样，在 Decoder 末端同分辨率融合：

```python
# ❌ 禁止
x_shallow = oam(x_shallow)    # 串联 → 高频被下采样污染
skip_l0 = enc_l0(x_shallow)

# ✅ 平行旁路
geom_prior = parallel_oam(x_shallow)  # 独立分支
skip_l0 = enc_l0(x_shallow)           # 主干不受干扰
# Decoder L0 完成后同分辨率, 由 TopologyGuidedDCN 直接消费方向先验
feat = topo_dcn(feat, geom_prior)     # OAM 方向码注入 DCN, 局部几何精修
```

---

## 附录：文件映射

| 文件 | 内容 |
|------|------|
| `PPBUNet_v1_x4.py` | 主网络 `PPBUNet` 及所有模块（CMW / HAT 瓶颈 / SimpleSkipFusion / TopologyGuidedDCN / SATUpsampler_v2 / CreviceAuxHead）|
| `hat.py` | `ResidualHybridAttentionGroup`: 窗口自注意力 + OCAB（瓶颈与 Decoder 共用）|
| `ps_mamba.py` | `PSMambaBlock`: 去门控 SSM（v1.5 起已不再被主网络引用，保留作历史参考）|
| `losses.py` | `CaelumLossV2` 损失函数库（13 子损失 + 3 阶段渐进 + 可选 GAN/正交/门控正则）|
| `dataset.py` | 数据管线（多种退化模式 + 混合采样）|
| `train.py` | 训练循环（含 mi_loss / aux_sr 接入、FP32 安全区间监控）|
| `inference.py` | 推理入口（调用 `switch_to_deploy()` 折叠 RepSR）|
| `diagnose_nan.py` / `replay_nan_batch.py` | FP16 NaN 诊断工具（Phase D 完整 backward + 瞬时权重复现）|

**配置入口**: `config['architecture'] = 'ppbunet_v1'`
