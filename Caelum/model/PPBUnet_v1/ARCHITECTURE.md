# PPBUNet v1.0 — Architecture & Design

> PPBUNet: **P**alette-**P**ainter-**B**rush **U-Net** for Anime Super-Resolution

本文档阐述 PPBUNet v1.0 的架构设计。实现文件为 [`PPBUNet_v1_x4.py`](PPBUNet_v1_x4.py)，依赖模块：[`ps_mamba.py`](ps_mamba.py)、[`hat.py`](hat.py)。

三阶段工作流以画家创作过程命名：
- **Palette** — 从低频特征提取全局配色方案
- **Painter** — 通过注意力驱动的 U-Net 解码重建结构
- **Brush**   — 在全分辨率下精修几何细节并上采样到 HR

---

## 目录

1. [信号模型：动漫插画的双态分布](#1-信号模型动漫插画的双态分布)
2. [宏观数据流](#2-宏观数据流)
3. [核心模块](#3-核心模块)
   - 3.1 [隐空间基底 (Latent Base)](#31-隐空间基底-latent-base)
   - 3.2 [结构重参数化块 (RepSRBlock)](#32-结构重参数化块-repsrblock)
   - 3.3 [平行几何方向流 (ParallelOAM)](#33-平行几何方向流-paralleloam)
   - 3.4 [显式频率路由 (FrequencyRouter)](#34-显式频率路由-frequencyrouter)
   - 3.5 [色彩原型提取 (ChromaticityPaletteExtractor)](#35-色彩原型提取-chromaticitypaletteextractor)
   - 3.6 [互信息特征提纯 (MIMFeatureFilter)](#36-互信息特征提纯-mimfeaturefilter)
   - 3.7 [黎曼流形对齐融合 (RiemannianManifoldAlignment)](#37-黎曼流形对齐融合-riemanniannanifoldalignment)
   - 3.8 [HAT 解码器](#38-hat-解码器)
   - 3.9 [角点感知可变形卷积 (CornerAwareDCN)](#39-角点感知可变形卷积-cornerawaredcn)
   - 3.10 [色彩调制 (PaletteModulation)](#310-色彩调制-palettemodulation)
   - 3.11 [各向异性流形感知上采样 (AMADSUpsampler)](#311-各向异性流形感知上采样-amadsupsampler)
4. [训练动力学](#4-训练动力学)
5. [开发法则 (AI Handoff Rules)](#5-开发法则-ai-handoff-rules)

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
[Input RGB] ─► RepSR ×2 ─►│                                                        │
               (shallow)  │                                                        │
                          │──┬── Enc L0 (RCAB ×2, 64ch) ──► ↓2×                    │
                          │  │       │skip_l0                                      │
                          │  │  Enc L1 (RCAB ×2, 128ch) ──► ↓2×                    │
                          │  │       │skip_l1                                      │
                          │  │  ┌────┴─── Bottleneck (256ch) ───────┐              │
                          │  │  │                                   │              │
                          │  │  │ FrequencyRouter (DW-LPF k=5)      │              │
                          │  │  │    │               │              │              │
                          │  │  │  Feat_DC         Feat_AC          │  ★ Palette   │
                          │  │  │    │               │              │              │
                          │  │  │  Palette        PS-Mamba ×N       │              │
                          │  │  │  Extractor         │              │              │
                          │  │  │    │               │              │              │
                          │  │  │  palette (B,256)   │              │              │
                          │  │  │    │       bn_conv(feat_ac)       │              │
                          │  │  │    │           + bn_in            │              │
                          │  │  │    │               │              │              │
                          │  │  └────│───────────────┘              │              │
                          │  │       │          │                   │              │
                          │  │  MIM(skip_l1, feat)                  │              │
                          │  │  RMA(skip_l1, feat)                  │              │
                          │  │       │                              │  ★ Painter   │
                          │  │  HAT Decoder L1 (128ch, heads×2)     │              │
                          │  │       │                              │              │
                          │  │  MIM(skip_l0, feat)                  │              │
                          │  │  RMA(skip_l0, feat)                  │              │
                          │  │       │                              │              │
                          │  │  HAT Decoder L0 (64ch)               │              │
                          │  │       │                              │              │
                          │  │       │                              │              │
  (ParallelOAM)           │  │       ▼                              │              │
                          │  └► GeometryFusion (γ=1e-2) ◄───────────│── geom_prior │
                          │     (cat→1×1→3×3→ Adding residuals)     │              │
                          │        │                                │              │
                          │     CornerAwareDCN (γ=1e-2)             │  ★ Brush     │
                          │        │                                │              │
                          │     PaletteModulation ◄── palette       │              │
                          │        │                                │              │
                          │     rendered_feat + x_shallow ◄────────────────────────┘
                          │        │                                
                          │     AMADSUpsampler (4×)                 
                          │        │                                
                          └───► [Output RGB]
```

**两条独立数据通道在 Decoder L0 后交汇：**

| 通道 | 路径 | 物理含义 |
|------|------|----------|
| **低频语义主干** | Shallow → Enc → Bottleneck → Dec | 全局语义结构、色彩语境、跨区域拓扑 |
| **高频几何旁路** | Shallow → ParallelOAM (H×W) | 0°/90°/45°/135° 绝对方向线条锐度 |

**关键设计决策：**

| 决策 | 代码位置 | 原因 |
|------|----------|------|
| 隐空间融合 | `rendered_feat + x_shallow` | 网络只需补充高频纹理，不生成反向噪声 |
| ParallelOAM 平行旁路 | 与 `enc_l0` 并行 | 高频方向信息绕过下采样，避免 Shannon-Nyquist 混叠 |
| 物理掩码对角核 | `mask_d45 = eye(7)` | Z² 网格上正交基底无法表示斜线，掩码核沿对角线连续 |
| MIM + RMA 替代 concat | `mim_l0/l1` + `fuse_0/1` | 互信息提纯过滤伪影 + 超球面流形对齐无损融合 |
| 因果链: Geom→DCN→Palette→Up | `forward()` 末段 | 方向先验 → 曲率精修 → 色彩渲染 → 上采样 |
| DC 仅提色彩，AC 仅走 Mamba | `freq_router` 双分支 | 频率解耦保证信息正交性 |

---

## 3. 核心模块

### 3.1 隐空间基底 (Latent Base)

| | 传统做法 | PPBUNet |
|---|---------|---------|
| **公式** | `out = network(x) + bicubic(x)` | `out = upsampler(feat_unet + x_shallow)` |
| **问题** | 网络被迫生成"反向噪声"抵消 bicubic 振铃，导致懒惰收敛 | 纯净高维流形空间中只需补充缺失的高频纹理 |

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

HAT 和 Mamba 从此不会被平滑色块的无用信息分散算力，专心追踪高频拓扑。

### 3.5 色彩原型提取 (ChromaticityPaletteExtractor)

从纯净 $Feat_{DC}$ 中用余弦相似度软聚类提取 16 个全局色彩原型，汇聚为 `(B, C)` 色彩向量。

> 🚨 **禁止用 `/ (H*W)` 计算均值！**
> 训练 Patch（64×64）与推理原图（2000×2000）空间尺寸差百倍。均值池化导致 Palette 向量数值偏移，推理时大面积平涂出现 DC 偏移。
> 必须除以 attention 分布权重总和，确保尺度不变性。

### 3.6 互信息特征提纯 (MIMFeatureFilter)

**vs 传统 U-Net 跳跃连接**：Encoder 特征携带大量高熵压缩伪影（JPEG 块效应、色带），直接 concat 会将伪影注入 Decoder。

MIMFeatureFilter 利用 InfoNCE 对比学习最大化 Encoder/Decoder 特征的互信息下界，学习动态门控过滤伪影：

- **通道门控**: GAP → FC → Sigmoid，按通道筛选有用特征
- **空间门控**: 7×7 Conv → Sigmoid，按空间位置筛选
- **Sobel 结构采样**: 仅从高频区域采样正负对，保护平坦区
- **透明启动**: bias=3.0 → sigmoid≈0.95，训练初期近乎不过滤
- **零推理开销**: InfoNCE 仅训练时计算

训练时通过 `model.mi_loss` 获取辅助损失（建议 λ=0.01）。

### 3.7 黎曼流形对齐融合 (RiemannianManifoldAlignment)

**vs 传统 `cat → Conv → ReLU`**：直接拼接 Encoder（高频线稿）和 Decoder（低频色彩）特征会产生流形坍缩，高频信号被低频淹没。

RMA 将特征投影至超球面 $S^{n-1}$，通过黎曼对数/指数映射在局部切空间完成无损对齐：

1. **径-向分解**: 方向 $d = f/\|f\|$ 用于语义对齐，幅值 $r = \|f\|$ 保留绝对色彩强度
2. **切空间融合**: `log_map(d_dec, d_enc)` → MLP → `exp_map(d_dec, v_fused)` 沿测地线映射回球面
3. **幅值门控重组**: Sigmoid 门控线性插值 $r_{enc}$ 与 $r_{dec}$

方向与幅值解耦处理，避免归一化摧毁色彩信息。

### 3.8 HAT 解码器

每级解码器堆叠 RHAG (Residual Hybrid Attention Group)，每组内含：

- **HybridAttentionBlock (HAB)**: 窗口自注意力 + 通道注意力 + MLP，交替常规/移位窗口
- **OverlappingCrossAttention (OCAB)**: 跨窗口边界注意力，消灭平涂区域的窗口网格伪影

| 层级 | 通道 | 注意力头 | 窗口大小 |
|------|------|---------|----------|
| Dec L1 | 128 | num_heads × 2 | window_size |
| Dec L0 | 64 | num_heads | window_size |

### 3.9 角点感知可变形卷积 (CornerAwareDCN)

放置位置: **GeometryFusion 之后、PaletteModulation 之前**。

动漫插画中最具辨识度的拓扑奇异点（发尖、褶皱交界、关节硬轮廓）对应二阶曲率极值。传统 3×3 刚性网格无法适应各向异性"V"型结构，L1 在此回归均值导致末端钝化。

DCNv2 打破刚性网格：
1. 轻量 DW Conv → 单通道 Sigmoid 角点热力图
2. 特征 + 角点先验 → 18 个坐标偏移 + 9 个注意力掩码
3. DCNv2 在变形网格上执行 3×3 卷积，采样点收缩包裹尖端

安全集成：零初始化 offset（第 0 步等价标准 Conv）+ 残差 LayerScale (γ=1e-2)。

> 🚨 **DCN 禁止放在 PaletteModulation 之后！**
> 色彩调制注入全局色度偏移后，角点先验会将色彩边界误判为几何角点。DCN 必须工作在纯结构特征上。

### 3.10 色彩调制 (PaletteModulation)

全局色彩向量经 `Linear → LeakyReLU` 投影后广播至空间维度，与特征 cat 后经 1×1 + 3×3 卷积，以 LayerScale (γ=1e-4) 加性融合。

不做任何归一化（无 AdaIN / InstanceNorm），完整保留绝对色彩与亮度信息。

### 3.11 各向异性流形感知上采样 (AMADSUpsampler)

| 特性 | AMADSUpsampler | 旧版 bilinear + 7×7 DWConv |
|------|---------------|---------------------------|
| 核心机制 | 全卷积几何 → 坐标偏移 → grid_sample | 固定插值核 |
| 振铃 | ✅ 无大核 sinc 旁瓣 | ❌ 7×7 DWConv Gibbs 振铃 |
| 空间连续性 | ✅ grid_sample 天生连续 | △ 依赖权重"碰巧"学到 |
| 各向异性 | ✅ 边缘门控 α(x) 调制偏移 | ❌ 各向同性 |
| 平坦区 | ✅ α≈0 → 纯双线性 | △ 需约束 |

关键设计：
- **DySample 机制**: 从"预测滤波核"升维至"预测流形坐标偏移"
- **边缘门控**: α(x) ∈ [0,1]，平坦区纯双线性，边缘释放各向异性偏移
- **相位零和残差**: Tanh 零初始化补偿插值高频衰减，max_neg_lobe 限幅防过冲
- **零初始化安全**: 偏移初始为零 → 第 0 步等价标准双线性上采样

---

## 4. 训练动力学

隐空间基底训练前 10 个 Epoch 属于混沌建构期。以下规则不可违反：

**MIM 辅助损失**：训练时 `model.mi_loss` 返回 InfoNCE 损失，建议加入总损失 (λ=0.01)。
推理时自动为零。

**优化器配置**：

| 配置项 | 值 | 原因 |
|--------|---|------|
| 优化器 | AdamW | — |
| weight_decay | 0 | 权重收缩退化对高频线条的捕捉能力 |
| clip_grad_norm | 1.0 | 驯服初期多分支梯度洪流 |

**收敛排查优先级**：

1. 检查 MIM 辅助损失是否从 Epoch 0 生效
2. 检查是否有误引入的 BN / IN 层
3. 检查 RepSR 是否意外引入归一化
4. 不要用 GAN 损失"提速"初期收敛

---

## 5. 开发法则 (AI Handoff Rules)

> **To the Future AI Assistant:**
> 接手分析或扩展此代码时必须强制遵守以下法则。

### Law 1 — NO BATCHNORM IN SR

禁止在主干网络引入 `BatchNorm`、`InstanceNorm` 或任何抹杀均值/方差的归一化。色彩的绝对准确性高于一切。

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

- $Feat_{DC}$ 仅用于计算 Palette
- $Feat_{AC}$ 仅交由 PS-Mamba / HAT 处理
- 禁止随意 concat 未解耦的低维特征

### Law 4 — DE-GATED HIGH-FREQUENCY PATH

PS-Mamba 或任何 Transformer 变体严禁在信号主干输出路径使用乘性门控 `y = y * silu(z)` 或破坏相位叠加的激活。高频特征必须保留加性残差直路：`out = mamba(x) + x`。

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
# Decoder L0 完成后同分辨率融合
feat = feat + gamma * geom_fusion(cat([feat, geom_prior]))
```

---

## 附录：文件映射

| 文件 | 内容 |
|------|------|
| `PPBUNet_v1_x4.py` | 主网络 `PPBUNet` 及所有模块 |
| `ps_mamba.py` | `PSMambaBlock`: 去门控 SSM, O(N) 全局高频拓扑追踪 |
| `hat.py` | `ResidualHybridAttentionGroup`: 窗口自注意力 + OCAB |
| `losses.py` | CaelumLossV2 损失函数库 (11 子损失) |
| `dataset.py` | 数据管线 (5 种退化模式 + 混合采样) |
| `train.py` | 训练循环 |
| `inference.py` | 推理入口 (调用 `switch_to_deploy()` 折叠 RepSR) |

**配置入口**: `config['architecture'] = 'ppbunet_v1'`
