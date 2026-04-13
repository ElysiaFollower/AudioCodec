Owner: ely
Status: draft
Last reviewed: 2026-04-13

# 基线架构说明

## 1. 设计原则

本次课程项目的架构选择以“快、稳、可解释”为最高优先级：

- 快：1 到 2 天内应能训练出可展示结果。
- 稳：尽量避免复杂对抗训练、扩散解码器、大规模先验模型。
- 可解释：模型结构应便于和传统 codec 做概念对照。

## 2. 推荐技术栈

- `Python 3.10+`
- `PyTorch`
- `torchaudio`
- `ffmpeg` 或系统自带 `ffmpeg` CLI
- `matplotlib` 用于频谱图和结果可视化

可选依赖：

- `vector-quantize-pytorch`
  作用：减少自己实现 `RVQ` 时的 bug 风险。
- `librosa`
  作用：补充音频特征与可视化工具。
- `pesq` / `pystoi`
  作用：补充客观质量指标；若安装不稳，可降级为 `SI-SNR + log-mel loss + 频谱图 + 听感样例`。

## 3. 最小模型结构

推荐从下列结构开始：

```text
waveform
  -> encoder (1D strided conv stack)
  -> latent frames
  -> RVQ bottleneck
  -> quantized latent
  -> decoder (1D transposed conv stack)
  -> reconstructed waveform
```

### 冻结起步配置

为避免实现阶段反复改结构，建议先冻结为下面这版：

- sample rate: `16 kHz`
- channels: `mono`
- encoder strides: `[5, 4, 4, 2]`
- total stride: `160`
- latent frame rate: `16000 / 160 = 100 Hz`
- latent dim: `128`
- codebook size: `256`
- RVQ stages: `8`

这意味着在不额外做熵编码时，离散 token 的名义码率约为：

`100 frame/s * 8 stages * log2(256) = 6400 bit/s = 6.4 kbps`

这个量级已经足够拿来和低码率 `MP3` 做课程项目对比。

### Encoder

- 输入：`16 kHz` 单声道波形
- 主体：若干层 `1D Conv + stride`
- 目标：把原始波形压成较低时间分辨率的 latent sequence

### Quantizer

- 使用 `Residual Vector Quantization`
- 推荐起步配置：
  - codebook size: `256`
  - codebook dim: `128`
  - RVQ stages: `8`

### Decoder

- 与 Encoder 基本对称
- 使用反卷积或上采样卷积恢复波形

## 4. 损失函数建议

最稳妥的组合是：

- 波形重建损失：`L1` 或 `L2`
- 频域损失：`multi-scale STFT loss`
- 量化相关损失：`commitment loss`

原因很简单：

- 只用 waveform loss，常常导致听感发闷或高频细节恢复差。
- 加上频域损失后，更容易得到可展示的频谱结果。

## 5. 为什么不把“严格 VAE”放到关键路径

如果课程项目强行坚持标准 `VAE` 形式，通常还要处理：

- KL 散度权重的平衡
- 随机采样带来的训练不稳定
- 与离散量化模块的耦合

这会显著增加试错成本，但并不直接提高课程展示价值。

因此建议将本次模型表述为：

`基于 autoencoder / VAE family 的 neural codec baseline，以 RVQ 作为离散瓶颈`

这样既不偏离原始选题方向，也能把工程难度控制在课程项目可承受范围内。

## 6. 数据与训练建议

### 数据选择

默认数据集固定为：`LibriSpeech dev-clean`

原因：

- 公开、常用、语音干净
- 原始数据就是 `16 kHz` 单声道 speech
- `torchaudio` 可直接下载和读取，实现成本最低
- 下载体量适中，适合课程项目

### 默认切分

为保证可复现，建议采用确定性切分：

1. 按文件路径或 utterance id 排序。
2. 依次累积时长，取前 `60 分钟` 作为训练集。
3. 接下来的 `10 分钟` 作为验证集。
4. 再接下来的 `10 分钟` 作为测试集。

如果时间非常紧，可以把训练集进一步缩到 `30 分钟`，但不建议比这更小。

### 训练样本形式

- 训练时从音频中随机裁 `2.0 s` 片段。
- 验证与测试时使用固定裁剪或整句推理。
- 若显存紧张，可先降到 `1.0 s` 片段做 smoke test。

### 默认训练预算

建议把训练分成两个预算层级：

#### Budget A：最小可交付

- overfit smoke test：`10` 条音频，`500 steps`
- main train：`5000 steps`
- batch size：`16 x 2.0 s` 片段（24 GB 显存）
- 若显存较小：改为 `8 x 2.0 s` 或 `16 x 1.0 s`

这个预算的目标不是追求最佳质量，而是尽快得到一版可听、可展示、可对比的结果。

#### Budget B：结果更稳一些

- 在 Budget A 跑通后，继续训练到 `10000 steps`
- 只在验证损失仍下降时继续
- 中间每 `500` 或 `1000` steps 导出重建样例，人工听感检查一次

### 默认优化设置

- optimizer: `AdamW`
- learning rate: `2e-4`
- weight decay: `1e-4`
- mixed precision: `开启`
- checkpoint 选择：以验证集 `L1 + STFT` 为主，不要只看训练损失

### 训练顺序

建议分三步：

1. 用 `5 到 10` 条音频做过拟合测试，确认模型与数据流正确。
2. 扩到小规模训练集，跑出第一版可听结果。
3. 再做码率、loss 或 RVQ stage 的小消融。

## 7. 评测设计

### 对比对象

- 你的 neural codec
- `MP3` baseline

### 评测维度

- 压缩率或等效码率
- `SI-SNR`
- 可选：`PESQ`、`STOI`
- 频谱图对比
- 听感样例

### 公平性建议

优先保证“压缩预算相近”而不是“配置形式相同”。

也就是说，你可以按接近的输出文件大小或等效码率来比较 neural codec 和 `MP3`，而不是执着于完全一致的内部参数。

对当前默认配置，一个合理的 `MP3` 对比起点是：

- `MP3 8 kbps`
- `MP3 16 kbps`

前者更接近当前 neural codec 的名义码率，后者可作为一个更强但仍低码率的参考。

## 8. Stretch goal

如果 baseline 跑通且仍有时间，只做一个扩展：

- 增加 `RVQ stages`
- 切换损失组合
- 引入轻量时序模块

不要同时开多个扩展方向，否则很容易在课程周期内失控。
