Owner: ely
Status: active
Last reviewed: 2026-05-05

# 当前 Codec Baseline

## 作用

本文档只记录当前科研实验使用的稳定 codec 基底。

它不是课程项目复盘，也不定义新的研究假设。真正的研究 idea 见：

- [研究想法](/Users/ely/workspace/research/audio/AudioCodec/docs/idea.md)

## 数据流

```text
waveform [B, 1, samples]
  -> SEANet encoder
  -> downsampled latent [B, 128, T]
  -> EMA residual vector quantizer
  -> RVQ codes [B, K, T]
  -> quantized embedding [B, 128, T]
  -> SEANet decoder
  -> reconstruction [B, 1, samples]
```

默认 speech 设置：

- sample rate: `16 kHz`
- channels: mono
- SEANet ratios: `[8, 5, 4, 2]`
- total stride: `320`
- frame rate: `50 Hz`
- latent dim: `128`
- codebook size: `1024`
- 每个 RVQ stage 每帧：`10 bits`
- nominal bitrate: `50 * K * 10 bit/s`

## 稳定配置

当前 baseline anchor：

- `configs/ablation-adversarial-msstft-balanced.json`: nominal `12 kbps`
- `configs/ablation-adversarial-msstft-balanced-8kbps.json`: nominal `8 kbps`
- `configs/ablation-adversarial-msstft-balanced-4kbps.json`: nominal `4 kbps`
- `configs/ablation-adversarial-msstft-balanced-2kbps.json`: nominal `2 kbps`

训练 recipe：

- `MS-STFT discriminator`
- feature matching loss
- balancer
- EMA RVQ with k-means init and dead-code replacement

## 为什么这个 baseline 重要

当前研究问题不是 neural codec 能否和 MP3 对比。那个阶段已经归档。

这个 baseline 重要，是因为它提供了一条稳定表示链：

```text
waveform -> latent -> RVQ embedding -> RVQ codes
```

这条链可以用来验证：

- 时间冗余在哪些层级可测；
- 上下文建模在哪些层级能转成保真率或压缩收益；
- Mamba 是否只是另一个上下文模型，还是在长序列/流式场景中有独立价值。

## 第一阶段应保持不变的内容

在 idea 尚未稳定前，不应该同时修改太多变量。第一阶段默认保持：

- 数据切分；
- sample rate 和 channels；
- SEANet ratios 和 frame rate；
- latent dim；
- codebook size；
- 训练 loss recipe；
- RVQ payload accounting。

下面这些不是普通的上下文模块变化，不能和核心 idea 混在一起：

- 降低 frame rate；
- 改 codebook size；
- 引入 mel/STFT/subband front-end；
- 加 semantic distillation；
- 给 bitstream 增加 side-channel information。

## 归档背景

课程阶段架构、报告和 benchmark 资料已经归档到：

- `docs/archive/course-project/`
