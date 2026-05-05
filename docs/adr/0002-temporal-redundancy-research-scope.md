Owner: ely
Status: accepted
Last reviewed: 2026-05-05

# ADR 0002: 围绕上下文插入位置定义科研问题

## 背景

仓库已经有一个可用的 speech codec baseline。下一阶段不再是证明 neural codec 能训练，也不再是完成课程项目式的 MP3 对比。

当前最重要的是把研究方向定义清楚，避免后续实现细节反过来扭曲 idea。

## 决策

当前科研 idea 定义为：

> 研究时间上下文建模应该在哪个 neural speech codec 表示层级进入。

表示链是：

```text
waveform -> downsampled latent -> RVQ embedding/codes -> code prior
```

项目必须显式比较两个假设：

- **早期上下文假设**：上下文应该靠近 waveform 进入，因为时间冗余在量化前最明显、信息也最完整。
- **后期上下文假设**：上下文在 downsampled latent 或 quantized codes 上更有用，因为序列更短、更结构化，也更接近 bitrate 或 entropy 收益。

Mamba 是候选上下文模型，不是整个科研 idea。

## 影响

实现计划、插入点和模型变体都是次级内容。它们必须服务于上面的假设对比。

当前 idea 的 source of truth 是：

- [研究想法](/Users/ely/workspace/research/audio/AudioCodec/docs/idea.md)

更早的细节草稿和过度具体的实验协议已经归档。它们可以在后续实现时参考，但不再约束当前 idea。
