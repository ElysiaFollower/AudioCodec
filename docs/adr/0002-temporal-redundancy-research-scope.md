Owner: ely
Status: accepted
Last reviewed: 2026-05-05

# ADR 0002: 围绕长程时间冗余定义科研问题

## 背景

仓库已经有一个可用的 speech codec baseline。下一阶段不再是证明 neural codec 能训练，也不再是完成课程项目式的 MP3 对比。

当前最重要的是把研究方向定义清楚，避免后续实现细节反过来扭曲 idea。

## 决策

当前科研 idea 定义为：

> 研究 neural speech codec 在已经具备局部时序建模的前提下，是否仍存在可利用的长程时间冗余；如果存在，这种长程冗余应该在哪个表示层级利用，才能转化为真实收益。

表示链是：

```text
waveform -> downsampled latent -> RVQ embedding/codes -> code prior
```

项目必须显式比较三个问题：

- **存在性问题**：局部卷积 / TCN / baseline LSTM 已经利用短程上下文后，medium / long / full utterance 级别的额外冗余是否仍然可测。
- **层级归因问题**：如果长程冗余存在，它在 early feature、downsampled latent、post-RVQ embedding、RVQ codes / code prior 哪一层最容易转化为收益。
- **收益归因问题**：这种收益到底是 reconstruction fidelity、rate-distortion、entropy bitrate、token efficiency，还是 long-context / streaming efficiency。

Local temporal modeling 是 baseline，不是贡献。Mamba 是候选长程上下文模型，不是整个科研 idea。

## 影响

实现计划、插入点和模型变体都是次级内容。它们必须服务于上面的存在性、层级归因和收益归因。

后续实验不得把小窗口卷积 / TCN 收益包装成长程时间冗余收益。`full utterance` 只能作为 offline upper bound；如果要写 streaming claim，必须使用 causal past context。

当前 idea 的 source of truth 是：

- [研究想法](/Users/ely/workspace/research/audio/AudioCodec/docs/idea.md)

更早的细节草稿和过度具体的实验协议已经归档。它们可以在后续实现时参考，但不再约束当前 idea。
