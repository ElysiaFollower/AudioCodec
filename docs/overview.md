Owner: ely
Status: active
Last reviewed: 2026-05-06

# AudioCodec 项目总览

## 当前项目定位

本仓库现在是一个科研工作区，核心问题只有一个：

> Neural speech codec 已经具备局部时序建模后，语音中是否仍存在可利用的长程时间冗余；如果存在，它应该在哪个表示层级被利用，才能转化为真实压缩收益或保真率收益？

当前 idea 的定义见：

- [研究想法](/Users/ely/workspace/research/audio/AudioCodec/docs/idea.md)

请先读这份文档。它是当前项目方向的唯一主入口。

## 这个项目现在不是什么

当前项目不是：

- 关于“打败 MP3”的课程项目报告；
- 已经固定好的工程开发路线；
- “Mamba 一定优于 Transformer”的模型替换项目；
- “给 codec 加局部时间上下文”的普通工程项目；
- “latent/code-level 一定优于 waveform-level”的预设结论；
- 从零重写 EnCodec、DAC 或 Mimi。

## 当前稳定基底

仓库已经有一个可用的 speech codec baseline：

```text
waveform -> SEANet encoder -> downsampled latent -> EMA RVQ -> RVQ codes -> quantized embedding -> SEANet decoder -> waveform
```

当前 baseline 说明见：

- [当前 codec baseline](/Users/ely/workspace/research/audio/AudioCodec/docs/architecture/current-codec-baseline.md)

这个 baseline 的作用是提供一条稳定的表示链，让研究问题可以被验证，而不是先重建整个 codec 系统。

## 活跃文档

活跃文档应该保持少而清晰：

- [研究想法](/Users/ely/workspace/research/audio/AudioCodec/docs/idea.md)
- [当前 codec baseline](/Users/ely/workspace/research/audio/AudioCodec/docs/architecture/current-codec-baseline.md)
- [ADR 0002](/Users/ely/workspace/research/audio/AudioCodec/docs/adr/0002-temporal-redundancy-research-scope.md)
- [上下文建模研究收集](/Users/ely/workspace/research/audio/AudioCodec/docs/research/context-modeling-intake.md)
- [长程时间冗余相关研究补充](/Users/ely/workspace/research/audio/AudioCodec/docs/research/long-range-redundancy-intake.md)
- [上下文建模实现与实验计划](/Users/ely/workspace/research/audio/AudioCodec/docs/research/context-modeling-experiment-plan.md)

归档或草稿实现笔记在：

- [文档归档](/Users/ely/workspace/research/audio/AudioCodec/docs/archive/README.md)

归档内容不再约束当前方向。它们可以作为参考，但不能覆盖 `docs/idea.md` 中定义的 idea。

## 当前下一步

当前 idea 和 E0-E2 go/no-go 工具链已基本成形。下一步是把仓库 clone 到 Linux 训练机，使用 self-contained pipeline 直接从当前分支产出实验资产：

- `scripts/run-context-prior-pipeline.sh` 默认先构建 manifest，并在需要时训练当前分支自己的 4kbps fixed-frame baseline checkpoint，再串起 representation export、frozen diagnostics、analytic prior、trained local/long prior、结果聚合和轻量结果包；
- `scripts/pack-context-results.sh` 会生成包含分析所需 `summary / metrics / manifest / run metadata` 和少量 `source / reconstruction` 试听对的可下载目录；
- 训练 checkpoint、representation tensor 和整批 reconstruction wav 默认保留在原始输出目录，不进入轻量下载包；
- 先看 `results/summary.json` 的 `go_no_go`，再决定是否进入 Phase 4 codec context training。

实现应服务于这个对照，而不是提前预设某个层级或某个模型族一定获胜。
