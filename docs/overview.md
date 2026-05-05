Owner: ely
Status: active
Last reviewed: 2026-05-05

# AudioCodec 项目总览

## 当前项目定位

本仓库现在是一个科研工作区，核心问题只有一个：

> 在 neural speech codec 中，应该在哪个表示层级引入大时间窗口的上下文建模，才能把普通局部卷积尚未利用的长程时间冗余转化为真实的压缩收益或保真率收益？

当前 idea 的定义见：

- [研究想法](/Users/ely/workspace/research/audio/AudioCodec/docs/idea.md)

请先读这份文档。它是当前项目方向的唯一主入口。

## 这个项目现在不是什么

当前项目不是：

- 关于“打败 MP3”的课程项目报告；
- 已经固定好的工程开发路线；
- “Mamba 一定优于 Transformer”的模型替换项目；
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
- [上下文建模实现与实验计划](/Users/ely/workspace/research/audio/AudioCodec/docs/research/context-modeling-experiment-plan.md)

归档或草稿实现笔记在：

- [文档归档](/Users/ely/workspace/research/audio/AudioCodec/docs/archive/README.md)

归档内容不再约束当前方向。它们可以作为参考，但不能覆盖 `docs/idea.md` 中定义的 idea。

## 当前下一步

当前 idea 已基本明确。下一步是一阶段研究收集和实验准备：

- 继续维护 `docs/research/context-modeling-intake.md` 中的论文和代码收集；
- 以 `docs/research/context-modeling-experiment-plan.md` 为下一阶段实现入口；
- 把 local conv/TCN 作为控制组，而不是主要贡献；
- 先实现 representation export、code-prior entropy baseline 和 latent/post-RVQ context smoke；
- 暂缓 SEANet early feature 拆分，等导出、结果表和 matched baseline 工具链稳定后再做。

实现应服务于这个对照，而不是提前预设某个层级或某个模型族一定获胜。
