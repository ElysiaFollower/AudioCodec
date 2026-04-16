

== Project Introduction

本节简要交代项目选题、任务目标、系统范围与开发环境，让读者先理解“我们为什么做 speech neural codec，以及最后要和谁比较”。 

=== Problem Definition

说明本项目聚焦 `16 kHz mono speech codec`，目标是在可控码率下实现高保真重建，并与传统音频压缩算法做 benchmark 对比。

=== Project Scope

说明本项目只做 speech 场景，不追求通用音乐 codec，不做流式部署和产品级端侧优化。

=== Why This Topic

说明选择 neural codec 的动机，包括传统 codec 在 speech 低码率条件下的局限，以及 learned codec 的潜在优势。

=== Project Deliverables

概述本项目最终交付包括模型实现、训练代码、benchmark 管线、试听样例和实验报告。

=== Development Environment

说明开发语言、框架、训练环境、关键依赖、`ffmpeg` 等外部工具，以及代码在本地和 Linux 训练机上的运行条件。

== Technical Details

=== Theory Background

介绍 neural codec 的基本思想，并解释本项目与严格 `VAE` 的关系、`RVQ` 的作用，以及感知型训练目标为什么重要。

=== Baseline Design and Early Simplification

说明最初为什么从 `Conv + RVQ + Decoder` 的可复现 baseline 出发，以及这种课程版设计的优势与局限。

=== Model Evolution

按发展历程说明架构如何从简单 baseline 逐步演进到 `SEANet + EMA RVQ`，并解释每一步改动的技术动机。

=== Final Model Architecture

详细介绍最终模型的组成，包括 `SEANet encoder/decoder`、`EMA RVQ`、离散 token 表示与码率控制方式。

=== Training Objective Evolution

说明训练目标如何从 `waveform + STFT (+ mel)` 逐步演进到 `adversarial + feature matching + balancer`，并解释为什么这些改动是质量提升的关键。

=== Key Debugging and Engineering Lessons

总结开发过程中影响结果的关键问题，包括 overfit 验机、数值稳定性、resume 风险、adversarial 训练语义对齐等。

=== Bitrate Ladder Design

说明为什么最终固定 backbone 和 loss，仅通过 `num_quantizers` 构造 `2 / 4 / 8 / 12 kbps` neural bitrate ladder。

=== Benchmark Pipeline Design

介绍为什么 benchmark 被独立放入 `evals/`，以及 manifest、traditional codec wrapper、统一 scoring 的设计原则。


== Experiment Results

=== Experimental Questions

明确本实验主要回答两个问题：同码率下 neural codec 是否更优，以及达到相近质量时传统 codec 需要多少实际码率。

=== Training Validation

展示从 overfit 验机到全数据训练成功的关键证据，说明模型确实可训练且重建质量随训练步数提升。

=== Benchmark Setup

说明 benchmark 使用的数据子集、四个 neural bitrate 点、传统 codec 组别、实际码率统计口径和所用评价指标。

=== Objective Results

用表格和图展示 `actual bitrate`、压缩率和质量指标，并突出 neural codec 在低码率端的表现。

=== Subjective Listening Findings

总结主观试听观察，包括传统 codec 在默认模式下的高保真表现，以及 `neural-2k` 中抽样发现的声线变化等典型现象。

=== Rate-Distortion Interpretation

解释 neural bitrate ladder 与 `Opus / MP3 / AAC` 曲线的相对位置，以及这些结果如何支持“neural codec 在低码率下具有优势”的结论。

=== Failure Cases and Limitations

诚实记录低码率下的声线漂移、传统 codec target bitrate 与 actual bitrate 脱钩等限制，说明当前结论的边界。

=== Final Conclusion

总结本项目最终验证了什么、没有验证什么，以及对传统 codec 与 neural codec 优劣的整体判断。

=== Future Work

提出下一步可改进方向，例如补 `ViSQOL`、完善主观听测、增加更多传统 codec 或进一步优化极低码率 timbre 保真。 


== References

列出课程项目中直接参考的论文、官方文档、开源实现和外部工具文档。
