Owner: ely
Status: accepted
Last reviewed: 2026-04-13

# ADR 0001: 课程项目先完成 VAE/RVQ 基线，再讨论 Mamba 扩展

## Context

仓库的长期愿景是构建更高压缩密度的 speech tokenizer，并探索更高效的时序建模方案。

但当前用户的实际约束是：

- 音频方向基础仍在建立中
- 课程项目时间只有 1 到 2 天
- 目标是快速完成一个可训练、可评测、可讲清楚的系统

在这一前提下，如果直接把 Mamba 集成进 codec 主干，会同时引入新的模型设计、训练不确定性和额外调参成本，风险过高。

## Decision

当前课程里程碑采用以下策略：

1. 先实现 `speech autoencoder + RVQ` 的 neural codec baseline。
2. 将与 `MP3` 的对比实验作为主要交付。
3. 将 Mamba 仅保留为未来工作或后续研究方向，不进入本次关键路径。

## Consequences

### Positive

- 更容易在课程周期内得到稳定结果。
- 更容易讲清楚 learned compression 与传统 codec 的差异。
- 为后续引入 Mamba 提供可复用的 baseline 和评测框架。

### Negative

- 本次作业的“模型创新度”会低于直接做新结构尝试。
- Presentation 中需要主动说明：当前工作是可复现、可分析的基线系统，而不是最终研究形态。

## Follow-up

当 baseline 和评测脚本稳定后，再考虑下面两类扩展：

- 用更强的时序模块替换部分卷积堆栈
- 面向离散 token 的下游建模与长音频实验
