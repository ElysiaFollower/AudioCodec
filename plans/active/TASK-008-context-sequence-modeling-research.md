Owner: ely
Status: active
Last reviewed: 2026-05-05

# TASK-008 一阶段：idea 相关研究收集与实验准备

## 目标

围绕 `docs/idea.md` 定义的研究问题，完成第一阶段研究收集和实验准备。

本任务的交付物不是直接实现某个上下文模块，而是让后续开发有清楚证据基础：

- 哪些相关研究支持或挑战 early context、latent context、post-RVQ context、code-prior context；
- 哪些代码库或实现路线可以复用；
- 哪些 baseline 必须 matched；
- 哪些指标和实验命令能验证 idea，而不是只验证模型能跑。

## 背景

当前项目 idea 已基本明确：

> 在 neural speech codec 中，应该在哪个表示层级引入时间上下文建模，才能把时间冗余转化为真实的压缩收益或保真率收益？

当前风险不在于没有实现脚手架，而在于过早固定实现路线会把研究问题缩窄成 “Mamba codec” 或 “latent/code-level context 一定更好”。

## 范围内

- 收集 neural speech codec、RVQ speech tokenizer、audio/speech codec entropy coding、codec token prior、sequence modeling for codec/token streams 相关论文；
- 收集可直接参考的代码库、模型配置和实验设置；
- 按表示层级整理证据：waveform/early feature、downsampled latent、post-RVQ embedding、discrete RVQ codes/code prior；
- 明确 TCN、LSTM、Transformer、Mamba/SSM 在比较矩阵中的角色；
- 形成下一阶段开发前置清单：数据、指标、脚本、模块边界、最小实验矩阵、排除项；
- 更新 `harness/feature_list.json`、`harness/progress.md`、必要 ADR 或长期文档。

## 范围外

- 直接实现 `TemporalMixer`、Mamba codec、post-RVQ refiner 或 entropy model；
- 固定最终实验矩阵；
- 运行完整训练或大规模 benchmark；
- 改变 stable codec baseline 的 frame rate、codebook size、loss recipe、front-end 或 RVQ payload accounting；
- 在没有 matched baseline 和验证结果前写论文级强 claim。

## 验收标准

- 研究收集产物能覆盖 `docs/idea.md` 中的两个竞争假设，而不是只支持单一路线；
- 每个候选方向都说明它对应哪类收益：保真率、rate-distortion、entropy、token efficiency 或推理效率；
- 至少列出一个简单 baseline 和一个 Transformer-family baseline，Mamba 不能单独出现；
- 下一阶段实现计划明确最小实验矩阵、验证命令、数据路径、指标和 out-of-scope；
- `./scripts/harness-check.sh` 通过；
- `harness/session-handoff.md` 能让新会话直接接手研究收集或进入实现准备。

## 当前下一步

从 `docs/idea.md` 出发，先建立研究收集表或文档骨架。每条材料至少记录：

- 论文/代码库；
- codec/tokenizer 类型；
- 上下文建模插入层级；
- 对 bitrate、entropy、保真率或效率的证据；
- 可复用实现点；
- 与当前 baseline 的差距和风险。
