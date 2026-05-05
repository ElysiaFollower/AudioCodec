<!--
职责：记录会影响后续 agent 决策的重要选择及其理由。
边界：不要记录每次小改动、聊天摘要或可从代码直接看出的事实。
-->

Owner: ely
Status: active
Last reviewed: 2026-05-05

# 决策日志

## 记录规则

重要决策必须写明：日期、决策、原因、否决方案、后续约束。

## 决策

### 2026-05-05 - 采用 repo-native harness

- 决策：采用 repo-native harness，仓库保存指令、状态、验证、交接和质量信息。
- 原因：降低冷启动成本、上下文丢失、范围漂移、验证缺口和返工。
- 否决方案：只依赖聊天 prompt 或单个巨型 `AGENTS.md`。
- 后续约束：项目事实必须进入仓库；重复失败优先转成测试、脚本或检查。

### 2026-05-05 - `docs/idea.md` 是当前研究 idea 的最高优先级来源

- 决策：当前研究问题以 `docs/idea.md` 为 source of truth，ADR 0002 记录范围决策，`docs/archive/` 只作历史参考。
- 原因：旧课程项目资料、第一版草稿和实现协议会把 agent 拉回“先实现模块”或“打败 MP3”的旧语境。
- 否决方案：让 README、归档课程报告或过度具体的 draft protocol 共同定义当前方向。
- 后续约束：任何会改变核心 idea 的改动，先更新 `docs/idea.md` 和 ADR，再改计划或实现。

### 2026-05-05 - 一阶段先做研究收集和实验准备

- 决策：idea 基本明确后，下一阶段不是直接大规模开发，而是围绕 idea 收集相关研究和代码，形成实验准备清单。
- 原因：当前关键风险是研究边界、matched baseline、插入层级和指标定义不清，而不是缺少某个模块脚手架。
- 否决方案：立即实现 Mamba-only codec、直接固定 latent/code-level context，或先扩展训练矩阵。
- 后续约束：一阶段产物必须覆盖 early context、latent context、post-RVQ context、code-prior context，并明确 TCN/LSTM/Transformer/Mamba 的比较位置。
