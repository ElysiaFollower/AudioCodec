<!--
职责：定义项目本地的 agent 任务完成评估标准。
边界：不要复制任务级验收标准；细节应链接到 active plans 和可执行检查。
-->

Owner: ely
Status: active
Last reviewed: 2026-05-05

# 评估 Rubric

宣告任务完成前，使用本 rubric 评估。

## 必要检查

- 范围：变更符合当前 active plan，没有扩展到无关实现、训练或归档资料重写。
- Idea 对齐：任何研究或实现工作都能回连到 `docs/idea.md` 的核心问题和两个竞争假设。
- Baseline 匹配：比较上下文模型时必须说明 matched baseline、控制变量和不变的 codec 设置。
- 指标边界：nominal bitrate、entropy-coded bitrate、token efficiency、保真率和效率分开记录。
- 证据：每个 `passing` 状态都有命令、结果和相关产物或观察。
- 测试：代码行为变化有最窄可靠测试；共享行为变化时扩大验证。
- 端到端：多组件或训练/评测流程必须跑通完整路径，不能只靠单元测试。
- 可观测：失败时有足够日志、错误上下文或过程工件定位问题。
- 文档：setup、职责、公共接口和非显然不变量已在来源附近更新。
- 交接：`harness/session-handoff.md` 写明当前状态、风险和下一步最佳动作。

## 失败偏置

证据不完整时，将任务标为 `active`、`blocked` 或未验证。不要只凭自信批准。
