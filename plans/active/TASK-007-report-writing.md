Owner: ely
Status: active
Last reviewed: 2026-04-16

# TASK-007 Report Writing

## Goal

围绕当前已经完成的模型实现、训练演进与 benchmark 结果，撰写一份可直接用于课程项目提交的报告正文初稿，并同步整理图表、表格与占位素材。

## Background

- `TASK-005` 已完成训练对齐，当前 neural codec 已能在 `2 / 4 / 8 / 12 kbps` 下实现可用至高保真的语音重建。
- `TASK-006` 已完成传统 codec benchmark，输出已经统一落在 `evals/outputs/`。
- 用户偏好使用 `Typst` 撰写报告，报告入口为 `docs/report/main.typ`。
- 当前最需要的不是继续扩展实验，而是把“设计演进 -> 关键 debug -> benchmark 结论”整理成清晰、可信、可追溯的论文式叙事。

## In Scope

- 把 `docs/report/main.typ` 从提纲扩展为正文初稿
- 生成或整理报告所需的关键表格与图
- 明确缺失素材的占位符与补充说明
- 让报告结构与项目真实发展过程一致，避免事后倒推式叙事

## Out Of Scope

- 在本阶段继续增加新的模型训练实验
- 在本阶段继续扩展 benchmark 维度
- 在本阶段追求排版终稿级微调

## Deliverables

1. 一份可编译的 `Typst` 报告正文初稿
2. 一组报告可直接引用的关键图表资产
3. 对缺失图片、试听图例或未来补充项的明确占位说明

## Acceptance Criteria

- 报告正文已经覆盖：背景、方法、训练演进、实验设计、结果分析、局限与未来工作
- 关键结论与仓库中的代码、配置、日志和 benchmark 结果一致
- 报告可以成功编译，或至少能明确指出剩余排版阻塞
