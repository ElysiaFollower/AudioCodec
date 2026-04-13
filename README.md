Owner: ely
Status: active
Last reviewed: 2026-04-13

# AudioCodec

本仓库用于推进一个面向 `speech` 的神经音频编解码项目。

- 长期方向：探索更高压缩密度、可服务于长上下文音频建模的 speech tokenizer，并为后续引入 Mamba 类时序建模做准备。
- 当前里程碑：在课程项目约束下，先完成一个 `Encoder-Decoder + RVQ` 的 speech neural codec baseline，并与 `MP3` 做压缩率和重建质量对比。

建议先读：

- [项目总览](./docs/overview.md)
- [基线架构说明](./docs/architecture/baseline-neural-codec.md)
- [当前执行计划](./plans/active/TASK-001-course-project-bootstrap.md)
- [范围决策 ADR](./docs/adr/0001-course-project-scope.md)

当前仓库以文档先行，后续代码、训练脚本和评测脚本应围绕以上文档继续补齐。
