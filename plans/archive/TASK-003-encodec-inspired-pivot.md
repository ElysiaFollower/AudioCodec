Owner: ely
Status: archived
Last reviewed: 2026-04-14

# TASK-003 Encodec-Inspired Pivot

## Archive Note

`encodec-inspired` 路线已经完成实现、训练与试听验证。结果表明单样本 overfit 可行，但全数据训练仍存在明显音质缺口；后续工作将以新的对齐计划继续推进，而不是继续沿用本计划。

## Goal

在保留已有 baseline 结果的前提下，引入 `encodec-inspired` 的高质量架构路线，把重建音频从“可辨认内容”提升到“可用”。

## Background

- 现有 baseline 与 mel 消融都已完成训练
- 主观试听表明：两者都存在明显电流音，尚未达到可用级别
- 外部参考实现 `MambaCodec/third_party/encodec` 已证明相同任务上存在明显更好的设计上限

## Constraints

- 剩余时间约 `1` 天
- 当前最重要目标是得到一版可听感验收的 speech codec
- 不再追加复杂 adversarial 训练或大范围重构训练框架

## In Scope

- 在本仓库内新增 `SEANet + EMA RVQ` 路线
- 保留现有 baseline 路线与配置
- 提供单独训练配置与最小验证
- 后续在 Linux 上重新训练并试听

## Out Of Scope

- 复刻完整 `Encodec` 全功能栈
- 分布式训练
- 熵编码 / LM 压缩
- 课程报告最终润色

## Acceptance Criteria

- 新架构可以用 `configs/encodec-inspired.json` 正常实例化和训练
- 单元测试覆盖 baseline 与新架构的前向 shape
- Linux 侧能直接用新配置发起训练
- 重建样例的主观音质明显优于当前 baseline

## Execution Notes

1. 先提交架构和配置工厂
2. 再在 Linux 侧用新配置跑 smoke 与主训练
3. 试听通过后，再进入 `MP3` 对比与最终汇报整理
