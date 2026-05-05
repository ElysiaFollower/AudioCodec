<!--
职责：总结仓库 harness 的健康状态和下一步维护动作。
边界：不要存放完整审计日志、任务历史或项目架构细节。
-->

Owner: ely
Status: active
Last reviewed: 2026-05-05

# Harness 质量

## 快照

- 上次审查：2026-05-05
- 审查者：Codex
- 总体状态：active harness refresh

## 健康信号

- `AGENTS.md` 长度：目标 50-150 行，当前作为短路由维护。
- WIP limit：1；当前唯一 active item 是 `long-range-redundancy-diagnostics-and-context-route`。
- 功能清单有效性：已按 `scripts/harness-check.sh` schema 补齐字段。
- 交接新鲜度：本轮结束前必须更新 `harness/session-handoff.md`。
- 验证命令健康度：harness check 和静态检查作为最低门禁；完整单测依赖 `audiocodec` conda 环境。
- 冷启动测试：`./init.sh` 能打印入口、命令和当前阶段。
- 端到端覆盖：真实训练 smoke 尚未在 Linux A100 上重新验证。
- 重复失败是否已执行化：AGENTS 膨胀、占位符、WIP>1、passing 缺 evidence 已由 `scripts/harness-check.sh` 检查。

## 维护队列

- Phase 1 完成后，把 representation export schema 和验证证据沉淀到 active spec、progress 和 handoff。
- 若新增代码路径，补对应单测或 smoke，并把验证命令写入 feature item。
- 若再次出现 agent 忘记更新 progress/handoff，优先增强 `scripts/harness-check.sh` 或任务完成 checklist。
- 若 `AGENTS.md` 超过 150 行，把专题规则迁移到 `docs/` 或 `harness/`。
