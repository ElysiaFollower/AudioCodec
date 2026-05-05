<!--
职责：定义本项目 harness 的运行时信号、过程工件和验证证据采集方式。
边界：不要存放完整日志；日志应由工具产生，本文只说明采集与解释规则。
-->

Owner: ely
Status: active
Last reviewed: 2026-05-05

# 可观测性

## 运行时信号

- CLI 启动/就绪：`PYTHONPATH=src python scripts/train_codec.py --help` 能打印训练入口参数。
- Harness 健康：`./scripts/harness-check.sh` 通过，且没有未解释的占位符或状态机错误。
- 单测健康：`python -m unittest discover -s tests -v` 在目标环境通过；macOS 本机可使用记录过的 OpenMP workaround。
- 训练 smoke：训练命令能创建 output dir、加载 config、读取少量 LibriSpeech 样本并完成 smoke step。
- Benchmark 健康：`evals/scripts/` 能构建 manifest、导出 codec 输出、运行传统 codec、计算指标。

## 研究过程信号

- 研究收集必须能回答它支持哪一层插入点：early waveform/feature、latent bottleneck、post-RVQ embedding、discrete code prior。
- 每篇论文或代码库记录其任务、数据、codec/tokenizer 设置、上下文模型、指标、与本项目 idea 的关系。
- 任何 claim 必须区分 nominal bitrate、entropy-coded bitrate、token efficiency、重建保真率和推理效率。
- Mamba/SSM 相关材料必须与至少一个非 Mamba baseline 放在同一比较框架里。

## 过程工件

- 任务合同：`plans/active/`
- 功能状态：`harness/feature_list.json`
- 进度日志：`harness/progress.md`
- 验证证据：feature item 的 `evidence`，以及 `harness/session-handoff.md` 中的命令结果摘要
- 长期决策：`harness/decisions.md` 或 `docs/adr/`
- 质量评估：`harness/evaluator-rubric.md` 和 `harness/quality.md`

## 面向 agent 的错误消息规则

验证失败时，错误消息应说明：

- 哪个命令失败；
- 失败的可观察症状；
- 最可能的检查位置；
- 下一步修复建议。

不要只写 “test failed”。
