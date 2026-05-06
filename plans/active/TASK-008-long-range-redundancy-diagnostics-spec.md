Owner: ely
Status: active
Last reviewed: 2026-05-06

# TASK-008 长程时间冗余诊断与上下文建模实现路线 Spec

## 目标

围绕 `docs/idea.md` 当前定义的研究主题，建立后续实现与实验的路线大纲：

> Neural speech codec 已经具备局部时序建模后，语音中是否仍存在可利用的长程时间冗余；如果存在，它应该在哪个表示层级被利用，才能转化为真实压缩收益或保真率收益？

本 spec 的作用不是立即实现模型代码，而是让后续 agent 可以按 phase 进入开发，并且始终区分两类任务：

- **实现任务**：新增脚本、模块、配置、schema、测试和 harness 支撑。
- **实验任务**：运行导出、诊断、训练、benchmark、聚合结果，并基于 gate 决定是否进入下一阶段。

## 当前主线

第一主线是 fixed-frame `SEANet + EMA RVQ` baseline 上的 long-range redundancy diagnostics。

必须先完成 `E0-E2` go/no-go diagnostics：

1. 导出长片段 / full utterance 的 `latent`、`quantized`、`codes` 和 reconstruction；
2. 比较 local / medium / long / full context 的 predictability / entropy；
3. 用 frozen RVQ codes 验证 code-prior entropy savings。

只有当 long/full context 相比最佳 local baseline 至少带来 `>=5%` estimated entropy bitrate 或 predictability improvement 时，才进入 codec context training。

## 固定边界

本任务不允许把研究问题偷换成普通工程扩容：

- 不直接实现 Mamba codec；
- 不把 local TCN / local conv 的收益写成长程时间冗余收益；
- 不改变 sample rate、frame rate、codebook size、RVQ stage 数、loss recipe 或 RVQ payload accounting；
- 不让 post-RVQ refiner 发送额外 side channel；
- 不把 code-prior entropy 改善写成 reconstruction fidelity 改善；
- 不在缺少 matched TCN / LSTM / Transformer baseline 时写 Mamba 优越性 claim；
- 不把 dynamic / variable frame-rate 混入 fixed-frame context 实验主线。

## Phase Roadmap

| Phase | 类型 | 目标 | 实现任务 | 实验任务 | Gate |
| --- | --- | --- | --- | --- | --- |
| Phase 0 | Spec / harness | 固化路线大纲 | 用本 spec 替换旧 active plan；更新 README、init、feature list、progress、handoff | 人工审查 spec 是否覆盖实现/实验边界 | spec 被用户确认 |
| Phase 1 | 实现 | Representation export 与结果 schema | 扩展 neural export，保存 `latent / quantized / codes / reconstruction`；写 `manifest.jsonl`、`run.json` 和 context metadata；记录 `context_scope`、`context_window_seconds`、`context_window_frames` | 用现有短样本做 sanity，只验证 shape、路径、metadata，不写科研结论 | export 可复现，schema 稳定 |
| Phase 2 | 实验优先 | Frozen representation redundancy diagnostics | 如需要，新增离线 diagnostics 脚本和 summary 聚合；不训练 codec | 比较 local / medium / long / full 在 latent、quantized、codes 上的 predictability / entropy | long/full 不优于 local 则停止或转向 |
| Phase 3 | 实现 + 实验 | Code-prior entropy baseline | 实现 unigram、previous-frame、local TCN、long Transformer prior；默认 token ordering 为 time-major、frame 内 coarse-to-fine stage conditioning | 在 frozen RVQ codes 上报告 bits-per-code、estimated entropy bitrate、entropy_savings_ratio | 达到 `>=5%` long-context 增益才进入 codec training |
| Phase 4 | 实现 + 实验 | Latent pre-RVQ context | 新增同形状 residual `TemporalMixer [B,C,T]->[B,C,T]`；先支持 identity、local TCN、long Transformer；接入 `latent_pre_rvq` | 4kbps anchor 上比较 identity/local/long 的 rate-distortion 与 reconstruction 指标 | long context 必须超过 matched local control |
| Phase 5 | 实现 + 实验 | Post-RVQ embedding context | 接入 decoder-side `post_rvq_embedding` refiner；加入 no-side-channel 检查 | 验证 nominal payload 不变时 fidelity 是否提升 | 只允许 claim 保真率，不允许 claim bitrate 降低 |
| Phase 6 | 条件实现 | Mamba / SSM 与 early feature | 仅在前面证明 long context 有收益后接入 Mamba；依赖缺失时清晰报错；early feature 拆 SEANet 作为后续高风险项 | 比较 Mamba vs Transformer 的质量、latency、memory；early feature 只做论文级对照 | Mamba 的价值必须来自效率或长窗表现 |
| Conditional Branch | 条件分支 | Dynamic / variable frame-rate | 不在主线立即实现；若 diagnostics 显示收益集中在静音、长元音、steady-state segment，再创建新任务 | 对标 TFC / CodecSlime 路线 | 另开 task，不混入 fixed-frame context 实验 |

## Phase 1 实现规格

当前实现状态：

- `evals/scripts/export_neural_codec.py` 已支持 `--save-representations`，可导出 `codes / latent / quantized / reconstruction`。
- `manifest.jsonl` 和 `run.json` 已写入 Phase 1 schema、context window metadata、frame/RVQ metadata 和 representation paths。
- `tests/test_evals_scripts.py` 已覆盖 metadata helper 和临时 checkpoint/audio 的 export integration sanity。

实现任务：

- 扩展 `evals/scripts/export_neural_codec.py` 或新增 context export 脚本，保留现有 neural benchmark 行为。
- 对每条 manifest row 保存：
  - `codes.pt`，shape `[1, K, T]`；
  - `latent.pt`，shape `[1, C, T]`；
  - `quantized.pt`，shape `[1, C, T]`；
  - reconstruction wav。
- 写出 `manifest.jsonl`、`run.json` 和后续聚合可消费的 result schema 字段。
- metadata 至少包含 `duration_seconds`、`sample_rate`、`num_frames`、`frame_rate`、`hop_length`、`num_quantizers`、`codebook_size`、`bits_per_code`、`nominal_bitrate_kbps`、`rvq_payload_bits`、`context_scope`、`context_window_seconds`、`context_window_frames`、`is_full_utterance`。

实验任务：

- 只用现有 `evals/data/manifests/test.jsonl` 做 sanity。
- 验证导出的 tensor shape、路径、metadata 和 reconstruction 能被现有 score 脚本消费。
- 不把 2 秒或短样本 sanity 写成长程结论。

## Phase 2 实验规格

当前实现状态：

- `evals/scripts/diagnose_representations.py` 已支持读取 Phase 1 export 目录，输出 `diagnostics.jsonl` 和 `summary.json`。
- 连续表示 `latent / quantized` 使用 past-window mean prediction，报告 `normalized_mse` 和 `predictability_score`。
- 离散 `codes` 使用 window reuse proxy，报告 `window_reuse_rate`、`previous_frame_match_rate` 和 `marginal_entropy_bits_per_code`。
- `tests/test_evals_scripts.py` 已覆盖 synthetic representation diagnostics 和 summary gate 输出。

实现任务：

- 新增 frozen-representation diagnostics 脚本时，输入必须是 Phase 1 的 export 目录。
- 输出 `diagnostics.jsonl` 和 `summary.json`，每行记录表示层级、context scope、窗口长度、metric 名称和值。
- local / medium / long / full 的窗口必须按当前表示层 frame rate 计算。

实验任务：

- 对 `latent`、`quantized`、`codes` 分别计算 predictability / entropy proxy。
- 必须包含最佳 local baseline；long/full 只和最佳 local 比，不和 no-context 比。
- 如果 long/full improvement `<5%`，记录为 go/no-go negative result，并暂停 codec context training。

## Phase 3 实现与实验规格

当前实现状态：

- `evals/scripts/evaluate_code_priors.py` 已支持读取 Phase 1 export 目录中的 frozen `codes.pt`。
- 当前实现 analytic `unigram` 和 `previous_frame` baselines；不训练 codec、不改变 reconstruction path。
- 输出 `train_metrics.jsonl`、`val_metrics.jsonl`、`summary.json` 和 `config.json`。
- Summary 记录 `stage_bits_per_code`、`bits_per_code`、`estimated_entropy_bitrate_kbps`、`entropy_savings_ratio`、`relative_improvement_vs_unigram` 和 `time_major_frame_stage_coarse_to_fine` token ordering。
- Analytic evaluator 的 summary 会把 `local_tcn` 和 `long_transformer` 标为需由训练脚本产生，避免把 analytic 结果伪装成 trained prior。
- `tests/test_evals_scripts.py` 已覆盖 synthetic RVQ codes 的 code-prior entropy summary。
- `evals/scripts/train_code_prior.py` 已支持训练 `local_tcn` 和 `long_transformer` prior，输入仍是 frozen `codes.pt`。
- 训练脚本输出 `train_metrics.jsonl`、`val_metrics.jsonl`、`summary.json`、`config.json` 和 `checkpoint.pt`，指标口径与 analytic prior 对齐。
- `mamba` 仍未进入主线；当前脚本不引入 SSM 依赖。
- `tests/test_evals_scripts.py` 已覆盖 synthetic RVQ codes 的 `local_tcn` training metrics 和 `long_transformer` smoke。
- `scripts/run-context-prior-pipeline.sh` 已串起 Phase 1 export、Phase 2 diagnostics、Phase 3 analytic prior 和 trained local/long prior。
- Pipeline 支持 `--dry-run`，可在没有真实 checkpoint 时验证命令拼装；真实运行时末尾默认调用轻量结果包脚本。
- `evals/scripts/collect_context_results.py` 已支持读取 pipeline 输出并生成 `results.jsonl`、`summary.csv` 和 `summary.json`。
- 结果聚合会记录 `relative_improvement_vs_local_or_unigram`、`gate_passed` 和 `go_no_go`，用于判断是否进入 codec context training。
- `scripts/pack-context-results.sh` 已支持把 pipeline 输出白名单复制成可下载小目录，只包含 `manifest / run metadata / summary / metrics / results`，不复制 `checkpoint.pt`、representation tensor 或 reconstruction wav。

实现任务：

- Code prior 只消费 frozen `codes.pt`，不改 reconstruction codec。
- 默认 token ordering：
  - time-major；
  - 同一 frame 内按 RVQ stage coarse-to-fine；
  - 所有输出必须记录 token ordering。
- 最小 prior 顺序：unigram per stage、previous-frame / local Markov、local TCN、long Transformer。
- Mamba prior 不作为 Phase 3 阻塞项；依赖未固定时标记 blocked。

实验任务：

- 报告 `bits_per_code`、`stage_bits_per_code`、`estimated_entropy_bitrate_kbps`、`entropy_savings_ratio`。
- 不报告 reconstruction fidelity gain。
- 达到 `>=5%` long-context improvement 才进入 Phase 4。
- 训练机跑完后优先下载 `download-bundles/<timestamp>/` 轻量结果包；除非需要复现实验，不默认下载 prior checkpoint 或 representation tensor。

## Phase 4-6 实现与实验规格

实现任务：

- `TemporalMixer` 统一接口为 `forward(x: Tensor[B, C, T]) -> Tensor[B, C, T]`。
- 默认 residual wrapper：`x_ctx = x + output_projection(mixer(input_projection(norm(x))))`。
- `identity` 必须保持形状和 payload 可解释。
- `latent_pre_rvq` 让 RVQ 消费 context 后的 latent；`post_rvq_embedding` 只在 decoder 侧消费 quantized context。
- `post_rvq_embedding` 必须有 no-side-channel 检查。
- Mamba / early feature 只在前面 gate 通过后实现。

实验任务：

- Phase 4 以 `configs/ablation-adversarial-msstft-balanced-4kbps.json` 为第一锚点。
- 先比较 identity、local TCN control、long Transformer，再考虑 LSTM/Mamba。
- Reconstruction 指标沿用 `evals/scripts/score_outputs.py`，并额外记录 context params、latency 和 memory。
- causal / non-causal 必须分开记录，full utterance 只能作为 offline upper bound。

## Conditional Branch

Dynamic / variable frame-rate 是重要后续分支，但不是当前 fixed-frame context 主线。

触发条件：

- Phase 2/3 显示冗余主要集中在静音、长元音或缓慢变化的 steady-state segment；
- long-context module 的收益更像“少发重复 frame/token”，而不是更强表示层级建模；
- fixed-frame code prior 有收益，但 codec context training 无法转成 RD 或 fidelity gain。

触发后动作：

- 新建独立 active task；
- 对标 TFC / CodecSlime；
- 重新定义 payload accounting 和 frame allocation schema；
- 不复用当前 fixed-frame context 实验的结论口径。

## 验收标准

- `plans/active` 只有本 spec 一个 active 文件；
- README 和 `init.sh` 指向本 spec；
- `harness/feature_list.json` 的唯一 active item 语义更新为 long-range diagnostics / context modeling route；
- 每个 phase 都清楚区分实现任务和实验任务；
- 下一位 agent 可以直接运行 Phase 1 真实 export sanity，并在通过后进入 Phase 2 diagnostics；
- 验证命令全部通过，失败项必须写入 `harness/session-handoff.md`。

## 验证命令

```bash
./scripts/harness-check.sh
git diff --check
bash -n scripts/pack-context-results.sh
bash -n scripts/run-context-prior-pipeline.sh
conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help
conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v
```

macOS 上 `KMP_DUPLICATE_LIB_OK=TRUE` 只作为本地 OpenMP workaround，不进入 Linux A100 训练命令。

## 下一步

下一步优先用已有 4kbps checkpoint 和可访问的 long/full utterance manifest 运行 `scripts/run-context-prior-pipeline.sh` 做真实 Phase 1 export + Phase 2 diagnostics + Phase 3 analytic/trained code-prior sanity。训练结束后先查看 `results/summary.json` 的 `go_no_go`，并下载 pipeline 自动生成的轻量结果包；Mamba 依赖未固定前不进入主线。
