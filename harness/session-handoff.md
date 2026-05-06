Owner: ely
Status: active
Last reviewed: 2026-05-06

# 会话交接

## 仓库状态

- 分支：`feat/context-modeling`
- 已有提交：
  - `8c54781 docs: define context modeling research baseline`
  - `11e4bf3 docs: add context modeling research intake`
  - `9faade8 docs: define context modeling experiment plan`
  - `267c771 docs: clarify temporal context insertion semantics`
  - `b57556e docs: refocus context modeling on long-range redundancy`
  - `705a275 docs: define long-range redundancy research theme`
  - `51a8b09 docs: add long-range redundancy research refresh`
  - `bf45862 docs: define long-range redundancy implementation spec`
  - `d242624 feat(evals): export codec representations for diagnostics`
  - `6ec80f2 feat(evals): add frozen representation diagnostics`
  - `ebb6077 feat(evals): add analytic code prior entropy baselines`
- 当前待提交目标：实现 Phase 3 trained code-prior baseline，继续积累本地可验证能力，避免依赖暂不可访问的 Linux 训练机。
- 本轮新增/修改范围：
  - added: `evals/scripts/train_code_prior.py`
  - modified: `evals/scripts/evaluate_code_priors.py`
  - modified: `tests/test_evals_scripts.py`
  - modified: `evals/README.md`
  - modified: `init.sh`
  - modified: `plans/active/TASK-008-long-range-redundancy-diagnostics-spec.md`
  - modified: `harness/feature_list.json`
  - modified: `harness/progress.md`
  - modified: `harness/session-handoff.md`

## 当前已验证状态

- `./init.sh` 通过，并打印新的 Phase 1 / Phase 2 / Phase 3 当前阶段提示。
- `./scripts/harness-check.sh` 通过，0 warnings。
- `git diff --check` 通过。
- `python3 -m json.tool harness/feature_list.json >/dev/null` 通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python evals/scripts/train_code_prior.py --help` 通过。
- `python3 -m py_compile evals/scripts/train_code_prior.py` 通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python evals/scripts/evaluate_code_priors.py --help` 通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python evals/scripts/diagnose_representations.py --help` 通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python evals/scripts/export_neural_codec.py --help` 通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help` 通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest tests.test_evals_scripts -v` 通过，14 tests OK。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v` 通过，33 tests OK。

## 本会话改动

- 新增 `evals/scripts/train_code_prior.py`：
  - 输入 Phase 1 export 目录；
  - 默认读取 `manifest.jsonl` 中的 `codes_path`；
  - 训练 `local_tcn` 或 `long_transformer` prior；
  - 输出 `train_metrics.jsonl`、`val_metrics.jsonl`、`summary.json`、`config.json` 和 `checkpoint.pt`。
- 训练脚本保持默认 token ordering：`time_major_frame_stage_coarse_to_fine`。
- 训练脚本报告：
  - `stage_bits_per_code`；
  - `bits_per_code`；
  - `estimated_entropy_bitrate_kbps`；
  - `entropy_savings_ratio`；
  - `context_scope` / `context_window_frames` / `context_window_seconds`。
- `local_tcn` 使用 causal dilated Conv1d；`long_transformer` 使用 causal Transformer encoder。
- `evaluate_code_priors.py` 的 blocked 文案已改成 `local_tcn` / `long_transformer` 由 `train_code_prior.py` 产生。
- `tests/test_evals_scripts.py` 新增 synthetic RVQ codes 测试，覆盖 local TCN training metrics 和 long Transformer smoke。
- `evals/README.md` 和 active spec 已记录训练型 code-prior 命令、输出和边界。

## 本会话决策

- 当前不等待 Linux 训练机；继续开发本地可验证工具。
- Phase 3 训练型 prior 只消费 frozen `codes.pt`，不训练 codec、不改 reconstruction path。
- 当前不引入 Mamba/SSM 依赖；Mamba 仍必须等 long-context 收益与依赖决策更清楚后再进入。
- 后续 bash 脚本现在可以开始写：它应串起 export、diagnostics、analytic prior 和 trained prior，但仍不跑真实 Linux smoke。

## 仍损坏或未验证

- 未用真实 4kbps checkpoint 和可访问的 long/full utterance manifest 跑完整 export+diagnostics+prior；当前 train prior test 使用 synthetic tensors。
- 未在本轮运行真实训练 smoke。
- 未在 Linux `4 x A100` 训练机重新验证 smoke。
- Mamba/SSM 依赖尚未固定。
- macOS 本地完整单测仍依赖 `KMP_DUPLICATE_LIB_OK=TRUE` workaround；这不应进入 Linux 训练命令。

## 清洁状态

- 本轮最终验证已完成，当前等待提交。
- 临时工件：本轮未创建模型输出、训练日志、下载缓存或调试脚本。

## 下一步最佳动作

编写统一 bash 脚本程式化执行 Phase 1 export、Phase 2 diagnostics、Phase 3 analytic prior 和 trained local/long prior；真实 Linux 训练机可用后，再用同一脚本跑 4kbps checkpoint 与 long/full utterance manifest。

## 命令

- 初始化：`./init.sh`
- Harness 检查：`./scripts/harness-check.sh`
- 聚焦验证：`git diff --check`
- Export CLI sanity：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python evals/scripts/export_neural_codec.py --help`
- Diagnostics CLI sanity：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python evals/scripts/diagnose_representations.py --help`
- Analytic code-prior CLI sanity：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python evals/scripts/evaluate_code_priors.py --help`
- Trained code-prior CLI sanity：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python evals/scripts/train_code_prior.py --help`
- Focused tests：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest tests.test_evals_scripts -v`
- CLI sanity：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help`
- 完整验证：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v`
- Linux A100 smoke：`PYTHONPATH=src python scripts/train_codec.py --config configs/ablation-adversarial-msstft-balanced.json --output-dir artifacts/smoke-b0-12k --smoke-test --limit-train-examples 10 --device cuda`
