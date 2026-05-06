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
- 当前待提交目标：实现 Phase 3 analytic code-prior entropy baseline，继续积累本地可验证能力，避免依赖暂不可访问的 Linux 训练机。
- 本轮新增/修改范围：
  - added: `evals/scripts/evaluate_code_priors.py`
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
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python evals/scripts/evaluate_code_priors.py --help` 通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python evals/scripts/diagnose_representations.py --help` 通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python evals/scripts/export_neural_codec.py --help` 通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help` 通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest tests.test_evals_scripts -v` 通过，12 tests OK。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v` 通过，31 tests OK。

## 本会话改动

- 新增 `evals/scripts/evaluate_code_priors.py`：
  - 输入 Phase 1 export 目录；
  - 默认读取 `manifest.jsonl` 中的 `codes_path`；
  - 输出 `code_priors/train_metrics.jsonl`、`code_priors/val_metrics.jsonl`、`code_priors/summary.json` 和 `code_priors/config.json`。
- 当前实现两个 analytic prior：
  - `unigram_per_stage`；
  - `previous_frame_markov_per_stage`，首帧用 unigram fallback。
- 输出指标：
  - `stage_bits_per_code`；
  - `bits_per_code`；
  - `estimated_entropy_bitrate_kbps`；
  - `entropy_savings_ratio`；
  - `relative_improvement_vs_unigram`。
- 统一记录 token ordering：`time_major_frame_stage_coarse_to_fine`。
- `summary.json` 中明确 `local_tcn`、`long_transformer` 和 `mamba` 尚未实现，不伪造训练型 prior 结果。
- `tests/test_evals_scripts.py` 新增 synthetic RVQ codes 测试，覆盖 self-eval、summary、blocked priors 和 previous-frame 相对 unigram 的 bitrate 关系。
- `evals/README.md` 和 active spec 已记录 Phase 3 analytic code-prior 命令、输出和边界。

## 本会话决策

- 当前不等待 Linux 训练机；继续开发本地可验证工具。
- Phase 3 先落地 analytic prior，形成可跑的 entropy accounting baseline。
- `local_tcn` 和 `long_transformer` 属于后续训练型 prior，不在当前 commit 中用启发式结果替代。
- Mamba/SSM 依赖未固定，仍不是 Phase 3 阻塞项。
- 后续 bash 脚本应在 Phase 3/4 工具链稳定后再写，统一串起 export、diagnostics、prior 和训练。

## 仍损坏或未验证

- 未用真实 4kbps checkpoint 和可访问的 long/full utterance manifest 跑完整 export+diagnostics+prior；当前 code-prior test 使用 synthetic tensors。
- 未在本轮运行真实训练 smoke。
- 未在 Linux `4 x A100` 训练机重新验证 smoke。
- Mamba/SSM 依赖尚未固定。
- macOS 本地完整单测仍依赖 `KMP_DUPLICATE_LIB_OK=TRUE` workaround；这不应进入 Linux 训练命令。

## 清洁状态

- 本轮最终验证已完成，当前等待提交。
- 临时工件：本轮未创建模型输出、训练日志、下载缓存或调试脚本。

## 下一步最佳动作

若本机有可用的真实 4kbps checkpoint 和 long/full utterance manifest，串一次 Phase 1 export + Phase 2 diagnostics + Phase 3 analytic prior sanity；否则继续 Phase 3 training prior，本地实现 `local_tcn` 或 `long_transformer`，输入仍是 frozen `codes.pt`，不要训练 codec，也不要引入 Mamba。

## 命令

- 初始化：`./init.sh`
- Harness 检查：`./scripts/harness-check.sh`
- 聚焦验证：`git diff --check`
- Export CLI sanity：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python evals/scripts/export_neural_codec.py --help`
- Diagnostics CLI sanity：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python evals/scripts/diagnose_representations.py --help`
- Code-prior CLI sanity：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python evals/scripts/evaluate_code_priors.py --help`
- Focused tests：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest tests.test_evals_scripts -v`
- CLI sanity：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help`
- 完整验证：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v`
- Linux A100 smoke：`PYTHONPATH=src python scripts/train_codec.py --config configs/ablation-adversarial-msstft-balanced.json --output-dir artifacts/smoke-b0-12k --smoke-test --limit-train-examples 10 --device cuda`
