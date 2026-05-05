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
- 当前工作状态：Phase 2 frozen representation diagnostics 工具已实现并通过本地验证，继续积累本地可验证能力，避免依赖暂不可访问的 Linux 训练机。
- 本轮新增/修改范围：
  - added: `evals/scripts/diagnose_representations.py`
  - modified: `tests/test_evals_scripts.py`
  - modified: `evals/README.md`
  - modified: `init.sh`
  - modified: `plans/active/TASK-008-long-range-redundancy-diagnostics-spec.md`
  - modified: `harness/feature_list.json`
  - modified: `harness/progress.md`
  - modified: `harness/session-handoff.md`

## 当前已验证状态

- `./init.sh` 通过，并打印新的 Phase 1 / Phase 2 当前阶段提示。
- `./scripts/harness-check.sh` 通过，0 warnings。
- `git diff --check` 通过。
- `python3 -m json.tool harness/feature_list.json >/dev/null` 通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python evals/scripts/diagnose_representations.py --help` 通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python evals/scripts/export_neural_codec.py --help` 通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help` 通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest tests.test_evals_scripts -v` 通过，11 tests OK。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v` 通过，30 tests OK。

## 本会话改动

- 新增 `evals/scripts/diagnose_representations.py`：
  - 输入 Phase 1 export 目录；
  - 默认读取 `manifest.jsonl` 中的 `latent_path`、`quantized_path`、`codes_path`；
  - 输出 `diagnostics/diagnostics.jsonl` 和 `diagnostics/summary.json`。
- 连续表示 `latent / quantized` 使用 past-window mean prediction，报告：
  - `mse`；
  - `variance`；
  - `normalized_mse`；
  - `predictability_score`。
- 离散 `codes` 使用 window reuse proxy，报告：
  - `window_reuse_rate`；
  - `previous_frame_match_rate`；
  - `marginal_entropy_bits_per_code`；
  - `distinct_codes_per_stage_mean`。
- 默认 context scopes：
  - `local = 1s`；
  - `medium = 5s`；
  - `long = 30s`；
  - `full_utterance = all past frames`。
- Summary 按 representation / context scope 聚合，并给出 long/full 相比 local 的 `>=5%` gate recommendation。
- `tests/test_evals_scripts.py` 新增 synthetic latent / quantized / codes diagnostics test，覆盖 `diagnostics.jsonl`、`summary.json` 和 gate 输出。
- `evals/README.md` 新增 Phase 2 diagnostics 命令，并明确 proxy metrics 不等于 Phase 3 learned code-prior entropy。
- Active spec、feature list、progress 和 `init.sh` 已更新为 Phase 1 export 与 Phase 2 diagnostics 工具均已实现。

## 本会话决策

- 当前不等待 Linux 训练机；继续开发本地可验证工具。
- Phase 2 diagnostics 只做 frozen representation proxy，不训练 codec、不训练 prior、不引入 Mamba。
- Continuous proxy 只用于前置筛查；真正 entropy bitrate 仍要等 Phase 3 code-prior baseline。
- 后续 bash 脚本应在 Phase 3/4 工具链稳定后再写，统一串起 export、diagnostics、prior 和训练，避免过早固化还在变化的命令。

## 仍损坏或未验证

- 未用真实 4kbps checkpoint 和可访问的 long/full utterance manifest 跑完整 export+diagnostics；当前 diagnostics test 使用 synthetic tensors。
- 未在本轮运行真实训练 smoke。
- 未在 Linux `4 x A100` 训练机重新验证 smoke。
- Mamba/SSM 依赖尚未固定。
- macOS 本地完整单测仍依赖 `KMP_DUPLICATE_LIB_OK=TRUE` workaround；这不应进入 Linux 训练命令。

## 清洁状态

- 本轮最终验证已完成。
- 临时工件：本轮未创建模型输出、训练日志、下载缓存或调试脚本。

## 下一步最佳动作

若本机有可用的真实 4kbps checkpoint 和 long/full utterance manifest，先串一次 Phase 1 export + Phase 2 diagnostics sanity；否则继续 Phase 3 code-prior entropy baseline 的本地开发，先实现 unigram / previous-frame prior，保持输入为 Phase 1 export 的 frozen `codes.pt`，不要训练 codec，也不要引入 Mamba。

## 命令

- 初始化：`./init.sh`
- Harness 检查：`./scripts/harness-check.sh`
- 聚焦验证：`git diff --check`
- Export CLI sanity：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python evals/scripts/export_neural_codec.py --help`
- Diagnostics CLI sanity：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python evals/scripts/diagnose_representations.py --help`
- Focused tests：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest tests.test_evals_scripts -v`
- CLI sanity：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help`
- 完整验证：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v`
- Linux A100 smoke：`PYTHONPATH=src python scripts/train_codec.py --config configs/ablation-adversarial-msstft-balanced.json --output-dir artifacts/smoke-b0-12k --smoke-test --limit-train-examples 10 --device cuda`
