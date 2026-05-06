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
  - `0120f6e feat(evals): add trained code prior baselines`
- 当前待提交目标：实现统一 pipeline 脚本，程式化串起 export、diagnostics、analytic prior 和 trained prior。
- 本轮新增/修改范围：
  - added: `scripts/run-context-prior-pipeline.sh`
  - modified: `tests/test_evals_scripts.py`
  - modified: `evals/README.md`
  - modified: `init.sh`
  - modified: `plans/active/TASK-008-long-range-redundancy-diagnostics-spec.md`
  - modified: `harness/feature_list.json`
  - modified: `harness/progress.md`
  - modified: `harness/session-handoff.md`

## 当前已验证状态

- `./init.sh` 通过，并打印新的 Phase 1 / Phase 2 / Phase 3 / pipeline 当前阶段提示。
- `./scripts/harness-check.sh` 通过，0 warnings。
- `git diff --check` 通过。
- `python3 -m json.tool harness/feature_list.json >/dev/null` 通过。
- `bash -n scripts/run-context-prior-pipeline.sh` 通过。
- `scripts/run-context-prior-pipeline.sh --manifest evals/data/manifests/test.jsonl --checkpoint /tmp/checkpoint.pt --output-root /tmp/context-pipeline --max-items 1 --train-steps 1 --dry-run` 通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help` 通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest tests.test_evals_scripts -v` 通过，15 tests OK。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v` 通过，34 tests OK。

## 本会话改动

- 新增 `scripts/run-context-prior-pipeline.sh`：
  - Phase 1：运行 `evals/scripts/export_neural_codec.py --save-representations`；
  - Phase 2：运行 `evals/scripts/diagnose_representations.py`；
  - Phase 3 analytic：运行 `evals/scripts/evaluate_code_priors.py --priors unigram previous_frame`；
  - Phase 3 trained：分别运行 `evals/scripts/train_code_prior.py --prior local_tcn` 和 `--prior long_transformer`。
- Pipeline 支持 `--dry-run`，可以在没有真实 checkpoint 时验证命令拼装。
- Pipeline 支持 `--skip-export`、`--skip-diagnostics`、`--skip-analytic-prior`、`--skip-trained-priors`。
- Pipeline 支持 `--max-items` 和训练参数：`--train-steps`、`--train-batch-size`、`--train-sequence-length`、`--train-eval-every`、`--train-device`。
- 未把 macOS `KMP_DUPLICATE_LIB_OK=TRUE` 写入 pipeline，避免污染 Linux 训练命令。
- `tests/test_evals_scripts.py` 新增 pipeline dry-run 测试，覆盖五段命令。
- `evals/README.md` 和 active spec 已记录 pipeline 命令和边界。

## 本会话决策

- 当前不等待 Linux 训练机；先保证真实训练前的命令编排可审查、可复用。
- Pipeline 使用当前 Python 环境执行；可通过 `PYTHON_BIN` 覆盖，不强绑 conda。
- Pipeline 默认运行 full utterance export metadata；如果只做 clip sanity，可用 `--not-full-utterance` 或调整 manifest。
- 真实 Linux 训练命令仍应避免 macOS OpenMP workaround。

## 仍损坏或未验证

- 未用真实 4kbps checkpoint 和可访问的 long/full utterance manifest 跑完整 pipeline；当前只验证 dry-run。
- 未在本轮运行真实训练 smoke。
- 未在 Linux `4 x A100` 训练机重新验证 smoke。
- Mamba/SSM 依赖尚未固定。
- macOS 本地完整单测仍依赖 `KMP_DUPLICATE_LIB_OK=TRUE` workaround；这不应进入 Linux 训练命令。

## 清洁状态

- 本轮最终验证已完成，当前等待提交。
- 临时工件：本轮未创建模型输出、训练日志、下载缓存或调试脚本。

## 下一步最佳动作

当真实 4kbps checkpoint 和 long/full utterance manifest 可访问时，运行 `scripts/run-context-prior-pipeline.sh` 做完整 sanity；如果 Linux 训练机仍不可用，下一步可以补结果聚合脚本，把 diagnostics、analytic prior 和 trained prior 的 summary 汇总成统一 results 表。

## 命令

- 初始化：`./init.sh`
- Harness 检查：`./scripts/harness-check.sh`
- 聚焦验证：`git diff --check`
- Pipeline syntax：`bash -n scripts/run-context-prior-pipeline.sh`
- Pipeline dry-run：`scripts/run-context-prior-pipeline.sh --manifest evals/data/manifests/test.jsonl --checkpoint /tmp/checkpoint.pt --output-root /tmp/context-pipeline --max-items 1 --train-steps 1 --dry-run`
- Focused tests：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest tests.test_evals_scripts -v`
- CLI sanity：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help`
- 完整验证：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v`
- Linux A100 smoke：`PYTHONPATH=src python scripts/train_codec.py --config configs/ablation-adversarial-msstft-balanced.json --output-dir artifacts/smoke-b0-12k --smoke-test --limit-train-examples 10 --device cuda`
