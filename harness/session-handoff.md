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
  - `3ccbbd1 chore(evals): add context prior pipeline script`
- 当前待提交目标：实现结果聚合脚本，统一 diagnostics、analytic prior 和 trained prior 输出，并给出 go/no-go。
- 本轮新增/修改范围：
  - added: `evals/scripts/collect_context_results.py`
  - modified: `scripts/run-context-prior-pipeline.sh`
  - modified: `tests/test_evals_scripts.py`
  - modified: `evals/README.md`
  - modified: `init.sh`
  - modified: `plans/active/TASK-008-long-range-redundancy-diagnostics-spec.md`
  - modified: `harness/feature_list.json`
  - modified: `harness/progress.md`
  - modified: `harness/session-handoff.md`

## 当前已验证状态

- `./init.sh` 通过，并打印新的 Phase 1 / Phase 2 / Phase 3 / pipeline / result collection 当前阶段提示。
- `./scripts/harness-check.sh` 通过，0 warnings。
- `git diff --check` 通过。
- `python3 -m json.tool harness/feature_list.json >/dev/null` 通过。
- `python3 -m py_compile evals/scripts/collect_context_results.py` 通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python evals/scripts/collect_context_results.py --help` 通过。
- `scripts/run-context-prior-pipeline.sh --manifest evals/data/manifests/test.jsonl --checkpoint /tmp/checkpoint.pt --output-root /tmp/context-pipeline --max-items 1 --train-steps 1 --dry-run` 通过，并包含 `collect_context_results.py`。
- `scripts/run-context-prior-pipeline.sh --manifest evals/data/manifests/test.jsonl --checkpoint /tmp/checkpoint.pt --output-root /tmp/context-pipeline --export-dir /tmp/custom-export --max-items 1 --train-steps 1 --dry-run` 通过，并确认 collect stage 使用 `--prior-root /tmp/context-pipeline/priors`。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help` 通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest tests.test_evals_scripts -v` 通过，16 tests OK。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v` 通过，35 tests OK。

## 本会话改动

- 新增 `evals/scripts/collect_context_results.py`：
  - 默认读取 `export_dir/diagnostics/summary.json`；
  - 默认读取 `export_dir/code_priors/summary.json`；
  - 默认读取 `export_dir.parent/priors/local-tcn/summary.json` 和 `export_dir.parent/priors/long-transformer/summary.json`；
  - 输出 `results.jsonl`、`summary.csv` 和 `summary.json`。
- 统一结果字段包括：
  - `stage`；
  - `representation`；
  - `prior_family`；
  - `context_scope`；
  - `bits_per_code`；
  - `estimated_entropy_bitrate_kbps`；
  - `entropy_savings_ratio`；
  - `relative_improvement_vs_local_or_unigram`；
  - `gate_passed`。
- `summary.json` 新增 `go_no_go`：
  - diagnostics gate 通过；
  - long/full prior 相比最佳 local/unigram reference 通过；
  - 两者同时满足才建议进入 codec context training。
- `scripts/run-context-prior-pipeline.sh` 已在最后自动调用结果聚合脚本。
- `tests/test_evals_scripts.py` 新增 synthetic summary 测试，覆盖结果表输出和 go/no-go 判定。
- `evals/README.md`、active spec、init 和 harness 已同步为结果聚合已实现。

## 本会话决策

- Go/no-go 不只看 prior：必须同时满足 representation diagnostics 和 code-prior long/full gate，避免只用 code entropy 结果跳到 codec context training。
- Long prior 的 reference 默认是最佳 local prior；若 local prior 缺失，退回 unigram。
- Pipeline 现在可以直接作为真实 4kbps checkpoint 的第一轮程序化 sanity 入口。

## 仍损坏或未验证

- 未用真实 4kbps checkpoint 和可访问的 long/full utterance manifest 跑完整 pipeline；当前只验证 synthetic summary 与 dry-run。
- 未在本轮运行真实训练 smoke。
- 未在 Linux `4 x A100` 训练机重新验证 smoke。
- Mamba/SSM 依赖尚未固定。
- macOS 本地完整单测仍依赖 `KMP_DUPLICATE_LIB_OK=TRUE` workaround；这不应进入 Linux 训练命令。

## 清洁状态

- 本轮最终验证已完成，当前等待提交。
- 临时工件：本轮未创建模型输出、训练日志、下载缓存或调试脚本。

## 下一步最佳动作

当真实 4kbps checkpoint 和 long/full utterance manifest 可访问时，运行 `scripts/run-context-prior-pipeline.sh` 做完整 sanity，然后查看 `results/summary.json` 的 `go_no_go`。如果 Linux 训练机仍不可用，可以先准备真实 manifest / checkpoint 路径配置文档。

## 命令

- 初始化：`./init.sh`
- Harness 检查：`./scripts/harness-check.sh`
- 聚焦验证：`git diff --check`
- Collect CLI sanity：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python evals/scripts/collect_context_results.py --help`
- Pipeline syntax：`bash -n scripts/run-context-prior-pipeline.sh`
- Pipeline dry-run：`scripts/run-context-prior-pipeline.sh --manifest evals/data/manifests/test.jsonl --checkpoint /tmp/checkpoint.pt --output-root /tmp/context-pipeline --max-items 1 --train-steps 1 --dry-run`
- Focused tests：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest tests.test_evals_scripts -v`
- CLI sanity：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help`
- 完整验证：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v`
- Linux A100 smoke：`PYTHONPATH=src python scripts/train_codec.py --config configs/ablation-adversarial-msstft-balanced.json --output-dir artifacts/smoke-b0-12k --smoke-test --limit-train-examples 10 --device cuda`
