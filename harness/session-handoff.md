Owner: ely
Status: active
Last reviewed: 2026-05-06

# 会话交接

## 仓库状态

- 分支：`feat/context-modeling`
- 最近提交：
  - `1536616 feat(evals): collect context modeling results`
  - `3ccbbd1 chore(evals): add context prior pipeline script`
  - `0120f6e feat(evals): add trained code prior baselines`
  - `ebb6077 feat(evals): add analytic code prior entropy baselines`
  - `6ec80f2 feat(evals): add frozen representation diagnostics`
  - `d242624 feat(evals): export codec representations for diagnostics`
  - `bf45862 docs: define long-range redundancy implementation spec`
- 本轮目标：新增轻量结果包脚本，并让一键 context prior pipeline 在训练、聚合结束后默认调用它。
- 本轮新增/修改范围：
  - added: `scripts/pack-context-results.sh`
  - modified: `scripts/run-context-prior-pipeline.sh`
  - modified: `tests/test_evals_scripts.py`
  - modified: `README.md`
  - modified: `docs/overview.md`
  - modified: `evals/README.md`
  - modified: `init.sh`
  - modified: `plans/active/TASK-008-long-range-redundancy-diagnostics-spec.md`
  - modified: `harness/bootstrap-contract.md`
  - modified: `harness/feature_list.json`
  - modified: `harness/progress.md`
  - modified: `harness/quality.md`
  - modified: `harness/session-handoff.md`

## 当前已验证状态

- `./init.sh` 通过，并打印轻量结果包当前阶段提示。
- `./scripts/harness-check.sh` 通过，0 warnings。
- `git diff --check` 通过。
- `python3 -m json.tool harness/feature_list.json >/dev/null` 通过。
- `bash -n scripts/pack-context-results.sh` 通过。
- `bash -n scripts/run-context-prior-pipeline.sh` 通过。
- `scripts/pack-context-results.sh --output-root /tmp/context-pipeline --export-dir /tmp/custom-export --bundle-dir /tmp/context-bundle --dry-run` 通过；无真实输出时只报告 missing，不创建 bundle。
- `scripts/run-context-prior-pipeline.sh --manifest evals/data/manifests/test.jsonl --checkpoint /tmp/checkpoint.pt --output-root /tmp/context-pipeline --export-dir /tmp/custom-export --max-items 1 --train-steps 1 --bundle-dir /tmp/context-bundle --dry-run` 通过，并包含 pack stage。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help` 通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest tests.test_evals_scripts -v` 通过，18 tests OK。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v` 通过，37 tests OK。

## 本会话改动

- 新增 `scripts/pack-context-results.sh`：
  - 默认读取 `evals/outputs/context-modeling`；
  - 默认打包到 `evals/outputs/context-modeling/download-bundles/<timestamp>/`；
  - 可用 `--bundle-dir` 固定训练机下载路径；
  - 写入 `BUNDLE_MANIFEST.txt`，记录来源目录、git commit、复制项、missing 项和排除范围。
- 打包脚本采用白名单复制：
  - export root: `manifest.jsonl`、`run.json`；
  - diagnostics: `summary.json`、`diagnostics.jsonl`；
  - analytic/trained code prior: `summary.json`、`config.json`、`train_metrics.jsonl`、`val_metrics.jsonl`；
  - results: `results.jsonl`、`summary.csv`、`summary.json`。
- 明确不复制：
  - trained prior `checkpoint.pt`；
  - `representations/*.pt`；
  - `reconstructions/*.wav`；
  - compressed audio outputs。
- `scripts/run-context-prior-pipeline.sh` 现在在 collect stage 后默认调用打包脚本，并支持：
  - `--skip-pack-results`
  - `--bundle-dir`
  - `--bundle-root`
  - `--bundle-run-id`
- `tests/test_evals_scripts.py` 新增轻量打包测试，覆盖实际复制和 dry-run 不落盘。
- README、overview、evals README、active spec、init 和 harness 已同步为“pipeline + result collection + lightweight bundle”阶段。

## 本会话决策

- 训练机原始输出目录仍保留完整结果和 checkpoint；轻量下载包只是额外产物，不改变原始目录结构。
- 默认分析只下载轻量包；除非要复现实验或继续训练，不默认下载 prior checkpoint、representation tensor 或 reconstruction wav。
- Pipeline 的 pack stage 可以被 `--skip-pack-results` 关闭，方便只跑原始实验输出。
- macOS 的 `KMP_DUPLICATE_LIB_OK=TRUE` 仍只用于本地验证，不写入 Linux 训练命令。

## 仍损坏或未验证

- 未用真实 4kbps checkpoint 和可访问的 long/full utterance manifest 跑完整 pipeline；当前只验证 synthetic output 和 dry-run。
- 未在 Linux `4 x A100` 训练机重新验证 smoke。
- 未验证真实长音频输出目录的轻量包大小；白名单策略应避免 heavy artifact，但实际大小仍取决于 manifest 和 metrics 行数。
- Mamba/SSM 依赖尚未固定。

## 清洁状态

- 本轮最终验证已完成。
- 未创建需要清理的模型输出、训练日志、下载缓存或调试脚本。
- 如果工作区不是 clean，优先检查本轮列出的文件；不要回退用户未授权的改动。

## 下一步最佳动作

把仓库 clone 到 Linux 训练机后，用真实 4kbps checkpoint 和 long/full utterance manifest 运行 `scripts/run-context-prior-pipeline.sh`。训练结束后先下载 `download-bundles/<timestamp>/` 或 `--bundle-dir` 指定的轻量结果目录，查看 `results/summary.json` 的 `go_no_go`，再决定是否进入 Phase 4 codec context training。

## 命令

- 初始化：`./init.sh`
- Harness 检查：`./scripts/harness-check.sh`
- 静态检查：`git diff --check`
- Pack syntax：`bash -n scripts/pack-context-results.sh`
- Pipeline syntax：`bash -n scripts/run-context-prior-pipeline.sh`
- Pack dry-run：`scripts/pack-context-results.sh --output-root /tmp/context-pipeline --export-dir /tmp/custom-export --bundle-dir /tmp/context-bundle --dry-run`
- Pipeline dry-run：`scripts/run-context-prior-pipeline.sh --manifest evals/data/manifests/test.jsonl --checkpoint /tmp/checkpoint.pt --output-root /tmp/context-pipeline --export-dir /tmp/custom-export --max-items 1 --train-steps 1 --bundle-dir /tmp/context-bundle --dry-run`
- Focused tests：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest tests.test_evals_scripts -v`
- CLI sanity：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help`
- 完整验证：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v`
- Linux A100 smoke：`PYTHONPATH=src python scripts/train_codec.py --config configs/ablation-adversarial-msstft-balanced.json --output-dir artifacts/smoke-b0-12k --smoke-test --limit-train-examples 10 --device cuda`
