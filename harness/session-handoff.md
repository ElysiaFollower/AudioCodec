Owner: ely
Status: active
Last reviewed: 2026-05-06

# 会话交接

## 仓库状态

- 分支：`feat/context-modeling`
- 最近提交：
  - `847186f chore: add linux cuda environment`
  - `04d3d02 chore: ignore training artifacts`
  - `5ef3d88 feat(evals): include audio pairs in result bundles`
  - `b62b8ae feat(evals): add lightweight result bundle packaging`
  - `1536616 feat(evals): collect context modeling results`
  - `3ccbbd1 chore(evals): add context prior pipeline script`
  - `0120f6e feat(evals): add trained code prior baselines`
  - `ebb6077 feat(evals): add analytic code prior entropy baselines`
- 本轮目标：按用户决策放弃历史 checkpoint/manifest 依赖，把 context prior 一键脚本改成当前分支自包含的实验入口。
- 本轮修改范围：
  - modified: `scripts/run-context-prior-pipeline.sh`
  - modified: `evals/scripts/build_manifest.py`
  - modified: `evals/scripts/_common.py`
  - modified: `tests/test_evals_scripts.py`
  - modified: `README.md`
  - modified: `evals/README.md`
  - modified: `docs/overview.md`
  - modified: `plans/active/TASK-008-long-range-redundancy-diagnostics-spec.md`
  - modified: `init.sh`
  - modified: `harness/bootstrap-contract.md`
  - modified: `harness/decisions.md`
  - modified: `harness/feature_list.json`
  - modified: `harness/progress.md`
  - modified: `harness/session-handoff.md`

## 当前已验证状态

- `./scripts/harness-check.sh` 通过，0 warnings。
- `git diff --check` 通过。
- `python3 -m json.tool harness/feature_list.json >/dev/null` 通过。
- `bash -n scripts/run-context-prior-pipeline.sh` 通过。
- `bash -n scripts/pack-context-results.sh` 通过。
- `scripts/run-context-prior-pipeline.sh --output-root /tmp/context-pipeline --max-items 1 --codec-steps 1 --train-steps 1 --dry-run` 通过，输出顺序包含 manifest build、codec training、export、diagnostics、analytic prior、local/long prior、collect、pack。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help` 通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest tests.test_evals_scripts -v` 通过，20 tests OK。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v` 通过，39 tests OK。

## 本会话改动

- `scripts/run-context-prior-pipeline.sh` 默认变成 self-contained pipeline：
  - 默认 config 为 `configs/ablation-adversarial-msstft-balanced-4kbps.json`；
  - 不传 `--manifest` 时，先构建 `output-root/manifests/test.jsonl`；
  - 不传 `--checkpoint` 时，先训练 baseline codec 到 `output-root/codec-baseline/checkpoints/best.pt`；
  - 后续再运行 representation export、frozen diagnostics、analytic code priors、trained local/long priors、结果聚合和轻量打包。
- 新增 pipeline 训练控制参数：`--codec-output-dir`、`--codec-steps`、`--codec-device`、`--codec-smoke-test`、`--limit-train-examples`、`--resume-codec-from`、`--skip-codec-training`、`--force-codec-training`。
- 新增 manifest 控制参数：`--dataset-root`、`--manifest-split`、`--manifest-limit`、`--skip-manifest-build`。
- `evals/scripts/build_manifest.py` 支持 `--dataset-root`，并通过 `_common.load_split_examples(..., dataset_root=...)` 覆盖 config 中的机器路径。
- `tests/test_evals_scripts.py` 新增默认 self-contained dry-run 覆盖，确认脚本默认拼出 manifest build 和 codec train stage。
- README、evals README、overview、TASK-008、bootstrap contract、decisions、feature list 和 progress 已同步新语义：`--checkpoint` / `--manifest` 只是复现或调试入口，不是默认实验入口。

## 本会话决策

- `feat/context-modeling` 分支必须自包含产出当前实验的 baseline 资产。
- 不再默认依赖历史分支、本地旧 `artifacts/` 或人工传入的 `/path/to/4kbps_checkpoint.pt`。
- 外部 checkpoint/manifest 只能作为显式复现或调试路径；训练机主命令应只需要数据集路径、设备、输出目录和训练参数。

## 仍损坏或未验证

- 未在 Linux A100 上真实运行 self-contained pipeline；本机只验证了 dry-run 和本地单测。
- 未验证 4kbps baseline 从零训练的耗时、收敛质量和最终 checkpoint 是否足以支撑 Phase 2/3 科研结论。
- 未验证真实长音频输出目录的轻量包大小；白名单策略应避免 heavy artifact，但实际大小仍取决于 manifest、metrics 行数和 `--audio-pairs` 数量。
- Mamba/SSM 依赖尚未固定。

## 清洁状态

- 本轮未创建需要清理的模型输出、训练日志或下载缓存。
- 训练输出、logs、download bundles、checkpoint 和 tensor 仍由 `.gitignore` 排除。
- 如果训练机上运行后出现大量 untracked artifact，优先确认是否位于 `evals/outputs/`、`logs/`、`artifacts/` 或其他已忽略目录；不要 commit 原始 checkpoint/tensor。

## 下一步最佳动作

训练机 `git pull` 后，在 `audiocodec-cu121` 环境中运行 self-contained pipeline。先 dry-run，再去掉 `--dry-run` 真跑。真实运行结束后优先下载 `evals/outputs/context-modeling/download-bundles/<timestamp>/`，查看 `results/summary.json` 的 `go_no_go` 和少量试听音频对。

## 命令

- 初始化：`./init.sh`
- Harness 检查：`./scripts/harness-check.sh`
- 静态检查：`git diff --check`
- Linux CUDA env：`conda env create -f environment-linux-cuda.yaml && conda activate audiocodec-cu121`
- CUDA sanity：`python -c "import torch, torchaudio; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`
- Pipeline dry-run：

```bash
CUDA_VISIBLE_DEVICES=4 PYTHON_BIN=python scripts/run-context-prior-pipeline.sh \
  --dataset-root /path/to/LibriSpeech/train-clean-100 \
  --output-root evals/outputs/context-modeling \
  --codec-device cuda \
  --device cuda \
  --train-device cuda \
  --train-steps 1000 \
  --train-sequence-length 512 \
  --train-batch-size 8 \
  --dry-run
```

- Pipeline real run：

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=4 PYTHON_BIN=python scripts/run-context-prior-pipeline.sh \
  --dataset-root /path/to/LibriSpeech/train-clean-100 \
  --output-root evals/outputs/context-modeling \
  --codec-device cuda \
  --device cuda \
  --train-device cuda \
  --train-steps 1000 \
  --train-sequence-length 512 \
  --train-batch-size 8 \
  2>&1 | tee logs/context-modeling-$(date +%Y%m%d-%H%M%S).log
```

- Focused tests：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest tests.test_evals_scripts -v`
- 完整验证：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v`
