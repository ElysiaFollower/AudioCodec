Owner: ely
Status: active
Last reviewed: 2026-05-06

# 会话交接

## 仓库状态

- 分支：`feat/context-modeling`
- 最近提交：
  - `b62b8ae feat(evals): add lightweight result bundle packaging`
  - `1536616 feat(evals): collect context modeling results`
  - `3ccbbd1 chore(evals): add context prior pipeline script`
  - `0120f6e feat(evals): add trained code prior baselines`
  - `ebb6077 feat(evals): add analytic code prior entropy baselines`
  - `6ec80f2 feat(evals): add frozen representation diagnostics`
  - `d242624 feat(evals): export codec representations for diagnostics`
- 本轮目标：训练机开跑前确认 git hygiene，避免训练输出需要 commit 或造成后续同步冲突。
- 本轮新增/修改范围：
  - modified: `.gitignore`
  - modified: `harness/progress.md`
  - modified: `harness/session-handoff.md`

## 当前已验证状态

- `./scripts/harness-check.sh` 通过，0 warnings。
- `git diff --check` 通过。
- `python3 -m json.tool harness/feature_list.json >/dev/null` 通过。
- `bash -n scripts/pack-context-results.sh` 通过。
- `bash -n scripts/run-context-prior-pipeline.sh` 通过。
- `git check-ignore -v` 确认 `/artifacts/`、`evals/outputs/`、`/runs/`、`logs/`、`checkpoints/`、`outputs/`、`data/` 和 download bundle 下的样例文件会被忽略。
- `git ls-files | rg '(^artifacts/|^evals/outputs/|^outputs/|^runs/|^logs/|^checkpoints/|^data/)'` 无输出，表示默认输出目录没有已跟踪文件。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help` 通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest tests.test_evals_scripts -v` 通过，19 tests OK。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v` 通过，38 tests OK。

## 本会话改动

- `.gitignore` 已补充通用训练/评测产物保护：
  - checkpoint/model tensor: `*.pt`、`*.pth`、`*.ckpt`、`*.safetensors`；
  - generated audio/media: `*.wav`、`*.flac`、`*.mp3`、`*.opus`、`*.aac`、`*.m4a`、`*.ogg`；
  - experiment trackers: `tensorboard/`、`wandb/`、`mlruns/`、`lightning_logs/`、`events.out.tfevents*`；
  - result transfer bundles and archives: `download-bundles/`、`*.tar`、`*.tar.gz`、`*.tgz`、`*.zip`。
- 这些规则不影响已跟踪的历史课程 demo wav；如未来确实要提交小型 curated media，需要显式 `git add -f`。

## 本会话决策

- 训练机原始输出目录仍保留完整结果和 checkpoint；轻量下载包只是额外产物，不改变原始目录结构。
- 训练机完成实验后不应 commit 原始输出。用轻量结果包传回分析材料；后续如要固化结果，应在本地从 bundle 摘要整理成小型文档或表格后再提交。
- macOS 的 `KMP_DUPLICATE_LIB_OK=TRUE` 仍只用于本地验证，不写入 Linux 训练命令。

## 仍损坏或未验证

- 未用真实 4kbps checkpoint 和可访问的 long/full utterance manifest 跑完整 pipeline；当前只验证 synthetic output 和 dry-run。
- 未在 Linux `4 x A100` 训练机重新验证 smoke。
- 未验证真实长音频输出目录的轻量包大小；白名单策略应避免 heavy artifact，但实际大小仍取决于 manifest、metrics 行数和 `--audio-pairs` 数量。
- Mamba/SSM 依赖尚未固定。

## 清洁状态

- 本轮最终验证已完成。
- 未创建需要清理的模型输出、训练日志、下载缓存或调试脚本。
- 如果工作区不是 clean，优先检查本轮列出的文件；不要回退用户未授权的改动。

## 下一步最佳动作

把仓库 clone 到 Linux 训练机后，用真实 4kbps checkpoint 和 long/full utterance manifest 运行 `scripts/run-context-prior-pipeline.sh`。训练结束后先下载 `download-bundles/<timestamp>/` 或 `--bundle-dir` 指定的轻量结果目录，查看 `results/summary.json` 的 `go_no_go`，并试听 `audio_pairs/` 中的少量 source/reconstruction 对，再决定是否进入 Phase 4 codec context training。

## 命令

- 初始化：`./init.sh`
- Harness 检查：`./scripts/harness-check.sh`
- 静态检查：`git diff --check`
- Pack syntax：`bash -n scripts/pack-context-results.sh`
- Pipeline syntax：`bash -n scripts/run-context-prior-pipeline.sh`
- Pack dry-run：`scripts/pack-context-results.sh --output-root /tmp/context-pipeline --export-dir /tmp/custom-export --bundle-dir /tmp/context-bundle --audio-pairs 1 --dry-run`
- Pipeline dry-run：`scripts/run-context-prior-pipeline.sh --manifest evals/data/manifests/test.jsonl --checkpoint /tmp/checkpoint.pt --output-root /tmp/context-pipeline --export-dir /tmp/custom-export --max-items 1 --train-steps 1 --bundle-dir /tmp/context-bundle --audio-pairs 1 --dry-run`
- Focused tests：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest tests.test_evals_scripts -v`
- CLI sanity：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help`
- 完整验证：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v`
- Linux A100 smoke：`PYTHONPATH=src python scripts/train_codec.py --config configs/ablation-adversarial-msstft-balanced.json --output-dir artifacts/smoke-b0-12k --smoke-test --limit-train-examples 10 --device cuda`
