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
- 本轮目标：让轻量结果包保留少量 source/reconstruction 试听对，同时继续避免下载包带上整批重文件。
- 本轮新增/修改范围：
  - modified: `scripts/pack-context-results.sh`
  - modified: `scripts/run-context-prior-pipeline.sh`
  - modified: `tests/test_evals_scripts.py`
  - modified: `README.md`
  - modified: `docs/overview.md`
  - modified: `evals/README.md`
  - modified: `init.sh`
  - modified: `plans/active/TASK-008-long-range-redundancy-diagnostics-spec.md`
  - modified: `harness/feature_list.json`
  - modified: `harness/progress.md`
  - modified: `harness/session-handoff.md`

## 当前已验证状态

- `./scripts/harness-check.sh` 通过，0 warnings。
- `git diff --check` 通过。
- `python3 -m json.tool harness/feature_list.json >/dev/null` 通过。
- `bash -n scripts/pack-context-results.sh` 通过。
- `bash -n scripts/run-context-prior-pipeline.sh` 通过。
- `scripts/run-context-prior-pipeline.sh --manifest evals/data/manifests/test.jsonl --checkpoint /tmp/checkpoint.pt --output-root /tmp/context-pipeline --export-dir /tmp/custom-export --max-items 1 --train-steps 1 --bundle-dir /tmp/context-bundle --audio-pairs 1 --dry-run` 通过，并包含 pack stage 的 `--audio-pairs 1`。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help` 通过。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest tests.test_evals_scripts -v` 通过，19 tests OK。
- `conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v` 通过，38 tests OK。

## 本会话改动

- `scripts/pack-context-results.sh` 现在默认从 export `manifest.jsonl` 解析前 3 条 `source_path / reconstruction_path`，复制到 `audio_pairs/001-<id>/source.*` 和 `audio_pairs/001-<id>/reconstruction.*`。
- 新增 pack 参数：
  - `--audio-pairs N`：调整试听对数量；
  - `--skip-audio-pairs`：关闭试听音频复制。
- `scripts/run-context-prior-pipeline.sh` 新增对应透传参数：
  - `--audio-pairs N`；
  - `--skip-audio-pairs`。
- 下载包仍采用白名单：
  - 复制 analysis metadata、summary、metrics、config、results；
  - 复制少量试听音频对；
  - 不复制 trained prior `checkpoint.pt`、`representations/*.pt`、整批 `reconstructions/*.wav` 或 compressed audio outputs。
- `tests/test_evals_scripts.py` 已覆盖默认试听对复制、`--skip-audio-pairs`、dry-run 不落盘和 pipeline dry-run 透传。
- README、overview、evals README、active spec、init 和 harness 已同步为“轻量结果包含试听音频对”。

## 本会话决策

- 训练机原始输出目录仍保留完整结果和 checkpoint；轻量下载包只是额外产物，不改变原始目录结构。
- 默认分析只下载轻量包；除非要复现实验或继续训练，不默认下载 prior checkpoint、representation tensor 或整批 reconstruction wav。
- 试听音频对默认 3 对，作为主观 sanity，不作为正式 benchmark 结论。
- Pipeline 的 pack stage 可以被 `--skip-pack-results` 关闭，方便只跑原始实验输出。
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
