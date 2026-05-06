Owner: ely
Status: active
Last reviewed: 2026-05-06

# 会话交接

## 仓库状态

- 分支：`feat/context-modeling`
- 最近提交：
  - `04d3d02 chore: ignore training artifacts`
  - `5ef3d88 feat(evals): include audio pairs in result bundles`
  - `b62b8ae feat(evals): add lightweight result bundle packaging`
  - `1536616 feat(evals): collect context modeling results`
  - `3ccbbd1 chore(evals): add context prior pipeline script`
  - `0120f6e feat(evals): add trained code prior baselines`
  - `ebb6077 feat(evals): add analytic code prior entropy baselines`
  - `6ec80f2 feat(evals): add frozen representation diagnostics`
  - `d242624 feat(evals): export codec representations for diagnostics`
- 本轮目标：修复训练机 PyTorch/CUDA/NCCL 环境定义，避免旧 pip torch 环境在 `import torch` 阶段失败。
- 本轮新增/修改范围：
  - added: `environment-linux-cuda.yaml`
  - modified: `environment.yaml`
  - modified: `README.md`
  - modified: `init.sh`
  - modified: `harness/bootstrap-contract.md`
  - modified: `harness/feature_list.json`
  - modified: `harness/progress.md`
  - modified: `harness/session-handoff.md`

## 当前已验证状态

- `./scripts/harness-check.sh` 通过，0 warnings。
- `git diff --check` 通过。
- `python3 -m json.tool harness/feature_list.json >/dev/null` 通过。
- `python3` 解析 `environment.yaml` 和 `environment-linux-cuda.yaml` YAML 通过。
- 未在本机创建 `audiocodec-cu121`，因为这是 Linux CUDA 训练机环境。

## 本会话改动

- 新增 `environment-linux-cuda.yaml`：
  - env name: `audiocodec-cu121`；
  - channels: `pytorch`、`nvidia`、`conda-forge`；
  - fixed packages: `pytorch=2.5.1`、`torchaudio=2.5.1`、`pytorch-cuda=12.1`；
  - pip section only保留 `tensorboard>=2.16,<3` 和 `-e .`。
- `environment.yaml` 保持跨平台本地开发用途，并加注释说明 Linux CUDA 训练机不要使用它安装 torch。
- README、init、bootstrap contract、feature list 和 progress 已同步环境入口。

## 本会话决策

- Linux A100 训练机使用 `environment-linux-cuda.yaml` 新建干净环境，不在旧 `audiocodec` 环境里原地修补 pip torch。
- `environment.yaml` 不直接改成 CUDA-only，避免破坏 macOS / CPU 本地开发和 harness 验证入口。
- macOS 的 `KMP_DUPLICATE_LIB_OK=TRUE` 仍只用于本地验证，不写入 Linux 训练命令。

## 仍损坏或未验证

- 未在训练机创建并验证 `audiocodec-cu121`。
- 未用真实 4kbps checkpoint 和可访问的 long/full utterance manifest 跑完整 pipeline。
- 未在 Linux A100 训练机重新验证 smoke。
- 未验证真实长音频输出目录的轻量包大小；白名单策略应避免 heavy artifact，但实际大小仍取决于 manifest、metrics 行数和 `--audio-pairs` 数量。
- Mamba/SSM 依赖尚未固定。

## 清洁状态

- 本轮最终验证已完成。
- 未创建需要清理的模型输出、训练日志、下载缓存或调试脚本。
- 如果工作区不是 clean，优先检查本轮列出的文件；不要回退用户未授权的改动。

## 下一步最佳动作

训练机先 `git pull`，然后创建 `audiocodec-cu121`：`conda env create -f environment-linux-cuda.yaml && conda activate audiocodec-cu121`。先运行 `python -c "import torch, torchaudio; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`，通过后再运行 `scripts/run-context-prior-pipeline.sh`。

## 命令

- 初始化：`./init.sh`
- Harness 检查：`./scripts/harness-check.sh`
- 静态检查：`git diff --check`
- Linux CUDA env：`conda env create -f environment-linux-cuda.yaml && conda activate audiocodec-cu121`
- CUDA sanity：`python -c "import torch, torchaudio; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`
- Pack syntax：`bash -n scripts/pack-context-results.sh`
- Pipeline syntax：`bash -n scripts/run-context-prior-pipeline.sh`
- Pack dry-run：`scripts/pack-context-results.sh --output-root /tmp/context-pipeline --export-dir /tmp/custom-export --bundle-dir /tmp/context-bundle --audio-pairs 1 --dry-run`
- Pipeline dry-run：`scripts/run-context-prior-pipeline.sh --manifest evals/data/manifests/test.jsonl --checkpoint /tmp/checkpoint.pt --output-root /tmp/context-pipeline --export-dir /tmp/custom-export --max-items 1 --train-steps 1 --bundle-dir /tmp/context-bundle --audio-pairs 1 --dry-run`
- Focused tests：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest tests.test_evals_scripts -v`
- CLI sanity：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python scripts/train_codec.py --help`
- 完整验证：`conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v`
- Linux A100 smoke：`PYTHONPATH=src python scripts/train_codec.py --config configs/ablation-adversarial-msstft-balanced.json --output-dir artifacts/smoke-b0-12k --smoke-test --limit-train-examples 10 --device cuda`
