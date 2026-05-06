#!/usr/bin/env sh
# 职责：初始化本地项目 harness，并打印最便宜且可靠的继续工作路径。
# 边界：不要安装全局工具、写入密钥、启动长运行训练，或意外修改项目源码。

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$repo_root"

echo "项目：AudioCodec speech neural codec research workspace"
echo "当前阶段：Phase 1 export、Phase 2 diagnostics、Phase 3 code-prior、pipeline、结果聚合与含试听对的轻量结果包已实现，下一步用真实 checkpoint/manifest 在训练机跑 sanity"
echo "技术栈：Python 3.11, PyTorch, torchaudio, SEANet-style encoder/decoder, EMA RVQ, evals scripts"
echo

if [ -x "./scripts/harness-check.sh" ]; then
  ./scripts/harness-check.sh
else
  echo "缺少可执行文件 scripts/harness-check.sh"
fi

cat <<'EOF'

建议先读：
- docs/idea.md
- plans/active/TASK-008-long-range-redundancy-diagnostics-spec.md
- harness/feature_list.json
- harness/session-handoff.md

环境安装/更新：
conda env update -f environment.yaml --prune
conda activate audiocodec

启动 / sanity：
PYTHONPATH=src python scripts/train_codec.py --help

聚焦验证：
./scripts/harness-check.sh
git diff --check
bash -n scripts/pack-context-results.sh
PYTHONPATH=src python scripts/train_codec.py --help

完整验证：
conda run -n audiocodec env PYTHONPATH=src KMP_DUPLICATE_LIB_OK=TRUE python -m unittest discover -s tests -v

Linux A100 smoke：
PYTHONPATH=src python scripts/train_codec.py --config configs/ablation-adversarial-msstft-balanced.json --output-dir artifacts/smoke-b0-12k --smoke-test --limit-train-examples 10 --device cuda
EOF
