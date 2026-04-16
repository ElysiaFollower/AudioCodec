Owner: ely
Status: active
Last reviewed: 2026-04-16

# Run Traditional Codec Benchmark

本文档固定当前 benchmark 的执行命令，用于对比：

- `Neural-12k`
- `Neural-8k`
- `Neural-4k`
- `Neural-2k`
- `Opus`
- `MP3`
- `AAC`
- `FLAC`

所有结果统一进入 `evals/outputs/`。

## Preconditions

先更新环境，补齐 `pystoi`：

```bash
conda env update -f environment.yaml --prune
conda activate audiocodec
```

确认以下前置资产已经存在：

- neural checkpoints:
  - `artifacts/linux-adversarial-msstft-balanced-100k/checkpoints/best.pt`
  - `artifacts/linux-adversarial-msstft-balanced-8kbps/checkpoints/best.pt`
  - `artifacts/linux-adversarial-msstft-balanced-4kbps/checkpoints/best.pt`
  - `artifacts/linux-adversarial-msstft-balanced-2kbps/checkpoints/best.pt`
- `ffmpeg` 支持：
  - `libmp3lame`
  - `libopus`
  - `aac`
  - `flac`

## Step 1: Build Manifest

先固定完整 test manifest：

```bash
python evals/scripts/build_manifest.py \
  --config configs/ablation-adversarial-msstft-balanced.json \
  --split test \
  --output evals/data/manifests/test.jsonl
```

如需快速试跑，可额外生成小子集：

```bash
python evals/scripts/build_manifest.py \
  --config configs/ablation-adversarial-msstft-balanced.json \
  --split test \
  --limit 50 \
  --output evals/data/manifests/test-50.jsonl
```

以下命令默认使用完整集 `evals/data/manifests/test.jsonl`。

## Step 2: Export Neural Codec Runs

### Neural-12k

```bash
python evals/scripts/export_neural_codec.py \
  --manifest evals/data/manifests/test.jsonl \
  --checkpoint artifacts/linux-adversarial-msstft-balanced-100k/checkpoints/best.pt \
  --output-dir evals/outputs/neural-12k \
  --codec-label neural-12k \
  --device cpu
```

### Neural-8k

```bash
python evals/scripts/export_neural_codec.py \
  --manifest evals/data/manifests/test.jsonl \
  --checkpoint artifacts/linux-adversarial-msstft-balanced-8kbps/checkpoints/best.pt \
  --output-dir evals/outputs/neural-8k \
  --codec-label neural-8k \
  --device cpu
```

### Neural-4k

```bash
python evals/scripts/export_neural_codec.py \
  --manifest evals/data/manifests/test.jsonl \
  --checkpoint artifacts/linux-adversarial-msstft-balanced-4kbps/checkpoints/best.pt \
  --output-dir evals/outputs/neural-4k \
  --codec-label neural-4k \
  --device cpu
```

### Neural-2k

```bash
python evals/scripts/export_neural_codec.py \
  --manifest evals/data/manifests/test.jsonl \
  --checkpoint artifacts/linux-adversarial-msstft-balanced-2kbps/checkpoints/best.pt \
  --output-dir evals/outputs/neural-2k \
  --codec-label neural-2k \
  --device cpu
```

## Step 3: Run FLAC Anchor

`FLAC` 不走目标码率，只做无损锚点：

```bash
python evals/scripts/run_traditional_codec.py \
  --manifest evals/data/manifests/test.jsonl \
  --codec flac \
  --output-dir evals/outputs/flac \
  --codec-label flac
```

## Step 4: Run Traditional Codecs In Bitrate Mode

说明：

- 这里的 `target bitrate` 只是统一控制输入
- 最终分析一律按 `actual bitrate` 和实际质量
- 不要求各 codec 精确命中 target bitrate

### MP3 Bitrate Sweep

```bash
python evals/scripts/run_traditional_codec.py --manifest evals/data/manifests/test.jsonl --codec mp3 --mode bitrate --bitrate-kbps 2  --output-dir evals/outputs/mp3-2k  --codec-label mp3-2k
python evals/scripts/run_traditional_codec.py --manifest evals/data/manifests/test.jsonl --codec mp3 --mode bitrate --bitrate-kbps 4  --output-dir evals/outputs/mp3-4k  --codec-label mp3-4k
python evals/scripts/run_traditional_codec.py --manifest evals/data/manifests/test.jsonl --codec mp3 --mode bitrate --bitrate-kbps 8  --output-dir evals/outputs/mp3-8k  --codec-label mp3-8k
python evals/scripts/run_traditional_codec.py --manifest evals/data/manifests/test.jsonl --codec mp3 --mode bitrate --bitrate-kbps 12 --output-dir evals/outputs/mp3-12k --codec-label mp3-12k
python evals/scripts/run_traditional_codec.py --manifest evals/data/manifests/test.jsonl --codec mp3 --mode bitrate --bitrate-kbps 16 --output-dir evals/outputs/mp3-16k --codec-label mp3-16k
```

### Opus Bitrate Sweep

```bash
python evals/scripts/run_traditional_codec.py --manifest evals/data/manifests/test.jsonl --codec opus --mode bitrate --bitrate-kbps 2  --output-dir evals/outputs/opus-2k  --codec-label opus-2k
python evals/scripts/run_traditional_codec.py --manifest evals/data/manifests/test.jsonl --codec opus --mode bitrate --bitrate-kbps 4  --output-dir evals/outputs/opus-4k  --codec-label opus-4k
python evals/scripts/run_traditional_codec.py --manifest evals/data/manifests/test.jsonl --codec opus --mode bitrate --bitrate-kbps 8  --output-dir evals/outputs/opus-8k  --codec-label opus-8k
python evals/scripts/run_traditional_codec.py --manifest evals/data/manifests/test.jsonl --codec opus --mode bitrate --bitrate-kbps 12 --output-dir evals/outputs/opus-12k --codec-label opus-12k
python evals/scripts/run_traditional_codec.py --manifest evals/data/manifests/test.jsonl --codec opus --mode bitrate --bitrate-kbps 16 --output-dir evals/outputs/opus-16k --codec-label opus-16k
```

### AAC Bitrate Sweep

```bash
python evals/scripts/run_traditional_codec.py --manifest evals/data/manifests/test.jsonl --codec aac --mode bitrate --bitrate-kbps 2  --output-dir evals/outputs/aac-2k  --codec-label aac-2k
python evals/scripts/run_traditional_codec.py --manifest evals/data/manifests/test.jsonl --codec aac --mode bitrate --bitrate-kbps 4  --output-dir evals/outputs/aac-4k  --codec-label aac-4k
python evals/scripts/run_traditional_codec.py --manifest evals/data/manifests/test.jsonl --codec aac --mode bitrate --bitrate-kbps 8  --output-dir evals/outputs/aac-8k  --codec-label aac-8k
python evals/scripts/run_traditional_codec.py --manifest evals/data/manifests/test.jsonl --codec aac --mode bitrate --bitrate-kbps 12 --output-dir evals/outputs/aac-12k --codec-label aac-12k
python evals/scripts/run_traditional_codec.py --manifest evals/data/manifests/test.jsonl --codec aac --mode bitrate --bitrate-kbps 16 --output-dir evals/outputs/aac-16k --codec-label aac-16k
```

## Step 5: Run Traditional Codecs In Default Mode

这些点用于观察各 codec 的自然 operating point，不人工指定码率。

### MP3 Default

```bash
python evals/scripts/run_traditional_codec.py \
  --manifest evals/data/manifests/test.jsonl \
  --codec mp3 \
  --mode default \
  --output-dir evals/outputs/mp3-default \
  --codec-label mp3-default
```

### Opus Default

```bash
python evals/scripts/run_traditional_codec.py \
  --manifest evals/data/manifests/test.jsonl \
  --codec opus \
  --mode default \
  --output-dir evals/outputs/opus-default \
  --codec-label opus-default
```

### AAC Default

```bash
python evals/scripts/run_traditional_codec.py \
  --manifest evals/data/manifests/test.jsonl \
  --codec aac \
  --mode default \
  --output-dir evals/outputs/aac-default \
  --codec-label aac-default
```

## Step 6: Score All Runs

```bash
python evals/scripts/score_outputs.py \
  --run-manifest evals/outputs/neural-12k/manifest.jsonl \
  --run-manifest evals/outputs/neural-8k/manifest.jsonl \
  --run-manifest evals/outputs/neural-4k/manifest.jsonl \
  --run-manifest evals/outputs/neural-2k/manifest.jsonl \
  --run-manifest evals/outputs/flac/manifest.jsonl \
  --run-manifest evals/outputs/mp3-2k/manifest.jsonl \
  --run-manifest evals/outputs/mp3-4k/manifest.jsonl \
  --run-manifest evals/outputs/mp3-8k/manifest.jsonl \
  --run-manifest evals/outputs/mp3-12k/manifest.jsonl \
  --run-manifest evals/outputs/mp3-16k/manifest.jsonl \
  --run-manifest evals/outputs/mp3-default/manifest.jsonl \
  --run-manifest evals/outputs/opus-2k/manifest.jsonl \
  --run-manifest evals/outputs/opus-4k/manifest.jsonl \
  --run-manifest evals/outputs/opus-8k/manifest.jsonl \
  --run-manifest evals/outputs/opus-12k/manifest.jsonl \
  --run-manifest evals/outputs/opus-16k/manifest.jsonl \
  --run-manifest evals/outputs/opus-default/manifest.jsonl \
  --run-manifest evals/outputs/aac-2k/manifest.jsonl \
  --run-manifest evals/outputs/aac-4k/manifest.jsonl \
  --run-manifest evals/outputs/aac-8k/manifest.jsonl \
  --run-manifest evals/outputs/aac-12k/manifest.jsonl \
  --run-manifest evals/outputs/aac-16k/manifest.jsonl \
  --run-manifest evals/outputs/aac-default/manifest.jsonl \
  --output-dir evals/outputs/scored-test
```

## Outputs

关键输出位于：

- `evals/outputs/scored-test/per_file_metrics.jsonl`
- `evals/outputs/scored-test/summary.csv`
- `evals/outputs/scored-test/summary.json`

## Notes

- `FLAC` 是无损锚点，不参与低码率 lossy 胜负叙事
- `Opus` 是 speech 低码率下的主传统对手
- `MP3` 是课程要求中的指定经典 baseline
- `AAC` 当前默认使用 `ffmpeg -c:a aac`
- 若某 codec 在极低 target bitrate 下明显不可用：
  - 可以保留该点作为 RD 曲线边缘点
  - 但不应把它作为 headline same-quality 结论的主证据
