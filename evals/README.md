Owner: ely
Status: active
Last reviewed: 2026-05-06

# Evals

本目录用于承载与模型主训练代码分离的 benchmark、传统 codec 基线、结果导出与对比分析脚本。

设计原则：

- `src/` 只放模型、训练与推理主逻辑
- `evals/` 只放评测相关代码与输出
- 传统 codec 调用逻辑不直接耦合进主训练入口

建议结构：

```text
evals/
  traditional_codecs/
    mp3/
  scripts/
  outputs/
```

当前推荐工作流：

1. 用 `evals/scripts/build_manifest.py` 固定 benchmark 样本集
2. 用 `evals/scripts/export_neural_codec.py` 导出 neural codec 重建结果
3. 用 `evals/scripts/run_traditional_codec.py` 生成 `MP3 / Opus / AAC / FLAC` 重建结果
   - 支持 `bitrate` mode：统一 target bitrate sweep
   - 支持 `default` mode：不人工指定码率，观察 codec-native operating point
4. 用 `evals/scripts/score_outputs.py` 统一计算压缩率和重建质量指标

长程时间冗余 Phase 1 representation export 在 neural export 基础上增加：

```bash
PYTHONPATH=src python evals/scripts/export_neural_codec.py \
  --manifest evals/data/manifests/test.jsonl \
  --checkpoint /path/to/checkpoint.pt \
  --output-dir evals/outputs/context-modeling/neural-4k-export \
  --codec-label neural-4k \
  --device auto \
  --save-representations \
  --context-scope full_utterance \
  --is-full-utterance
```

该模式会在 `manifest.jsonl` 中记录 `latent_path`、`quantized_path`、`codes_path`、`context_scope`、`context_window_seconds`、`context_window_frames`、`frame_rate`、`hop_length`、`nominal_bitrate_kbps` 和 `rvq_payload_bits` 等字段。短样本 sanity 不能写成长程结论。

Phase 2 frozen representation diagnostics 消费上述 export 目录：

```bash
PYTHONPATH=src python evals/scripts/diagnose_representations.py \
  --export-dir evals/outputs/context-modeling/neural-4k-export \
  --representations latent quantized codes
```

该脚本输出 `diagnostics/diagnostics.jsonl` 和 `diagnostics/summary.json`。连续表示使用 past-window mean prediction 的 `normalized_mse` / `predictability_score`，离散 codes 使用 `window_reuse_rate`、`previous_frame_match_rate` 和 `marginal_entropy_bits_per_code`。这些是 Phase 2 proxy diagnostics，不等于 Phase 3 learned code-prior entropy。

Phase 3 code-prior entropy baseline 只消费 frozen RVQ codes，不改 reconstruction codec：

```bash
PYTHONPATH=src python evals/scripts/evaluate_code_priors.py \
  --export-dir evals/outputs/context-modeling/neural-4k-export \
  --priors unigram previous_frame
```

该脚本输出 `code_priors/train_metrics.jsonl`、`code_priors/val_metrics.jsonl`、`code_priors/summary.json` 和 `code_priors/config.json`。当前已实现 analytic `unigram` 和 `previous_frame` baselines，报告 `stage_bits_per_code`、`bits_per_code`、`estimated_entropy_bitrate_kbps` 和 `entropy_savings_ratio`，并记录 `time_major_frame_stage_coarse_to_fine` token ordering。`local_tcn` 和 `long_transformer` 由训练脚本负责，`mamba` 仍不进入主线。

训练型 code prior 同样只消费 frozen RVQ codes：

```bash
PYTHONPATH=src python evals/scripts/train_code_prior.py \
  --export-dir evals/outputs/context-modeling/neural-4k-export \
  --prior local_tcn \
  --output-dir evals/outputs/context-modeling/priors/local-tcn \
  --steps 1000 \
  --sequence-length 512 \
  --batch-size 8
```

将 `--prior` 改为 `long_transformer` 可跑长窗 Transformer prior。该脚本写出 `train_metrics.jsonl`、`val_metrics.jsonl`、`summary.json`、`config.json` 和 `checkpoint.pt`；它只报告 entropy / token metrics，不报告 reconstruction fidelity。

结果聚合脚本读取 pipeline 输出并生成统一结果表：

```bash
PYTHONPATH=src python evals/scripts/collect_context_results.py \
  --export-dir evals/outputs/context-modeling/neural-4k-export \
  --output-dir evals/outputs/context-modeling/results
```

该脚本输出 `results.jsonl`、`summary.csv` 和 `summary.json`，统一记录 `stage`、`representation`、`prior_family`、`context_scope`、`bits_per_code`、`estimated_entropy_bitrate_kbps`、`entropy_savings_ratio`、`relative_improvement_vs_local_or_unigram` 和 `gate_passed`。`summary.json` 中的 `go_no_go` 用于判断是否进入 codec context training。

完整 pipeline 可用一个 bash 脚本串起 export、diagnostics、analytic prior、trained prior、结果聚合和轻量打包：

```bash
scripts/run-context-prior-pipeline.sh \
  --manifest evals/data/manifests/test.jsonl \
  --checkpoint /path/to/checkpoint.pt \
  --output-root evals/outputs/context-modeling \
  --codec-label neural-4k \
  --train-steps 1000 \
  --train-sequence-length 512 \
  --train-batch-size 8
```

本机没有真实 checkpoint 时可先用 `--dry-run` 验证命令拼装。Pipeline 会在训练后自动调用结果聚合和轻量打包脚本。Linux 训练命令不要写入 macOS 的 `KMP_DUPLICATE_LIB_OK=TRUE` workaround。

训练输出目录会包含 trained prior 的 `checkpoint.pt`、representation tensor 和 reconstruction wav。分析时默认不需要下载这些重文件；pipeline 末尾会自动调用轻量打包脚本：

```bash
scripts/pack-context-results.sh \
  --output-root evals/outputs/context-modeling \
  --export-dir evals/outputs/context-modeling/neural-4k-export
```

默认输出到 `evals/outputs/context-modeling/download-bundles/<timestamp>/`。这个目录只白名单复制 `manifest.jsonl`、`run.json`、diagnostics/code-prior/results 的 `summary.json`、`summary.csv`、`results.jsonl`、`train_metrics.jsonl`、`val_metrics.jsonl` 和 `config.json`，并写入 `BUNDLE_MANIFEST.txt`。它不会复制 `checkpoint.pt`、`*.pt` representation tensor、reconstruction wav 或压缩音频。若训练机下载路径需要固定，可在 pipeline 中使用 `--bundle-dir /path/to/download-bundle`；若只想保留原始输出，可用 `--skip-pack-results`。

完整命令见：

- [run-traditional-codec-benchmark.md](/Users/ely/workspace/research/audio/AudioCodec/docs/archive/course-project/how-to/run-traditional-codec-benchmark.md)

第一版已经支持：

- deterministic manifest
- neural codec export
- `MP3 / Opus / AAC / FLAC` encode-decode baseline
- 汇总：
  - `actual_bitrate_kbps`
  - `compression_ratio_vs_pcm16`
  - `log_spectral_distance`
  - `multi_scale_stft`
  - `si_sdr_db`
  - `stoi`（若环境中安装了 `pystoi`）

说明：

- `ViSQOL` 暂未并入当前默认脚本，因为它通常需要额外外部依赖和单独安装流程
- neural codec 的压缩成本统计使用 `RVQ payload bytes`，而不是 checkpoint 大小
