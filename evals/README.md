Owner: ely
Status: active
Last reviewed: 2026-04-15

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

完整命令见：

- [run-traditional-codec-benchmark.md](/Users/ely/workspace/research/audio/AudioCodec/docs/how-to/run-traditional-codec-benchmark.md)

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
