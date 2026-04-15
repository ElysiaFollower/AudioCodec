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

当前优先事项：

1. 落地 `MP3` baseline
2. 统一 neural codec 与传统 codec 的对比输入输出
3. 汇总压缩率、码率与保真度指标
