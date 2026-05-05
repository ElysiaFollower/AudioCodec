Owner: ely
Status: active
Last reviewed: 2026-05-05

# 启动上下文序列建模研究

## 1. 第一天不要做的事

不要先做：

- 重写整个 codec 仓库。
- 直接改 waveform front-end。
- 直接上 Mamba-3 新实现并阻塞在 CUDA/Triton 编译。
- 同时引入 semantic loss、mel/STFT front-end、DDP 和新数据集。
- 只跑 Mamba，不跑 Transformer/TCN/LSTM 对照。

第一天目标是让实验变量可控。

## 2. 推荐分支策略

在当前 `AudioCodec` 仓库新开分支推进：

```bash
git checkout -b research/context-sequence-modeling
```

理由：

- 当前仓库已有稳定训练、导出、benchmark 闭环。
- `MambaCodec` 更像脚手架和第三方源码收集区，不适合作为第一版实验基底。

## 3. 开发机与训练机边界

当前默认工作流：

- macOS 开发机：用于文档、代码编辑、轻量单元测试、shape test 和小规模 smoke。
- Linux 训练机：将仓库 clone 到拥有 `4 x A100` 的机器上，执行正式训练、benchmark export、长上下文和效率实验。

因此：

- macOS 上的 OpenMP / torch / torchaudio 兼容问题只影响本地验证便利性，不应被记录成训练实验阻塞。
- 训练结果、step time、显存、RTF 和长音频效率必须以 Linux A100 环境为准。
- 第一阶段优先把 4 张 A100 当作 4 个独立单卡实验并行使用，而不是先改 DDP。

## 4. 环境检查

```bash
conda activate audiocodec
PYTHONPATH=src python scripts/train_codec.py --help
PYTHONPATH=src python -m unittest discover -s tests -v
```

如果要接 Mamba，优先验证稳定 fallback：

```bash
python - <<'PY'
try:
    from mamba_ssm import Mamba
    print("mamba_ssm available", Mamba)
except Exception as exc:
    print("mamba_ssm unavailable", repr(exc))
PY
```

不要让 Mamba-3 编译风险阻塞主实验。可以先用 Mamba/Mamba2 或临时 identity/fallback 接口把实验框架跑通。

## 5. Day 1 implementation checklist

### Step 1: 冻结 baseline

确认这些配置仍是 anchor：

- `configs/ablation-adversarial-msstft-balanced.json`
- `configs/ablation-adversarial-msstft-balanced-8kbps.json`
- `configs/ablation-adversarial-msstft-balanced-4kbps.json`
- `configs/ablation-adversarial-msstft-balanced-2kbps.json`

先跑 smoke：

```bash
PYTHONPATH=src python scripts/train_codec.py \
  --config configs/ablation-adversarial-msstft-balanced.json \
  --output-dir artifacts/smoke-b0-12k \
  --smoke-test \
  --limit-train-examples 10 \
  --device auto
```

### Step 2: 实现 TemporalMixer 配置

新增配置字段时保持向后兼容，默认行为必须等价当前 baseline：

```text
model.temporal_mixer_kind = "lstm"
model.temporal_mixer_insertion = "bottleneck"
model.post_rvq_refiner_kind = "none"
```

第一版只要求：

- `none`
- `lstm`
- `mamba` if dependency available

随后再加：

- `tcn`
- `transformer`

### Step 3: 先跑 B1

创建 no-mixer config，不要先调 Mamba：

```text
seanet_lstm_layers = 0
```

目的：

- 估计当前 `SkipLSTM` 的真实贡献。
- 如果 no-mixer 几乎不掉，Mamba 替换 bottleneck 的收益空间有限。

### Step 4: 接 M1 latent Mamba

接口要求：

```text
input/output: [B, C, T]
internal: [B, T, C]
```

只做 shape、dtype、forward/backward smoke，不直接长跑。

### Step 5: 接 R-MA post-RVQ refiner

位置：

```text
rvq_output.quantized -> refiner -> decoder
```

必须确认：

- `rvq_output.codes` 不变。
- `decode(codes)` 使用同一个 refiner。
- 没有额外 side information 进入 bitstream。

## 6. 第一张实验表

第一周目标是得到这张表，不是追求最终最优：

| run_id | insertion | model | bitrate | steps | params | msstft | lsd | stoi | step_time | notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| B0-4k | baseline | LSTM | 4 | 20k | | | | | | |
| B1-4k | bottleneck | none | 4 | 20k | | | | | | |
| L-MA-4k | latent | Mamba | 4 | 20k | | | | | | |
| R-MA-4k | post-RVQ | Mamba | 4 | 20k | | | | | | |
| B0-12k | baseline | LSTM | 12 | 20k | | | | | | |
| L-MA-12k | latent | Mamba | 12 | 20k | | | | | | |
| R-MA-12k | post-RVQ | Mamba | 12 | 20k | | | | | | |

之后再补 TCN/Transformer matched baseline。

## 7. 记录纪律

每次实验必须保存：

- resolved config。
- metrics JSONL。
- checkpoint。
- exported benchmark manifest。
- summary CSV/JSON。
- run table row。

如果某个实验没有 matched baseline，结论只能写成观察，不得写成证明。

## 8. 推荐下一步代码改动顺序

1. `src/audiocodec/config.py`: 增加 temporal/refiner 配置字段。
2. `src/audiocodec/models/temporal.py`: 新增 `TemporalMixer`。
3. `src/audiocodec/models/seanet.py`: 让 `SkipLSTM` 替换为 mixer factory。
4. `src/audiocodec/models/codec.py`: 加 post-RVQ refiner 路径。
5. `tests/test_codec_model.py`: 加 none/mamba/refiner shape round trip。
6. `evals/scripts/`: 增加 code dump 和 entropy/code usage 脚本。
