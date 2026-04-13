"""Project configuration objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


def _product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


@dataclass(slots=True)
class DatasetConfig:
    root: str = "/path/to/LibriSpeech/dev-clean"
    train_minutes: int = 60
    val_minutes: int = 10
    test_minutes: int = 10


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 16_000
    channels: int = 1
    train_clip_seconds: float = 2.0
    eval_clip_seconds: float = 2.0


@dataclass(slots=True)
class ModelConfig:
    base_channels: int = 32
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 4)
    encoder_strides: tuple[int, ...] = (5, 4, 4, 2)
    latent_dim: int = 128
    residual_layers_per_stage: int = 2
    input_kernel_size: int = 7
    output_kernel_size: int = 7

    def __post_init__(self) -> None:
        self.channel_multipliers = tuple(self.channel_multipliers)
        self.encoder_strides = tuple(self.encoder_strides)
        if len(self.channel_multipliers) != len(self.encoder_strides):
            raise ValueError("channel_multipliers and encoder_strides must have the same length.")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if self.residual_layers_per_stage <= 0:
            raise ValueError("residual_layers_per_stage must be positive.")

    @property
    def stage_channels(self) -> tuple[int, ...]:
        return tuple(self.base_channels * multiplier for multiplier in self.channel_multipliers)

    @property
    def hop_length(self) -> int:
        return _product(self.encoder_strides)


@dataclass(slots=True)
class RVQConfig:
    codebook_size: int = 256
    num_quantizers: int = 8
    commitment_weight: float = 0.25
    codebook_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.codebook_size <= 1:
            raise ValueError("codebook_size must be greater than 1.")
        if self.num_quantizers <= 0:
            raise ValueError("num_quantizers must be positive.")


@dataclass(slots=True)
class LossConfig:
    waveform_weight: float = 1.0
    stft_weight: float = 1.0
    fft_sizes: tuple[int, ...] = (512, 1_024, 2_048)
    mel_weight: float = 0.0
    mel_n_mels: int = 80
    mel_fft_size: int = 1_024
    mel_hop_length: int = 256
    mel_f_min: float = 0.0
    mel_f_max: float | None = None

    def __post_init__(self) -> None:
        self.fft_sizes = tuple(self.fft_sizes)
        if self.mel_weight < 0:
            raise ValueError("mel_weight must be non-negative.")
        if self.mel_n_mels <= 0:
            raise ValueError("mel_n_mels must be positive.")
        if self.mel_fft_size <= 0:
            raise ValueError("mel_fft_size must be positive.")
        if self.mel_hop_length <= 0:
            raise ValueError("mel_hop_length must be positive.")


@dataclass(slots=True)
class OptimizationConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    smoke_test_steps: int = 500
    main_steps: int = 5_000
    mixed_precision: bool = True
    eval_interval: int = 500
    checkpoint_interval: int = 500
    log_interval: int = 50
    num_workers: int = 0
    max_eval_batches: int = 8
    seed: int = 1337


@dataclass(slots=True)
class CodecExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    quantizer: RVQConfig = field(default_factory=RVQConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    @property
    def frame_rate(self) -> int:
        return self.audio.sample_rate // self.model.hop_length

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "CodecExperimentConfig":
        return cls(
            dataset=DatasetConfig(**payload.get("dataset", {})),
            audio=AudioConfig(**payload.get("audio", {})),
            model=ModelConfig(**payload.get("model", {})),
            quantizer=RVQConfig(**payload.get("quantizer", {})),
            loss=LossConfig(**payload.get("loss", {})),
            optimization=OptimizationConfig(**payload.get("optimization", {})),
        )


def load_experiment_config(path: str | Path) -> CodecExperimentConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text())
    return CodecExperimentConfig.from_dict(payload)
