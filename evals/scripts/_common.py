from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audiocodec.config import CodecExperimentConfig, load_experiment_config
from audiocodec.data.librispeech import build_librispeech_splits
from audiocodec.data.splits import AudioExample


def _load_torchaudio():
    import torchaudio

    return torchaudio


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | Path) -> list[dict]:
    payload: list[dict] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                payload.append(json.loads(line))
    return payload


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    with output_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def derive_item_id(source_path: str | Path, dataset_root: str | Path | None = None) -> str:
    path = Path(source_path).expanduser().resolve()
    if dataset_root is not None:
        root = Path(dataset_root).expanduser().resolve()
        try:
            relative = path.relative_to(root)
            return relative.with_suffix("").as_posix().replace("/", "__")
        except ValueError:
            pass
    return path.with_suffix("").name


def pcm16_bytes(num_samples: int, channels: int = 1) -> int:
    return int(num_samples) * int(channels) * 2


def bytes_to_kbps(num_bytes: int, duration_seconds: float) -> float:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive.")
    return (float(num_bytes) * 8.0) / float(duration_seconds) / 1000.0


def rvq_payload_bytes(num_frames: int, num_quantizers: int, codebook_size: int) -> int:
    bits_per_code = math.ceil(math.log2(codebook_size))
    total_bits = int(num_frames) * int(num_quantizers) * bits_per_code
    return math.ceil(total_bits / 8)


def load_audio(
    path: str | Path,
    sample_rate: int,
    channels: int,
):
    torchaudio = _load_torchaudio()
    try:
        waveform, source_sample_rate = torchaudio.load(str(path))
        if source_sample_rate != sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_sample_rate, sample_rate)
        if channels == 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.shape[0] != channels:
            raise ValueError(f"Expected {channels} channels, got {waveform.shape[0]} for {path}.")
        return waveform
    except Exception as exc:
        message = str(exc)
        if "TorchCodec" not in message and "torchcodec" not in message:
            raise
        return _load_audio_with_ffmpeg(path, sample_rate=sample_rate, channels=channels)


def _load_audio_with_ffmpeg(path: str | Path, sample_rate: int, channels: int) -> torch.Tensor:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    waveform = torch.frombuffer(bytearray(result.stdout), dtype=torch.float32).clone()
    if waveform.numel() % channels != 0:
        raise ValueError(f"Decoded audio from {path} is not divisible by channel count {channels}.")
    return waveform.view(-1, channels).transpose(0, 1).contiguous()


def save_audio(path: str | Path, waveform, sample_rate: int) -> None:
    torchaudio = _load_torchaudio()
    output_path = Path(path)
    ensure_directory(output_path.parent)
    try:
        torchaudio.save(str(output_path), waveform.detach().float().cpu(), sample_rate)
    except Exception as exc:
        message = str(exc)
        if "TorchCodec" not in message and "torchcodec" not in message:
            raise
        _save_audio_with_ffmpeg(output_path, waveform, sample_rate=sample_rate)


def _save_audio_with_ffmpeg(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    output_path = Path(path)
    audio = waveform.detach().float().cpu().contiguous()
    if audio.ndim != 2:
        raise ValueError("waveform must have shape [channels, samples].")
    channels = audio.shape[0]
    interleaved = audio.transpose(0, 1).contiguous().numpy().tobytes()
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-i",
        "-",
        str(output_path),
    ]
    subprocess.run(command, input=interleaved, check=True)


def resolve_device(device: str):
    import torch

    requested = device.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def run_ffmpeg(args: list[str]) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        *args,
    ]
    subprocess.run(command, check=True)


def load_split_examples(config_path: str | Path, split: str) -> tuple[CodecExperimentConfig, list[AudioExample]]:
    config = load_experiment_config(config_path)
    splits = build_librispeech_splits(
        root=config.dataset.root,
        train_minutes=config.dataset.train_minutes,
        val_minutes=config.dataset.val_minutes,
        test_minutes=config.dataset.test_minutes,
    )
    if split == "train":
        return config, splits.train
    if split == "val":
        return config, splits.val
    if split == "test":
        return config, splits.test
    raise ValueError(f"Unsupported split: {split}")


def examples_to_manifest_rows(
    config: CodecExperimentConfig,
    examples: list[AudioExample],
    limit: int | None = None,
) -> list[dict]:
    dataset_root = Path(config.dataset.root).expanduser().resolve()
    selected = examples[:limit] if limit is not None else examples
    rows: list[dict] = []
    for example in selected:
        num_samples = int(round(example.duration_seconds * config.audio.sample_rate))
        rows.append(
            {
                "id": derive_item_id(example.path, dataset_root=dataset_root),
                "source_path": str(example.path),
                "duration_seconds": example.duration_seconds,
                "num_samples": num_samples,
                "sample_rate": config.audio.sample_rate,
                "channels": config.audio.channels,
                "pcm16_bytes": pcm16_bytes(num_samples, channels=config.audio.channels),
            }
        )
    return rows
