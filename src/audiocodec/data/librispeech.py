"""LibriSpeech-style dataset discovery and segment sampling."""

from __future__ import annotations

import json
from pathlib import Path
import random
import subprocess

import torch
from torch.utils.data import Dataset

from .splits import AudioExample, DatasetSplits, build_duration_capped_splits, build_overfit_splits


def _load_torchaudio():
    import torchaudio

    return torchaudio


def _get_audio_duration_seconds(path: Path) -> float:
    torchaudio = _load_torchaudio()

    errors: list[Exception] = []
    info_fn = getattr(torchaudio, "info", None)
    if callable(info_fn):
        try:
            info = info_fn(str(path))
            return info.num_frames / info.sample_rate
        except Exception as exc:
            errors.append(exc)

    try:
        waveform, sample_rate = torchaudio.load(str(path))
        return waveform.shape[-1] / sample_rate
    except Exception as exc:
        errors.append(exc)

    try:
        return _get_audio_duration_seconds_with_ffprobe(path)
    except Exception as exc:
        errors.append(exc)
        messages = "; ".join(str(error) for error in errors)
        raise RuntimeError(
            f"Could not read audio duration for {path}; tried torchaudio and ffprobe. Errors: {messages}"
        ) from exc


def _get_audio_duration_seconds_with_ffprobe(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    payload = json.loads(result.stdout)
    duration_seconds = float(payload["format"]["duration"])
    if duration_seconds <= 0:
        raise ValueError(f"ffprobe reported non-positive duration for {path}: {duration_seconds}")
    return duration_seconds


def _load_audio_with_ffmpeg(path: Path, sample_rate: int, channels: int) -> torch.Tensor:
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


def discover_librispeech_examples(
    root: str | Path,
    max_duration_seconds: float | None = None,
) -> list[AudioExample]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")

    examples: list[AudioExample] = []
    elapsed = 0.0
    for path in sorted(root_path.rglob("*.flac")):
        duration_seconds = _get_audio_duration_seconds(path)
        examples.append(AudioExample(path=path, duration_seconds=duration_seconds))
        elapsed += duration_seconds
        if max_duration_seconds is not None and elapsed >= max_duration_seconds:
            break

    if not examples:
        raise FileNotFoundError(f"No .flac files were found under {root_path}")
    return examples


def build_librispeech_splits(
    root: str | Path,
    train_minutes: int,
    val_minutes: int,
    test_minutes: int,
) -> DatasetSplits:
    required_seconds = float((train_minutes + val_minutes + test_minutes) * 60)
    examples = discover_librispeech_examples(root, max_duration_seconds=required_seconds)
    return build_duration_capped_splits(
        examples=examples,
        train_minutes=train_minutes,
        val_minutes=val_minutes,
        test_minutes=test_minutes,
    )


def build_single_file_overfit_splits(audio_path: str | Path) -> DatasetSplits:
    path = Path(audio_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Overfit audio path does not exist: {path}")
    example = AudioExample(path=path, duration_seconds=_get_audio_duration_seconds(path))
    return build_overfit_splits(example)


class SpeechSegmentDataset(Dataset):
    """Loads LibriSpeech utterances and returns fixed-length waveform clips."""

    def __init__(
        self,
        examples: list[AudioExample],
        sample_rate: int,
        channels: int,
        clip_seconds: float | None,
        random_crop: bool,
    ) -> None:
        self.examples = examples
        self.sample_rate = sample_rate
        self.channels = channels
        self.clip_seconds = clip_seconds
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.examples)

    def _load_audio(self, path: Path) -> torch.Tensor:
        torchaudio = _load_torchaudio()

        try:
            waveform, source_sample_rate = torchaudio.load(str(path))
        except Exception:
            return _load_audio_with_ffmpeg(path, sample_rate=self.sample_rate, channels=self.channels)

        if source_sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_sample_rate, self.sample_rate)

        if self.channels == 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.shape[0] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, but {path} has {waveform.shape[0]} channels."
            )
        return waveform

    def _crop_or_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.clip_seconds is None:
            return waveform

        clip_samples = int(self.clip_seconds * self.sample_rate)
        waveform_length = waveform.shape[-1]
        if waveform_length < clip_samples:
            pad_amount = clip_samples - waveform_length
            return torch.nn.functional.pad(waveform, (0, pad_amount))
        if waveform_length == clip_samples:
            return waveform

        if self.random_crop:
            start = random.randint(0, waveform_length - clip_samples)
        else:
            start = 0
        return waveform[..., start : start + clip_samples]

    def __getitem__(self, index: int) -> torch.Tensor:
        example = self.examples[index]
        waveform = self._load_audio(example.path)
        return self._crop_or_pad(waveform)
