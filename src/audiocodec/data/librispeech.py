"""LibriSpeech-style dataset discovery and segment sampling."""

from __future__ import annotations

from pathlib import Path
import random

import torch
from torch.utils.data import Dataset

from .splits import AudioExample, DatasetSplits, build_duration_capped_splits, build_overfit_splits


def _load_torchaudio():
    import torchaudio

    return torchaudio


def _get_audio_duration_seconds(path: Path) -> float:
    torchaudio = _load_torchaudio()

    info_fn = getattr(torchaudio, "info", None)
    if callable(info_fn):
        info = info_fn(str(path))
        return info.num_frames / info.sample_rate

    waveform, sample_rate = torchaudio.load(str(path))
    return waveform.shape[-1] / sample_rate


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

        waveform, source_sample_rate = torchaudio.load(str(path))
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
