"""Pure-Python helpers for deterministic duration-capped splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class AudioExample:
    path: Path
    duration_seconds: float


@dataclass(slots=True)
class DatasetSplits:
    train: list[AudioExample]
    val: list[AudioExample]
    test: list[AudioExample]


def build_overfit_splits(example: AudioExample) -> DatasetSplits:
    """Reuse one example across train/val/test for debugging overfit behavior."""

    return DatasetSplits(train=[example], val=[example], test=[example])


def _take_examples_until_duration(
    examples: list[AudioExample],
    target_seconds: float,
    start_index: int,
) -> tuple[list[AudioExample], int]:
    selected: list[AudioExample] = []
    elapsed = 0.0
    index = start_index
    while index < len(examples) and elapsed < target_seconds:
        example = examples[index]
        selected.append(example)
        elapsed += example.duration_seconds
        index += 1
    if elapsed < target_seconds:
        raise ValueError(
            f"Not enough audio to satisfy split duration target {target_seconds:.1f}s. "
            f"Collected only {elapsed:.1f}s."
        )
    return selected, index


def build_duration_capped_splits(
    examples: Iterable[AudioExample],
    train_minutes: int,
    val_minutes: int,
    test_minutes: int,
) -> DatasetSplits:
    ordered_examples = sorted(examples, key=lambda example: str(example.path))
    index = 0

    train, index = _take_examples_until_duration(ordered_examples, train_minutes * 60, index)
    val, index = _take_examples_until_duration(ordered_examples, val_minutes * 60, index)
    test, _ = _take_examples_until_duration(ordered_examples, test_minutes * 60, index)

    return DatasetSplits(train=train, val=val, test=test)
