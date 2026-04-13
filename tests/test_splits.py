"""Tests for deterministic duration-capped split helpers."""

from pathlib import Path
import unittest

from audiocodec.data.splits import AudioExample, build_duration_capped_splits


class SplitBuilderTests(unittest.TestCase):
    def test_build_duration_capped_splits_respects_sorted_paths(self) -> None:
        examples = [
            AudioExample(path=Path("b.flac"), duration_seconds=40),
            AudioExample(path=Path("a.flac"), duration_seconds=30),
            AudioExample(path=Path("c.flac"), duration_seconds=50),
            AudioExample(path=Path("d.flac"), duration_seconds=70),
        ]

        splits = build_duration_capped_splits(
            examples=examples,
            train_minutes=1,
            val_minutes=1,
            test_minutes=0,
        )

        self.assertEqual([example.path.name for example in splits.train], ["a.flac", "b.flac"])
        self.assertEqual([example.path.name for example in splits.val], ["c.flac", "d.flac"])

    def test_build_duration_capped_splits_raises_when_audio_is_insufficient(self) -> None:
        examples = [AudioExample(path=Path("a.flac"), duration_seconds=10)]
        with self.assertRaises(ValueError):
            build_duration_capped_splits(examples=examples, train_minutes=1, val_minutes=1, test_minutes=1)


if __name__ == "__main__":
    unittest.main()
