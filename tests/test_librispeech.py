"""Tests for LibriSpeech dataset audio backend fallbacks."""

from __future__ import annotations

from pathlib import Path
import subprocess
import struct
import unittest
from unittest.mock import patch

from audiocodec.data import librispeech
from audiocodec.data.librispeech import SpeechSegmentDataset
from audiocodec.data.splits import AudioExample


class _FailingTorchaudio:
    @staticmethod
    def info(path: str):
        raise RuntimeError(f"Couldn't find appropriate backend to handle uri {path}")

    @staticmethod
    def load(path: str):
        raise RuntimeError(f"Couldn't find appropriate backend to handle uri {path}")


class LibriSpeechAudioFallbackTests(unittest.TestCase):
    def test_duration_falls_back_to_ffprobe_when_torchaudio_backend_is_missing(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"format": {"duration": "1.25"}}',
            stderr="",
        )

        with (
            patch.object(librispeech, "_load_torchaudio", return_value=_FailingTorchaudio),
            patch.object(librispeech.subprocess, "run", return_value=completed) as run,
        ):
            duration_seconds = librispeech._get_audio_duration_seconds(Path("sample.flac"))

        self.assertEqual(duration_seconds, 1.25)
        command = run.call_args.args[0]
        self.assertEqual(command[0], "ffprobe")
        self.assertIn("sample.flac", command)

    def test_dataset_load_falls_back_to_ffmpeg_when_torchaudio_backend_is_missing(self) -> None:
        samples = struct.pack("<4f", 0.0, 0.25, -0.25, 0.5)
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=samples, stderr=b"")
        dataset = SpeechSegmentDataset(
            examples=[AudioExample(path=Path("sample.flac"), duration_seconds=1.0)],
            sample_rate=16000,
            channels=1,
            clip_seconds=None,
            random_crop=False,
        )

        with (
            patch.object(librispeech, "_load_torchaudio", return_value=_FailingTorchaudio),
            patch.object(librispeech.subprocess, "run", return_value=completed) as run,
        ):
            waveform = dataset[0]

        self.assertEqual(tuple(waveform.shape), (1, 4))
        self.assertAlmostEqual(float(waveform[0, 1]), 0.25)
        command = run.call_args.args[0]
        self.assertEqual(command[0], "ffmpeg")
        self.assertIn("sample.flac", command)


if __name__ == "__main__":
    unittest.main()
