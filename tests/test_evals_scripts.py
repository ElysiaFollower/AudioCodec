from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


EVALS_SCRIPTS = Path(__file__).resolve().parents[1] / "evals" / "scripts"
if str(EVALS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EVALS_SCRIPTS))

from _common import bytes_to_kbps, derive_item_id, pcm16_bytes, rvq_payload_bytes
from _metrics import compute_log_spectral_distance, compute_multi_scale_stft, compute_si_sdr_db


class EvalHelpersTest(unittest.TestCase):
    def test_pcm16_bytes(self) -> None:
        self.assertEqual(pcm16_bytes(16000, channels=1), 32000)

    def test_bytes_to_kbps(self) -> None:
        self.assertAlmostEqual(bytes_to_kbps(3000, 2.0), 12.0)

    def test_rvq_payload_bytes(self) -> None:
        self.assertEqual(rvq_payload_bytes(num_frames=100, num_quantizers=24, codebook_size=1024), 3000)

    def test_derive_item_id_uses_relative_path(self) -> None:
        root = Path("/tmp/librispeech")
        path = root / "speaker" / "chapter" / "utt.flac"
        self.assertEqual(derive_item_id(path, dataset_root=root), "speaker__chapter__utt")


class EvalMetricsTest(unittest.TestCase):
    def test_identical_waveforms_have_near_zero_distance(self) -> None:
        waveform = torch.randn(1, 16000)
        self.assertLess(compute_log_spectral_distance(waveform, waveform), 1e-4)
        self.assertLess(compute_multi_scale_stft(waveform, waveform), 1e-6)

    def test_identical_waveforms_have_high_si_sdr(self) -> None:
        waveform = torch.randn(1, 16000)
        self.assertGreater(compute_si_sdr_db(waveform, waveform), 60.0)


if __name__ == "__main__":
    unittest.main()
