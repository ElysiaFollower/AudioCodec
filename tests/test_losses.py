"""Tests for codec loss behavior."""

import unittest

import torch

from audiocodec.losses import CodecLoss


class CodecLossTests(unittest.TestCase):
    def test_codec_loss_reports_zero_mel_when_disabled(self) -> None:
        criterion = CodecLoss(
            sample_rate=16_000,
            waveform_weight=1.0,
            stft_weight=1.0,
            mel_weight=0.0,
            commitment_weight=0.25,
            codebook_weight=1.0,
            fft_sizes=(512,),
            mel_n_mels=80,
            mel_fft_size=1024,
            mel_hop_length=256,
            mel_f_min=0.0,
            mel_f_max=None,
        )

        target = torch.randn(2, 1, 2048)
        reconstruction = torch.randn(2, 1, 2048)
        losses = criterion(
            reconstruction=reconstruction,
            target=target,
            commitment_loss=torch.tensor(0.1),
            codebook_loss=torch.tensor(0.2),
        )

        self.assertIn("mel_loss", losses)
        self.assertEqual(float(losses["mel_loss"].item()), 0.0)


if __name__ == "__main__":
    unittest.main()
