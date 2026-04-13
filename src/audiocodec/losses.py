"""Losses for training the baseline speech codec."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MultiScaleSTFTLoss(nn.Module):
    def __init__(self, fft_sizes: tuple[int, ...] = (512, 1_024, 2_048), eps: float = 1e-5) -> None:
        super().__init__()
        self.fft_sizes = fft_sizes
        self.eps = eps

    def _magnitude_spectrogram(self, waveform: torch.Tensor, fft_size: int) -> torch.Tensor:
        batch_size, channels, samples = waveform.shape
        flattened = waveform.reshape(batch_size * channels, samples)
        window = torch.hann_window(fft_size, device=waveform.device, dtype=waveform.dtype)
        spectrogram = torch.stft(
            flattened,
            n_fft=fft_size,
            hop_length=fft_size // 4,
            win_length=fft_size,
            window=window,
            return_complex=True,
        )
        return spectrogram.abs()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = prediction.new_zeros(())
        for fft_size in self.fft_sizes:
            prediction_mag = self._magnitude_spectrogram(prediction, fft_size)
            target_mag = self._magnitude_spectrogram(target, fft_size)
            magnitude_loss = F.l1_loss(prediction_mag, target_mag)
            log_magnitude_loss = F.l1_loss(
                torch.log(prediction_mag + self.eps),
                torch.log(target_mag + self.eps),
            )
            loss = loss + magnitude_loss + log_magnitude_loss
        return loss / len(self.fft_sizes)


class CodecLoss(nn.Module):
    def __init__(
        self,
        waveform_weight: float,
        stft_weight: float,
        commitment_weight: float,
        codebook_weight: float,
        fft_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.waveform_weight = waveform_weight
        self.stft_weight = stft_weight
        self.commitment_weight = commitment_weight
        self.codebook_weight = codebook_weight
        self.stft_loss = MultiScaleSTFTLoss(fft_sizes=fft_sizes)

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        commitment_loss: torch.Tensor,
        codebook_loss: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        waveform_loss = F.l1_loss(reconstruction, target)
        stft_loss = self.stft_loss(reconstruction, target)
        total_loss = (
            self.waveform_weight * waveform_loss
            + self.stft_weight * stft_loss
            + self.commitment_weight * commitment_loss
            + self.codebook_weight * codebook_loss
        )
        return {
            "total_loss": total_loss,
            "waveform_loss": waveform_loss,
            "stft_loss": stft_loss,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
        }

