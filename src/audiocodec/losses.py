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


def _load_torchaudio():
    import torchaudio

    return torchaudio


class MelSpectrogramLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_mels: int = 80,
        fft_size: int = 1_024,
        hop_length: int = 256,
        f_min: float = 0.0,
        f_max: float | None = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.eps = eps
        self._mel_transform = None

    def _get_transform(self, waveform: torch.Tensor) -> nn.Module:
        if self._mel_transform is None:
            torchaudio = _load_torchaudio()
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.fft_size,
                hop_length=self.hop_length,
                win_length=self.fft_size,
                f_min=self.f_min,
                f_max=self.f_max,
                n_mels=self.n_mels,
                power=1.0,
                center=True,
                norm="slaney",
                mel_scale="slaney",
            )
        return self._mel_transform.to(device=waveform.device, dtype=waveform.dtype)

    def _mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        batch_size, channels, samples = waveform.shape
        flattened = waveform.reshape(batch_size * channels, samples)
        transform = self._get_transform(waveform)
        return transform(flattened)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction_mel = self._mel_spectrogram(prediction)
        target_mel = self._mel_spectrogram(target)
        mel_loss = F.l1_loss(prediction_mel, target_mel)
        log_mel_loss = F.l1_loss(
            torch.log(prediction_mel + self.eps),
            torch.log(target_mel + self.eps),
        )
        return mel_loss + log_mel_loss


class CodecLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        waveform_weight: float,
        stft_weight: float,
        mel_weight: float,
        commitment_weight: float,
        codebook_weight: float,
        fft_sizes: tuple[int, ...],
        mel_n_mels: int,
        mel_fft_size: int,
        mel_hop_length: int,
        mel_f_min: float,
        mel_f_max: float | None,
    ) -> None:
        super().__init__()
        self.waveform_weight = waveform_weight
        self.stft_weight = stft_weight
        self.mel_weight = mel_weight
        self.commitment_weight = commitment_weight
        self.codebook_weight = codebook_weight
        self.stft_loss = MultiScaleSTFTLoss(fft_sizes=fft_sizes)
        self.mel_loss = None
        if mel_weight > 0:
            self.mel_loss = MelSpectrogramLoss(
                sample_rate=sample_rate,
                n_mels=mel_n_mels,
                fft_size=mel_fft_size,
                hop_length=mel_hop_length,
                f_min=mel_f_min,
                f_max=mel_f_max,
            )

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        commitment_loss: torch.Tensor,
        codebook_loss: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        waveform_loss = F.l1_loss(reconstruction, target)
        stft_loss = self.stft_loss(reconstruction, target)
        mel_loss = reconstruction.new_zeros(())
        if self.mel_loss is not None:
            mel_loss = self.mel_loss(reconstruction, target)
        total_loss = (
            self.waveform_weight * waveform_loss
            + self.stft_weight * stft_loss
            + self.mel_weight * mel_loss
            + self.commitment_weight * commitment_loss
            + self.codebook_weight * codebook_loss
        )
        return {
            "total_loss": total_loss,
            "waveform_loss": waveform_loss,
            "stft_loss": stft_loss,
            "mel_loss": mel_loss,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
        }
