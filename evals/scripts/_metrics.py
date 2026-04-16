from __future__ import annotations

import math

import torch

from audiocodec.losses import MultiScaleSTFTLoss


def align_waveforms(reference: torch.Tensor, degraded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    min_length = min(reference.shape[-1], degraded.shape[-1])
    if min_length <= 0:
        raise ValueError("Waveforms must have a positive number of samples.")
    return reference[..., :min_length].float(), degraded[..., :min_length].float()


def compute_si_sdr_db(reference: torch.Tensor, degraded: torch.Tensor, eps: float = 1e-8) -> float:
    reference, degraded = align_waveforms(reference, degraded)
    target = torch.sum(degraded * reference, dim=-1, keepdim=True) * reference
    target = target / (torch.sum(reference**2, dim=-1, keepdim=True) + eps)
    noise = degraded - target
    ratio = (torch.sum(target**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    return float(10.0 * torch.log10(ratio + eps).mean().item())


def compute_log_spectral_distance(
    reference: torch.Tensor,
    degraded: torch.Tensor,
    fft_size: int = 512,
    hop_length: int = 128,
    eps: float = 1e-7,
) -> float:
    reference, degraded = align_waveforms(reference, degraded)
    window = torch.hann_window(fft_size, device=reference.device, dtype=reference.dtype)
    reference_spec = torch.stft(
        reference.squeeze(0),
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=fft_size,
        window=window,
        return_complex=True,
    ).abs()
    degraded_spec = torch.stft(
        degraded.squeeze(0),
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=fft_size,
        window=window,
        return_complex=True,
    ).abs()
    reference_log = 20.0 * torch.log10(reference_spec + eps)
    degraded_log = 20.0 * torch.log10(degraded_spec + eps)
    frame_rms = torch.sqrt(torch.mean((reference_log - degraded_log) ** 2, dim=0))
    return float(frame_rms.mean().item())


def compute_multi_scale_stft(reference: torch.Tensor, degraded: torch.Tensor) -> float:
    reference, degraded = align_waveforms(reference, degraded)
    loss = MultiScaleSTFTLoss()
    value = loss(degraded.unsqueeze(0), reference.unsqueeze(0))
    return float(value.item())


def compute_stoi_or_none(reference: torch.Tensor, degraded: torch.Tensor, sample_rate: int) -> float | None:
    try:
        from pystoi import stoi as pystoi_stoi
    except ModuleNotFoundError:
        return None
    reference, degraded = align_waveforms(reference, degraded)
    return float(
        pystoi_stoi(
            reference.squeeze().detach().cpu().numpy(),
            degraded.squeeze().detach().cpu().numpy(),
            sample_rate,
            extended=False,
        )
    )
