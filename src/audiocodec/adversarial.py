"""Adversarial training utilities for speech codec experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _load_torchaudio():
    import torchaudio

    return torchaudio


def _get_2d_padding(
    kernel_size: tuple[int, int],
    dilation: tuple[int, int] = (1, 1),
) -> tuple[int, int]:
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )


def _apply_conv2d_norm(conv: nn.Conv2d, norm: str) -> nn.Module:
    if norm == "none":
        return conv
    if norm == "weight_norm":
        return torch.nn.utils.parametrizations.weight_norm(conv)
    raise ValueError(f"Unsupported conv2d norm: {norm}")


class NormConv2d(nn.Module):
    """Minimal 2D conv wrapper matching the msstft discriminator needs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        dilation: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        norm: str = "weight_norm",
    ) -> None:
        super().__init__()
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
        )
        self.conv = _apply_conv2d_norm(conv, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DiscriminatorSTFT(nn.Module):
    """Single-scale STFT discriminator."""

    def __init__(
        self,
        filters: int,
        in_channels: int = 1,
        out_channels: int = 1,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        max_filters: int = 1024,
        filters_scale: int = 1,
        kernel_size: tuple[int, int] = (3, 9),
        dilations: tuple[int, ...] = (1, 2, 4),
        stride: tuple[int, int] = (1, 2),
        normalized: bool = True,
        norm: str = "weight_norm",
        activation_negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        torchaudio = _load_torchaudio()

        self.in_channels = in_channels
        self.activation = nn.LeakyReLU(negative_slope=activation_negative_slope)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window_fn=torch.hann_window,
            normalized=normalized,
            center=False,
            pad_mode=None,
            power=None,
        )

        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(
                spec_channels,
                filters,
                kernel_size=kernel_size,
                padding=_get_2d_padding(kernel_size),
                norm=norm,
            )
        )

        in_chs = min(filters_scale * filters, max_filters)
        for index, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (index + 1)) * filters, max_filters)
            self.convs.append(
                NormConv2d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=(dilation, 1),
                    padding=_get_2d_padding(kernel_size, (dilation, 1)),
                    norm=norm,
                )
            )
            in_chs = out_chs

        out_chs = min((filters_scale ** (len(dilations) + 1)) * filters, max_filters)
        square_kernel = (kernel_size[0], kernel_size[0])
        self.convs.append(
            NormConv2d(
                in_chs,
                out_chs,
                kernel_size=square_kernel,
                padding=_get_2d_padding(square_kernel),
                norm=norm,
            )
        )
        self.conv_post = NormConv2d(
            out_chs,
            out_channels,
            kernel_size=square_kernel,
            padding=_get_2d_padding(square_kernel),
            norm=norm,
        )

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        spectrogram = self.spec_transform(waveform)
        spectrogram = torch.cat([spectrogram.real, spectrogram.imag], dim=1)
        spectrogram = spectrogram.permute(0, 1, 3, 2)

        feature_maps: list[torch.Tensor] = []
        hidden = spectrogram
        for layer in self.convs:
            hidden = self.activation(layer(hidden))
            feature_maps.append(hidden)

        logits = self.conv_post(hidden)
        return logits, feature_maps


class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-scale STFT discriminator used by Encodec-style training."""

    def __init__(
        self,
        filters: int,
        in_channels: int = 1,
        out_channels: int = 1,
        n_ffts: tuple[int, ...] = (1024, 2048, 512),
        hop_lengths: tuple[int, ...] = (256, 512, 128),
        win_lengths: tuple[int, ...] = (1024, 2048, 512),
        norm: str = "weight_norm",
    ) -> None:
        super().__init__()
        if not (len(n_ffts) == len(hop_lengths) == len(win_lengths)):
            raise ValueError("n_ffts, hop_lengths, and win_lengths must have the same length.")
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorSTFT(
                    filters=filters,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_fft=n_ffts[index],
                    hop_length=hop_lengths[index],
                    win_length=win_lengths[index],
                    norm=norm,
                )
                for index in range(len(n_ffts))
            ]
        )

    def forward(
        self,
        waveform: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        logits: list[torch.Tensor] = []
        feature_maps: list[list[torch.Tensor]] = []
        for discriminator in self.discriminators:
            logit, fmap = discriminator(waveform)
            logits.append(logit)
            feature_maps.append(fmap)
        return logits, feature_maps


def _mse_real_loss(logits: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(logits, torch.ones_like(logits))


def _mse_fake_loss(logits: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(logits, torch.zeros_like(logits))


def _mse_generator_loss(logits: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(logits, torch.ones_like(logits))


def _hinge_real_loss(logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - logits).mean()


def _hinge_fake_loss(logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 + logits).mean()


def _hinge_generator_loss(logits: torch.Tensor) -> torch.Tensor:
    return -logits.mean()


def discriminator_adversarial_loss(
    fake_logits: list[torch.Tensor],
    real_logits: list[torch.Tensor],
    loss_type: str = "hinge",
) -> torch.Tensor:
    if len(fake_logits) != len(real_logits):
        raise ValueError("fake_logits and real_logits must have the same number of scales.")
    if not fake_logits:
        raise ValueError("At least one discriminator scale is required.")

    total = fake_logits[0].new_zeros(())
    for fake_scale, real_scale in zip(fake_logits, real_logits):
        if loss_type == "hinge":
            total = total + _hinge_fake_loss(fake_scale) + _hinge_real_loss(real_scale)
        elif loss_type == "mse":
            total = total + _mse_fake_loss(fake_scale) + _mse_real_loss(real_scale)
        else:
            raise ValueError(f"Unsupported adversarial loss_type: {loss_type}")
    return total / len(fake_logits)


def generator_adversarial_loss(
    fake_logits: list[torch.Tensor],
    loss_type: str = "hinge",
) -> torch.Tensor:
    if not fake_logits:
        raise ValueError("At least one discriminator scale is required.")

    total = fake_logits[0].new_zeros(())
    for fake_scale in fake_logits:
        if loss_type == "hinge":
            total = total + _hinge_generator_loss(fake_scale)
        elif loss_type == "mse":
            total = total + _mse_generator_loss(fake_scale)
        else:
            raise ValueError(f"Unsupported adversarial loss_type: {loss_type}")
    return total / len(fake_logits)


class FeatureMatchingLoss(nn.Module):
    """Average L1 feature matching across discriminator scales."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        fake_feature_maps: list[list[torch.Tensor]],
        real_feature_maps: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        if len(fake_feature_maps) != len(real_feature_maps):
            raise ValueError("fake_feature_maps and real_feature_maps must match in length.")
        if not fake_feature_maps:
            raise ValueError("At least one discriminator scale is required.")

        total = fake_feature_maps[0][0].new_zeros(())
        for fake_scale_maps, real_scale_maps in zip(fake_feature_maps, real_feature_maps):
            if len(fake_scale_maps) != len(real_scale_maps):
                raise ValueError("Feature map depths must match between fake and real paths.")
            scale_total = total.new_zeros(())
            for fake_map, real_map in zip(fake_scale_maps, real_scale_maps):
                scale_total = scale_total + F.l1_loss(fake_map, real_map.detach())
            total = total + scale_total / len(fake_scale_maps)
        return total / len(fake_feature_maps)
