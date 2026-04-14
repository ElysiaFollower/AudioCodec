"""Encodec-inspired SEANet building blocks for offline speech codec training."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm


def _apply_conv_norm(module: nn.Module, norm: str) -> nn.Module:
    if norm == "weight_norm":
        return weight_norm(module)
    if norm == "none":
        return module
    raise ValueError(f"Unsupported convolution norm: {norm}")


def _get_extra_padding(length: int, kernel_size: int, stride: int, padding_total: int) -> int:
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return max(0, ideal_length - length)


def _pad1d(x: torch.Tensor, padding_left: int, padding_right: int, mode: str = "reflect") -> torch.Tensor:
    if mode != "reflect":
        return F.pad(x, (padding_left, padding_right), mode=mode)

    length = x.shape[-1]
    max_pad = max(padding_left, padding_right)
    extra_pad = 0
    if length <= max_pad:
        extra_pad = max_pad - length + 1
        x = F.pad(x, (0, extra_pad))
    padded = F.pad(x, (padding_left, padding_right), mode=mode)
    if extra_pad:
        padded = padded[..., : padded.shape[-1] - extra_pad]
    return padded


def _unpad1d(x: torch.Tensor, padding_left: int, padding_right: int) -> torch.Tensor:
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class SConv1d(nn.Module):
    """Conv1d wrapper with Encodec-style asymmetric padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        norm: str = "weight_norm",
        pad_mode: str = "reflect",
    ) -> None:
        super().__init__()
        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )
        self.conv = _apply_conv_norm(conv, norm)
        self.pad_mode = pad_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        dilation = self.conv.dilation[0]
        effective_kernel_size = (kernel_size - 1) * dilation + 1
        padding_total = effective_kernel_size - stride
        extra_padding = _get_extra_padding(x.shape[-1], effective_kernel_size, stride, padding_total)
        padding_right = padding_total // 2
        padding_left = padding_total - padding_right
        padded = _pad1d(x, padding_left, padding_right + extra_padding, mode=self.pad_mode)
        return self.conv(padded)


class SConvTranspose1d(nn.Module):
    """ConvTranspose1d wrapper that trims the exact padding added by `SConv1d`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        norm: str = "weight_norm",
    ) -> None:
        super().__init__()
        conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv = _apply_conv_norm(conv, norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        padding_total = kernel_size - stride
        padding_right = padding_total // 2
        padding_left = padding_total - padding_right
        y = self.conv(x)
        return _unpad1d(y, padding_left, padding_right)


class SkipLSTM(nn.Module):
    """LSTM over temporal frames with a residual skip."""

    def __init__(self, dimension: int, num_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(dimension, dimension, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.permute(2, 0, 1)
        output, _ = self.lstm(sequence)
        return (output + sequence).permute(1, 2, 0)


class SEANetResidualBlock(nn.Module):
    """Residual block with a dilated convolution and a bottleneck projection."""

    def __init__(
        self,
        channels: int,
        norm: str,
        dilation: int,
        kernel_size: int,
        compress: int,
    ) -> None:
        super().__init__()
        hidden_channels = max(1, channels // compress)
        self.block = nn.Sequential(
            _make_activation(),
            SConv1d(
                channels,
                hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                norm=norm,
            ),
            _make_activation(),
            SConv1d(hidden_channels, channels, kernel_size=1, norm=norm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


def _make_activation() -> nn.ELU:
    return nn.ELU(alpha=1.0)


class SEANetEncoder(nn.Module):
    """Encodec-style SEANet encoder for speech."""

    def __init__(
        self,
        channels: int,
        dimension: int,
        n_filters: int,
        ratios: Iterable[int],
        residual_layers: int,
        residual_kernel_size: int,
        dilation_base: int,
        compress: int,
        lstm_layers: int,
        kernel_size: int,
        last_kernel_size: int,
        norm: str,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.ratios = tuple(reversed(tuple(ratios)))
        self.hop_length = math.prod(self.ratios)

        layers: list[nn.Module] = [SConv1d(channels, n_filters, kernel_size=kernel_size, norm=norm)]
        current_multiplier = 1
        for ratio in self.ratios:
            current_channels = current_multiplier * n_filters
            for residual_layer in range(residual_layers):
                layers.append(
                    SEANetResidualBlock(
                        channels=current_channels,
                        norm=norm,
                        dilation=dilation_base**residual_layer,
                        kernel_size=residual_kernel_size,
                        compress=compress,
                    )
                )
            layers.extend(
                [
                    _make_activation(),
                    SConv1d(
                        current_channels,
                        current_channels * 2,
                        kernel_size=ratio * 2,
                        stride=ratio,
                        norm=norm,
                    ),
                ]
            )
            current_multiplier *= 2

        current_channels = current_multiplier * n_filters
        if lstm_layers:
            layers.append(SkipLSTM(current_channels, num_layers=lstm_layers))
        layers.extend(
            [
                _make_activation(),
                SConv1d(current_channels, dimension, kernel_size=last_kernel_size, norm=norm),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SEANetDecoder(nn.Module):
    """Mirror decoder for `SEANetEncoder`."""

    def __init__(
        self,
        channels: int,
        dimension: int,
        n_filters: int,
        ratios: Iterable[int],
        residual_layers: int,
        residual_kernel_size: int,
        dilation_base: int,
        compress: int,
        lstm_layers: int,
        kernel_size: int,
        last_kernel_size: int,
        norm: str,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.ratios = tuple(ratios)
        self.hop_length = math.prod(self.ratios)

        multiplier = 2 ** len(self.ratios)
        current_channels = multiplier * n_filters
        layers: list[nn.Module] = [SConv1d(dimension, current_channels, kernel_size=kernel_size, norm=norm)]
        if lstm_layers:
            layers.append(SkipLSTM(current_channels, num_layers=lstm_layers))

        for ratio in self.ratios:
            next_channels = current_channels // 2
            layers.extend(
                [
                    _make_activation(),
                    SConvTranspose1d(
                        current_channels,
                        next_channels,
                        kernel_size=ratio * 2,
                        stride=ratio,
                        norm=norm,
                    ),
                ]
            )
            for residual_layer in range(residual_layers):
                layers.append(
                    SEANetResidualBlock(
                        channels=next_channels,
                        norm=norm,
                        dilation=dilation_base**residual_layer,
                        kernel_size=residual_kernel_size,
                        compress=compress,
                    )
                )
            current_channels = next_channels

        layers.extend(
            [
                _make_activation(),
                SConv1d(current_channels, channels, kernel_size=last_kernel_size, norm=norm),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
