"""Reusable neural network blocks for 1D audio models."""

from __future__ import annotations

import math

from torch import nn


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def normalization(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_group_count(channels), channels)


def stride_to_conv_params(stride: int) -> tuple[int, int, int]:
    kernel_size = 2 * stride
    padding = math.ceil(stride / 2)
    output_padding = 2 * padding - stride
    return kernel_size, padding, output_padding


class ResidualUnit1d(nn.Module):
    """A small residual block used before and after sampling changes."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            normalization(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, inputs):
        return inputs + self.block(inputs)

