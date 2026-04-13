"""Residual vector quantization modules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class RVQOutput:
    quantized: torch.Tensor
    codes: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor


class Codebook(nn.Module):
    """A single nearest-neighbor codebook."""

    def __init__(self, dimension: int, size: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.size = size
        self.embedding = nn.Embedding(size, dimension)
        nn.init.uniform_(self.embedding.weight, -1.0 / size, 1.0 / size)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, channels, frames].")

        batch_size, channels, frames = inputs.shape
        flat_inputs = inputs.transpose(1, 2).reshape(-1, channels)
        codebook = self.embedding.weight

        distances = (
            flat_inputs.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs @ codebook.t()
            + codebook.pow(2).sum(dim=1)
        )
        indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(indices).view(batch_size, frames, channels).transpose(1, 2)
        return quantized, indices.view(batch_size, frames)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 2:
            raise ValueError("indices must have shape [batch, frames].")
        return self.embedding(indices).transpose(1, 2)


class ResidualVectorQuantizer(nn.Module):
    """Greedy residual vector quantization with straight-through gradients."""

    def __init__(self, dimension: int, num_quantizers: int, codebook_size: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebooks = nn.ModuleList(
            [Codebook(dimension=dimension, size=codebook_size) for _ in range(num_quantizers)]
        )

    def forward(self, latents: torch.Tensor) -> RVQOutput:
        if latents.ndim != 3:
            raise ValueError("latents must have shape [batch, channels, frames].")

        residual = latents
        quantized_sum = torch.zeros_like(latents)
        codes: list[torch.Tensor] = []
        commitment_loss = latents.new_zeros(())
        codebook_loss = latents.new_zeros(())

        for codebook in self.codebooks:
            stage_input = residual
            stage_quantized, stage_codes = codebook(stage_input)
            codes.append(stage_codes)

            commitment_loss = commitment_loss + F.mse_loss(stage_input, stage_quantized.detach())
            codebook_loss = codebook_loss + F.mse_loss(stage_quantized, stage_input.detach())

            quantized_sum = quantized_sum + stage_quantized
            residual = stage_input - stage_quantized.detach()

        quantized = latents + (quantized_sum - latents).detach()
        return RVQOutput(
            quantized=quantized,
            codes=torch.stack(codes, dim=1),
            commitment_loss=commitment_loss / self.num_quantizers,
            codebook_loss=codebook_loss / self.num_quantizers,
        )

    @torch.no_grad()
    def encode(self, latents: torch.Tensor) -> torch.Tensor:
        residual = latents
        codes: list[torch.Tensor] = []
        for codebook in self.codebooks:
            quantized, stage_codes = codebook(residual)
            codes.append(stage_codes)
            residual = residual - quantized
        return torch.stack(codes, dim=1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 3:
            raise ValueError("codes must have shape [batch, num_quantizers, frames].")
        if codes.shape[1] != self.num_quantizers:
            raise ValueError("codes.shape[1] must match num_quantizers.")

        batch_size, _, frames = codes.shape
        decoded = self.codebooks[0].embedding.weight.new_zeros(batch_size, self.dimension, frames)
        for stage, codebook in enumerate(self.codebooks):
            decoded = decoded + codebook.decode(codes[:, stage, :])
        return decoded

