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


def _sample_vectors(samples: torch.Tensor, num_samples: int) -> torch.Tensor:
    if samples.shape[0] >= num_samples:
        indices = torch.randperm(samples.shape[0], device=samples.device)[:num_samples]
    else:
        indices = torch.randint(0, samples.shape[0], (num_samples,), device=samples.device)
    return samples[indices]


@torch.no_grad()
def _run_kmeans(samples: torch.Tensor, num_centroids: int, num_iters: int) -> tuple[torch.Tensor, torch.Tensor]:
    if samples.shape[0] == 0:
        raise ValueError("k-means initialization requires at least one sample.")

    centroids = _sample_vectors(samples, num_centroids)
    for _ in range(num_iters):
        distances = (
            samples.pow(2).sum(dim=1, keepdim=True)
            - 2 * samples @ centroids.t()
            + centroids.pow(2).sum(dim=1)
        )
        assignments = torch.argmin(distances, dim=1)
        counts = torch.bincount(assignments, minlength=num_centroids)
        updated = centroids.clone()
        for centroid_index in range(num_centroids):
            mask = assignments == centroid_index
            if mask.any():
                updated[centroid_index] = samples[mask].mean(dim=0)
        centroids = updated
    return centroids, counts


class EMACodebook(nn.Module):
    """Euclidean codebook with k-means initialization and EMA updates."""

    def __init__(
        self,
        dimension: int,
        size: int,
        decay: float,
        kmeans_init: bool,
        kmeans_iters: int,
        dead_code_threshold: int,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.size = size
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.dead_code_threshold = dead_code_threshold
        initial_embeddings = torch.zeros(size, dimension) if kmeans_init else torch.empty(size, dimension)
        if not kmeans_init:
            nn.init.kaiming_uniform_(initial_embeddings)
        self.register_buffer("initialized", torch.tensor([not kmeans_init], dtype=torch.bool))
        self.register_buffer("embedding", initial_embeddings)
        self.register_buffer("embedding_avg", initial_embeddings.clone())
        self.register_buffer("cluster_size", torch.zeros(size))

    @torch.no_grad()
    def _maybe_initialize(self, flat_inputs: torch.Tensor) -> None:
        if bool(self.initialized.item()):
            return
        centroids, counts = _run_kmeans(flat_inputs.to(self.embedding.dtype), self.size, self.kmeans_iters)
        self.embedding.copy_(centroids)
        self.embedding_avg.copy_(centroids)
        self.cluster_size.copy_(counts.to(self.cluster_size.dtype))
        self.initialized.fill_(True)

    def _compute_indices(self, flat_inputs: torch.Tensor) -> torch.Tensor:
        distances = (
            flat_inputs.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs @ self.embedding.t()
            + self.embedding.pow(2).sum(dim=1)
        )
        return torch.argmin(distances, dim=1)

    @torch.no_grad()
    def _replace_dead_codes(self, flat_inputs: torch.Tensor) -> None:
        if self.dead_code_threshold == 0:
            return
        dead_mask = self.cluster_size < self.dead_code_threshold
        if not dead_mask.any():
            return
        replacements = _sample_vectors(flat_inputs, int(dead_mask.sum().item())).to(self.embedding.dtype)
        self.embedding[dead_mask] = replacements

    @torch.no_grad()
    def _ema_update(self, flat_inputs: torch.Tensor, indices: torch.Tensor) -> None:
        one_hot = F.one_hot(indices, num_classes=self.size).to(flat_inputs.dtype)
        cluster_size = one_hot.sum(dim=0)
        embedding_sum = one_hot.transpose(0, 1) @ flat_inputs

        self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1.0 - self.decay)
        self.embedding_avg.mul_(self.decay).add_(embedding_sum, alpha=1.0 - self.decay)

        smoothed_cluster_size = (self.cluster_size + 1e-5) / (
            self.cluster_size.sum() + self.size * 1e-5
        ) * self.cluster_size.sum().clamp_min(1.0)
        normalized = self.embedding_avg / smoothed_cluster_size.unsqueeze(1)
        self.embedding.copy_(normalized)

    def _lookup(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.embedding)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, channels, frames].")

        input_dtype = inputs.dtype
        batch_size, channels, frames = inputs.shape
        flat_inputs = inputs.transpose(1, 2).reshape(-1, channels).to(self.embedding.dtype)
        self._maybe_initialize(flat_inputs)

        flat_indices = self._compute_indices(flat_inputs)
        quantized = self._lookup(flat_indices)

        if self.training:
            self._replace_dead_codes(flat_inputs)
            self._ema_update(flat_inputs, flat_indices)

        commitment_loss = F.mse_loss(flat_inputs.float(), quantized.detach().float())
        if self.training:
            quantized = flat_inputs + (quantized - flat_inputs).detach()
        quantized = quantized.view(batch_size, frames, channels).transpose(1, 2).to(input_dtype)
        return quantized, flat_indices.view(batch_size, frames), commitment_loss

    @torch.no_grad()
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, channels, frames = inputs.shape
        flat_inputs = inputs.transpose(1, 2).reshape(-1, channels).to(self.embedding.dtype)
        self._maybe_initialize(flat_inputs)
        return self._compute_indices(flat_inputs).view(batch_size, frames)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 2:
            raise ValueError("indices must have shape [batch, frames].")
        return self._lookup(indices).transpose(1, 2)


class EMAResidualVectorQuantizer(nn.Module):
    """Residual vector quantizer with k-means init and EMA codebook updates."""

    def __init__(
        self,
        dimension: int,
        num_quantizers: int,
        codebook_size: int,
        decay: float,
        kmeans_init: bool,
        kmeans_iters: int,
        dead_code_threshold: int,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebooks = nn.ModuleList(
            [
                EMACodebook(
                    dimension=dimension,
                    size=codebook_size,
                    decay=decay,
                    kmeans_init=kmeans_init,
                    kmeans_iters=kmeans_iters,
                    dead_code_threshold=dead_code_threshold,
                )
                for _ in range(num_quantizers)
            ]
        )

    def forward(self, latents: torch.Tensor) -> RVQOutput:
        residual = latents
        quantized_sum = torch.zeros_like(latents)
        codes: list[torch.Tensor] = []
        commitment_loss = latents.new_zeros(())

        for codebook in self.codebooks:
            stage_quantized, stage_codes, stage_commitment = codebook(residual)
            quantized_sum = quantized_sum + stage_quantized
            residual = residual - stage_quantized
            commitment_loss = commitment_loss + stage_commitment
            codes.append(stage_codes)

        return RVQOutput(
            quantized=quantized_sum,
            codes=torch.stack(codes, dim=1),
            commitment_loss=commitment_loss / self.num_quantizers,
            codebook_loss=latents.new_zeros(()),
        )

    @torch.no_grad()
    def encode(self, latents: torch.Tensor) -> torch.Tensor:
        residual = latents
        codes: list[torch.Tensor] = []
        for codebook in self.codebooks:
            stage_codes = codebook.encode(residual)
            stage_quantized = codebook.decode(stage_codes).to(residual.dtype)
            codes.append(stage_codes)
            residual = residual - stage_quantized
        return torch.stack(codes, dim=1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 3:
            raise ValueError("codes must have shape [batch, num_quantizers, frames].")
        if codes.shape[1] != self.num_quantizers:
            raise ValueError("codes.shape[1] must match num_quantizers.")
        batch_size, _, frames = codes.shape
        decoded = self.codebooks[0].embedding.new_zeros(batch_size, self.dimension, frames)
        for stage, codebook in enumerate(self.codebooks):
            decoded = decoded + codebook.decode(codes[:, stage, :])
        return decoded
