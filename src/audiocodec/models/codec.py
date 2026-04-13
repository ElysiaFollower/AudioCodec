"""Baseline Conv + RVQ + Conv codec model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from audiocodec.config import CodecExperimentConfig, ModelConfig, RVQConfig

from .blocks import ResidualUnit1d, normalization, stride_to_conv_params
from .quantizer import ResidualVectorQuantizer


@dataclass(slots=True)
class EncodedAudio:
    codes: torch.Tensor
    original_length: int
    padded_length: int


@dataclass(slots=True)
class CodecOutput:
    reconstruction: torch.Tensor
    codes: torch.Tensor
    latent: torch.Tensor
    quantized: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    original_length: int
    padded_length: int


class ConvEncoder(nn.Module):
    def __init__(self, channels: int, config: ModelConfig) -> None:
        super().__init__()
        self.channels = channels
        self.config = config
        self.input_conv = nn.Conv1d(
            channels,
            config.base_channels,
            kernel_size=config.input_kernel_size,
            padding=config.input_kernel_size // 2,
        )

        stages = []
        current_channels = config.base_channels
        for stride, next_channels in zip(config.encoder_strides, config.stage_channels):
            block_layers = [ResidualUnit1d(current_channels) for _ in range(config.residual_layers_per_stage)]
            kernel_size, padding, _ = stride_to_conv_params(stride)
            block_layers.extend(
                [
                    normalization(current_channels),
                    nn.SiLU(),
                    nn.Conv1d(
                        current_channels,
                        next_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                ]
            )
            stages.append(nn.Sequential(*block_layers))
            current_channels = next_channels

        self.stages = nn.ModuleList(stages)
        self.output = nn.Sequential(
            normalization(current_channels),
            nn.SiLU(),
            nn.Conv1d(current_channels, config.latent_dim, kernel_size=3, padding=1),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        hidden = self.input_conv(waveform)
        for stage in self.stages:
            hidden = stage(hidden)
        return self.output(hidden)


class ConvDecoder(nn.Module):
    def __init__(self, channels: int, config: ModelConfig) -> None:
        super().__init__()
        stage_channels = config.stage_channels
        current_channels = stage_channels[-1]
        self.input_conv = nn.Conv1d(config.latent_dim, current_channels, kernel_size=3, padding=1)

        upsample_channels = list(reversed(stage_channels[:-1])) + [config.base_channels]
        stages = []
        for stride, next_channels in zip(reversed(config.encoder_strides), upsample_channels):
            kernel_size, padding, output_padding = stride_to_conv_params(stride)
            block_layers = [
                normalization(current_channels),
                nn.SiLU(),
                nn.ConvTranspose1d(
                    current_channels,
                    next_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                ),
            ]
            block_layers.extend(
                [ResidualUnit1d(next_channels) for _ in range(config.residual_layers_per_stage)]
            )
            stages.append(nn.Sequential(*block_layers))
            current_channels = next_channels

        self.stages = nn.ModuleList(stages)
        self.output = nn.Sequential(
            normalization(current_channels),
            nn.SiLU(),
            nn.Conv1d(
                current_channels,
                channels,
                kernel_size=config.output_kernel_size,
                padding=config.output_kernel_size // 2,
            ),
        )

    def forward(self, quantized: torch.Tensor) -> torch.Tensor:
        hidden = self.input_conv(quantized)
        for stage in self.stages:
            hidden = stage(hidden)
        return self.output(hidden)


class ConvRVQCodec(nn.Module):
    """Speech codec baseline with a convolutional autoencoder and RVQ bottleneck."""

    def __init__(self, sample_rate: int, channels: int, model_config: ModelConfig, rvq_config: RVQConfig) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self.model_config = model_config
        self.rvq_config = rvq_config

        self.encoder = ConvEncoder(channels=channels, config=model_config)
        self.quantizer = ResidualVectorQuantizer(
            dimension=model_config.latent_dim,
            num_quantizers=rvq_config.num_quantizers,
            codebook_size=rvq_config.codebook_size,
        )
        self.decoder = ConvDecoder(channels=channels, config=model_config)

    @classmethod
    def from_config(cls, config: CodecExperimentConfig) -> "ConvRVQCodec":
        return cls(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            model_config=config.model,
            rvq_config=config.quantizer,
        )

    @property
    def hop_length(self) -> int:
        return self.model_config.hop_length

    @property
    def frame_rate(self) -> int:
        return self.sample_rate // self.hop_length

    def _pad_waveform(self, waveform: torch.Tensor) -> tuple[torch.Tensor, int]:
        if waveform.ndim != 3:
            raise ValueError("waveform must have shape [batch, channels, samples].")
        if waveform.shape[1] != self.channels:
            raise ValueError("waveform channel count does not match the model configuration.")

        original_length = waveform.shape[-1]
        remainder = original_length % self.hop_length
        if remainder == 0:
            return waveform, 0
        pad_amount = self.hop_length - remainder
        return F.pad(waveform, (0, pad_amount)), pad_amount

    def encode(self, waveform: torch.Tensor) -> EncodedAudio:
        padded_waveform, pad_amount = self._pad_waveform(waveform)
        latent = self.encoder(padded_waveform)
        codes = self.quantizer.encode(latent)
        return EncodedAudio(
            codes=codes,
            original_length=waveform.shape[-1],
            padded_length=waveform.shape[-1] + pad_amount,
        )

    def decode(self, codes: torch.Tensor, length: int | None = None) -> torch.Tensor:
        quantized = self.quantizer.decode(codes)
        reconstruction = self.decoder(quantized)
        if length is not None:
            reconstruction = reconstruction[..., :length]
        return reconstruction

    def forward(self, waveform: torch.Tensor) -> CodecOutput:
        padded_waveform, pad_amount = self._pad_waveform(waveform)
        latent = self.encoder(padded_waveform)
        rvq_output = self.quantizer(latent)
        reconstruction = self.decoder(rvq_output.quantized)
        if pad_amount:
            reconstruction = reconstruction[..., :-pad_amount]
        return CodecOutput(
            reconstruction=reconstruction,
            codes=rvq_output.codes,
            latent=latent,
            quantized=rvq_output.quantized,
            commitment_loss=rvq_output.commitment_loss,
            codebook_loss=rvq_output.codebook_loss,
            original_length=waveform.shape[-1],
            padded_length=padded_waveform.shape[-1],
        )

