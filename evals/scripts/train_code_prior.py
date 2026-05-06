#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

import torch
from torch import nn
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import ensure_directory, write_jsonl
from evaluate_code_priors import (
    CODE_PRIOR_SCHEMA_VERSION,
    TOKEN_ORDERING,
    PriorMetadata,
    _load_sequences,
    _metadata,
)


PRIOR_CHOICES = ("local_tcn", "long_transformer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen RVQ code-prior baselines.")
    parser.add_argument("--export-dir", type=Path, required=True, help="Phase 1 neural export directory.")
    parser.add_argument("--manifest", type=Path, default=None, help="Training manifest. Defaults to export-dir/manifest.jsonl.")
    parser.add_argument(
        "--eval-export-dir",
        type=Path,
        default=None,
        help="Optional separate Phase 1 export directory for held-out evaluation.",
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=None,
        help="Optional held-out eval manifest. Defaults to eval-export-dir/manifest.jsonl when provided.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prior", choices=PRIOR_CHOICES, default="local_tcn")
    parser.add_argument("--codebook-size", type=int, default=None)
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def _resolve_device(device: str) -> torch.device:
    requested = device.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _flatten_codes(codes: torch.Tensor) -> torch.Tensor:
    batch, quantizers, frames = codes.shape
    return codes.permute(0, 2, 1).reshape(batch, frames * quantizers).long()


def _flatten_sequences(sequences) -> list[torch.Tensor]:
    flattened: list[torch.Tensor] = []
    for sequence in sequences:
        for item in _flatten_codes(sequence.codes):
            if item.numel() >= 2:
                flattened.append(item.contiguous())
    if not flattened:
        raise ValueError("No code sequences with at least two tokens were found.")
    return flattened


def _sample_batch(
    sequences: list[torch.Tensor],
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    generator: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")
    eligible = [sequence for sequence in sequences if int(sequence.numel()) >= sequence_length]
    if eligible:
        sample_pool = eligible
        chunk_length = sequence_length
    else:
        chunk_length = max(int(sequence.numel()) for sequence in sequences)
        sample_pool = [sequence for sequence in sequences if int(sequence.numel()) >= chunk_length]
    inputs = []
    targets = []
    target_positions = []
    for _ in range(batch_size):
        sequence = sample_pool[generator.randrange(len(sample_pool))]
        max_start = int(sequence.numel()) - chunk_length
        start = generator.randrange(max_start + 1) if max_start > 0 else 0
        chunk = sequence[start : start + chunk_length]
        inputs.append(chunk[:-1])
        targets.append(chunk[1:])
        target_positions.append(torch.arange(start + 1, start + chunk_length, dtype=torch.long))
    return (
        torch.stack(inputs).to(device),
        torch.stack(targets).to(device),
        torch.stack(target_positions).to(device),
    )


def _all_eval_batches(
    sequences: list[torch.Tensor],
    *,
    sequence_length: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    batches = []
    for sequence in sequences:
        step = max(2, sequence_length)
        for start in range(0, int(sequence.numel()) - 1, step - 1):
            end = min(start + step, int(sequence.numel()))
            chunk = sequence[start:end]
            if chunk.numel() < 2:
                continue
            positions = torch.arange(start + 1, end, dtype=torch.long)
            batches.append((chunk[:-1].unsqueeze(0).to(device), chunk[1:].unsqueeze(0).to(device), positions.unsqueeze(0).to(device)))
    return batches


class CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class LocalTCNPrior(nn.Module):
    def __init__(
        self,
        *,
        codebook_size: int,
        num_quantizers: int,
        embedding_dim: int,
        hidden_dim: int,
        layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.code_embedding = nn.Embedding(codebook_size, embedding_dim)
        self.stage_embedding = nn.Embedding(num_quantizers, embedding_dim)
        self.input = nn.Linear(embedding_dim, hidden_dim)
        blocks = []
        for index in range(layers):
            dilation = 2**index
            blocks.append(
                nn.Sequential(
                    CausalConv1d(hidden_dim, kernel_size=kernel_size, dilation=dilation),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, codebook_size)
        self.receptive_tokens = 1 + (kernel_size - 1) * sum(2**index for index in range(layers))

    def forward(self, tokens: torch.Tensor, stages: torch.Tensor) -> torch.Tensor:
        x = self.code_embedding(tokens) + self.stage_embedding(stages)
        x = self.input(x).transpose(1, 2)
        for block in self.blocks:
            x = x + block(x)
        x = self.norm(x.transpose(1, 2))
        return self.output(x)


class LongTransformerPrior(nn.Module):
    def __init__(
        self,
        *,
        codebook_size: int,
        num_quantizers: int,
        max_sequence_length: int,
        embedding_dim: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.code_embedding = nn.Embedding(codebook_size, embedding_dim)
        self.stage_embedding = nn.Embedding(num_quantizers, embedding_dim)
        self.position_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.output = nn.Linear(embedding_dim, codebook_size)
        self.receptive_tokens = max_sequence_length

    def forward(self, tokens: torch.Tensor, stages: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0).expand_as(tokens)
        x = self.code_embedding(tokens) + self.stage_embedding(stages) + self.position_embedding(positions)
        mask = torch.triu(
            torch.full((tokens.shape[1], tokens.shape[1]), float("-inf"), device=tokens.device),
            diagonal=1,
        )
        x = self.encoder(x, mask=mask)
        return self.output(self.norm(x))


def _build_model(args: argparse.Namespace, metadata: PriorMetadata) -> nn.Module:
    if args.prior == "local_tcn":
        return LocalTCNPrior(
            codebook_size=metadata.codebook_size,
            num_quantizers=metadata.num_quantizers,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
        )
    return LongTransformerPrior(
        codebook_size=metadata.codebook_size,
        num_quantizers=metadata.num_quantizers,
        max_sequence_length=args.sequence_length - 1,
        embedding_dim=args.embedding_dim,
        layers=args.layers,
        heads=args.transformer_heads,
        dropout=args.dropout,
    )


def _stage_ids(positions: torch.Tensor, num_quantizers: int) -> torch.Tensor:
    return positions.remainder(num_quantizers)


def _loss_and_stage_bits(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_positions: torch.Tensor,
    num_quantizers: int,
) -> tuple[torch.Tensor, list[float | None], list[int]]:
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    stage_ids = _stage_ids(target_positions, num_quantizers)
    stage_bits: list[float | None] = []
    stage_counts: list[int] = []
    for stage in range(num_quantizers):
        mask = stage_ids.eq(stage)
        stage_counts.append(int(mask.sum().item()))
        if not mask.any():
            stage_bits.append(None)
            continue
        stage_bits.append(float((losses[mask].mean() / math.log(2.0)).item()))
    return losses.mean(), stage_bits, stage_counts


def _metric_row(
    *,
    prior: str,
    dataset_split: str,
    step: int,
    stage_bits: list[float | None],
    stage_counts: list[int],
    loss_nats: float,
    metadata: PriorMetadata,
    context_window_frames: int,
    context_scope: str,
) -> dict:
    valid_stage_bits = [float(value) for value in stage_bits if value is not None]
    bits_per_code = sum(valid_stage_bits) / len(valid_stage_bits) if valid_stage_bits else None
    estimated_bitrate = float(metadata.frame_rate * sum(valid_stage_bits)) / 1000.0
    return {
        "schema_version": CODE_PRIOR_SCHEMA_VERSION,
        "prior_family": prior,
        "prior_name": prior,
        "dataset_split": dataset_split,
        "step": step,
        "token_ordering": TOKEN_ORDERING,
        "context_scope": context_scope,
        "context_window_frames": context_window_frames,
        "context_window_seconds": context_window_frames / float(metadata.frame_rate),
        "loss_nats": loss_nats,
        "bits_per_code": bits_per_code,
        "stage_bits_per_code": stage_bits,
        "stage_token_counts": stage_counts,
        "estimated_entropy_bitrate_kbps": estimated_bitrate,
        "nominal_bitrate_kbps": metadata.nominal_bitrate_kbps,
        "entropy_savings_ratio": 1.0 - estimated_bitrate / metadata.nominal_bitrate_kbps,
        "frame_rate": metadata.frame_rate,
        "num_quantizers": metadata.num_quantizers,
        "codebook_size": metadata.codebook_size,
        "nominal_bits_per_code": metadata.bits_per_code,
    }


@torch.no_grad()
def _evaluate(
    *,
    model: nn.Module,
    sequences: list[torch.Tensor],
    metadata: PriorMetadata,
    args: argparse.Namespace,
    device: torch.device,
    dataset_split: str,
    step: int,
    context_window_frames: int,
    context_scope: str,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    stage_loss_bits = [0.0 for _ in range(metadata.num_quantizers)]
    stage_counts = [0 for _ in range(metadata.num_quantizers)]
    for inputs, targets, target_positions in _all_eval_batches(
        sequences,
        sequence_length=args.sequence_length,
        device=device,
    ):
        stages = _stage_ids(target_positions - 1, metadata.num_quantizers)
        logits = model(inputs, stages)
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)
        total_loss += float(losses.sum().item())
        total_tokens += int(losses.numel())
        target_stages = _stage_ids(target_positions, metadata.num_quantizers)
        for stage in range(metadata.num_quantizers):
            mask = target_stages.eq(stage)
            if mask.any():
                stage_loss_bits[stage] += float((losses[mask] / math.log(2.0)).sum().item())
                stage_counts[stage] += int(mask.sum().item())
    stage_bits = [
        stage_loss_bits[stage] / stage_counts[stage] if stage_counts[stage] else None
        for stage in range(metadata.num_quantizers)
    ]
    return _metric_row(
        prior=args.prior,
        dataset_split=dataset_split,
        step=step,
        stage_bits=stage_bits,
        stage_counts=stage_counts,
        loss_nats=total_loss / float(total_tokens),
        metadata=metadata,
        context_window_frames=context_window_frames,
        context_scope=context_scope,
    )


def train_code_prior(
    *,
    args: argparse.Namespace,
    export_dir: Path,
    manifest_path: Path,
    eval_export_dir: Path | None,
    eval_manifest_path: Path | None,
    output_dir: Path,
) -> dict:
    if args.steps < 1:
        raise ValueError("steps must be positive.")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if args.sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")
    torch.manual_seed(args.seed)
    generator = random.Random(args.seed)
    device = _resolve_device(args.device)

    train_sequences_raw = _load_sequences(manifest_path, max_items=args.max_train_items)
    resolved_eval_manifest = eval_manifest_path
    if resolved_eval_manifest is None and eval_export_dir is not None:
        resolved_eval_manifest = eval_export_dir / "manifest.jsonl"
    eval_sequences_raw = (
        _load_sequences(resolved_eval_manifest, max_items=args.max_eval_items)
        if resolved_eval_manifest is not None
        else train_sequences_raw
    )
    metadata = _metadata(
        train_sequences=train_sequences_raw,
        eval_sequences=eval_sequences_raw,
        codebook_size_override=args.codebook_size,
    )
    train_sequences = _flatten_sequences(train_sequences_raw)
    eval_sequences = _flatten_sequences(eval_sequences_raw)
    model = _build_model(args, metadata).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    context_tokens = int(getattr(model, "receptive_tokens", args.sequence_length - 1))
    context_window_frames = max(1, math.ceil(context_tokens / metadata.num_quantizers))
    context_scope = "local" if args.prior == "local_tcn" else "long"

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for step in range(1, args.steps + 1):
        model.train()
        inputs, targets, target_positions = _sample_batch(
            train_sequences,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            device=device,
            generator=generator,
        )
        input_stages = _stage_ids(target_positions - 1, metadata.num_quantizers)
        logits = model(inputs, input_stages)
        loss, stage_bits, stage_counts = _loss_and_stage_bits(
            logits=logits,
            targets=targets,
            target_positions=target_positions,
            num_quantizers=metadata.num_quantizers,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step == args.steps or step % args.eval_every == 0:
            train_rows.append(
                _metric_row(
                    prior=args.prior,
                    dataset_split="train",
                    step=step,
                    stage_bits=stage_bits,
                    stage_counts=stage_counts,
                    loss_nats=float(loss.item()),
                    metadata=metadata,
                    context_window_frames=context_window_frames,
                    context_scope=context_scope,
                )
            )
            val_rows.append(
                _evaluate(
                    model=model,
                    sequences=eval_sequences,
                    metadata=metadata,
                    args=args,
                    device=device,
                    dataset_split="heldout" if resolved_eval_manifest is not None else "self_eval",
                    step=step,
                    context_window_frames=context_window_frames,
                    context_scope=context_scope,
                )
            )

    ensure_directory(output_dir)
    write_jsonl(output_dir / "train_metrics.jsonl", train_rows)
    write_jsonl(output_dir / "val_metrics.jsonl", val_rows)
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "schema_version": CODE_PRIOR_SCHEMA_VERSION,
            "prior_family": args.prior,
            "model": model.state_dict(),
            "step": args.steps,
            "metadata": metadata.__dict__,
            "token_ordering": TOKEN_ORDERING,
        },
        checkpoint_path,
    )
    config = {
        "schema_version": CODE_PRIOR_SCHEMA_VERSION,
        "export_dir": str(export_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "eval_export_dir": str(eval_export_dir.resolve()) if eval_export_dir is not None else None,
        "eval_manifest_path": str(resolved_eval_manifest.resolve()) if resolved_eval_manifest is not None else None,
        "prior": args.prior,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "kernel_size": args.kernel_size,
        "transformer_heads": args.transformer_heads,
        "dropout": args.dropout,
        "device": str(device),
        "seed": args.seed,
        "token_ordering": TOKEN_ORDERING,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=True))
    summary = {
        "schema_version": CODE_PRIOR_SCHEMA_VERSION,
        "prior_family": args.prior,
        "prior_name": args.prior,
        "token_ordering": TOKEN_ORDERING,
        "frame_rate": metadata.frame_rate,
        "num_quantizers": metadata.num_quantizers,
        "codebook_size": metadata.codebook_size,
        "nominal_bits_per_code": metadata.bits_per_code,
        "nominal_bitrate_kbps": metadata.nominal_bitrate_kbps,
        "context_scope": context_scope,
        "context_window_frames": context_window_frames,
        "context_window_seconds": context_window_frames / float(metadata.frame_rate),
        "train_items": len(train_sequences_raw),
        "eval_items": len(eval_sequences_raw),
        "final_train": train_rows[-1],
        "final_val": val_rows[-1],
        "checkpoint_path": str(checkpoint_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True))
    return summary


def main() -> None:
    args = parse_args()
    export_dir = args.export_dir.resolve()
    manifest_path = args.manifest.resolve() if args.manifest is not None else export_dir / "manifest.jsonl"
    eval_export_dir = args.eval_export_dir.resolve() if args.eval_export_dir is not None else None
    eval_manifest_path = args.eval_manifest.resolve() if args.eval_manifest is not None else None
    output_dir = args.output_dir.resolve() if args.output_dir is not None else export_dir / f"{args.prior}_prior"
    summary = train_code_prior(
        args=args,
        export_dir=export_dir,
        manifest_path=manifest_path,
        eval_export_dir=eval_export_dir,
        eval_manifest_path=eval_manifest_path,
        output_dir=output_dir,
    )
    final_val = summary["final_val"]
    print(
        f"Wrote {args.prior} prior to {output_dir}; "
        f"final val bits/code={final_val['bits_per_code']:.4f}, "
        f"estimated bitrate={final_val['estimated_entropy_bitrate_kbps']:.4f} kbps"
    )


if __name__ == "__main__":
    main()
