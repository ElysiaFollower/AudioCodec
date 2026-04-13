"""Training loop for the baseline speech codec."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from audiocodec.config import CodecExperimentConfig
from audiocodec.data.librispeech import SpeechSegmentDataset, build_librispeech_splits
from audiocodec.losses import CodecLoss
from audiocodec.models.codec import ConvRVQCodec


def _load_torchaudio():
    import torchaudio

    return torchaudio


@dataclass(slots=True)
class TrainingArtifacts:
    output_dir: Path
    resolved_config_path: Path
    metrics_path: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str | None) -> torch.device:
    if requested_device and requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cycle_dataloader(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch


def build_dataloaders(
    config: CodecExperimentConfig,
    limit_train_examples: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    splits = build_librispeech_splits(
        root=config.dataset.root,
        train_minutes=config.dataset.train_minutes,
        val_minutes=config.dataset.val_minutes,
        test_minutes=config.dataset.test_minutes,
    )

    train_examples = splits.train
    if limit_train_examples is not None:
        train_examples = train_examples[:limit_train_examples]
        if not train_examples:
            raise ValueError("limit_train_examples truncated the training split to zero examples.")

    train_dataset = SpeechSegmentDataset(
        examples=train_examples,
        sample_rate=config.audio.sample_rate,
        channels=config.audio.channels,
        clip_seconds=config.audio.train_clip_seconds,
        random_crop=True,
    )
    val_dataset = SpeechSegmentDataset(
        examples=splits.val,
        sample_rate=config.audio.sample_rate,
        channels=config.audio.channels,
        clip_seconds=config.audio.eval_clip_seconds,
        random_crop=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.optimization.batch_size,
        shuffle=True,
        num_workers=config.optimization.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.optimization.batch_size,
        shuffle=False,
        num_workers=config.optimization.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader


def build_loss(config: CodecExperimentConfig) -> CodecLoss:
    return CodecLoss(
        waveform_weight=config.loss.waveform_weight,
        stft_weight=config.loss.stft_weight,
        commitment_weight=config.quantizer.commitment_weight,
        codebook_weight=config.quantizer.codebook_weight,
        fft_sizes=config.loss.fft_sizes,
    )


def append_metrics(metrics_path: Path, payload: dict) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: CodecExperimentConfig,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config.to_dict(),
            "metrics": metrics,
        },
        path,
    )


def save_reconstruction_examples(
    sample_dir: Path,
    step: int,
    source: torch.Tensor,
    reconstruction: torch.Tensor,
    sample_rate: int,
) -> None:
    torchaudio = _load_torchaudio()

    sample_dir.mkdir(parents=True, exist_ok=True)
    source_path = sample_dir / f"step-{step:06d}-source.wav"
    recon_path = sample_dir / f"step-{step:06d}-reconstruction.wav"
    torchaudio.save(str(source_path), source[0].cpu().clamp(-1.0, 1.0), sample_rate)
    torchaudio.save(str(recon_path), reconstruction[0].cpu().clamp(-1.0, 1.0), sample_rate)


@torch.no_grad()
def evaluate(
    model: ConvRVQCodec,
    criterion: CodecLoss,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor] | None]:
    model.eval()

    totals: dict[str, float] = {}
    num_batches = 0
    preview: tuple[torch.Tensor, torch.Tensor] | None = None
    for batch in dataloader:
        batch = batch.to(device)
        output = model(batch)
        losses = criterion(
            reconstruction=output.reconstruction,
            target=batch,
            commitment_loss=output.commitment_loss,
            codebook_loss=output.codebook_loss,
        )
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.item())
        if preview is None:
            preview = (batch.detach().cpu(), output.reconstruction.detach().cpu())
        num_batches += 1
        if num_batches >= max_batches:
            break

    if num_batches == 0:
        raise ValueError("Validation dataloader produced zero batches.")

    averages = {name: total / num_batches for name, total in totals.items()}
    return averages, preview


def train_codec(
    config: CodecExperimentConfig,
    output_dir: str | Path,
    steps: int | None = None,
    smoke_test: bool = False,
    limit_train_examples: int | None = None,
    device: str | None = None,
) -> TrainingArtifacts:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    set_seed(config.optimization.seed)
    resolved_device = resolve_device(device)
    train_loader, val_loader = build_dataloaders(config, limit_train_examples=limit_train_examples)

    model = ConvRVQCodec.from_config(config).to(resolved_device)
    criterion = build_loss(config).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )

    amp_enabled = config.optimization.mixed_precision and resolved_device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    train_iterator = cycle_dataloader(train_loader)

    resolved_config_path = output_path / "resolved_config.json"
    metrics_path = output_path / "metrics.jsonl"
    resolved_config_path.write_text(json.dumps(config.to_dict(), indent=2))

    total_steps = steps
    if total_steps is None:
        total_steps = config.optimization.smoke_test_steps if smoke_test else config.optimization.main_steps

    best_val_loss = float("inf")
    for step in range(1, total_steps + 1):
        started_at = time.perf_counter()
        model.train()
        batch = next(train_iterator).to(resolved_device)

        optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with autocast_context:
            output = model(batch)
            losses = criterion(
                reconstruction=output.reconstruction,
                target=batch,
                commitment_loss=output.commitment_loss,
                codebook_loss=output.codebook_loss,
            )
            total_loss = losses["total_loss"]

        if amp_enabled:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        step_metrics = {name: float(value.item()) for name, value in losses.items()}
        step_metrics["step_time_seconds"] = time.perf_counter() - started_at

        if step == 1 or step % config.optimization.log_interval == 0:
            append_metrics(metrics_path, {"split": "train", "step": step, **step_metrics})
            print(
                f"[train] step={step} total={step_metrics['total_loss']:.4f} "
                f"waveform={step_metrics['waveform_loss']:.4f} stft={step_metrics['stft_loss']:.4f}"
            )

        should_evaluate = (
            step == total_steps
            or step % config.optimization.eval_interval == 0
            or step % config.optimization.checkpoint_interval == 0
        )
        if should_evaluate:
            val_metrics, preview = evaluate(
                model=model,
                criterion=criterion,
                dataloader=val_loader,
                device=resolved_device,
                max_batches=config.optimization.max_eval_batches,
            )
            append_metrics(metrics_path, {"split": "val", "step": step, **val_metrics})
            print(
                f"[val] step={step} total={val_metrics['total_loss']:.4f} "
                f"waveform={val_metrics['waveform_loss']:.4f} stft={val_metrics['stft_loss']:.4f}"
            )

            if preview is not None:
                save_reconstruction_examples(
                    sample_dir=output_path / "samples",
                    step=step,
                    source=preview[0],
                    reconstruction=preview[1],
                    sample_rate=config.audio.sample_rate,
                )

            checkpoint_path = output_path / "checkpoints" / f"step-{step:06d}.pt"
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                step=step,
                config=config,
                metrics=val_metrics,
            )

            if val_metrics["total_loss"] < best_val_loss:
                best_val_loss = val_metrics["total_loss"]
                save_checkpoint(
                    output_path / "checkpoints" / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config=config,
                    metrics=val_metrics,
                )
            model.train()

    return TrainingArtifacts(
        output_dir=output_path,
        resolved_config_path=resolved_config_path,
        metrics_path=metrics_path,
    )
