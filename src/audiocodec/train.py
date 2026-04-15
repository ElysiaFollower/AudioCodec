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

from audiocodec.adversarial import (
    FeatureMatchingLoss,
    MultiScaleSTFTDiscriminator,
    discriminator_adversarial_loss,
    generator_adversarial_loss,
)
from audiocodec.balancer import Balancer
from audiocodec.config import CodecExperimentConfig
from audiocodec.data.librispeech import (
    SpeechSegmentDataset,
    build_librispeech_splits,
    build_single_file_overfit_splits,
)
from audiocodec.losses import CodecLoss
from audiocodec.models.codec import BaseCodecModel, build_codec_model


def _load_torchaudio():
    import torchaudio

    return torchaudio


@dataclass(slots=True)
class TrainingArtifacts:
    output_dir: Path
    resolved_config_path: Path
    metrics_path: Path
    tensorboard_dir: Path | None = None


@dataclass(slots=True)
class AdversarialComponents:
    discriminator: MultiScaleSTFTDiscriminator
    optimizer: torch.optim.Optimizer
    feature_matching_loss: FeatureMatchingLoss


def build_balancer(config: CodecExperimentConfig) -> Balancer | None:
    if not config.balancer.enabled:
        return None

    weights: dict[str, float] = {}
    if config.loss.waveform_weight > 0:
        weights["waveform_loss"] = config.loss.waveform_weight
    if config.loss.stft_weight > 0:
        weights["stft_loss"] = config.loss.stft_weight
    if config.loss.mel_weight > 0:
        weights["mel_loss"] = config.loss.mel_weight
    if config.adversarial.enabled and config.adversarial.adversarial_weight > 0:
        weights["generator_adversarial_loss"] = config.adversarial.adversarial_weight
    if config.adversarial.enabled and config.adversarial.feature_matching_weight > 0:
        weights["feature_matching_loss"] = config.adversarial.feature_matching_weight
    return Balancer(
        weights=weights,
        balance_grads=config.balancer.balance_grads,
        total_norm=config.balancer.total_norm,
        ema_decay=config.balancer.ema_decay,
        per_batch_item=config.balancer.per_batch_item,
        epsilon=config.balancer.epsilon,
    )


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
    overfit_example_path: str | None = None,
) -> tuple[DataLoader, DataLoader]:
    overfit_mode = overfit_example_path is not None
    if overfit_mode:
        splits = build_single_file_overfit_splits(overfit_example_path)
    else:
        splits = build_librispeech_splits(
            root=config.dataset.root,
            train_minutes=config.dataset.train_minutes,
            val_minutes=config.dataset.val_minutes,
            test_minutes=config.dataset.test_minutes,
        )

    train_examples = splits.train
    if limit_train_examples is not None and not overfit_mode:
        train_examples = train_examples[:limit_train_examples]
        if not train_examples:
            raise ValueError("limit_train_examples truncated the training split to zero examples.")

    train_dataset = SpeechSegmentDataset(
        examples=train_examples,
        sample_rate=config.audio.sample_rate,
        channels=config.audio.channels,
        clip_seconds=config.audio.train_clip_seconds,
        random_crop=not overfit_mode,
    )
    val_dataset = SpeechSegmentDataset(
        examples=train_examples if overfit_mode else splits.val,
        sample_rate=config.audio.sample_rate,
        channels=config.audio.channels,
        clip_seconds=config.audio.eval_clip_seconds,
        random_crop=False,
    )

    batch_size = 1 if overfit_mode else config.optimization.batch_size
    num_workers = 0 if overfit_mode else config.optimization.num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not overfit_mode,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader


def build_loss(config: CodecExperimentConfig) -> CodecLoss:
    return CodecLoss(
        sample_rate=config.audio.sample_rate,
        waveform_weight=config.loss.waveform_weight,
        stft_weight=config.loss.stft_weight,
        mel_weight=config.loss.mel_weight,
        commitment_weight=config.quantizer.commitment_weight,
        codebook_weight=config.quantizer.codebook_weight,
        fft_sizes=config.loss.fft_sizes,
        mel_n_mels=config.loss.mel_n_mels,
        mel_fft_size=config.loss.mel_fft_size,
        mel_hop_length=config.loss.mel_hop_length,
        mel_f_min=config.loss.mel_f_min,
        mel_f_max=config.loss.mel_f_max,
    )


def build_adversarial_components(
    config: CodecExperimentConfig,
    device: torch.device,
) -> AdversarialComponents | None:
    if not config.adversarial.enabled:
        return None

    discriminator = MultiScaleSTFTDiscriminator(
        filters=config.adversarial.discriminator_filters,
        in_channels=config.audio.channels,
        n_ffts=config.adversarial.n_ffts,
        hop_lengths=config.adversarial.hop_lengths,
        win_lengths=config.adversarial.win_lengths,
    ).to(device)
    optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.adversarial.discriminator_learning_rate,
        betas=config.adversarial.discriminator_betas,
        weight_decay=config.adversarial.discriminator_weight_decay,
    )
    return AdversarialComponents(
        discriminator=discriminator,
        optimizer=optimizer,
        feature_matching_loss=FeatureMatchingLoss(),
    )


def build_generator_optimizer(
    model: nn.Module,
    config: CodecExperimentConfig,
) -> torch.optim.Optimizer:
    if config.optimization.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.optimization.learning_rate,
            betas=config.optimization.betas,
            weight_decay=config.optimization.weight_decay,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        betas=config.optimization.betas,
        weight_decay=config.optimization.weight_decay,
    )


def append_metrics(metrics_path: Path, payload: dict) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def create_tensorboard_writer(output_dir: Path, enabled: bool):
    if not enabled:
        return None

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorBoard logging was requested, but the active environment does not provide it. "
            "Install `tensorboard` or run without `--tensorboard`."
        ) from exc

    log_dir = output_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: CodecExperimentConfig,
    metrics: dict[str, float],
    discriminator: nn.Module | None = None,
    discriminator_optimizer: torch.optim.Optimizer | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config.to_dict(),
        "metrics": metrics,
    }
    if discriminator is not None:
        payload["discriminator"] = discriminator.state_dict()
    if discriminator_optimizer is not None:
        payload["discriminator_optimizer"] = discriminator_optimizer.state_dict()
    torch.save(payload, path)


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
    source_audio = source[0].detach().float().cpu().clamp(-1.0, 1.0)
    reconstruction_audio = reconstruction[0].detach().float().cpu().clamp(-1.0, 1.0)
    torchaudio.save(str(source_path), source_audio, sample_rate)
    torchaudio.save(str(recon_path), reconstruction_audio, sample_rate)


@torch.no_grad()
def evaluate(
    model: BaseCodecModel,
    criterion: CodecLoss,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int,
    amp_enabled: bool,
) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor] | None]:
    model.eval()

    totals: dict[str, float] = {}
    num_batches = 0
    preview: tuple[torch.Tensor, torch.Tensor] | None = None
    for batch in dataloader:
        batch = batch.to(device)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with autocast_context:
            output = model(batch)
        losses = criterion(
            reconstruction=output.reconstruction.float(),
            target=batch.float(),
            commitment_loss=output.commitment_loss.float(),
            codebook_loss=output.codebook_loss.float(),
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


def _ensure_finite_metrics(metrics: dict[str, torch.Tensor] | dict[str, float], split: str, step: int) -> None:
    for name, value in metrics.items():
        numeric = float(value.item()) if isinstance(value, torch.Tensor) else float(value)
        if not torch.isfinite(torch.tensor(numeric)):
            raise FloatingPointError(f"Non-finite {split} metric `{name}` detected at step {step}.")


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def train_codec(
    config: CodecExperimentConfig,
    output_dir: str | Path,
    steps: int | None = None,
    smoke_test: bool = False,
    limit_train_examples: int | None = None,
    overfit_example_path: str | None = None,
    device: str | None = None,
    tensorboard: bool = False,
) -> TrainingArtifacts:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(
            f"Output directory {output_path} already contains metrics.jsonl. "
            "Use a fresh --output-dir because resume/mixed-run logging is not supported."
        )

    set_seed(config.optimization.seed)
    resolved_device = resolve_device(device)
    train_loader, val_loader = build_dataloaders(
        config,
        limit_train_examples=limit_train_examples,
        overfit_example_path=overfit_example_path,
    )

    model = build_codec_model(config).to(resolved_device)
    criterion = build_loss(config).to(resolved_device)
    adversarial_components = build_adversarial_components(config, device=resolved_device)
    balancer = build_balancer(config)
    optimizer = build_generator_optimizer(model=model, config=config)

    amp_enabled = (
        config.optimization.mixed_precision
        and resolved_device.type == "cuda"
        and adversarial_components is None
        and balancer is None
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    train_iterator = cycle_dataloader(train_loader)
    writer = create_tensorboard_writer(output_path, enabled=tensorboard)

    resolved_config_path = output_path / "resolved_config.json"
    resolved_payload = config.to_dict()
    if overfit_example_path is not None:
        resolved_payload.setdefault("debug", {})
        resolved_payload["debug"].update(
            {
                "overfit_example_path": str(Path(overfit_example_path).expanduser().resolve()),
                "effective_batch_size": 1,
                "effective_num_workers": 0,
                "random_crop": False,
            }
        )
    resolved_payload.setdefault("runtime", {})
    resolved_payload["runtime"]["amp_enabled"] = amp_enabled
    resolved_config_path.write_text(json.dumps(resolved_payload, indent=2))
    if writer is not None:
        writer.add_text("config/json", json.dumps(resolved_payload, indent=2))

    total_steps = steps
    if total_steps is None:
        total_steps = config.optimization.smoke_test_steps if smoke_test else config.optimization.main_steps

    best_val_loss = float("inf")
    try:
        for step in range(1, total_steps + 1):
            started_at = time.perf_counter()
            model.train()
            if adversarial_components is not None:
                adversarial_components.discriminator.train()
            batch = next(train_iterator).to(resolved_device)

            optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            )
            with autocast_context:
                output = model(batch)
            # Keep loss computation in fp32 so STFT/mel terms don't hit complex-half instabilities.
            losses = criterion(
                reconstruction=output.reconstruction.float(),
                target=batch.float(),
                commitment_loss=output.commitment_loss.float(),
                codebook_loss=output.codebook_loss.float(),
            )
            objective_total_loss = losses["total_loss"]
            quantizer_total_loss = (
                config.quantizer.commitment_weight * losses["commitment_loss"]
                + config.quantizer.codebook_weight * losses["codebook_loss"]
            )

            discriminator_loss: torch.Tensor | None = None
            generator_adv_loss_value: torch.Tensor | None = None
            feature_matching_loss_value: torch.Tensor | None = None

            if adversarial_components is not None:
                discriminator = adversarial_components.discriminator

                adversarial_components.optimizer.zero_grad(set_to_none=True)
                fake_logits_for_d, _ = discriminator(output.reconstruction.detach().float())
                real_logits_for_d, _ = discriminator(batch.float())
                discriminator_loss = discriminator_adversarial_loss(
                    fake_logits=fake_logits_for_d,
                    real_logits=real_logits_for_d,
                    loss_type=config.adversarial.loss_type,
                )
                _ensure_finite_metrics(
                    {"discriminator_loss": discriminator_loss},
                    split="train",
                    step=step,
                )
                discriminator_loss.backward()
                adversarial_components.optimizer.step()

                _set_requires_grad(discriminator, False)
                fake_logits_for_g, fake_feature_maps = discriminator(output.reconstruction.float())
                with torch.no_grad():
                    _, real_feature_maps = discriminator(batch.float())
                generator_adv_loss_value = generator_adversarial_loss(
                    fake_logits=fake_logits_for_g,
                    loss_type=config.adversarial.loss_type,
                )
                feature_matching_loss_value = adversarial_components.feature_matching_loss(
                    fake_feature_maps=fake_feature_maps,
                    real_feature_maps=real_feature_maps,
                )
                _set_requires_grad(discriminator, True)

            balanced_losses: dict[str, torch.Tensor] = {}
            if balancer is not None:
                if config.loss.waveform_weight > 0:
                    balanced_losses["waveform_loss"] = losses["waveform_loss"]
                if config.loss.stft_weight > 0:
                    balanced_losses["stft_loss"] = losses["stft_loss"]
                if config.loss.mel_weight > 0:
                    balanced_losses["mel_loss"] = losses["mel_loss"]
                if generator_adv_loss_value is not None and config.adversarial.adversarial_weight > 0:
                    balanced_losses["generator_adversarial_loss"] = generator_adv_loss_value
                if feature_matching_loss_value is not None and config.adversarial.feature_matching_weight > 0:
                    balanced_losses["feature_matching_loss"] = feature_matching_loss_value

            if balancer is not None:
                if quantizer_total_loss.requires_grad:
                    quantizer_total_loss.backward(retain_graph=True)
                balanced_total_loss = balancer.backward(
                    losses=balanced_losses,
                    input_tensor=output.reconstruction.float(),
                )
                total_loss = balanced_total_loss + quantizer_total_loss.detach()
            elif adversarial_components is not None:
                total_loss = (
                    objective_total_loss
                    + config.adversarial.adversarial_weight * generator_adv_loss_value
                    + config.adversarial.feature_matching_weight * feature_matching_loss_value
                )
            else:
                total_loss = objective_total_loss

            finite_metrics: dict[str, torch.Tensor] = dict(losses)
            finite_metrics["generator_total_loss"] = total_loss
            if discriminator_loss is not None:
                finite_metrics["discriminator_loss"] = discriminator_loss
            if generator_adv_loss_value is not None:
                finite_metrics["generator_adversarial_loss"] = generator_adv_loss_value
            if feature_matching_loss_value is not None:
                finite_metrics["feature_matching_loss"] = feature_matching_loss_value
            _ensure_finite_metrics(finite_metrics, split="train", step=step)

            if amp_enabled:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                if balancer is None:
                    total_loss.backward()
                optimizer.step()

            step_metrics = {name: float(value.item()) for name, value in losses.items()}
            step_metrics["generator_total_loss"] = float(total_loss.item())
            if discriminator_loss is not None:
                step_metrics["discriminator_loss"] = float(discriminator_loss.item())
            if generator_adv_loss_value is not None:
                step_metrics["generator_adversarial_loss"] = float(generator_adv_loss_value.item())
            if feature_matching_loss_value is not None:
                step_metrics["feature_matching_loss"] = float(feature_matching_loss_value.item())
            if balancer is not None:
                step_metrics.update(balancer.metrics)
            step_metrics["step_time_seconds"] = time.perf_counter() - started_at

            if step == 1 or step % config.optimization.log_interval == 0:
                append_metrics(metrics_path, {"split": "train", "step": step, **step_metrics})
                if writer is not None:
                    for name, value in step_metrics.items():
                        writer.add_scalar(f"train/{name}", value, global_step=step)
                message = (
                    f"[train] step={step} total={step_metrics['generator_total_loss']:.4f} "
                    f"waveform={step_metrics['waveform_loss']:.4f} stft={step_metrics['stft_loss']:.4f}"
                )
                if "generator_adversarial_loss" in step_metrics:
                    message += (
                        f" adv={step_metrics['generator_adversarial_loss']:.4f}"
                        f" feat={step_metrics['feature_matching_loss']:.4f}"
                        f" d={step_metrics['discriminator_loss']:.4f}"
                    )
                print(message)

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
                    amp_enabled=amp_enabled,
                )
                _ensure_finite_metrics(val_metrics, split="val", step=step)
                append_metrics(metrics_path, {"split": "val", "step": step, **val_metrics})
                if writer is not None:
                    for name, value in val_metrics.items():
                        writer.add_scalar(f"val/{name}", value, global_step=step)
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
                    discriminator=(
                        adversarial_components.discriminator if adversarial_components is not None else None
                    ),
                    discriminator_optimizer=(
                        adversarial_components.optimizer if adversarial_components is not None else None
                    ),
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
                        discriminator=(
                            adversarial_components.discriminator if adversarial_components is not None else None
                        ),
                        discriminator_optimizer=(
                            adversarial_components.optimizer if adversarial_components is not None else None
                        ),
                    )
                model.train()
    finally:
        if writer is not None:
            writer.close()

    tensorboard_dir = output_path / "tensorboard" if tensorboard else None
    return TrainingArtifacts(
        output_dir=output_path,
        resolved_config_path=resolved_config_path,
        metrics_path=metrics_path,
        tensorboard_dir=tensorboard_dir,
    )
