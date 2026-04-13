#!/usr/bin/env python3
"""CLI entry point for training the baseline speech codec."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from audiocodec.config import load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Conv + RVQ speech codec baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline-train"))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--limit-train-examples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--tensorboard", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.dataset_root is not None:
        config.dataset.root = args.dataset_root

    try:
        from audiocodec.train import train_codec
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "torchaudio"}:
            raise SystemExit(
                "Training requires `torch` and `torchaudio` to be installed in the active environment."
            ) from exc
        raise

    train_codec(
        config=config,
        output_dir=args.output_dir,
        steps=args.steps,
        smoke_test=args.smoke_test,
        limit_train_examples=args.limit_train_examples,
        device=args.device,
        tensorboard=args.tensorboard,
    )


if __name__ == "__main__":
    main()
