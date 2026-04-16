#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _common import examples_to_manifest_rows, load_split_examples, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic benchmark manifest from the dataset split.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, examples = load_split_examples(args.config, args.split)
    rows = examples_to_manifest_rows(config, examples, limit=args.limit)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
