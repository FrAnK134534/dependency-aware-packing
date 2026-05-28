#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dapacking.dataset_card import load_dataset_card_inputs, render_dataset_card


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a dataset card for a frozen corpus.")
    parser.add_argument("--documents", required=True, type=Path)
    parser.add_argument("--edges", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source-manifest", default="")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents, edges = load_dataset_card_inputs(args.documents, args.edges)
    report = render_dataset_card(
        args.name,
        documents,
        edges,
        split_dir=args.split_dir,
        source_manifest=args.source_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"documents={len(documents)}")
    print(f"edges={len(edges)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
