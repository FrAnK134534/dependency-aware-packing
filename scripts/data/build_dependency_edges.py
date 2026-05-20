#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dapacking.edges import build_dependency_edges, write_dependency_edges
from dapacking.io import read_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dependency_edges.jsonl from documents.jsonl.")
    parser.add_argument("--input", required=True, type=Path, help="Input documents JSONL.")
    parser.add_argument("--output", required=True, type=Path, help="Output dependency edges JSONL.")
    parser.add_argument("--min-score", type=float, default=0.11)
    parser.add_argument("--include-same-repo-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = read_documents(args.input)
    edges = build_dependency_edges(
        documents,
        min_score=args.min_score,
        include_same_repo_only=args.include_same_repo_only,
    )
    write_dependency_edges(args.output, edges)
    print(f"documents={len(documents)}")
    print(f"edges={len(edges)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
