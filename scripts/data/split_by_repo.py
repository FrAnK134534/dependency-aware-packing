#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dapacking.edges import read_dependency_edges, write_dependency_edges
from dapacking.io import read_documents, write_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split documents by repository to avoid leakage.")
    parser.add_argument("--documents", required=True, type=Path)
    parser.add_argument("--edges", type=Path, help="Optional dependency edges JSONL.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_ratio = args.train_ratio + args.validation_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise SystemExit("Split ratios must sum to 1.0.")

    documents = read_documents(args.documents)
    repos = sorted({document.repo for document in documents})
    rng = random.Random(args.seed)
    rng.shuffle(repos)

    train_end = int(len(repos) * args.train_ratio)
    validation_end = train_end + int(len(repos) * args.validation_ratio)
    split_repos = {
        "train": set(repos[:train_end]),
        "validation": set(repos[train_end:validation_end]),
        "test": set(repos[validation_end:]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_docids: dict[str, set[str]] = {}
    for split, repo_set in split_repos.items():
        split_documents = [document for document in documents if document.repo in repo_set]
        split_docids[split] = {document.docid for document in split_documents}
        write_documents(args.output_dir / f"{split}_docs.jsonl", split_documents)
        print(f"{split}_repos={len(repo_set)} {split}_documents={len(split_documents)}")

    if args.edges:
        edges = read_dependency_edges(args.edges)
        for split, docids in split_docids.items():
            split_edges = [
                edge
                for edge in edges
                if edge.source_docid in docids and edge.target_docid in docids
            ]
            write_dependency_edges(args.output_dir / f"{split}_edges.jsonl", split_edges)
            print(f"{split}_edges={len(split_edges)}")


if __name__ == "__main__":
    main()
