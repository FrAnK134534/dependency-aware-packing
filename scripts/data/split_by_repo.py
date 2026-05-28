#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(
        description="Split documents by repository/collection group to avoid leakage."
    )
    parser.add_argument("--documents", required=True, type=Path)
    parser.add_argument("--edges", type=Path, help="Optional dependency edges JSONL.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--group-key",
        choices=("auto", "repo", "collection", "document_id"),
        default="auto",
        help="Group key used for leakage-safe splitting. 'auto' prefers repo, then collection.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_ratio = args.train_ratio + args.validation_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise SystemExit("Split ratios must sum to 1.0.")

    documents = read_documents(args.documents)
    groups = sorted({_group_value(document, args.group_key) for document in documents})
    rng = random.Random(args.seed)
    rng.shuffle(groups)

    train_count, validation_count, _test_count = _split_counts(
        len(groups),
        args.train_ratio,
        args.validation_ratio,
        args.test_ratio,
    )
    train_end = train_count
    validation_end = train_end + validation_count
    split_groups = {
        "train": set(groups[:train_end]),
        "validation": set(groups[train_end:validation_end]),
        "test": set(groups[validation_end:]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_docids: dict[str, set[str]] = {}
    for split, group_set in split_groups.items():
        split_documents = [
            document
            for document in documents
            if _group_value(document, args.group_key) in group_set
        ]
        split_docids[split] = {document.docid for document in split_documents}
        write_documents(args.output_dir / f"{split}_docs.jsonl", split_documents)
        print(f"{split}_groups={len(group_set)} {split}_documents={len(split_documents)}")

    split_manifest = {
        split: sorted(group_set) for split, group_set in split_groups.items()
    }
    (args.output_dir / "split_groups.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

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


def _group_value(document, group_key: str) -> str:
    if group_key == "repo":
        return document.repo or "__missing_repo__"
    if group_key == "collection":
        return document.collection or "__missing_collection__"
    if group_key == "document_id":
        return document.document_id or document.docid
    return document.repo or document.collection or document.document_id or "__unknown__"


def _split_counts(
    total: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    train_count = int(total * train_ratio)
    validation_count = int(total * validation_ratio)

    if train_ratio > 0 and train_count == 0:
        train_count = 1
    if validation_ratio > 0 and total - train_count > 1 and validation_count == 0:
        validation_count = 1

    test_count = total - train_count - validation_count
    if test_ratio > 0 and test_count == 0 and total >= 3:
        if train_count >= validation_count and train_count > 1:
            train_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        test_count = total - train_count - validation_count

    if test_count < 0:
        overflow = -test_count
        validation_reduction = min(validation_count, overflow)
        validation_count -= validation_reduction
        overflow -= validation_reduction
        train_count = max(train_count - overflow, 0)
        test_count = total - train_count - validation_count
    return train_count, validation_count, test_count


if __name__ == "__main__":
    main()
