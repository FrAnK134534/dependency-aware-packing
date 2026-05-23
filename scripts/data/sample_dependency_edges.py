#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dapacking.edges import read_dependency_edges
from dapacking.io import write_jsonl
from dapacking.io import read_documents
from dapacking.review import EdgeReviewConfig, sample_edge_review_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample dependency edges for manual review."
    )
    parser.add_argument("--documents", required=True, type=Path)
    parser.add_argument("--edges", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strong-only", action="store_true")
    parser.add_argument("--excerpt-chars", type=int, default=360)
    parser.add_argument("--format", choices=("csv", "jsonl"), default="csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = read_documents(args.documents)
    edges = read_dependency_edges(args.edges)
    records = sample_edge_review_records(
        documents,
        edges,
        EdgeReviewConfig(
            sample_size=args.sample_size,
            seed=args.seed,
            include_weak=not args.strong_only,
            excerpt_chars=args.excerpt_chars,
        ),
    )
    write_records(args.output, records, args.format)
    strong_count = sum(1 for record in records if record["is_strong"])
    print(f"documents={len(documents)}")
    print(f"edges={len(edges)}")
    print(f"sampled={len(records)}")
    print(f"sampled_strong_edges={strong_count}")
    print(f"output={args.output}")


def write_records(path: Path, records: list[dict[str, object]], output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl":
        write_jsonl(path, records)
        return

    fieldnames = list(records[0]) if records else [
        "review_id",
        "relation",
        "labels",
        "is_strong",
        "weight",
        "source_docid",
        "target_docid",
        "source_repo",
        "target_repo",
        "source_path",
        "target_path",
        "source_type",
        "target_type",
        "source_excerpt",
        "target_excerpt",
        "manual_reasonable",
        "manual_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
