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

from dapacking.edge_annotation import annotate_edge_review_record
from dapacking.io import read_documents


ASSISTANT_FIELDS = [
    "assistant_review_label",
    "assistant_confidence",
    "assistant_error_type",
    "assistant_note",
    "assistant_annotation_policy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add assistant-assisted edge-review suggestions to a sampled review CSV."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--documents", type=Path, help="Optional full document JSONL for stronger evidence checks.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--write-manual-columns",
        action="store_true",
        help="Also copy assistant labels into manual_* columns. Use only for a clearly disclosed assisted audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents_by_id = {}
    if args.documents:
        documents_by_id = {document.docid: document for document in read_documents(args.documents)}

    with args.input.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    counts: dict[str, int] = {}
    for row in rows:
        annotation = annotate_edge_review_record(row, documents_by_id)
        row["assistant_review_label"] = annotation.label
        row["assistant_confidence"] = f"{annotation.confidence:.2f}"
        row["assistant_error_type"] = annotation.error_type
        row["assistant_note"] = annotation.note
        row["assistant_annotation_policy"] = "assistant_suggestion_not_human_review"
        counts[annotation.label] = counts.get(annotation.label, 0) + 1

        if args.write_manual_columns:
            row["manual_reasonable"] = annotation.label
            row["manual_confidence"] = f"{annotation.confidence:.2f}"
            row["manual_error_type"] = annotation.error_type
            row["manual_note"] = f"[assistant-assisted] {annotation.note}"

    fieldnames = list(rows[0]) if rows else []
    for field in ASSISTANT_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"records={len(rows)}")
    for label in sorted(counts):
        print(f"{label}={counts[label]}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
