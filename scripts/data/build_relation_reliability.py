#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dapacking.audit import read_review_records, summarize_review_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build relation-level reliability YAML from manually annotated edge review. "
            "Assistant review labels are intentionally ignored."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--rate",
        choices=["supportive", "strict"],
        default="supportive",
        help="Use yes+partial or yes-only rate as relation reliability.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_review_records(args.input)
    summaries = summarize_review_records(records)

    reliability = {}
    summary_rows = {}
    for summary in summaries:
        rate = summary.supportive_rate if args.rate == "supportive" else summary.strict_precision
        reliability[summary.relation] = round(rate, 4)
        summary_rows[summary.relation] = summary.to_row()

    payload = {
        "source_review_file": str(args.input),
        "rate": args.rate,
        "manual_label_fields": ["manual_reasonable", "review_label", "manual_label", "label"],
        "ignored_label_fields": ["assistant_review_label", "assistant_label", "assistant_reasonable"],
        "relation_reliability": reliability,
        "relation_review_summary": summary_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=True)

    print(f"review_records={len(records)}")
    print(f"relations={len(summaries)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
