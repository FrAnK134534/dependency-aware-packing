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

from dapacking.audit import read_review_records, render_review_markdown, summarize_review_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize manual dependency edge review labels.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_review_records(args.input)
    summaries = summarize_review_records(records)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = [summary.to_row() for summary in summaries]
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    report = render_review_markdown(summaries, records)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
