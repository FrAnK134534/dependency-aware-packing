#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dapacking.stats import PackingSummary, summarize_packed_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize packed JSONL files.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Packed JSONL files.")
    parser.add_argument("--edges", type=Path, help="Optional dependency edges JSONL for coverage metrics.")
    parser.add_argument("--output", type=Path, help="Optional CSV output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [summarize_packed_file(path, args.edges) for path in args.inputs]
    rows = [summary.to_row() for summary in summaries]

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print_markdown_table(summaries)


def print_markdown_table(summaries: list[PackingSummary]) -> None:
    headers = [
        "method",
        "samples",
        "total_tokens",
        "avg_tokens",
        "avg_docs",
        "avg_dep",
        "order_dep",
        "edge_cov",
        "w_edge_cov",
        "strong_cov",
        "weak_cov",
        "avg_util",
        "avg_trunc",
        "same_repo",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for summary in summaries:
        print(
            "| "
            + " | ".join(
                [
                    summary.method,
                    str(summary.samples),
                    str(summary.total_tokens),
                    f"{summary.avg_tokens:.2f}",
                    f"{summary.avg_num_docs:.2f}",
                    f"{summary.avg_dependency_score:.4f}",
                    f"{summary.avg_order_dependency:.4f}",
                    f"{summary.edge_coverage:.4f}",
                    f"{summary.weighted_edge_coverage:.4f}",
                    f"{summary.weighted_strong_edge_coverage:.4f}",
                    f"{summary.weighted_weak_edge_coverage:.4f}",
                    f"{summary.avg_token_utilization:.4f}",
                    f"{summary.avg_truncation_rate:.4f}",
                    f"{summary.same_repo_pair_ratio:.4f}",
                ]
            )
            + " |"
        )


if __name__ == "__main__":
    main()
