#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether dependency-aware gains are not just utilization artifacts."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--method", default="dependency_aware_v2_strong_first")
    parser.add_argument(
        "--baselines",
        default="random,length_aware,same_repo,bm25,semantic,datasculpt_lite",
    )
    parser.add_argument("--utilization-tolerance", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.summary)
    method_row = rows.get(args.method)
    if method_row is None:
        raise SystemExit(f"Method not found in summary: {args.method}")

    report_rows = []
    for baseline in [item.strip() for item in args.baselines.split(",") if item.strip()]:
        baseline_row = rows.get(baseline)
        if baseline_row is None:
            continue
        report_rows.append(_compare(method_row, baseline_row, args.utilization_tolerance))

    _print_markdown(report_rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
            writer.writeheader()
            writer.writerows(report_rows)


def _read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row["method"]): row for row in rows}


def _compare(
    method_row: dict[str, str],
    baseline_row: dict[str, str],
    utilization_tolerance: float,
) -> dict[str, object]:
    method_util = _float(method_row, "avg_token_utilization")
    baseline_util = _float(baseline_row, "avg_token_utilization")
    method_strong = _float(method_row, "weighted_strong_edge_coverage")
    baseline_strong = _float(baseline_row, "weighted_strong_edge_coverage")
    method_redundancy = _float(method_row, "avg_redundant_pair_rate")
    baseline_redundancy = _float(baseline_row, "avg_redundant_pair_rate")
    return {
        "method": method_row["method"],
        "baseline": baseline_row["method"],
        "utilization_delta": round(method_util - baseline_util, 4),
        "strong_coverage_delta": round(method_strong - baseline_strong, 4),
        "redundancy_delta": round(method_redundancy - baseline_redundancy, 4),
        "matched_utilization": method_util + utilization_tolerance >= baseline_util,
        "strong_coverage_improved": method_strong > baseline_strong,
    }


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except ValueError:
        return 0.0


def _print_markdown(rows: list[dict[str, object]]) -> None:
    print(
        "| method | baseline | utilization_delta | strong_coverage_delta | "
        "redundancy_delta | matched_utilization | strong_coverage_improved |"
    )
    print("| --- | --- | ---: | ---: | ---: | --- | --- |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["baseline"]),
                    f"{float(row['utilization_delta']):.4f}",
                    f"{float(row['strong_coverage_delta']):.4f}",
                    f"{float(row['redundancy_delta']):.4f}",
                    str(row["matched_utilization"]),
                    str(row["strong_coverage_improved"]),
                ]
            )
            + " |"
        )


if __name__ == "__main__":
    main()
