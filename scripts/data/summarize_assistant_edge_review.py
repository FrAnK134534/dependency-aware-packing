#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize assistant-assisted edge review suggestions.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    summaries = _summaries(rows)
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "relation",
                    "reviewed",
                    "yes",
                    "partial",
                    "no",
                    "unknown",
                    "strict_precision_suggestion",
                    "supportive_rate_suggestion",
                    "avg_confidence",
                ],
            )
            writer.writeheader()
            writer.writerows(summaries)

    report = _render_markdown(summaries)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(report, encoding="utf-8")
    print(report, end="")


def _summaries(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        relation = row.get("primary_relation") or row.get("relation") or "unknown"
        grouped.setdefault(relation, []).append(row)

    summaries: list[dict[str, object]] = []
    for relation in sorted(grouped):
        relation_rows = grouped[relation]
        labels = [row.get("assistant_review_label", "unknown").strip().lower() for row in relation_rows]
        yes = labels.count("yes")
        partial = labels.count("partial")
        no = labels.count("no")
        unknown = len(labels) - yes - partial - no
        judged = max(yes + partial + no, 1)
        confidences = []
        for row in relation_rows:
            try:
                confidences.append(float(row.get("assistant_confidence", "0") or 0))
            except ValueError:
                confidences.append(0.0)
        summaries.append(
            {
                "relation": relation,
                "reviewed": len(relation_rows),
                "yes": yes,
                "partial": partial,
                "no": no,
                "unknown": unknown,
                "strict_precision_suggestion": round(yes / judged, 4),
                "supportive_rate_suggestion": round((yes + partial) / judged, 4),
                "avg_confidence": round(sum(confidences) / max(len(confidences), 1), 4),
            }
        )
    return summaries


def _render_markdown(summaries: list[dict[str, object]]) -> str:
    lines = [
        "# Assistant-Assisted Edge Review",
        "",
        "These labels are assistant suggestions, not independent human annotations.",
        "Use them to speed up review, then manually confirm labels before treating",
        "the numbers as paper-quality edge precision.",
        "",
        (
            "| relation | reviewed | yes | partial | no | unknown | "
            "strict_precision_suggestion | supportive_rate_suggestion | avg_confidence |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["relation"]),
                    str(row["reviewed"]),
                    str(row["yes"]),
                    str(row["partial"]),
                    str(row["no"]),
                    str(row["unknown"]),
                    f"{float(row['strict_precision_suggestion']):.4f}",
                    f"{float(row['supportive_rate_suggestion']):.4f}",
                    f"{float(row['avg_confidence']):.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
