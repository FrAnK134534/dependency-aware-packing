from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReviewRelationSummary:
    relation: str
    reviewed: int
    yes: int
    partial: int
    no: int
    unknown: int
    strict_precision: float
    supportive_rate: float

    def to_row(self) -> dict[str, object]:
        return {
            "relation": self.relation,
            "reviewed": self.reviewed,
            "yes": self.yes,
            "partial": self.partial,
            "no": self.no,
            "unknown": self.unknown,
            "strict_precision": round(self.strict_precision, 4),
            "supportive_rate": round(self.supportive_rate, 4),
        }


def read_review_records(path: str | Path) -> list[dict[str, Any]]:
    review_path = Path(path)
    if review_path.suffix.lower() == ".jsonl":
        with review_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with review_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_review_records(records: list[dict[str, Any]]) -> list[ReviewRelationSummary]:
    grouped: dict[str, list[str]] = {}
    for record in records:
        relation = _relation(record)
        grouped.setdefault(relation, []).append(_label(record))

    summaries: list[ReviewRelationSummary] = []
    for relation in sorted(grouped):
        labels = grouped[relation]
        yes = labels.count("yes")
        partial = labels.count("partial")
        no = labels.count("no")
        unknown = len(labels) - yes - partial - no
        judged = max(yes + partial + no, 1)
        summaries.append(
            ReviewRelationSummary(
                relation=relation,
                reviewed=len(labels),
                yes=yes,
                partial=partial,
                no=no,
                unknown=unknown,
                strict_precision=yes / judged,
                supportive_rate=(yes + partial) / judged,
            )
        )
    return summaries


def render_review_markdown(
    summaries: list[ReviewRelationSummary],
    records: list[dict[str, Any]],
) -> str:
    lines = [
        "# Edge Review Audit",
        "",
        "This report summarizes manual edge-review labels. `strict_precision` uses only",
        "`yes` as correct; `supportive_rate` counts both `yes` and `partial` as usable",
        "for context-gain validation.",
        "",
        (
            "| relation | reviewed | yes | partial | no | unknown | "
            "strict_precision | supportive_rate |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    summary.relation,
                    str(summary.reviewed),
                    str(summary.yes),
                    str(summary.partial),
                    str(summary.no),
                    str(summary.unknown),
                    f"{summary.strict_precision:.4f}",
                    f"{summary.supportive_rate:.4f}",
                ]
            )
            + " |"
        )

    error_counts = _error_counts(records)
    if error_counts:
        lines.extend(["", "## Common Error Types", ""])
        for error_type, count in sorted(error_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{error_type}`: {count}")
    return "\n".join(lines) + "\n"


def _relation(record: dict[str, Any]) -> str:
    return str(record.get("primary_relation") or record.get("relation") or "unknown")


def _label(record: dict[str, Any]) -> str:
    for key in ("manual_reasonable", "review_label", "manual_label", "label"):
        value = str(record.get(key, "")).strip().lower()
        if value:
            if value in {"y", "true", "1"}:
                return "yes"
            if value in {"p", "partly"}:
                return "partial"
            if value in {"n", "false", "0"}:
                return "no"
            return value
    return "unknown"


def _error_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        error_type = str(record.get("manual_error_type", "")).strip()
        if not error_type:
            continue
        counts[error_type] = counts.get(error_type, 0) + 1
    return counts
