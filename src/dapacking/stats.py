from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dapacking.io import read_jsonl


@dataclass(frozen=True)
class PackingSummary:
    path: str
    method: str
    samples: int
    total_tokens: int
    avg_tokens: float
    avg_num_docs: float
    avg_dependency_score: float
    avg_token_utilization: float
    avg_truncation_rate: float

    def to_row(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "method": self.method,
            "samples": self.samples,
            "total_tokens": self.total_tokens,
            "avg_tokens": round(self.avg_tokens, 4),
            "avg_num_docs": round(self.avg_num_docs, 4),
            "avg_dependency_score": round(self.avg_dependency_score, 4),
            "avg_token_utilization": round(self.avg_token_utilization, 4),
            "avg_truncation_rate": round(self.avg_truncation_rate, 4),
        }


def summarize_packed_file(path: str | Path) -> PackingSummary:
    records = read_jsonl(path)
    if not records:
        return PackingSummary(str(path), "unknown", 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    stats = [record.get("stats", {}) for record in records]
    method = str(records[0].get("method", "unknown"))
    total_tokens = sum(int(item.get("tokens", 0)) for item in stats)

    return PackingSummary(
        path=str(path),
        method=method,
        samples=len(records),
        total_tokens=total_tokens,
        avg_tokens=_average(item.get("tokens", 0) for item in stats),
        avg_num_docs=_average(item.get("num_docs", 0) for item in stats),
        avg_dependency_score=_average(item.get("dependency_score", 0) for item in stats),
        avg_token_utilization=_average(item.get("token_utilization", 0) for item in stats),
        avg_truncation_rate=_average(item.get("truncation_rate", 0) for item in stats),
    )


def _average(values: Any) -> float:
    numeric_values = [float(value) for value in values]
    return sum(numeric_values) / max(len(numeric_values), 1)
