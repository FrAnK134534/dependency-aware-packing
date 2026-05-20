from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dapacking.edges import DependencyEdge, read_dependency_edges
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
    avg_order_dependency: float = 0.0
    edge_coverage: float = 0.0
    weighted_edge_coverage: float = 0.0
    same_repo_pair_ratio: float = 0.0

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
            "avg_order_dependency": round(self.avg_order_dependency, 4),
            "edge_coverage": round(self.edge_coverage, 4),
            "weighted_edge_coverage": round(self.weighted_edge_coverage, 4),
            "same_repo_pair_ratio": round(self.same_repo_pair_ratio, 4),
        }


def summarize_packed_file(path: str | Path, edges_path: str | Path | None = None) -> PackingSummary:
    records = read_jsonl(path)
    if not records:
        return PackingSummary(str(path), "unknown", 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    stats = [record.get("stats", {}) for record in records]
    method = str(records[0].get("method", "unknown"))
    total_tokens = sum(int(item.get("tokens", 0)) for item in stats)
    edges = read_dependency_edges(edges_path) if edges_path else []
    edge_metrics = _edge_metrics(records, edges)

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
        avg_order_dependency=edge_metrics["avg_order_dependency"],
        edge_coverage=edge_metrics["edge_coverage"],
        weighted_edge_coverage=edge_metrics["weighted_edge_coverage"],
        same_repo_pair_ratio=_average(_same_repo_pair_ratio(record.get("docids", [])) for record in records),
    )


def _average(values: Any) -> float:
    numeric_values = [float(value) for value in values]
    return sum(numeric_values) / max(len(numeric_values), 1)


def _edge_metrics(records: list[dict], edges: list[DependencyEdge]) -> dict[str, float]:
    if not edges:
        return {
            "avg_order_dependency": 0.0,
            "edge_coverage": 0.0,
            "weighted_edge_coverage": 0.0,
        }

    edge_index = {(edge.source_docid, edge.target_docid): edge.weight for edge in edges}
    all_edge_keys = set(edge_index)
    covered: set[tuple[str, str]] = set()
    order_scores: list[float] = []

    for record in records:
        docids = [str(docid) for docid in record.get("docids", [])]
        positions = {docid: index for index, docid in enumerate(docids)}
        for source, target in all_edge_keys:
            if source in positions and target in positions and positions[source] < positions[target]:
                covered.add((source, target))

        if len(docids) < 2:
            order_scores.append(0.0)
            continue

        sample_score = 0.0
        for target_index in range(1, len(docids)):
            target = docids[target_index]
            best = 0.0
            for source in docids[:target_index]:
                best = max(best, edge_index.get((source, target), 0.0))
            sample_score += best
        order_scores.append(sample_score / (len(docids) - 1))

    total_weight = sum(edge_index.values())
    covered_weight = sum(edge_index[key] for key in covered)
    return {
        "avg_order_dependency": _average(order_scores),
        "edge_coverage": len(covered) / max(len(all_edge_keys), 1),
        "weighted_edge_coverage": covered_weight / max(total_weight, 1e-12),
    }


def _same_repo_pair_ratio(docids: list[str]) -> float:
    if len(docids) < 2:
        return 0.0
    total = 0
    same = 0
    repos = [_repo_from_docid(str(docid)) for docid in docids]
    for i, repo_a in enumerate(repos):
        for repo_b in repos[i + 1 :]:
            total += 1
            if repo_a and repo_a == repo_b:
                same += 1
    return same / max(total, 1)


def _repo_from_docid(docid: str) -> str:
    if ":" not in docid:
        return ""
    return docid.split(":", 1)[0]
