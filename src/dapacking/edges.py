from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dapacking.dependency import DEFAULT_WEIGHTS, dependency_score
from dapacking.documents import Document
from dapacking.io import read_jsonl, write_jsonl


@dataclass(frozen=True)
class DependencyEdge:
    source_docid: str
    target_docid: str
    relation: str
    weight: float
    metadata: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "source_docid": self.source_docid,
            "target_docid": self.target_docid,
            "relation": self.relation,
            "weight": self.weight,
            "metadata": self.metadata,
        }


def build_dependency_edges(
    documents: list[Document],
    weights: dict[str, float] | None = None,
    min_score: float = 0.11,
    include_same_repo_only: bool = False,
) -> list[DependencyEdge]:
    weights = weights or DEFAULT_WEIGHTS
    edges: list[DependencyEdge] = []

    for source in documents:
        for target in documents:
            if source.docid == target.docid:
                continue
            if source.repo and target.repo and source.repo != target.repo:
                continue

            evidence = dependency_score(source, target, weights)
            if not evidence.labels:
                continue
            if not include_same_repo_only and evidence.labels == ("same_repo",):
                continue
            if evidence.score < min_score:
                continue

            edges.append(
                DependencyEdge(
                    source_docid=source.docid,
                    target_docid=target.docid,
                    relation="+".join(evidence.labels),
                    weight=round(evidence.score, 6),
                    metadata={
                        "repo": source.repo or target.repo,
                        "source_path": source.path,
                        "target_path": target.path,
                        "labels": list(evidence.labels),
                    },
                )
            )

    return edges


def read_dependency_edges(path: str | Path) -> list[DependencyEdge]:
    edges: list[DependencyEdge] = []
    for record in read_jsonl(path):
        edges.append(
            DependencyEdge(
                source_docid=str(record["source_docid"]),
                target_docid=str(record["target_docid"]),
                relation=str(record.get("relation", "")),
                weight=float(record.get("weight", 0.0)),
                metadata=dict(record.get("metadata", {})),
            )
        )
    return edges


def write_dependency_edges(path: str | Path, edges: list[DependencyEdge]) -> None:
    write_jsonl(path, (edge.to_json() for edge in edges))
