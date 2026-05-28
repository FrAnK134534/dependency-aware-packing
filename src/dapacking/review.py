from __future__ import annotations

import random
from dataclasses import dataclass

from dapacking.dependency import WEAK_DEPENDENCY_LABELS, has_strong_dependency
from dapacking.documents import Document
from dapacking.edges import DependencyEdge


@dataclass(frozen=True)
class EdgeReviewConfig:
    sample_size: int = 50
    seed: int = 42
    include_weak: bool = True
    excerpt_chars: int = 360
    per_relation_sample_size: int = 0


def sample_edge_review_records(
    documents: list[Document],
    edges: list[DependencyEdge],
    config: EdgeReviewConfig | None = None,
) -> list[dict[str, object]]:
    config = config or EdgeReviewConfig()
    document_by_id = {document.docid: document for document in documents}
    candidates = [
        edge
        for edge in edges
        if edge.source_docid in document_by_id
        and edge.target_docid in document_by_id
        and (config.include_weak or has_strong_dependency(_edge_labels(edge)))
    ]

    rng = random.Random(config.seed)
    if config.per_relation_sample_size > 0:
        candidates = _balanced_by_relation(candidates, config.per_relation_sample_size, rng)
    elif config.sample_size > 0 and len(candidates) > config.sample_size:
        candidates = rng.sample(candidates, config.sample_size)
    else:
        candidates = list(candidates)
        rng.shuffle(candidates)

    records: list[dict[str, object]] = []
    for index, edge in enumerate(candidates):
        source = document_by_id[edge.source_docid]
        target = document_by_id[edge.target_docid]
        labels = _edge_labels(edge)
        primary_relation = _primary_relation(labels)
        records.append(
            {
                "review_id": f"edge_review_{index:05d}",
                "relation": edge.relation,
                "primary_relation": primary_relation,
                "labels": ",".join(labels),
                "is_strong": has_strong_dependency(labels),
                "weight": edge.weight,
                "source_docid": source.docid,
                "target_docid": target.docid,
                "source_repo": source.repo,
                "target_repo": target.repo,
                "source_path": source.path,
                "target_path": target.path,
                "source_type": source.source_type,
                "target_type": target.source_type,
                "source_excerpt": _excerpt(source.content, config.excerpt_chars),
                "target_excerpt": _excerpt(target.content, config.excerpt_chars),
                "manual_reasonable": "",
                "manual_confidence": "",
                "manual_error_type": "",
                "manual_note": "",
            }
        )
    return records


def _balanced_by_relation(
    edges: list[DependencyEdge],
    per_relation_sample_size: int,
    rng: random.Random,
) -> list[DependencyEdge]:
    grouped: dict[str, list[DependencyEdge]] = {}
    for edge in edges:
        relation = _primary_relation(_edge_labels(edge))
        grouped.setdefault(relation, []).append(edge)

    sampled: list[DependencyEdge] = []
    for relation in sorted(grouped):
        relation_edges = list(grouped[relation])
        rng.shuffle(relation_edges)
        sampled.extend(relation_edges[:per_relation_sample_size])
    rng.shuffle(sampled)
    return sampled


def _edge_labels(edge: DependencyEdge) -> list[str]:
    labels = edge.metadata.get("labels")
    if isinstance(labels, list):
        return [str(label) for label in labels]
    return [label for label in edge.relation.split("+") if label]


def _primary_relation(labels: list[str]) -> str:
    for label in labels:
        if label not in WEAK_DEPENDENCY_LABELS:
            return label
    return labels[0] if labels else "unknown"


def _excerpt(content: str, max_chars: int) -> str:
    text = " ".join(content.split())
    if len(text) <= max_chars:
        return text
    return text[: max(max_chars - 3, 0)] + "..."
