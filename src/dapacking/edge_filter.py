from __future__ import annotations

from collections import Counter
from typing import Iterable

from dapacking.dependency import DEFAULT_WEIGHTS
from dapacking.edges import DependencyEdge, RELATION_ORDER
from dapacking.relation_config import RelationConfig


def filter_dependency_edges(
    edges: Iterable[DependencyEdge],
    relation_config: RelationConfig,
    weights: dict[str, float] | None = None,
    min_weight: float = 0.0,
) -> list[DependencyEdge]:
    weights = weights or DEFAULT_WEIGHTS
    filtered_edges: list[DependencyEdge] = []
    allowed = set(relation_config.allowed_relations)

    for edge in edges:
        original_labels = edge_labels(edge)
        kept_labels = tuple(label for label in relation_order(original_labels) if label in allowed)
        if not kept_labels:
            continue

        relation_reliability = {
            label: float(relation_config.relation_reliability.get(label, 1.0))
            for label in kept_labels
        }
        new_weight = sum(
            weights.get(label, 0.0) * relation_reliability[label] for label in kept_labels
        )
        if new_weight <= min_weight:
            continue

        metadata = dict(edge.metadata)
        metadata.update(
            {
                "original_relation": edge.relation,
                "original_weight": edge.weight,
                "original_labels": list(original_labels),
                "relation_reliability": relation_reliability,
                "filtered_by_relation_config": relation_config.name,
                "labels": list(kept_labels),
            }
        )
        filtered_edges.append(
            DependencyEdge(
                source_docid=edge.source_docid,
                target_docid=edge.target_docid,
                relation="+".join(kept_labels),
                weight=round(new_weight, 6),
                metadata=metadata,
            )
        )

    return filtered_edges


def edge_labels(edge: DependencyEdge) -> tuple[str, ...]:
    labels = tuple(str(label) for label in edge.metadata.get("labels", []) if str(label))
    if labels:
        return labels
    return tuple(label for label in edge.relation.split("+") if label)


def relation_order(labels: Iterable[str]) -> tuple[str, ...]:
    label_set = set(labels)
    ordered = [label for label in RELATION_ORDER if label in label_set]
    ordered.extend(sorted(label_set - set(RELATION_ORDER)))
    return tuple(ordered)


def relation_counts(edges: Iterable[DependencyEdge]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for edge in edges:
        for label in edge_labels(edge):
            counts[label] += 1
    return counts
