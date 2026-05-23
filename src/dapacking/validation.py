from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from dapacking.dependency import WEAK_DEPENDENCY_LABELS, has_strong_dependency
from dapacking.documents import Document
from dapacking.edges import DependencyEdge
from dapacking.tokenization import truncate_to_tokens


@dataclass(frozen=True)
class DependencyValidationConfig:
    max_examples_per_relation: int = 200
    seed: int = 42
    include_weak: bool = False
    max_source_tokens: int = 2048
    max_target_tokens: int = 2048


def build_dependency_validation_records(
    documents: list[Document],
    edges: list[DependencyEdge],
    config: DependencyValidationConfig | None = None,
) -> list[dict[str, object]]:
    config = config or DependencyValidationConfig()
    document_by_id = {document.docid: document for document in documents}
    grouped_edges: dict[str, list[DependencyEdge]] = defaultdict(list)

    for edge in edges:
        if edge.source_docid not in document_by_id or edge.target_docid not in document_by_id:
            continue
        labels = _edge_labels(edge)
        if not config.include_weak and not has_strong_dependency(labels):
            continue
        grouped_edges[_primary_relation(labels)].append(edge)

    rng = random.Random(config.seed)
    records: list[dict[str, object]] = []
    for relation in sorted(grouped_edges):
        relation_edges = list(grouped_edges[relation])
        rng.shuffle(relation_edges)
        if config.max_examples_per_relation > 0:
            relation_edges = relation_edges[: config.max_examples_per_relation]
        for edge in relation_edges:
            source = document_by_id[edge.source_docid]
            target = document_by_id[edge.target_docid]
            records.append(_make_validation_record(len(records), edge, source, target, config))

    return records


def _make_validation_record(
    index: int,
    edge: DependencyEdge,
    source: Document,
    target: Document,
    config: DependencyValidationConfig,
) -> dict[str, object]:
    labels = _edge_labels(edge)
    source_text, source_overflow = truncate_to_tokens(source.content, config.max_source_tokens)
    target_text, target_overflow = truncate_to_tokens(target.content, config.max_target_tokens)
    target_header = _target_header(target)
    source_block = _document_block(source, source_text)

    return {
        "sample_id": f"depval_{index:06d}",
        "relation": edge.relation,
        "primary_relation": _primary_relation(labels),
        "labels": labels,
        "weight": edge.weight,
        "source_docid": source.docid,
        "target_docid": target.docid,
        "source_path": source.path,
        "target_path": target.path,
        "source_type": source.source_type,
        "target_type": target.source_type,
        "context_with_dependency": (
            f"{source_block}\n\n"
            f"The next document depends on the context above.\n"
            f"{target_header}"
        ),
        "context_without_dependency": target_header,
        "target_text": target_text,
        "metadata": {
            "repo": source.repo or target.repo,
            "source_truncated_tokens": source_overflow,
            "target_truncated_tokens": target_overflow,
        },
    }


def _document_block(document: Document, content: str) -> str:
    return f"<doc id=\"{document.docid}\" path=\"{document.path}\">\n{content}\n</doc>"


def _target_header(document: Document) -> str:
    return f"<doc id=\"{document.docid}\" path=\"{document.path}\">\n"


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
