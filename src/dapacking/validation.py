from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dapacking.dependency import WEAK_DEPENDENCY_LABELS, has_strong_dependency
from dapacking.documents import Document
from dapacking.edges import DependencyEdge
from dapacking.io import read_jsonl
from dapacking.tokenization import truncate_to_tokens


@dataclass(frozen=True)
class DependencyValidationConfig:
    max_examples_per_relation: int = 200
    seed: int = 42
    include_weak: bool = False
    max_source_tokens: int = 2048
    max_target_tokens: int = 2048
    review_annotations: dict[tuple[str, str, str], dict[str, Any]] | None = None
    allowed_review_labels: tuple[str, ...] = ("yes", "partial")
    min_review_confidence: float = 0.6
    allow_unreviewed_backfill: bool = False


@dataclass(frozen=True)
class ControlValidationConfig:
    max_examples_per_control: int = 200
    seed: int = 42
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
        review = _matching_review_annotation(edge, labels, config.review_annotations)
        if config.review_annotations is not None:
            if review is None and not config.allow_unreviewed_backfill:
                continue
            if review is not None and not _review_allowed(review, config):
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
            review = _matching_review_annotation(
                edge,
                _edge_labels(edge),
                config.review_annotations,
            )
            records.append(
                _make_validation_record(len(records), edge, source, target, config, review)
            )

    return records


def build_control_validation_records(
    documents: list[Document],
    edges: list[DependencyEdge],
    config: ControlValidationConfig | None = None,
) -> list[dict[str, object]]:
    config = config or ControlValidationConfig()
    rng = random.Random(config.seed)
    edge_pairs = _edge_pair_set(edges)

    same_group_pairs = [
        (source, target)
        for source in documents
        for target in documents
        if source.docid != target.docid
        and _document_group(source)
        and _document_group(source) == _document_group(target)
        and (source.docid, target.docid) not in edge_pairs
        and (target.docid, source.docid) not in edge_pairs
    ]
    cross_group_pairs = [
        (source, target)
        for source in documents
        for target in documents
        if source.docid != target.docid
        and _document_group(source)
        and _document_group(target)
        and _document_group(source) != _document_group(target)
    ]

    records: list[dict[str, object]] = []
    records.extend(
        _sample_control_records(
            same_group_pairs,
            "same_group_non_edge",
            len(records),
            rng,
            config,
        )
    )
    records.extend(
        _sample_control_records(
            cross_group_pairs,
            "random_cross_group",
            len(records),
            rng,
            config,
        )
    )
    return records


def _make_validation_record(
    index: int,
    edge: DependencyEdge,
    source: Document,
    target: Document,
    config: DependencyValidationConfig,
    review: dict[str, Any] | None = None,
    bridge_text: str = "The next document depends on the context above.",
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
            f"{bridge_text}\n"
            f"{target_header}"
        ),
        "context_without_dependency": target_header,
        "target_text": target_text,
        "metadata": {
            "repo": source.repo or target.repo,
            "source_truncated_tokens": source_overflow,
            "target_truncated_tokens": target_overflow,
            "review_label": _review_label(review),
            "review_confidence": _review_confidence(review),
            "evidence_relation": _primary_relation(labels),
        },
    }


def _sample_control_records(
    pairs: list[tuple[Document, Document]],
    control_type: str,
    start_index: int,
    rng: random.Random,
    config: ControlValidationConfig,
) -> list[dict[str, object]]:
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    if config.max_examples_per_control > 0:
        shuffled = shuffled[: config.max_examples_per_control]

    records: list[dict[str, object]] = []
    for offset, (source, target) in enumerate(shuffled):
        edge = DependencyEdge(
            source_docid=source.docid,
            target_docid=target.docid,
            relation=control_type,
            weight=0.0,
            metadata={"labels": [control_type]},
        )
        record = _make_validation_record(
            start_index + offset,
            edge,
            source,
            target,
            DependencyValidationConfig(
                max_source_tokens=config.max_source_tokens,
                max_target_tokens=config.max_target_tokens,
            ),
            bridge_text="The next document is paired as a context-gain control.",
        )
        record["metadata"]["control_type"] = control_type
        record["primary_relation"] = control_type
        records.append(record)
    return records


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


def _edge_pair_set(edges: list[DependencyEdge]) -> set[tuple[str, str]]:
    return {(edge.source_docid, edge.target_docid) for edge in edges}


def _document_group(document: Document) -> str:
    for key in ("repo", "collection", "document_set", "dataset"):
        value = str(document.metadata.get(key, "")).strip()
        if value:
            return value
    return ""


def read_review_annotations(path: str | Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    review_path = Path(path)
    if review_path.suffix.lower() == ".jsonl":
        records = read_jsonl(review_path)
    else:
        with review_path.open("r", encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))

    annotations: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        source = str(record.get("source_docid", "")).strip()
        target = str(record.get("target_docid", "")).strip()
        relation = str(record.get("primary_relation", record.get("relation", ""))).strip()
        if not source or not target:
            continue
        label = _annotation_label(record)
        confidence = _annotation_confidence(record, label)
        annotation = dict(record)
        annotation["review_label"] = label
        annotation["review_confidence"] = confidence
        annotations[(source, target, relation)] = annotation
        annotations.setdefault((source, target, ""), annotation)
    return annotations


def _matching_review_annotation(
    edge: DependencyEdge,
    labels: list[str],
    annotations: dict[tuple[str, str, str], dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if annotations is None:
        return None
    primary = _primary_relation(labels)
    keys = [
        (edge.source_docid, edge.target_docid, primary),
        (edge.source_docid, edge.target_docid, edge.relation),
        (edge.source_docid, edge.target_docid, ""),
    ]
    for key in keys:
        annotation = annotations.get(key)
        if annotation is not None:
            return annotation
    return None


def _review_allowed(review: dict[str, Any], config: DependencyValidationConfig) -> bool:
    label = _review_label(review)
    confidence = _review_confidence(review)
    allowed = {item.lower() for item in config.allowed_review_labels}
    return label in allowed and confidence >= config.min_review_confidence


def _annotation_label(record: dict[str, Any]) -> str:
    for key in ("review_label", "manual_reasonable", "manual_label", "label"):
        value = str(record.get(key, "")).strip().lower()
        if value:
            return value
    return ""


def _annotation_confidence(record: dict[str, Any], label: str) -> float:
    for key in ("review_confidence", "manual_confidence", "confidence"):
        value = record.get(key)
        if value is not None and value != "":
            try:
                return float(value)
            except (TypeError, ValueError):
                break
    return {"yes": 1.0, "partial": 0.7, "no": 0.0}.get(label, 0.0)


def _review_label(review: dict[str, Any] | None) -> str:
    if review is None:
        return ""
    return str(review.get("review_label", "")).lower()


def _review_confidence(review: dict[str, Any] | None) -> float:
    if review is None:
        return 0.0
    try:
        return float(review.get("review_confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0
