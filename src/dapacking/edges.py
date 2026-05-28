from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from dapacking.dependency import (
    CODE_SUFFIXES,
    DEFAULT_WEIGHTS,
    has_api_doc_usage_relation,
    has_config_script_relation,
    has_citation_relation,
    has_definition_usage_relation,
    has_docs_code_relation,
    has_equation_or_figure_reference_relation,
    has_example_code_relation,
    has_hyperlink_relation,
    has_import_relation,
    has_readme_code_relation,
    has_same_collection,
    has_same_directory,
    has_same_document,
    has_same_domain,
    has_section_neighbor,
    has_test_source_relation,
)
from dapacking.documents import Document
from dapacking.io import read_jsonl, write_jsonl


RELATION_ORDER = tuple(DEFAULT_WEIGHTS)


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
    by_group: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        by_group[_document_group_key(document)].append(document)

    for group_documents in by_group.values():
        edges.extend(
            _group_dependency_edges(group_documents, weights, min_score, include_same_repo_only)
        )

    return edges


def _group_dependency_edges(
    documents: list[Document],
    weights: dict[str, float],
    min_score: float,
    include_same_repo_only: bool,
) -> list[DependencyEdge]:
    labels_by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    doc_by_id = {document.docid: document for document in documents}
    code_docs = [document for document in documents if document.suffix in CODE_SUFFIXES]
    source_docs = [
        document for document in documents if document.source_type in {"source", "script"}
    ]
    test_docs = [document for document in documents if document.source_type == "test"]
    readme_docs = [document for document in documents if document.source_type == "readme"]
    docs_docs = [document for document in documents if document.source_type == "docs"]
    example_docs = [document for document in documents if document.source_type == "example"]
    config_docs = [document for document in documents if document.source_type == "config"]
    script_docs = [document for document in documents if document.source_type == "script"]
    api_docs = [
        document
        for document in documents
        if document.source_type in {"api_doc", "docs", "technical_doc"}
    ]
    usage_docs = [
        document
        for document in documents
        if document.source_type
        in {"api_doc", "usage", "example", "tutorial", "docs", "technical_doc", "text_section"}
    ]

    if include_same_repo_only:
        for source in documents:
            for target in documents:
                if source.docid != target.docid:
                    _add_labels(labels_by_pair, source, target, "same_repo")

    by_parent: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        by_parent[document.parent].append(document)
    for siblings in by_parent.values():
        if len(siblings) < 2:
            continue
        for source in siblings:
            for target in siblings:
                if source.docid != target.docid and has_same_directory(source, target):
                    _add_labels(labels_by_pair, source, target, "same_directory")

    by_document_id: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        document_id = str(document.metadata.get("document_id", ""))
        if document_id:
            by_document_id[document_id].append(document)
    for document_sections in by_document_id.values():
        if len(document_sections) < 2:
            continue
        for source in document_sections:
            for target in document_sections:
                if source.docid == target.docid:
                    continue
                if has_same_document(source, target):
                    _add_labels(labels_by_pair, source, target, "same_document")
                if has_section_neighbor(source, target):
                    _add_labels(labels_by_pair, source, target, "section_neighbor")

    for source in code_docs:
        source_hints = _import_hints(source)
        for target in code_docs:
            if source.docid == target.docid:
                continue
            if not _contains_any_hint(target.content.lower(), source_hints):
                continue
            if has_import_relation(source, target):
                _add_labels(labels_by_pair, source, target, "import_relation")

    for source in source_docs:
        source_stem = PurePosixPath(source.path).stem.replace("test_", "").replace("_test", "")
        if len(source_stem) < 3 or source_stem in {"__init__", "conftest"}:
            continue
        for target in test_docs:
            if source.docid != target.docid and has_test_source_relation(source, target):
                _add_labels(labels_by_pair, source, target, "test_source_relation")

    for source in readme_docs:
        for target in code_docs:
            if source.docid != target.docid and has_readme_code_relation(source, target):
                _add_labels(labels_by_pair, source, target, "readme_code_relation")

    for source in docs_docs:
        for target in code_docs:
            if source.docid != target.docid and has_docs_code_relation(source, target):
                _add_labels(labels_by_pair, source, target, "docs_code_relation")

    for source in config_docs:
        for target in script_docs:
            if source.docid != target.docid and has_config_script_relation(source, target):
                _add_labels(labels_by_pair, source, target, "config_script_relation")

    for source in example_docs:
        for target in code_docs:
            if source.docid != target.docid and has_example_code_relation(source, target):
                _add_labels(labels_by_pair, source, target, "example_code_relation")

    for source in api_docs:
        for target in usage_docs:
            if source.docid != target.docid and has_api_doc_usage_relation(source, target):
                _add_labels(labels_by_pair, source, target, "api_doc_usage_relation")

    for source in documents:
        for target in documents:
            if source.docid == target.docid:
                continue
            if has_hyperlink_relation(source, target):
                _add_labels(labels_by_pair, source, target, "hyperlink_relation")
            if has_citation_relation(source, target):
                _add_labels(labels_by_pair, source, target, "citation_relation")
            if has_definition_usage_relation(source, target):
                _add_labels(labels_by_pair, source, target, "definition_usage_relation")
            if has_equation_or_figure_reference_relation(source, target):
                _add_labels(labels_by_pair, source, target, "equation_or_figure_reference_relation")

    edges: list[DependencyEdge] = []
    for (source_docid, target_docid), labels in labels_by_pair.items():
        ordered_labels = tuple(label for label in RELATION_ORDER if label in labels)
        if not ordered_labels:
            continue
        if not include_same_repo_only and ordered_labels == ("same_repo",):
            continue
        score = sum(weights.get(label, 0.0) for label in ordered_labels)
        if score < min_score:
            continue
        source = doc_by_id[source_docid]
        target = doc_by_id[target_docid]
        edges.append(_make_edge(source, target, ordered_labels, score))
    return edges


def _add_labels(
    labels_by_pair: dict[tuple[str, str], set[str]],
    source: Document,
    target: Document,
    label: str,
) -> None:
    labels = labels_by_pair[(source.docid, target.docid)]
    labels.add(label)
    if source.repo and source.repo == target.repo:
        labels.add("same_repo")
    if has_same_collection(source, target):
        labels.add("same_collection")
    if has_same_document(source, target):
        labels.add("same_document")
    if has_section_neighbor(source, target):
        labels.add("section_neighbor")
    if has_same_domain(source, target):
        labels.add("same_domain")


def _make_edge(
    source: Document,
    target: Document,
    labels: tuple[str, ...],
    score: float,
) -> DependencyEdge:
    return DependencyEdge(
        source_docid=source.docid,
        target_docid=target.docid,
        relation="+".join(labels),
        weight=round(score, 6),
        metadata={
            "repo": source.repo or target.repo,
            "collection": source.collection or target.collection,
            "source_path": source.path,
            "target_path": target.path,
            "source_document_id": source.metadata.get("document_id", ""),
            "target_document_id": target.metadata.get("document_id", ""),
            "source_url": source.metadata.get("url", ""),
            "target_url": target.metadata.get("url", ""),
            "labels": list(labels),
        },
    )


def _document_group_key(document: Document) -> str:
    for key in ("repo", "collection", "document_set", "dataset"):
        value = str(document.metadata.get(key, "")).strip()
        if value:
            return value
    return "__unknown__"


def _import_hints(document: Document) -> set[str]:
    path = PurePosixPath(document.path)
    stem = path.stem
    no_suffix = path.with_suffix("")
    module = ".".join(part for part in no_suffix.parts if part not in {"src", "lib", "."})
    hints = {stem.lower(), module.lower()}
    return {hint for hint in hints if len(hint) >= 2}


def _contains_any_hint(text: str, hints: set[str]) -> bool:
    return any(hint in text for hint in hints)


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
