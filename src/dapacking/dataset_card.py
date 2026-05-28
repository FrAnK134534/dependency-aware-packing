from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from dapacking.dependency import WEAK_DEPENDENCY_LABELS, has_strong_dependency
from dapacking.documents import Document
from dapacking.edges import DependencyEdge, read_dependency_edges
from dapacking.io import read_documents


def render_dataset_card(
    name: str,
    documents: list[Document],
    edges: list[DependencyEdge] | None = None,
    split_dir: str | Path | None = None,
    source_manifest: str = "",
) -> str:
    edges = edges or []
    lines = [
        f"# Dataset Card: {name}",
        "",
        "## Purpose",
        "",
        "This dataset is used to compare long-context packing strategies under a",
        "fixed model, tokenizer, context length, and training-token budget.",
        "",
        "## Sources",
        "",
        f"- source_manifest: `{source_manifest or 'not_recorded'}`",
        f"- documents: {len(documents)}",
        f"- groups: {len({_group_value(document) for document in documents})}",
        f"- edges: {len(edges)}",
        "",
        "## Document Distribution",
        "",
    ]
    lines.extend(_counter_table("source_type", _counter(documents, "source_type")))
    lines.extend(_counter_table("language", _counter(documents, "language")))
    lines.extend(_counter_table("license", _counter(documents, "license")))
    lines.extend(
        _counter_table("collection_or_repo", Counter(_group_value(doc) for doc in documents))
    )

    if edges:
        relation_counts = Counter(_primary_relation(edge) for edge in edges)
        strong_count = sum(1 for edge in edges if has_strong_dependency(_edge_labels(edge)))
        weak_count = len(edges) - strong_count
        lines.extend(
            [
                "",
                "## Dependency Edge Distribution",
                "",
                f"- strong_edges: {strong_count}",
                f"- weak_edges: {weak_count}",
                "",
            ]
        )
        lines.extend(_counter_table("relation", relation_counts))

    if split_dir:
        lines.extend(["", "## Split Audit", ""])
        lines.extend(_split_audit(Path(split_dir)))

    lines.extend(
        [
            "",
            "## Leakage Policy",
            "",
            "- Split by repository or external collection/document group, not by individual file.",
            "- Do not train on held-out evaluation repositories or collections.",
            "- Keep edge review and context-gain validation records tied to the same split.",
            "",
            "## Known Limitations",
            "",
            "- Dependency edges are rule-based and must be audited before formal training.",
            "- Non-code PDF extraction is text-first and may lose table/equation structure.",
            (
                "- Generalization data supports method scope, while repo-main remains "
                "the primary evidence."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def load_dataset_card_inputs(
    documents_path: str | Path,
    edges_path: str | Path | None = None,
) -> tuple[list[Document], list[DependencyEdge]]:
    documents = read_documents(documents_path)
    edges = read_dependency_edges(edges_path) if edges_path else []
    return documents, edges


def _counter(documents: list[Document], metadata_key: str) -> Counter[str]:
    return Counter(
        str(document.metadata.get(metadata_key, "unknown") or "unknown")
        for document in documents
    )


def _counter_table(name: str, counts: Counter[str]) -> list[str]:
    lines = [f"### {name}", "", f"| {name} | count |", "| --- | ---: |"]
    for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{value}` | {count} |")
    lines.append("")
    return lines


def _split_audit(split_dir: Path) -> list[str]:
    split_groups_path = split_dir / "split_groups.json"
    if not split_groups_path.exists():
        return ["- split_groups.json: missing"]
    split_groups = json.loads(split_groups_path.read_text(encoding="utf-8"))
    overlaps: list[str] = []
    split_names = sorted(split_groups)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = sorted(set(split_groups[left]) & set(split_groups[right]))
            if overlap:
                overlaps.append(f"{left}/{right}: {len(overlap)}")
    lines = [f"- split_groups.json: `{split_groups_path}`"]
    for split in split_names:
        lines.append(f"- {split}_groups: {len(split_groups[split])}")
    lines.append(f"- group_overlap: {'none' if not overlaps else ', '.join(overlaps)}")
    return lines


def _group_value(document: Document) -> str:
    return document.repo or document.collection or document.document_id or "__unknown__"


def _edge_labels(edge: DependencyEdge) -> list[str]:
    labels = edge.metadata.get("labels")
    if isinstance(labels, list):
        return [str(label) for label in labels]
    return [label for label in edge.relation.split("+") if label]


def _primary_relation(edge: DependencyEdge) -> str:
    labels = _edge_labels(edge)
    for label in labels:
        if label not in WEAK_DEPENDENCY_LABELS:
            return label
    return labels[0] if labels else "unknown"
