from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping

from dapacking.dependency import GENERIC_STEMS, WEAK_DEPENDENCY_LABELS
from dapacking.documents import Document


@dataclass(frozen=True)
class AssistedEdgeAnnotation:
    label: str
    confidence: float
    error_type: str
    note: str


def annotate_edge_review_record(
    record: Mapping[str, object],
    documents_by_id: Mapping[str, Document] | None = None,
) -> AssistedEdgeAnnotation:
    """Create a transparent assistant-assisted edge review suggestion.

    This is intentionally not called a manual label. The goal is to reduce
    review workload while preserving a clean distinction from human audit.
    """

    documents_by_id = documents_by_id or {}
    source_docid = str(record.get("source_docid", ""))
    target_docid = str(record.get("target_docid", ""))
    source = documents_by_id.get(source_docid)
    target = documents_by_id.get(target_docid)
    relation = _primary_relation(record)

    source_text = _review_text(record, source, "source")
    target_text = _review_text(record, target, "target")
    source_path = _path(record, source, "source")
    target_path = _path(record, target, "target")
    source_type = str(record.get("source_type") or (source.source_type if source else ""))
    target_type = str(record.get("target_type") or (target.source_type if target else ""))

    if relation in WEAK_DEPENDENCY_LABELS:
        return AssistedEdgeAnnotation(
            "partial",
            0.5,
            "",
            "weak structural co-location edge; useful only as a weak prior",
        )

    exact_evidence = _exact_cross_mention(source_path, source_text, target_path, target_text)
    if relation == "import_relation":
        return _annotate_import(source_path, target_text, exact_evidence)
    if relation == "test_source_relation":
        return _annotate_test_source(source_path, target_path, target_text, exact_evidence)
    if relation in {"readme_code_relation", "docs_code_relation", "example_code_relation"}:
        return _annotate_doc_code(relation, target_path, source_text, exact_evidence)
    if relation == "config_script_relation":
        return _annotate_config_script(source_path, target_path, source_text, target_text)
    if relation == "hyperlink_relation":
        return _annotate_link(target_path, source_text, exact_evidence)
    if relation == "citation_relation":
        return _annotate_citation(record, source_text, target)
    if relation == "definition_usage_relation":
        return _annotate_definition_usage(source, target_text)
    if relation == "api_doc_usage_relation":
        return _annotate_api_doc_usage(source, target_text, source_type, target_type)
    if relation == "equation_or_figure_reference_relation":
        return _annotate_object_reference(source, target_text)

    if exact_evidence:
        return AssistedEdgeAnnotation("yes", 0.72, "", "explicit path or filename cross-reference")
    return AssistedEdgeAnnotation(
        "partial",
        0.45,
        "uncertain_relation",
        f"relation {relation} is not covered by the assistant rule set",
    )


def _annotate_import(
    source_path: str,
    target_text: str,
    exact_evidence: bool,
) -> AssistedEdgeAnnotation:
    module = _module_alias(source_path)
    stem = _stem(source_path)
    patterns = [
        rf"\bimport\s+{re.escape(module)}\b",
        rf"\bfrom\s+{re.escape(module)}\s+import\b",
    ]
    if stem and stem not in GENERIC_STEMS:
        patterns.extend(
            [
                rf"\bimport\s+{re.escape(stem)}\b",
                rf"\bfrom\s+{re.escape(stem)}\s+import\b",
            ]
        )
    if any(re.search(pattern, target_text, flags=re.IGNORECASE) for pattern in patterns):
        return AssistedEdgeAnnotation("yes", 0.92, "", "target explicitly imports source module")
    if exact_evidence or _nongeneric_stem_in_text(source_path, target_text):
        return AssistedEdgeAnnotation("partial", 0.66, "", "target mentions source path/module but import form is not visible")
    return AssistedEdgeAnnotation("no", 0.72, "missing_import_evidence", "no import/path evidence visible")


def _annotate_test_source(
    source_path: str,
    target_path: str,
    target_text: str,
    exact_evidence: bool,
) -> AssistedEdgeAnnotation:
    source_stem = _stem(source_path).replace("test_", "").replace("_test", "")
    target_name = PurePosixPath(target_path).name.lower()
    if exact_evidence or _has_path_alias(source_path, target_text):
        return AssistedEdgeAnnotation("yes", 0.88, "", "test contains import/path evidence for source")
    if source_stem and source_stem not in GENERIC_STEMS and source_stem in target_name:
        return AssistedEdgeAnnotation("yes", 0.78, "", "test filename matches non-generic source stem")
    if "test" in target_name or "/test" in target_path.lower() or "tests/" in target_path.lower():
        return AssistedEdgeAnnotation("partial", 0.58, "", "target is a test but exact source evidence is weak")
    return AssistedEdgeAnnotation("no", 0.7, "missing_test_evidence", "target is not clearly a test of source")


def _annotate_doc_code(
    relation: str,
    target_path: str,
    source_text: str,
    exact_evidence: bool,
) -> AssistedEdgeAnnotation:
    if exact_evidence or _has_path_alias(target_path, source_text):
        return AssistedEdgeAnnotation("yes", 0.86, "", "documentation explicitly mentions target code path/module")
    if _nongeneric_stem_in_text(target_path, source_text):
        return AssistedEdgeAnnotation("partial", 0.62, "", "documentation mentions target symbol/stem but not full path")
    return AssistedEdgeAnnotation(
        "no",
        0.72,
        "missing_doc_code_evidence",
        f"{relation} lacks visible target path/module/symbol evidence",
    )


def _annotate_config_script(
    config_path: str,
    script_path: str,
    config_text: str,
    script_text: str,
) -> AssistedEdgeAnnotation:
    script_name = PurePosixPath(script_path).name.lower()
    config_name = PurePosixPath(config_path).name.lower()
    config_lower = config_text.lower()
    script_lower = script_text.lower()
    if script_path.lower() in config_lower or config_path.lower() in script_lower:
        return AssistedEdgeAnnotation("yes", 0.9, "", "config/script explicitly references the other path")
    if script_name and script_name in config_lower and re.search(
        rf"\b(?:python|python3|bash|sh|run|script|command|cmd)\b[^\n]*{re.escape(script_name)}",
        config_lower,
    ):
        return AssistedEdgeAnnotation("yes", 0.84, "", "config/workflow explicitly calls script")
    if config_name and config_name in script_lower and re.search(
        rf"\b(?:open|load|read|parse|yaml\.safe_load|json\.load)\b[^\n]*{re.escape(config_name)}",
        script_lower,
    ):
        return AssistedEdgeAnnotation("yes", 0.84, "", "script explicitly reads config")
    if script_name in config_lower or config_name in script_lower:
        return AssistedEdgeAnnotation("partial", 0.6, "", "filename mention exists but call/read evidence is weak")
    return AssistedEdgeAnnotation("no", 0.75, "missing_config_script_evidence", "no explicit call/read evidence")


def _annotate_link(
    target_path: str,
    source_text: str,
    exact_evidence: bool,
) -> AssistedEdgeAnnotation:
    if exact_evidence or target_path.lower() in source_text.lower():
        return AssistedEdgeAnnotation("yes", 0.86, "", "source explicitly links or mentions target")
    return AssistedEdgeAnnotation("no", 0.68, "missing_link_evidence", "no target URL/path visible")


def _annotate_citation(
    record: Mapping[str, object],
    source_text: str,
    target: Document | None,
) -> AssistedEdgeAnnotation:
    target_title = str((target.metadata.get("title") if target else "") or "").strip().lower()
    doi = str((target.metadata.get("doi") if target else "") or "").strip().lower()
    if doi and doi in source_text.lower():
        return AssistedEdgeAnnotation("yes", 0.9, "", "source explicitly cites target DOI")
    if len(target_title) >= 12 and target_title in source_text.lower():
        return AssistedEdgeAnnotation("yes", 0.84, "", "source explicitly cites target title")
    if "citation_relation" in str(record.get("labels", "")):
        return AssistedEdgeAnnotation("partial", 0.55, "", "citation edge exists but excerpt lacks full citation evidence")
    return AssistedEdgeAnnotation("no", 0.7, "missing_citation_evidence", "no DOI/title/bibkey evidence visible")


def _annotate_definition_usage(
    source: Document | None,
    target_text: str,
) -> AssistedEdgeAnnotation:
    term = _definition_term(source)
    if term and re.search(rf"(?<!\w){re.escape(term)}(?!\w)", target_text, flags=re.IGNORECASE):
        return AssistedEdgeAnnotation("yes", 0.82, "", "target explicitly uses source definition term")
    return AssistedEdgeAnnotation("partial", 0.52, "weak_definition_evidence", "definition term is not clearly visible")


def _annotate_api_doc_usage(
    source: Document | None,
    target_text: str,
    source_type: str,
    target_type: str,
) -> AssistedEdgeAnnotation:
    terms = _api_terms(source)
    for term in terms:
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", target_text, flags=re.IGNORECASE):
            return AssistedEdgeAnnotation("yes", 0.84, "", "target explicitly uses API term from source")
    if source_type in {"api_doc", "docs", "technical_doc"} and target_type in {"example", "usage", "tutorial", "docs"}:
        return AssistedEdgeAnnotation("partial", 0.5, "", "source/target roles match API-usage pattern, but exact term is weak")
    return AssistedEdgeAnnotation("no", 0.68, "missing_api_usage_evidence", "no API term usage visible")


def _annotate_object_reference(
    source: Document | None,
    target_text: str,
) -> AssistedEdgeAnnotation:
    title = str((source.metadata.get("section_title") if source else "") or "")
    matches = re.findall(r"\b(?:Figure|Fig\.|Table|Equation|Eq\.)\s*\(?[A-Za-z0-9.\-]+\)?", title)
    for label in matches:
        if label.lower() in target_text.lower():
            return AssistedEdgeAnnotation("yes", 0.82, "", "target explicitly refers to source object label")
    return AssistedEdgeAnnotation("partial", 0.5, "weak_object_reference", "object label is not clearly visible")


def _review_text(
    record: Mapping[str, object],
    document: Document | None,
    side: str,
) -> str:
    if document is not None:
        return document.content
    return str(record.get(f"{side}_excerpt", ""))


def _path(record: Mapping[str, object], document: Document | None, side: str) -> str:
    if document is not None:
        return document.path
    return str(record.get(f"{side}_path", ""))


def _primary_relation(record: Mapping[str, object]) -> str:
    relation = str(record.get("primary_relation") or "")
    if relation:
        return relation
    labels = [label for label in str(record.get("labels", "")).split(",") if label]
    for label in labels:
        if label not in WEAK_DEPENDENCY_LABELS:
            return label
    return str(record.get("relation", "unknown")).split("+", 1)[0]


def _exact_cross_mention(
    source_path: str,
    source_text: str,
    target_path: str,
    target_text: str,
) -> bool:
    return _has_path_alias(source_path, target_text) or _has_path_alias(target_path, source_text)


def _has_path_alias(path: str, text: str) -> bool:
    text_lower = text.lower()
    aliases = {path, PurePosixPath(path).name, _module_alias(path)}
    return any(alias and len(alias) >= 3 and alias.lower() in text_lower for alias in aliases)


def _nongeneric_stem_in_text(path: str, text: str) -> bool:
    stem = _stem(path)
    return bool(stem and stem not in GENERIC_STEMS and re.search(rf"(?<!\w){re.escape(stem)}(?!\w)", text, flags=re.IGNORECASE))


def _module_alias(path: str) -> str:
    pure = PurePosixPath(path)
    no_suffix = pure.with_suffix("")
    parts = [part for part in no_suffix.parts if part not in {"src", "lib", "."}]
    return ".".join(parts)


def _stem(path: str) -> str:
    return PurePosixPath(path).stem.lower()


def _definition_term(document: Document | None) -> str:
    if document is None:
        return ""
    for key in ("term", "concept", "definition_term"):
        value = str(document.metadata.get(key, "")).strip()
        if value:
            return value
    match = re.search(
        r"(?im)^\s*(?:definition|define|term)\s*[:\-]\s*([A-Za-z][\w \-]{2,80})",
        document.content,
    )
    return match.group(1).strip() if match else ""


def _api_terms(document: Document | None) -> set[str]:
    if document is None:
        return set()
    values: set[str] = set()
    for key in ("api_name", "api_names", "symbols", "term"):
        value = document.metadata.get(key)
        if isinstance(value, (list, tuple, set)):
            values.update(str(item) for item in value)
        elif value:
            values.add(str(value))
    title = str(document.metadata.get("section_title", document.metadata.get("title", ""))).strip()
    if title:
        values.add(title)
    return {value for value in values if len(value) >= 3 and value.lower() not in GENERIC_STEMS}
