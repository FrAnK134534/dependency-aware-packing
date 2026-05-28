from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlparse

from dapacking.documents import Document


DEFAULT_WEIGHTS = {
    "import_relation": 1.0,
    "test_source_relation": 0.9,
    "api_doc_usage_relation": 0.8,
    "citation_relation": 0.85,
    "hyperlink_relation": 0.75,
    "config_script_relation": 0.75,
    "readme_code_relation": 0.65,
    "docs_code_relation": 0.65,
    "definition_usage_relation": 0.65,
    "example_code_relation": 0.55,
    "equation_or_figure_reference_relation": 0.55,
    "same_directory": 0.25,
    "same_document": 0.2,
    "section_neighbor": 0.12,
    "same_repo": 0.1,
    "same_collection": 0.1,
    "same_domain": 0.08,
}
WEAK_DEPENDENCY_LABELS = frozenset(
    {
        "same_directory",
        "same_document",
        "section_neighbor",
        "same_repo",
        "same_collection",
        "same_domain",
    }
)

CODE_SUFFIXES = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".cpp", ".c", ".h"}
CONFIG_NAMES = {
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "package.json",
    "cargo.toml",
    "go.mod",
    "makefile",
    "dockerfile",
}
GENERIC_STEMS = frozenset(
    {
        "__init__",
        "api",
        "app",
        "base",
        "client",
        "common",
        "config",
        "configs",
        "constants",
        "core",
        "data",
        "dataset",
        "datasets",
        "helper",
        "helpers",
        "index",
        "io",
        "main",
        "manager",
        "parser",
        "schema",
        "schemas",
        "server",
        "settings",
        "test",
        "tests",
        "types",
        "util",
        "utils",
    }
)


@dataclass(frozen=True)
class DependencyEvidence:
    score: float
    labels: tuple[str, ...]


def has_strong_dependency(labels: tuple[str, ...] | list[str]) -> bool:
    return any(label not in WEAK_DEPENDENCY_LABELS for label in labels)


def dependency_score(
    source: Document,
    target: Document,
    weights: dict[str, float] | None = None,
) -> DependencyEvidence:
    """Estimate whether `source` can help predict or understand `target`."""

    weights = weights or DEFAULT_WEIGHTS
    labels: list[str] = []
    score = 0.0

    checks = {
        "import_relation": has_import_relation(source, target),
        "test_source_relation": has_test_source_relation(source, target),
        "api_doc_usage_relation": has_api_doc_usage_relation(source, target),
        "citation_relation": has_citation_relation(source, target),
        "hyperlink_relation": has_hyperlink_relation(source, target),
        "config_script_relation": has_config_script_relation(source, target),
        "readme_code_relation": has_readme_code_relation(source, target),
        "docs_code_relation": has_docs_code_relation(source, target),
        "definition_usage_relation": has_definition_usage_relation(source, target),
        "example_code_relation": has_example_code_relation(source, target),
        "equation_or_figure_reference_relation": has_equation_or_figure_reference_relation(
            source, target
        ),
        "same_directory": has_same_directory(source, target),
        "same_document": has_same_document(source, target),
        "section_neighbor": has_section_neighbor(source, target),
        "same_repo": bool(source.repo and source.repo == target.repo),
        "same_collection": has_same_collection(source, target),
        "same_domain": has_same_domain(source, target),
    }

    for label, passed in checks.items():
        if passed:
            labels.append(label)
            score += weights.get(label, 0.0)

    return DependencyEvidence(score=score, labels=tuple(labels))


def has_import_relation(source: Document, target: Document) -> bool:
    if source.repo and target.repo and source.repo != target.repo:
        return False
    if source.suffix not in CODE_SUFFIXES or target.suffix not in CODE_SUFFIXES:
        return False

    source_stem = PurePosixPath(source.path).stem
    source_module = _path_to_module(source.path)
    raw_module = _full_path_to_module(source.path)
    import_patterns = [
        rf"\bimport\s+{re.escape(source_module)}\b",
        rf"\bfrom\s+{re.escape(source_module)}\s+import\b",
        rf"\bimport\s+{re.escape(raw_module)}\b",
        rf"\bfrom\s+{re.escape(raw_module)}\s+import\b",
        rf"require\(['\"](.*/)?{re.escape(source_stem)}['\"]\)",
    ]
    if source_stem not in GENERIC_STEMS:
        import_patterns.extend(
            [
                rf"\bimport\s+{re.escape(source_stem)}\b",
                rf"\bfrom\s+{re.escape(source_stem)}\s+import\b",
            ]
        )
    return any(re.search(pattern, target.content) for pattern in import_patterns)


def has_test_source_relation(source: Document, target: Document) -> bool:
    if source.repo and target.repo and source.repo != target.repo:
        return False
    if source.source_type not in {"source", "script"}:
        return False
    if not _is_test_document(target):
        return False

    source_stem = _clean_stem(source.path)
    if len(source_stem) < 3 or source_stem in {"__init__", "conftest"}:
        return False

    evidence = _has_import_or_path_evidence(source, target)
    if source_stem in GENERIC_STEMS:
        return evidence
    return evidence or source_stem in target.filename.lower()


def has_readme_code_relation(source: Document, target: Document) -> bool:
    if source.repo and target.repo and source.repo != target.repo:
        return False
    if source.source_type != "readme" or target.suffix not in CODE_SUFFIXES:
        return False
    return _document_mentions_code_target(source, target)


def has_docs_code_relation(source: Document, target: Document) -> bool:
    if source.repo and target.repo and source.repo != target.repo:
        return False
    if source.source_type != "docs" or target.suffix not in CODE_SUFFIXES:
        return False
    return _document_mentions_code_target(source, target)


def has_config_script_relation(source: Document, target: Document) -> bool:
    if source.repo and target.repo and source.repo != target.repo:
        return False
    if not _is_config_document(source) or not _is_script_document(target):
        return False
    return _config_calls_script(source, target) or _script_reads_config(target, source)


def has_example_code_relation(source: Document, target: Document) -> bool:
    if source.repo and target.repo and source.repo != target.repo:
        return False
    if source.source_type != "example" or target.suffix not in CODE_SUFFIXES:
        return False
    return _document_mentions_code_target(source, target)


def has_hyperlink_relation(source: Document, target: Document) -> bool:
    links = _metadata_values(source, "links", "out_links", "hrefs", "references")
    target_aliases = _target_link_aliases(target)
    if not target_aliases:
        return False

    for link in links:
        normalized_link = _normalize_link(link)
        if normalized_link and any(
            normalized_link == _normalize_link(alias) for alias in target_aliases
        ):
            return True

    source_content = source.content.lower()
    return any(alias and alias.lower() in source_content for alias in target_aliases)


def has_citation_relation(source: Document, target: Document) -> bool:
    source_refs = _metadata_values(source, "citations", "citation_keys", "references", "bibkeys")
    source_is_citation_context = source.source_type in {"paper", "paper_section", "academic_paper"}
    target_is_citable = target.source_type in {"paper", "paper_section", "academic_paper"}
    if not source_refs and not (source_is_citation_context and target_is_citable):
        return False

    target_refs = _metadata_values(target, "doi", "citation_key", "bibkey", "paper_id")
    target_refs.extend(_metadata_values(target, "title"))

    source_blob = " ".join(source_refs + [source.content]).lower()
    for ref in target_refs:
        normalized = ref.strip().lower()
        if len(normalized) >= 4 and normalized in source_blob:
            return True

    target_title = str(target.metadata.get("title", "")).strip().lower()
    return bool(len(target_title) >= 12 and target_title in source.content.lower())


def has_definition_usage_relation(source: Document, target: Document) -> bool:
    if source.docid == target.docid:
        return False
    if not _same_document_or_collection(source, target):
        return False
    term = _definition_term(source)
    if not term:
        return False
    if _is_definition_document(target):
        return False
    if not _term_in_text(term, target.content):
        return False
    source_index = _section_index(source)
    target_index = _section_index(target)
    return source_index is None or target_index is None or source_index <= target_index


def has_api_doc_usage_relation(source: Document, target: Document) -> bool:
    if source.docid == target.docid:
        return False
    if not _same_document_or_collection(source, target):
        return False
    if not _is_api_document(source):
        return False
    if target.source_type not in {
        "api_doc",
        "usage",
        "example",
        "tutorial",
        "docs",
        "technical_doc",
        "text_section",
    }:
        return False

    api_terms = _api_terms(source)
    if not api_terms:
        return False
    return any(_term_in_code_or_text(term, target.content) for term in api_terms)


def has_equation_or_figure_reference_relation(source: Document, target: Document) -> bool:
    if source.docid == target.docid:
        return False
    if not _same_document_or_collection(source, target):
        return False
    labels = _object_labels(source)
    if not labels:
        return False
    target_text = target.content.lower()
    return any(label.lower() in target_text for label in labels)


def has_same_directory(source: Document, target: Document) -> bool:
    return bool(source.repo and source.repo == target.repo and source.parent == target.parent)


def has_same_document(source: Document, target: Document) -> bool:
    source_doc = _explicit_document_id(source)
    target_doc = _explicit_document_id(target)
    return bool(source_doc and source_doc == target_doc and source.docid != target.docid)


def has_section_neighbor(source: Document, target: Document) -> bool:
    if not has_same_document(source, target):
        return False
    source_index = _section_index(source)
    target_index = _section_index(target)
    return (
        source_index is not None
        and target_index is not None
        and abs(source_index - target_index) == 1
    )


def has_same_collection(source: Document, target: Document) -> bool:
    source_collection = str(source.metadata.get("collection", ""))
    target_collection = str(target.metadata.get("collection", ""))
    return bool(source_collection and source_collection == target_collection)


def has_same_domain(source: Document, target: Document) -> bool:
    source_domain = _domain(source)
    target_domain = _domain(target)
    return bool(source_domain and source_domain == target_domain)


def _path_to_module(path: str) -> str:
    pure_path = PurePosixPath(path)
    no_suffix = pure_path.with_suffix("")
    parts = [part for part in no_suffix.parts if part not in {"src", "lib", "."}]
    return ".".join(parts)


def _full_path_to_module(path: str) -> str:
    pure_path = PurePosixPath(path)
    no_suffix = pure_path.with_suffix("")
    parts = [part for part in no_suffix.parts if part != "."]
    return ".".join(parts)


def _clean_stem(path: str) -> str:
    return PurePosixPath(path).stem.replace("test_", "").replace("_test", "").lower()


def _is_test_document(document: Document) -> bool:
    target_name = document.filename.lower()
    target_path = document.path.lower()
    return (
        document.source_type == "test"
        or "test" in target_name
        or "/test" in target_path
        or "tests/" in target_path
    )


def _is_config_document(document: Document) -> bool:
    return document.source_type == "config" or document.filename.lower() in CONFIG_NAMES


def _is_script_document(document: Document) -> bool:
    target_name = document.filename.lower()
    return document.source_type == "script" or target_name in {
        "train.sh",
        "run.sh",
        "train.py",
        "run.py",
    }


def _has_import_or_path_evidence(source: Document, target: Document) -> bool:
    text = target.content.lower()
    aliases = _code_target_aliases(source, include_symbols=False)
    if any(alias and alias.lower() in text for alias in aliases):
        return True
    return has_import_relation(source, target)


def _document_mentions_code_target(source: Document, target: Document) -> bool:
    text = source.content.lower()
    for alias in _code_target_aliases(target, include_symbols=False):
        if alias and alias.lower() in text:
            return True
    for symbol in _code_symbols(target):
        if len(symbol) >= 3 and symbol.lower() not in GENERIC_STEMS and _symbol_in_text(
            symbol, source.content
        ):
            return True
    return False


def _code_target_aliases(document: Document, include_symbols: bool = True) -> set[str]:
    aliases = {document.path, document.filename}
    module = _path_to_module(document.path)
    raw_module = _full_path_to_module(document.path)
    if "." in module:
        aliases.add(module)
    if "." in raw_module:
        aliases.add(raw_module)
    metadata_module = str(document.metadata.get("module", ""))
    metadata_module_path = str(document.metadata.get("module_path", ""))
    for alias in (metadata_module, metadata_module_path):
        if "." in alias or alias.lower() not in GENERIC_STEMS:
            aliases.add(alias)
    if include_symbols:
        aliases.update(_code_symbols(document))
    return {alias.strip() for alias in aliases if alias and len(alias.strip()) >= 3}


def _code_symbols(document: Document) -> set[str]:
    symbols = set(_metadata_values(document, "symbols", "api_names", "exported_symbols"))
    patterns = [
        r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        r"\bfunction\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(",
        r"\bfunc\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        r"\bconst\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=",
        r"\bexport\s+(?:class|function|const)\s+([A-Za-z_$][A-Za-z0-9_$]*)",
    ]
    for pattern in patterns:
        symbols.update(re.findall(pattern, document.content))
    return {symbol for symbol in symbols if len(symbol) >= 3}


def _config_calls_script(config: Document, script: Document) -> bool:
    text = config.content.lower()
    if script.path.lower() in text:
        return True
    script_name = script.filename.lower()
    if script_name not in text:
        return False
    call_patterns = [
        rf"\b(?:python|python3|bash|sh|zsh|make|run|script|command|cmd|entrypoint)"
        rf"\b[^\n]*{re.escape(script_name)}",
        rf"{re.escape(script_name)}[^\n]*\b(?:args|with|--|:)",
    ]
    return any(re.search(pattern, text) for pattern in call_patterns)


def _script_reads_config(script: Document, config: Document) -> bool:
    text = script.content.lower()
    if config.path.lower() in text or config.filename.lower() in text:
        return True
    config_stem = PurePosixPath(config.path).stem.lower()
    if config_stem in GENERIC_STEMS or len(config_stem) < 3:
        return False
    read_patterns = [
        rf"\b(?:open|load|read|parse|from_pretrained|yaml\.safe_load|json\.load)"
        rf"\b[^\n]*{re.escape(config_stem)}",
        rf"{re.escape(config_stem)}[^\n]*\b(?:yaml|yml|json|toml|ini|cfg)\b",
    ]
    return any(re.search(pattern, text) for pattern in read_patterns)


def _metadata_values(document: Document, *keys: str) -> list[str]:
    values: list[str] = []
    for key in keys:
        value = document.metadata.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            values.extend(str(item) for item in value if item is not None)
        else:
            values.append(str(value))
    return [value for value in values if value]


def _target_link_aliases(document: Document) -> set[str]:
    aliases = {document.docid, document.path}
    aliases.update(
        _metadata_values(document, "url", "canonical_url", "anchor", "source_url", "location")
    )
    return {alias for alias in aliases if alias}


def _normalize_link(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc:
        return parsed._replace(fragment="").geturl().rstrip("/").lower()
    return value.rstrip("/").lower()


def _same_document_or_collection(source: Document, target: Document) -> bool:
    return (
        has_same_document(source, target)
        or has_same_collection(source, target)
        or bool(source.repo and source.repo == target.repo)
    )


def _definition_term(document: Document) -> str:
    if not _is_definition_document(document):
        return ""
    for key in ("term", "concept", "definition_term"):
        value = str(document.metadata.get(key, "")).strip()
        if value:
            return value
    match = re.search(
        r"(?im)^\s*(?:definition|define|term)\s*[:\-]\s*([A-Za-z][\w \-]{2,80})",
        document.content,
    )
    if match:
        return match.group(1).strip()
    return ""


def _is_definition_document(document: Document) -> bool:
    role = str(document.metadata.get("role", document.metadata.get("section_type", ""))).lower()
    if role in {"definition", "concept", "glossary"}:
        return True
    title = str(document.metadata.get("section_title", document.metadata.get("title", ""))).lower()
    return "definition" in title or "glossary" in title


def _api_terms(document: Document) -> set[str]:
    terms = set(_metadata_values(document, "api_name", "api_names", "symbols", "term"))
    title = str(document.metadata.get("section_title", document.metadata.get("title", ""))).strip()
    if title:
        terms.add(title)
    return {term for term in terms if len(term) >= 3 and term.lower() not in GENERIC_STEMS}


def _is_api_document(document: Document) -> bool:
    title = str(document.metadata.get("section_title", document.metadata.get("title", ""))).lower()
    if any(marker in title for marker in ("usage", "example", "tutorial")):
        return False
    return document.source_type == "api_doc" or bool(
        _metadata_values(document, "api_name", "api_names", "symbols")
    )


def _object_labels(document: Document) -> set[str]:
    labels = set(
        _metadata_values(document, "object_label", "figure_id", "table_id", "equation_id", "label")
    )
    if not labels:
        title = str(document.metadata.get("section_title", "")).strip()
        match = re.search(
            r"\b((?:Figure|Fig\.|Table|Equation|Eq\.)\s*\(?[A-Za-z0-9.\-]+\)?)",
            title,
        )
        if match:
            labels.add(match.group(1))
    expanded: set[str] = set()
    for label in labels:
        expanded.add(label)
        if re.match(r"^[A-Za-z0-9.\-]+$", label):
            expanded.update(
                {
                    f"Figure {label}",
                    f"Fig. {label}",
                    f"Table {label}",
                    f"Equation {label}",
                    f"Eq. ({label})",
                }
            )
    return {label for label in expanded if len(label) >= 3}


def _term_in_text(term: str, text: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, flags=re.IGNORECASE))


def _symbol_in_text(symbol: str, text: str) -> bool:
    flags = 0 if any(char.isupper() for char in symbol) else re.IGNORECASE
    return bool(re.search(rf"(?<!\w){re.escape(symbol)}(?!\w)", text, flags=flags))


def _term_in_code_or_text(term: str, text: str) -> bool:
    if _term_in_text(term, text):
        return True
    if "." in term:
        return term.lower() in text.lower()
    return bool(re.search(rf"\b{re.escape(term)}\s*\(", text, flags=re.IGNORECASE))


def _explicit_document_id(document: Document) -> str:
    return str(document.metadata.get("document_id", ""))


def _section_index(document: Document) -> int | None:
    value = document.metadata.get("section_index")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _domain(document: Document) -> str:
    url = str(document.metadata.get("url", document.metadata.get("source_url", "")))
    if not url:
        return ""
    return urlparse(url).netloc.lower()
