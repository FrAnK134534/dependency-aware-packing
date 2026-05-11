from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from dapacking.documents import Document


DEFAULT_WEIGHTS = {
    "import_relation": 1.0,
    "test_source_relation": 0.9,
    "readme_code_relation": 0.6,
    "config_script_relation": 0.5,
    "same_directory": 0.25,
    "same_repo": 0.1,
}

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


@dataclass(frozen=True)
class DependencyEvidence:
    score: float
    labels: tuple[str, ...]


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
        "readme_code_relation": has_readme_code_relation(source, target),
        "config_script_relation": has_config_script_relation(source, target),
        "same_directory": has_same_directory(source, target),
        "same_repo": bool(source.repo and source.repo == target.repo),
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
    import_patterns = (
        rf"\bimport\s+{re.escape(source_module)}\b",
        rf"\bfrom\s+{re.escape(source_module)}\s+import\b",
        rf"\bimport\s+{re.escape(source_stem)}\b",
        rf"\bfrom\s+{re.escape(source_stem)}\s+import\b",
        rf"require\(['\"](.*/)?{re.escape(source_stem)}['\"]\)",
    )
    return any(re.search(pattern, target.content) for pattern in import_patterns)


def has_test_source_relation(source: Document, target: Document) -> bool:
    if source.repo and target.repo and source.repo != target.repo:
        return False

    source_stem = PurePosixPath(source.path).stem.replace("test_", "").replace("_test", "")
    target_name = target.filename.lower()
    target_path = target.path.lower()
    if "test" not in target_name and "/test" not in target_path and "tests/" not in target_path:
        return False
    return source_stem.lower() in target.content.lower() or source_stem.lower() in target_name


def has_readme_code_relation(source: Document, target: Document) -> bool:
    if source.repo and target.repo and source.repo != target.repo:
        return False
    return source.filename.lower().startswith("readme") and target.suffix in CODE_SUFFIXES


def has_config_script_relation(source: Document, target: Document) -> bool:
    if source.repo and target.repo and source.repo != target.repo:
        return False
    source_name = source.filename.lower()
    target_name = target.filename.lower()
    is_config = source_name in CONFIG_NAMES or source.suffix in {".yaml", ".yml", ".toml", ".ini", ".cfg"}
    is_script = target.suffix in CODE_SUFFIXES or target_name in {"train.sh", "run.sh"}
    return is_config and is_script


def has_same_directory(source: Document, target: Document) -> bool:
    return bool(source.repo and source.repo == target.repo and source.parent == target.parent)


def _path_to_module(path: str) -> str:
    pure_path = PurePosixPath(path)
    no_suffix = pure_path.with_suffix("")
    parts = [part for part in no_suffix.parts if part not in {"src", "lib", "."}]
    return ".".join(parts)
