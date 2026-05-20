from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dapacking.documents import Document

CODE_SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
}

DOC_SUFFIXES = {".md", ".rst", ".txt", ".adoc"}
CONFIG_SUFFIXES = {".yaml", ".yml", ".toml", ".json", ".ini", ".cfg", ".env"}
SCRIPT_SUFFIXES = {".sh", ".bash", ".zsh", ".ps1"}

CONFIG_NAMES = {
    "dockerfile",
    "makefile",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "package.json",
    "cargo.toml",
    "go.mod",
    "environment.yml",
}

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
    "target",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
}


@dataclass(frozen=True)
class CorpusBuildConfig:
    max_file_bytes: int = 1_000_000
    include_unknown: bool = False
    max_docs_per_repo: int | None = None


def build_documents_from_repos(
    repo_roots: list[str | Path],
    config: CorpusBuildConfig | None = None,
) -> list[Document]:
    config = config or CorpusBuildConfig()
    documents: list[Document] = []

    for root_like in repo_roots:
        root = Path(root_like).resolve()
        repo_name = root.name
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Repository root does not exist or is not a directory: {root}")

        repo_documents: list[Document] = []
        for path in sorted(_iter_repo_files(root)):
            if path.stat().st_size > config.max_file_bytes:
                continue

            source_type = classify_source_type(path.relative_to(root))
            if source_type == "unknown" and not config.include_unknown:
                continue

            content = _read_text(path)
            if content is None or not content.strip():
                continue

            relative_path = path.relative_to(root).as_posix()
            repo_documents.append(
                Document(
                    docid=f"{repo_name}:{relative_path}",
                    content=content,
                    metadata={
                        "repo": repo_name,
                        "path": relative_path,
                        "language": infer_language(path),
                        "source_type": source_type,
                        "bytes": path.stat().st_size,
                    },
                )
            )

        documents.extend(_limit_repo_documents(repo_documents, config.max_docs_per_repo))

    return documents


def _limit_repo_documents(documents: list[Document], limit: int | None) -> list[Document]:
    if limit is None or limit <= 0 or len(documents) <= limit:
        return documents
    return sorted(documents, key=_document_priority)[:limit]


def _document_priority(document: Document) -> tuple[int, str]:
    source_priority = {
        "readme": 0,
        "config": 1,
        "source": 2,
        "test": 3,
        "docs": 4,
        "example": 5,
        "script": 6,
        "unknown": 7,
    }
    return source_priority.get(document.source_type, 99), document.path


def classify_source_type(relative_path: str | Path) -> str:
    path = Path(relative_path)
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    suffix = path.suffix.lower()

    if name.startswith("readme"):
        return "readme"
    if "test" in name or "tests" in parts or "test" in parts or "__tests__" in parts:
        return "test"
    if "docs" in parts or "doc" in parts or "documentation" in parts:
        return "docs"
    if "examples" in parts or "example" in parts:
        return "example"
    if "scripts" in parts or suffix in SCRIPT_SUFFIXES or name in {"train.py", "run.py", "main.py"}:
        return "script"
    if name in CONFIG_NAMES or suffix in CONFIG_SUFFIXES:
        return "config"
    if suffix in CODE_SUFFIXES:
        return "source"
    if suffix in DOC_SUFFIXES:
        return "docs"
    return "unknown"


def infer_language(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".md": "markdown",
        ".rst": "rst",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".json": "json",
        ".sh": "shell",
    }.get(suffix, suffix.lstrip(".") or "text")


def _iter_repo_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS for part in relative_parts):
            continue
        files.append(path)
    return files


def _read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode("latin-1")
        except UnicodeDecodeError:
            return None
