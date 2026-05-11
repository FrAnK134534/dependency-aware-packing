from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any


@dataclass(frozen=True)
class Document:
    docid: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def repo(self) -> str:
        return str(self.metadata.get("repo", ""))

    @property
    def path(self) -> str:
        path = self.metadata.get("path")
        if path:
            return str(path)
        if ":" in self.docid:
            return self.docid.split(":", 1)[1]
        return self.docid

    @property
    def suffix(self) -> str:
        return PurePosixPath(self.path).suffix.lower()

    @property
    def parent(self) -> str:
        return str(PurePosixPath(self.path).parent)

    @property
    def filename(self) -> str:
        return PurePosixPath(self.path).name


@dataclass
class PackedSample:
    sample_id: str
    method: str
    docids: list[str]
    content: str
    stats: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "method": self.method,
            "docids": self.docids,
            "content": self.content,
            "stats": self.stats,
        }
