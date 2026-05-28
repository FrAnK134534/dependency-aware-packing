from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dapacking.io import read_jsonl


@dataclass(frozen=True)
class ManifestEntry:
    source_id: str
    source_kind: str
    location: str
    collection: str
    license: str = ""
    title: str = ""
    source_type: str = ""
    document_id: str = ""
    metadata: dict[str, Any] | None = None


def read_manifest(path: str | Path) -> list[ManifestEntry]:
    manifest_path = Path(path)
    suffix = manifest_path.suffix.lower()
    if suffix == ".jsonl":
        records = read_jsonl(manifest_path)
    elif suffix in {".tsv", ".csv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle, delimiter=delimiter))
    else:
        raise ValueError(f"Unsupported manifest format: {manifest_path}")
    return [_entry_from_record(record, manifest_path) for record in records]


def _entry_from_record(record: dict[str, Any], manifest_path: Path) -> ManifestEntry:
    source_id = str(record.get("source_id", "")).strip()
    source_kind = str(record.get("source_kind", "")).strip()
    location = str(record.get("location", "")).strip()
    collection = str(record.get("collection", "")).strip()
    if not source_id:
        raise ValueError(f"Manifest row in {manifest_path} is missing source_id")
    if not source_kind:
        raise ValueError(f"Manifest row for {source_id} is missing source_kind")
    if not location:
        raise ValueError(f"Manifest row for {source_id} is missing location")
    if not collection:
        collection = "external"

    known = {
        "source_id",
        "source_kind",
        "location",
        "collection",
        "license",
        "title",
        "source_type",
        "document_id",
    }
    metadata = {
        key: value
        for key, value in record.items()
        if key not in known and value is not None and value != ""
    }
    return ManifestEntry(
        source_id=source_id,
        source_kind=source_kind,
        location=location,
        collection=collection,
        license=str(record.get("license", "")).strip(),
        title=str(record.get("title", "")).strip(),
        source_type=str(record.get("source_type", "")).strip(),
        document_id=str(record.get("document_id", "")).strip(),
        metadata=metadata,
    )
