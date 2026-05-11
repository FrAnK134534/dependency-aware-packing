from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from dapacking.documents import Document, PackedSample


def read_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc
    return records


def read_documents(path: str | Path) -> list[Document]:
    documents: list[Document] = []
    for record in read_jsonl(path):
        documents.append(
            Document(
                docid=str(record["docid"]),
                content=str(record["content"]),
                metadata=dict(record.get("metadata", {})),
            )
        )
    return documents


def write_samples(path: str | Path, samples: Iterable[PackedSample]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_json(), ensure_ascii=False) + "\n")
