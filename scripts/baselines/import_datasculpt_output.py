#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dapacking.documents import Document, PackedSample
from dapacking.io import read_documents, read_jsonl, write_samples
from dapacking.packers import average_dependency_score, average_semantic_metrics, format_document_for_window
from dapacking.stats import summarize_packed_file
from dapacking.tokenization import configure_tokenizer, count_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import original DataSculpt context windows as dapacking samples.")
    parser.add_argument("--datasculpt-output-folder", required=True, type=Path)
    parser.add_argument("--documents", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--method-name", default="datasculpt_original")
    parser.add_argument("--tokenizer", default="simple")
    parser.add_argument("--tokenizer-local-files-only", action="store_true")
    parser.add_argument("--edges", type=Path)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_tokenizer(args.tokenizer, local_files_only=args.tokenizer_local_files_only)
    source_documents = {document.docid: document for document in read_documents(args.documents)}
    samples = _read_datasculpt_windows(args, source_documents)
    write_samples(args.output, samples)

    print(f"samples={len(samples)}")
    print(f"output={args.output}")

    if args.summary:
        summary = summarize_packed_file(args.output, args.edges)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary.to_row()))
            writer.writeheader()
            writer.writerow(summary.to_row())
        print(f"summary={args.summary}")


def _read_datasculpt_windows(
    args: argparse.Namespace,
    source_documents: dict[str, Document],
) -> list[PackedSample]:
    samples: list[PackedSample] = []
    for path in sorted(args.datasculpt_output_folder.rglob("*")):
        if path.is_dir():
            continue
        for record in read_jsonl(path):
            docs = record.get("docs", [])
            if not isinstance(docs, list) or not docs:
                continue
            total_token_num = int(float(record.get("total_token_num", 0) or 0))
            samples.append(_make_sample(args, len(samples), docs, source_documents, total_token_num))
    return samples


def _make_sample(
    args: argparse.Namespace,
    index: int,
    rows: list[dict],
    source_documents: dict[str, Document],
    datasculpt_total_token_num: int,
) -> PackedSample:
    docids: list[str] = []
    chunk_documents: list[Document] = []
    content_parts: list[str] = []
    for row_index, row in enumerate(rows):
        docid = str(row.get("source_id") or row.get("docid") or row.get("id") or f"unknown:{row_index}")
        chunk = str(row.get("chunk") or row.get("content") or row.get("text") or "")
        if docid not in docids:
            docids.append(docid)
        original = source_documents.get(docid)
        metadata = dict(original.metadata) if original else dict(row.get("metadata", {}))
        chunk_documents.append(Document(docid=docid, content=chunk or (original.content if original else ""), metadata=metadata))
        content_parts.append(format_document_for_window(chunk_documents[-1]))

    content = "\n\n".join(content_parts)
    tokens = count_tokens(content)
    dep_score = average_dependency_score(_unique_documents(chunk_documents))
    semantic_similarity, redundant_pair_rate = average_semantic_metrics(_unique_documents(chunk_documents))
    return PackedSample(
        sample_id=f"{args.method_name}_{index:06d}",
        method=args.method_name,
        docids=docids,
        content=content,
        stats={
            "tokens": tokens,
            "num_docs": len(docids),
            "dependency_score": round(dep_score, 4),
            "token_utilization": round(tokens / max(args.max_tokens, 1), 4),
            "truncated_tokens": 0,
            "truncation_rate": 0.0,
            "semantic_similarity": round(semantic_similarity, 4),
            "redundant_pair_rate": round(redundant_pair_rate, 4),
            "tokenizer": args.tokenizer,
            "datasculpt_total_token_num": datasculpt_total_token_num,
        },
    )


def _unique_documents(documents: list[Document]) -> list[Document]:
    by_id: dict[str, Document] = {}
    for document in documents:
        by_id.setdefault(document.docid, document)
    return list(by_id.values())


if __name__ == "__main__":
    main()
