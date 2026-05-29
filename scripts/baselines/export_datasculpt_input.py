#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dapacking.io import read_documents, write_jsonl
from dapacking.tokenization import configure_tokenizer, count_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export dapacking Document JSONL for the original DataSculpt pipeline.")
    parser.add_argument("--documents", required=True, type=Path)
    parser.add_argument("--output-folder", required=True, type=Path)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--tokenizer", default="simple")
    parser.add_argument("--tokenizer-local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_tokenizer(args.tokenizer, local_files_only=args.tokenizer_local_files_only)
    documents = read_documents(args.documents)
    args.output_folder.mkdir(parents=True, exist_ok=True)

    shard_count = 0
    for start in range(0, len(documents), args.shard_size):
        shard = documents[start : start + args.shard_size]
        rows = []
        for document in shard:
            rows.append(
                {
                    "docid": document.docid,
                    "content": document.content,
                    "metadata": document.metadata,
                    "token_len": count_tokens(document.content),
                }
            )
        output = args.output_folder / f"part-{shard_count:05d}"
        write_jsonl(output, rows)
        shard_count += 1

    print(f"documents={len(documents)}")
    print(f"shards={shard_count}")
    print(f"output_folder={args.output_folder}")


if __name__ == "__main__":
    main()
