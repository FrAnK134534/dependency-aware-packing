#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dapacking.io import read_documents, write_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple document JSONL files.")
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents_by_id = {}
    for path in args.input:
        for document in read_documents(path):
            documents_by_id.setdefault(document.docid, document)
    documents = list(documents_by_id.values())
    write_documents(args.output, documents)
    print(f"inputs={len(args.input)}")
    print(f"documents={len(documents)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
