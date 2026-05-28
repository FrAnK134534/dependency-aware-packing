#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dapacking.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge dependency edge JSONL files without recomputing existing edges."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=Path,
        help="Input dependency_edges.jsonl path. May be passed multiple times.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seen: set[tuple[str, str, str]] = set()
    records: list[dict[str, object]] = []
    for path in args.input:
        for record in read_jsonl(path):
            key = (
                str(record.get("source_docid", "")),
                str(record.get("target_docid", "")),
                str(record.get("relation", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            records.append(record)

    write_jsonl(args.output, records)
    print(f"inputs={len(args.input)}")
    print(f"edges={len(records)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
