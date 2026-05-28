#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dapacking.edges import read_dependency_edges
from dapacking.io import read_documents, write_jsonl
from dapacking.tokenization import configure_tokenizer
from dapacking.validation import ControlValidationConfig, build_control_validation_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build same-group non-edge and random cross-group context-gain controls."
    )
    parser.add_argument("--documents", required=True, type=Path)
    parser.add_argument("--edges", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-examples-per-control", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-tokens", type=int, default=2048)
    parser.add_argument("--max-target-tokens", type=int, default=2048)
    parser.add_argument("--tokenizer", default="simple")
    parser.add_argument("--tokenizer-local-files-only", action="store_true")
    parser.add_argument("--tokenizer-trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_tokenizer(
        args.tokenizer,
        local_files_only=args.tokenizer_local_files_only,
        trust_remote_code=args.tokenizer_trust_remote_code,
    )
    documents = read_documents(args.documents)
    edges = read_dependency_edges(args.edges)
    records = build_control_validation_records(
        documents,
        edges,
        ControlValidationConfig(
            max_examples_per_control=args.max_examples_per_control,
            seed=args.seed,
            max_source_tokens=args.max_source_tokens,
            max_target_tokens=args.max_target_tokens,
        ),
    )
    write_jsonl(args.output, records)
    relation_counts = Counter(str(record["primary_relation"]) for record in records)
    print(f"documents={len(documents)}")
    print(f"edges={len(edges)}")
    print(f"control_records={len(records)}")
    for relation, count in sorted(relation_counts.items()):
        print(f"control.{relation}={count}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
