#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dapacking.edge_filter import filter_dependency_edges, relation_counts
from dapacking.edges import read_dependency_edges, write_dependency_edges
from dapacking.relation_config import load_relation_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter dependency_edges.jsonl to a high-precision relation set."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--relation-config",
        required=True,
        type=Path,
        help="YAML relation config, e.g. configs/relations/main_high_precision.yaml.",
    )
    parser.add_argument("--min-weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    relation_config = load_relation_config(args.relation_config)
    edges = read_dependency_edges(args.input)
    filtered = filter_dependency_edges(
        edges,
        relation_config=relation_config,
        min_weight=args.min_weight,
    )
    write_dependency_edges(args.output, filtered)

    print(f"relation_config={relation_config.name}")
    print(f"input_edges={len(edges)}")
    print(f"output_edges={len(filtered)}")
    print("output_relation_counts=" + _format_counts(relation_counts(filtered)))
    print(f"output={args.output}")


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "{}"
    return ",".join(f"{relation}:{counts[relation]}" for relation in sorted(counts))


if __name__ == "__main__":
    main()
