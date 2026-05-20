#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dapacking.io import read_documents, write_samples
from dapacking.packers import PackingConfig, build_packer
from dapacking.stats import summarize_packed_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multiple packing baselines from one document file.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--methods",
        default=(
            "random,length_aware,same_repo,bm25,semantic,datasculpt_lite,"
            "dependency_aware,dependency_aware_v2_token_fit,"
            "dependency_aware_no_same_directory,dependency_aware_no_same_repo,"
            "dependency_aware_strong_edges_only"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-dependency-score", type=float, default=0.11)
    parser.add_argument("--min-similarity-score", type=float, default=0.01)
    parser.add_argument("--candidate-pool-size", type=int, default=80)
    parser.add_argument("--redundancy-threshold", type=float, default=0.72)
    parser.add_argument("--edges", type=Path, help="Optional edge file for summary coverage metrics.")
    parser.add_argument("--summary", type=Path, help="Optional summary CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = read_documents(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]

    summaries = []
    for method in methods:
        output_path = args.output_dir / f"{method}_{args.max_tokens}.jsonl"
        packer = build_packer(
            PackingConfig(
                method=method,
                max_tokens=args.max_tokens,
                seed=args.seed,
                min_dependency_score=args.min_dependency_score,
                min_similarity_score=args.min_similarity_score,
                candidate_pool_size=args.candidate_pool_size,
                redundancy_threshold=args.redundancy_threshold,
            )
        )
        samples = packer.pack(documents)
        write_samples(output_path, samples)
        summary = summarize_packed_file(output_path, args.edges)
        summaries.append(summary)
        print(
            f"method={method} samples={summary.samples} tokens={summary.total_tokens} "
            f"avg_dep={summary.avg_dependency_score:.4f} output={output_path}",
            flush=True,
        )

    if args.summary:
        import csv

        args.summary.parent.mkdir(parents=True, exist_ok=True)
        rows = [summary.to_row() for summary in summaries]
        with args.summary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"summary={args.summary}", flush=True)


if __name__ == "__main__":
    main()
