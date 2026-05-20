#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dapacking.io import read_documents, write_samples
from dapacking.packers import PackingConfig, build_packer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate packed long-context samples.")
    parser.add_argument("--config", type=Path, help="Optional YAML config path.")
    parser.add_argument("--input", type=Path, help="Input JSONL documents.")
    parser.add_argument("--output", type=Path, help="Output JSONL packed samples.")
    parser.add_argument(
        "--method",
        choices=[
            "random",
            "length_aware",
            "same_repo",
            "bm25",
            "semantic",
            "datasculpt_lite",
            "dependency_aware",
            "dependency_aware_v2_token_fit",
            "dependency_aware_no_same_directory",
            "dependency_aware_no_same_repo",
            "dependency_aware_strong_edges_only",
        ],
        help="Packing method.",
    )
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens per packed sample.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--min-dependency-score",
        type=float,
        default=None,
        help="Minimum dependency score for dependency-aware packing.",
    )
    parser.add_argument(
        "--min-similarity-score",
        type=float,
        default=None,
        help="Minimum TF-IDF similarity for semantic packing.",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=None,
        help="Maximum ranked candidates considered per anchor; use 0 for all candidates.",
    )
    parser.add_argument(
        "--redundancy-threshold",
        type=float,
        default=None,
        help="Pair similarity threshold where DataSculpt-lite starts penalizing redundancy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    input_path = args.input or Path(config.get("input_path", ""))
    output_path = args.output or Path(config.get("output_path", ""))
    method = args.method or config.get("method", "dependency_aware")
    max_tokens = args.max_tokens or int(config.get("max_tokens", 4096))
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    min_dependency_score = (
        args.min_dependency_score
        if args.min_dependency_score is not None
        else float(config.get("min_dependency_score", 0.11))
    )
    min_similarity_score = (
        args.min_similarity_score
        if args.min_similarity_score is not None
        else float(config.get("min_similarity_score", 0.01))
    )
    candidate_pool_size = (
        args.candidate_pool_size
        if args.candidate_pool_size is not None
        else int(config.get("candidate_pool_size", 80))
    )
    redundancy_threshold = (
        args.redundancy_threshold
        if args.redundancy_threshold is not None
        else float(config.get("redundancy_threshold", 0.72))
    )

    if not input_path:
        raise SystemExit("--input or input_path in config is required")
    if not output_path:
        raise SystemExit("--output or output_path in config is required")

    documents = read_documents(input_path)
    packer = build_packer(
        PackingConfig(
            method=method,
            max_tokens=max_tokens,
            seed=seed,
            dependency_weights=config.get("dependency_weights"),
            min_dependency_score=min_dependency_score,
            min_similarity_score=min_similarity_score,
            candidate_pool_size=candidate_pool_size,
            redundancy_threshold=redundancy_threshold,
        )
    )
    samples = packer.pack(documents)
    write_samples(output_path, samples)

    total_tokens = sum(sample.stats["tokens"] for sample in samples)
    avg_dep = sum(sample.stats["dependency_score"] for sample in samples) / max(len(samples), 1)
    print(f"method={method}")
    print(f"documents={len(documents)}")
    print(f"samples={len(samples)}")
    print(f"tokens={total_tokens}")
    print(f"avg_dependency_score={avg_dep:.4f}")
    print(f"output={output_path}")


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    main()
