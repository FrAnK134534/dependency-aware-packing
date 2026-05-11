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
        choices=["random", "length_aware", "same_repo", "bm25", "dependency_aware"],
        help="Packing method.",
    )
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens per packed sample.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    input_path = args.input or Path(config.get("input_path", ""))
    output_path = args.output or Path(config.get("output_path", ""))
    method = args.method or config.get("method", "dependency_aware")
    max_tokens = args.max_tokens or int(config.get("max_tokens", 4096))
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))

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
