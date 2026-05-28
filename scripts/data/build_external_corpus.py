#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dapacking.collectors import ExternalCorpusConfig, build_external_documents
from dapacking.io import write_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build non-code documents JSONL from a manifest of papers, web pages, "
            "docs, and text."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--fetch-urls", action="store_true")
    parser.add_argument("--follow-same-domain-once", action="store_true")
    parser.add_argument("--max-follow-links", type=int, default=20)
    parser.add_argument("--request-timeout", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = build_external_documents(
        args.manifest,
        ExternalCorpusConfig(
            base_dir=args.base_dir or args.manifest.parent,
            fetch_urls=args.fetch_urls,
            follow_same_domain_once=args.follow_same_domain_once,
            max_follow_links=args.max_follow_links,
            request_timeout=args.request_timeout,
        ),
    )
    write_documents(args.output, documents)
    print(f"documents={len(documents)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
