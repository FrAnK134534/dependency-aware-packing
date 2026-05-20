#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dapacking.corpus import CorpusBuildConfig, build_documents_from_repos
from dapacking.io import write_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build documents.jsonl from local repository roots.")
    parser.add_argument("--repo", action="append", type=Path, default=[], help="Repository root. Repeatable.")
    parser.add_argument("--manifest", type=Path, help="Text file with one repository root per line.")
    parser.add_argument("--output", required=True, type=Path, help="Output documents JSONL path.")
    parser.add_argument("--max-file-bytes", type=int, default=1_000_000)
    parser.add_argument("--include-unknown", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_roots = list(args.repo)
    if args.manifest:
        repo_roots.extend(_read_manifest(args.manifest))
    if not repo_roots:
        raise SystemExit("Provide at least one --repo or --manifest.")

    documents = build_documents_from_repos(
        repo_roots,
        CorpusBuildConfig(
            max_file_bytes=args.max_file_bytes,
            include_unknown=args.include_unknown,
        ),
    )
    write_documents(args.output, documents)
    print(f"repos={len(repo_roots)}")
    print(f"documents={len(documents)}")
    print(f"output={args.output}")


def _read_manifest(path: Path) -> list[Path]:
    roots: list[Path] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                roots.append(Path(line))
    return roots


if __name__ == "__main__":
    main()
