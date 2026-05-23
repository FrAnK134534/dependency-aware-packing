#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect cap sensitivity packing summaries into one CSV."
    )
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    for summary_path in sorted(args.input_root.glob(f"cap_*/packed/train_{args.max_tokens}/summary.csv")):
        cap = _cap_from_path(summary_path)
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({"max_docs_per_repo": cap, **row})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["max_docs_per_repo"]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"summary_files={len({row['max_docs_per_repo'] for row in rows})}")
    print(f"rows={len(rows)}")
    print(f"output={args.output}")


def _cap_from_path(path: Path) -> str:
    for part in path.parts:
        match = re.fullmatch(r"cap_(\d+)", part)
        if match:
            return match.group(1)
    return "unknown"


if __name__ == "__main__":
    main()
