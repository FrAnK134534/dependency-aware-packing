#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepoSpec:
    name: str
    url: str
    ref: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone a TSV repository set and write a local path manifest."
    )
    parser.add_argument("--input", required=True, type=Path, help="TSV file: name, url, optional ref.")
    parser.add_argument("--repo-dir", default=Path("data/raw/repos"), type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Optional first-N repo limit for smoke tests.")
    parser.add_argument("--force-fetch", action="store_true", help="Fetch existing repositories.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = read_repo_specs(args.input)
    if args.limit:
        specs = specs[: args.limit]

    args.repo_dir.mkdir(parents=True, exist_ok=True)
    local_paths: list[Path] = []
    failures: list[tuple[RepoSpec, str]] = []

    for spec in specs:
        target = args.repo_dir / spec.name
        try:
            ensure_repo(spec, target, args.depth, args.force_fetch)
            local_paths.append(target)
        except subprocess.CalledProcessError as exc:
            failures.append((spec, str(exc)))

    write_manifest(args.output_manifest, local_paths)
    print(f"requested_repos={len(specs)}")
    print(f"available_repos={len(local_paths)}")
    print(f"output_manifest={args.output_manifest}")

    if failures:
        print("clone_failures:", file=sys.stderr)
        for spec, message in failures:
            print(f"- {spec.name}: {message}", file=sys.stderr)
        raise SystemExit(1)


def read_repo_specs(path: Path) -> list[RepoSpec]:
    specs: list[RepoSpec] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) not in {2, 3}:
                raise ValueError(f"Invalid repo spec on line {line_no}: {line}")
            name, url = fields[:2]
            ref = fields[2] if len(fields) == 3 else None
            specs.append(RepoSpec(name=name, url=url, ref=ref))
    return specs


def ensure_repo(spec: RepoSpec, target: Path, depth: int, force_fetch: bool) -> None:
    if target.exists():
        if force_fetch:
            run(["git", "-C", str(target), "fetch", "--depth", str(depth), "origin"])
            if spec.ref:
                run(["git", "-C", str(target), "checkout", spec.ref])
        print(f"exists {target}")
        return

    command = ["git", "clone", "--depth", str(depth), spec.url, str(target)]
    if spec.ref:
        command[2:2] = ["--branch", spec.ref]
    run(command)


def write_manifest(path: Path, local_paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for local_path in local_paths:
            handle.write(f"{local_path}\n")


def run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
