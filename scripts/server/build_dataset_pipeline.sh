#!/usr/bin/env bash
set -euo pipefail

REPO_MANIFEST=${1:-data/raw/repo_manifest.txt}
OUTPUT_DIR=${2:-data/processed}
MAX_DOCS_PER_REPO=${MAX_DOCS_PER_REPO:-}

mkdir -p "${OUTPUT_DIR}"

BUILD_ARGS=(
  --manifest "${REPO_MANIFEST}"
  --output "${OUTPUT_DIR}/documents.jsonl"
)

if [[ -n "${MAX_DOCS_PER_REPO}" ]]; then
  BUILD_ARGS+=(--max-docs-per-repo "${MAX_DOCS_PER_REPO}")
fi

python scripts/data/build_repo_corpus.py \
  "${BUILD_ARGS[@]}"

python scripts/data/build_dependency_edges.py \
  --input "${OUTPUT_DIR}/documents.jsonl" \
  --output "${OUTPUT_DIR}/dependency_edges.jsonl"

python scripts/data/split_by_repo.py \
  --documents "${OUTPUT_DIR}/documents.jsonl" \
  --edges "${OUTPUT_DIR}/dependency_edges.jsonl" \
  --output-dir "${OUTPUT_DIR}/splits"

echo "Dataset pipeline complete: ${OUTPUT_DIR}"
