#!/usr/bin/env bash
set -euo pipefail

REPO_MANIFEST=${1:-data/raw/repo_manifest.txt}
OUTPUT_DIR=${2:-data/processed}

mkdir -p "${OUTPUT_DIR}"

python scripts/data/build_repo_corpus.py \
  --manifest "${REPO_MANIFEST}" \
  --output "${OUTPUT_DIR}/documents.jsonl"

python scripts/data/build_dependency_edges.py \
  --input "${OUTPUT_DIR}/documents.jsonl" \
  --output "${OUTPUT_DIR}/dependency_edges.jsonl"

python scripts/data/split_by_repo.py \
  --documents "${OUTPUT_DIR}/documents.jsonl" \
  --edges "${OUTPUT_DIR}/dependency_edges.jsonl" \
  --output-dir "${OUTPUT_DIR}/splits"

echo "Dataset pipeline complete: ${OUTPUT_DIR}"
