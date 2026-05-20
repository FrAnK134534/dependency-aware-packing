#!/usr/bin/env bash
set -euo pipefail

SPLIT=${1:-train}
MAX_TOKENS=${2:-8192}
PROCESSED_DIR=${3:-data/processed/splits}
OUTPUT_DIR=${4:-data/processed/packed/${SPLIT}_${MAX_TOKENS}}

DOCS="${PROCESSED_DIR}/${SPLIT}_docs.jsonl"
EDGES="${PROCESSED_DIR}/${SPLIT}_edges.jsonl"

python scripts/run_packing_matrix.py \
  --input "${DOCS}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-tokens "${MAX_TOKENS}" \
  --edges "${EDGES}" \
  --summary "${OUTPUT_DIR}/summary.csv"

echo "Packing matrix complete: ${OUTPUT_DIR}"
