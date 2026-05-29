#!/usr/bin/env bash
set -euo pipefail

MANIFEST=${1:?Usage: bash scripts/data/build_external_dataset_pipeline.sh MANIFEST OUTPUT_DIR [DATASET_NAME]}
OUTPUT_DIR=${2:?Usage: bash scripts/data/build_external_dataset_pipeline.sh MANIFEST OUTPUT_DIR [DATASET_NAME]}
DATASET_NAME=${3:-external-generalization-v1}

FETCH_URLS=${FETCH_URLS:-0}
FOLLOW_SAME_DOMAIN_ONCE=${FOLLOW_SAME_DOMAIN_ONCE:-0}
MAX_FOLLOW_LINKS=${MAX_FOLLOW_LINKS:-10}
GROUP_KEY=${GROUP_KEY:-document_id}

mkdir -p "${OUTPUT_DIR}"

EXTERNAL_ARGS=(
  --manifest "${MANIFEST}"
  --output "${OUTPUT_DIR}/documents.jsonl"
)

if [[ "${FETCH_URLS}" == "1" ]]; then
  EXTERNAL_ARGS+=(--fetch-urls)
fi
if [[ "${FOLLOW_SAME_DOMAIN_ONCE}" == "1" ]]; then
  EXTERNAL_ARGS+=(--follow-same-domain-once --max-follow-links "${MAX_FOLLOW_LINKS}")
fi

python scripts/data/build_external_corpus.py "${EXTERNAL_ARGS[@]}"

python scripts/data/build_dependency_edges.py \
  --input "${OUTPUT_DIR}/documents.jsonl" \
  --output "${OUTPUT_DIR}/dependency_edges.jsonl"

python scripts/data/split_by_repo.py \
  --documents "${OUTPUT_DIR}/documents.jsonl" \
  --edges "${OUTPUT_DIR}/dependency_edges.jsonl" \
  --output-dir "${OUTPUT_DIR}/splits" \
  --group-key "${GROUP_KEY}"

python scripts/data/write_dataset_card.py \
  --documents "${OUTPUT_DIR}/documents.jsonl" \
  --edges "${OUTPUT_DIR}/dependency_edges.jsonl" \
  --split-dir "${OUTPUT_DIR}/splits" \
  --name "${DATASET_NAME}" \
  --source-manifest "${MANIFEST}" \
  --output "${OUTPUT_DIR}/DATASET_CARD.md"

echo "External dataset pipeline complete: ${OUTPUT_DIR}"
