#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR=${1:-data/processed/repo_main_v1}
MAX_TOKENS=${2:-8192}
TOKENIZER=${3:-Qwen/Qwen2.5-Coder-7B}

RELATION_CONFIG=${RELATION_CONFIG:-configs/relations/main_high_precision.yaml}
PACK_SPLITS=${PACK_SPLITS:-"train validation"}
METHODS=${METHODS:-same_repo,bm25,datasculpt_lite,dependency_aware_high_precision_only,dependency_aware_high_precision_random_order,dependency_aware_high_precision_reverse_order,dependency_aware_v2_strong_first}

TOKENIZER_ARGS=()
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  TOKENIZER_ARGS+=(--tokenizer-local-files-only)
fi
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  TOKENIZER_ARGS+=(--tokenizer-trust-remote-code)
fi

for SPLIT in ${PACK_SPLITS}; do
  DOCS="${DATASET_DIR}/splits/${SPLIT}_docs.jsonl"
  EDGES="${DATASET_DIR}/splits/${SPLIT}_edges.jsonl"
  FILTERED_EDGES="${DATASET_DIR}/splits/${SPLIT}_edges_high_precision.jsonl"
  PACKED_DIR="${DATASET_DIR}/packed/${SPLIT}_${MAX_TOKENS}_high_precision"

  if [[ ! -f "${DOCS}" ]]; then
    echo "Missing documents: ${DOCS}" >&2
    exit 2
  fi
  if [[ ! -f "${EDGES}" ]]; then
    echo "Missing edges: ${EDGES}" >&2
    exit 2
  fi

  python scripts/data/filter_dependency_edges.py \
    --input "${EDGES}" \
    --relation-config "${RELATION_CONFIG}" \
    --output "${FILTERED_EDGES}"

  python scripts/run_packing_matrix.py \
    --input "${DOCS}" \
    --edges "${FILTERED_EDGES}" \
    --relation-config "${RELATION_CONFIG}" \
    --output-dir "${PACKED_DIR}" \
    --methods "${METHODS}" \
    --max-tokens "${MAX_TOKENS}" \
    --tokenizer "${TOKENIZER}" \
    "${TOKENIZER_ARGS[@]}" \
    --summary "${PACKED_DIR}/summary.csv"
done

echo "High-precision freeze complete: ${DATASET_DIR}/packed"
