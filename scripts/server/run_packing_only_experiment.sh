#!/usr/bin/env bash
set -euo pipefail

REPO_SET=${1:-configs/datasets/python50_repos.tsv}
OUTPUT_DIR=${2:-data/processed/python50}
MAX_TOKENS=${3:-8192}
TOKENIZER=${4:-simple}
METHODS=${5:-random,bm25,semantic,datasculpt_lite,dependency_aware_v2_token_fit,dependency_aware_v2_strong_first,dependency_aware_strong_edges_only}

RAW_REPO_DIR=${RAW_REPO_DIR:-data/raw/repos}
MAX_DOCS_PER_REPO=${MAX_DOCS_PER_REPO:-300}
MANIFEST_STEM=$(basename "${REPO_SET%.*}")
LOCAL_MANIFEST=${LOCAL_MANIFEST:-data/raw/${MANIFEST_STEM}_local_manifest.txt}
PACK_SPLITS=${PACK_SPLITS:-train}

python scripts/data/clone_repo_manifest.py \
  --input "${REPO_SET}" \
  --repo-dir "${RAW_REPO_DIR}" \
  --output-manifest "${LOCAL_MANIFEST}"

MAX_DOCS_PER_REPO="${MAX_DOCS_PER_REPO}" \
  bash scripts/server/build_dataset_pipeline.sh "${LOCAL_MANIFEST}" "${OUTPUT_DIR}"

for SPLIT in ${PACK_SPLITS}; do
  PACKED_DIR="${OUTPUT_DIR}/packed/${SPLIT}_${MAX_TOKENS}"
  python scripts/run_packing_matrix.py \
    --input "${OUTPUT_DIR}/splits/${SPLIT}_docs.jsonl" \
    --output-dir "${PACKED_DIR}" \
    --methods "${METHODS}" \
    --max-tokens "${MAX_TOKENS}" \
    --tokenizer "${TOKENIZER}" \
    --edges "${OUTPUT_DIR}/splits/${SPLIT}_edges.jsonl" \
    --summary "${PACKED_DIR}/summary.csv"
done

echo "Packing-only experiment complete: ${OUTPUT_DIR}/packed"
