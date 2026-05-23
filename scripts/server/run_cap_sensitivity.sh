#!/usr/bin/env bash
set -euo pipefail

REPO_SET=${1:-configs/datasets/python50_repos.tsv}
OUTPUT_ROOT=${2:-data/processed/cap_sensitivity}
MAX_TOKENS=${3:-8192}
TOKENIZER=${4:-simple}
METHODS=${5:-random,bm25,datasculpt_lite,dependency_aware_v2_strong_first}

CAP_VALUES=${CAP_VALUES:-"100 300 500"}
COMBINED_SUMMARY="${OUTPUT_ROOT}/cap_sensitivity_summary.csv"

for CAP in ${CAP_VALUES}; do
  echo "== Running cap sensitivity: MAX_DOCS_PER_REPO=${CAP} =="
  MAX_DOCS_PER_REPO="${CAP}" bash scripts/server/run_packing_only_experiment.sh \
    "${REPO_SET}" \
    "${OUTPUT_ROOT}/cap_${CAP}" \
    "${MAX_TOKENS}" \
    "${TOKENIZER}" \
    "${METHODS}"
done

python scripts/analysis/collect_cap_summaries.py \
  --input-root "${OUTPUT_ROOT}" \
  --max-tokens "${MAX_TOKENS}" \
  --output "${COMBINED_SUMMARY}"

echo "Cap sensitivity complete: ${COMBINED_SUMMARY}"
