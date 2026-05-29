#!/usr/bin/env bash
set -euo pipefail

DOCUMENTS=${1:?Usage: bash scripts/baselines/run_datasculpt_original.sh DOCUMENTS EDGES OUTPUT_DIR DATASCULPT_ROOT [MAX_TOKENS]}
EDGES=${2:?Usage: bash scripts/baselines/run_datasculpt_original.sh DOCUMENTS EDGES OUTPUT_DIR DATASCULPT_ROOT [MAX_TOKENS]}
OUTPUT_DIR=${3:?Usage: bash scripts/baselines/run_datasculpt_original.sh DOCUMENTS EDGES OUTPUT_DIR DATASCULPT_ROOT [MAX_TOKENS]}
DATASCULPT_ROOT=${4:?Usage: bash scripts/baselines/run_datasculpt_original.sh DOCUMENTS EDGES OUTPUT_DIR DATASCULPT_ROOT [MAX_TOKENS]}
MAX_TOKENS=${5:-8192}

TOKENIZER=${TOKENIZER:-simple}
DELTA=${DELTA:-0.5}
EPSILON=${EPSILON:-0.5}
ITER_T=${ITER_T:-2}
SHARD_SIZE=${SHARD_SIZE:-5000}

INPUT_DIR="${OUTPUT_DIR}/datasculpt_input"
WORK_DIR="${OUTPUT_DIR}/datasculpt_work"
PACKED_OUTPUT="${OUTPUT_DIR}/datasculpt_original_${MAX_TOKENS}.jsonl"
SUMMARY_OUTPUT="${OUTPUT_DIR}/datasculpt_original_summary.csv"

python scripts/baselines/export_datasculpt_input.py \
  --documents "${DOCUMENTS}" \
  --output-folder "${INPUT_DIR}" \
  --shard-size "${SHARD_SIZE}" \
  --tokenizer "${TOKENIZER}"

PYTHON_BIN="${PYTHON_BIN:-python}" \
  bash "${DATASCULPT_ROOT}/src/run_datasculpt_pipeline.sh" \
  "${MAX_TOKENS}" \
  "${DELTA}" \
  "${EPSILON}" \
  "${ITER_T}" \
  "${INPUT_DIR}" \
  "${WORK_DIR}"

python scripts/baselines/import_datasculpt_output.py \
  --datasculpt-output-folder "${WORK_DIR}/output/data_sculpt" \
  --documents "${DOCUMENTS}" \
  --output "${PACKED_OUTPUT}" \
  --max-tokens "${MAX_TOKENS}" \
  --tokenizer "${TOKENIZER}" \
  --edges "${EDGES}" \
  --summary "${SUMMARY_OUTPUT}"

echo "DataSculpt original baseline complete:"
echo "  packed:  ${PACKED_OUTPUT}"
echo "  summary: ${SUMMARY_OUTPUT}"
