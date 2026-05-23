#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-}
TOKENIZER=${TOKENIZER:-${MODEL}}
TRAIN_FILE=${TRAIN_FILE:-}
VALIDATION_FILE=${VALIDATION_FILE:-}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/training/qlora_smoke}
RUN_NAME=${RUN_NAME:-dapacking_7b_8k_qlora_smoke}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-8192}
MAX_STEPS=${MAX_STEPS:-100}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-512}
MAX_VALIDATION_SAMPLES=${MAX_VALIDATION_SAMPLES:-128}
LEARNING_RATE=${LEARNING_RATE:-0.0002}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-16}
LOGGING_STEPS=${LOGGING_STEPS:-10}
EVAL_STEPS=${EVAL_STEPS:-50}
SAVE_STEPS=${SAVE_STEPS:-100}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
TARGET_MODULES=${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
REPORT_TO=${REPORT_TO:-none}

if [[ -z "${MODEL}" ]]; then
  echo "MODEL is required, for example MODEL=Qwen/Qwen2.5-Coder-7B" >&2
  exit 2
fi

if [[ -z "${TRAIN_FILE}" ]]; then
  echo "TRAIN_FILE is required, for example TRAIN_FILE=data/processed/python50_qwen/packed/train_8192/dependency_aware_v2_strong_first_8192.jsonl" >&2
  exit 2
fi

VALIDATION_ARGS=()
if [[ -n "${VALIDATION_FILE}" ]]; then
  VALIDATION_ARGS=(--validation-file "${VALIDATION_FILE}")
fi

LOCAL_FILES_ARGS=()
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  LOCAL_FILES_ARGS=(--local-files-only)
fi

TRUST_REMOTE_ARGS=()
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  TRUST_REMOTE_ARGS=(--trust-remote-code)
fi

BF16_ARGS=()
if [[ "${BF16:-1}" == "1" ]]; then
  BF16_ARGS=(--bf16)
fi

LOAD_4BIT_ARGS=()
if [[ "${LOAD_IN_4BIT:-1}" == "1" ]]; then
  LOAD_4BIT_ARGS=(--load-in-4bit)
fi

GRADIENT_CHECKPOINTING_ARGS=()
if [[ "${GRADIENT_CHECKPOINTING:-1}" == "1" ]]; then
  GRADIENT_CHECKPOINTING_ARGS=(--gradient-checkpointing)
fi

torchrun --nproc_per_node="${NPROC_PER_NODE}" scripts/training/train_causal_lm_qlora.py \
  --model "${MODEL}" \
  --tokenizer "${TOKENIZER}" \
  --train-file "${TRAIN_FILE}" \
  "${VALIDATION_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --max-seq-length "${MAX_SEQ_LENGTH}" \
  --max-steps "${MAX_STEPS}" \
  --max-train-samples "${MAX_TRAIN_SAMPLES}" \
  --max-validation-samples "${MAX_VALIDATION_SAMPLES}" \
  --learning-rate "${LEARNING_RATE}" \
  --per-device-train-batch-size "${MICRO_BATCH_SIZE}" \
  --per-device-eval-batch-size "${EVAL_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --logging-steps "${LOGGING_STEPS}" \
  --eval-steps "${EVAL_STEPS}" \
  --save-steps "${SAVE_STEPS}" \
  --lora-r "${LORA_R}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout "${LORA_DROPOUT}" \
  --target-modules "${TARGET_MODULES}" \
  --report-to "${REPORT_TO}" \
  "${BF16_ARGS[@]}" \
  "${LOAD_4BIT_ARGS[@]}" \
  "${GRADIENT_CHECKPOINTING_ARGS[@]}" \
  "${LOCAL_FILES_ARGS[@]}" \
  "${TRUST_REMOTE_ARGS[@]}"
