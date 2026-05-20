#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/training/7b_8k_qlora.yaml}

echo "This is a launch template for the 8-GPU NVLink server."
echo "Config: ${CONFIG}"
echo
echo "After the server training stack is selected, replace this template with one of:"
echo "  accelerate launch ..."
echo "  torchrun --nproc_per_node=8 ..."
echo "  deepspeed ..."
echo
echo "Required invariant: only the packed training data path should change across packing baselines."
echo "Keep model, tokenizer, context length, token budget, LoRA/QLoRA config, optimizer,"
echo "effective batch size, validation set, and evaluation protocol fixed."
