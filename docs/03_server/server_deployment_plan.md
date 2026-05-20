# 8-GPU NVLink Server Deployment Plan

This document describes how the project should be deployed on a single-node
8-GPU NVLink server.

## 1. Server Role

The server is used to make experiments systematic, not to redefine the project
contribution.

The contribution remains:

```text
dependency-aware packing for long-context adaptation
```

The server enables:

```text
7B + 8K main experiments
selected 7B + 16K or 13B + 8K extensions
multiple packing baselines
ablation experiments
more stable throughput and memory behavior
```

## 2. Recommended Software Stack

Recommended base environment:

```text
Python 3.10+
PyTorch with CUDA support
transformers
datasets
accelerate
peft
bitsandbytes, if QLoRA is used
flash-attn, if supported by the server CUDA stack
deepspeed or FSDP
wandb or tensorboard for logging
```

The exact CUDA, driver, and PyTorch versions should be recorded in every
experiment log.

## 3. Deployment Steps

Clone the repository:

```bash
git clone https://github.com/FrAnK134534/dependency-aware-packing.git
cd dependency-aware-packing
```

Create an environment:

```bash
conda create -n dapacking python=3.10 -y
conda activate dapacking
python -m pip install -e ".[dev]"
```

Install training dependencies after confirming the server CUDA version:

```bash
python -m pip install torch transformers datasets accelerate peft deepspeed
```

Install optional dependencies only when compatible:

```bash
python -m pip install bitsandbytes flash-attn
```

Run local smoke tests:

```bash
PYTHONPATH=src python -m pytest -q
```

Run packing smoke test:

```bash
python scripts/run_packing.py \
  --input data/examples/code_docs.jsonl \
  --output outputs/dependency_smoke.jsonl \
  --method dependency_aware \
  --max-tokens 512
```

## 4. Experiment Stages

### Stage 1: Packing-Only Validation

Goal:

```text
Verify that all packing methods generate valid outputs and measurable
statistics before any expensive model training.
```

Methods:

```text
random
length_aware
same_repo
bm25
semantic
datasculpt_lite
dependency_aware
dependency_aware_v2_token_fit
dependency_aware_v2_strong_first
dependency_aware_no_same_directory
dependency_aware_no_same_repo
dependency_aware_strong_edges_only
```

Output:

```text
packed JSONL files
packing summary CSV
dependency edge coverage table
```

For final pretraining data generation, use the target model tokenizer rather
than the lightweight local tokenizer:

```bash
python scripts/run_packing_matrix.py \
  --input data/processed/splits/train_docs.jsonl \
  --output-dir data/processed/packed/train_8192_qwen_tokens \
  --max-tokens 8192 \
  --tokenizer Qwen/Qwen2.5-Coder-7B \
  --edges data/processed/splits/train_edges.jsonl \
  --summary data/processed/packed/train_8192_qwen_tokens/summary.csv
```

If the server cannot access HuggingFace, download or cache the tokenizer first
and add `--tokenizer-local-files-only`.

### Stage 2: 7B + 8K Smoke Training

Goal:

```text
Ensure the training stack works on 8 GPUs.
```

Run only:

```text
random
dependency_aware
```

Use a small token budget first, for example:

```text
5M to 10M tokens
```

### Stage 3: 7B + 8K Main Baseline Comparison

Goal:

```text
Compare all packing baselines under a fixed token budget.
```

Suggested budget:

```text
50M / 100M / 200M tokens, depending on server availability
```

Variables that must stay fixed:

```text
base model
context length
training tokens
optimizer
learning rate
LoRA/QLoRA config
effective batch size
validation set
evaluation protocol
```

### Stage 4: Extension Runs

Only after Stage 3 is stable:

```text
7B + 16K
13B + 8K
BM25 + structure reranking
Semantic + structure reranking
dependency ablations
```

## 5. Training Configuration Principles

Prefer LoRA/QLoRA first:

```text
The project requires many controlled runs.
Full fine-tuning is too expensive for the first experimental loop.
```

Use distributed training:

```text
torchrun / accelerate / deepspeed / FSDP
```

Record:

```text
number of GPUs
GPU model
NVLink availability
CUDA version
PyTorch version
training precision
micro batch size
gradient accumulation
effective batch size
tokens/sec
peak memory
training hours
```

## 6. Run Naming Convention

Use names that encode the important variables:

```text
{model}_{ctx}_{budget}_{packing}_{date}
```

Example:

```text
qwen2.5-coder-7b_8k_100m_dependency_2026-05-20
```

## 7. Required Artifacts per Run

Each run should save:

```text
config yaml
git commit hash
packing input path
training data manifest
validation data manifest
training log
loss curve
evaluation results
GPU memory and throughput summary
checkpoint or adapter path
```

Large outputs should not be committed to git. Save them under server storage and
commit only small manifests, configs, and summarized results.

## 8. Failure Handling

If 7B + 8K OOMs:

```text
reduce micro batch size
increase gradient accumulation
enable gradient checkpointing
use QLoRA instead of LoRA
verify FlashAttention
reduce max sequence length for smoke testing
```

If throughput is too low:

```text
check dataloader bottlenecks
pre-tokenize data
verify NVLink visibility
avoid excessive CPU-side packing during training
cache packed datasets
```

If metrics do not improve:

```text
inspect packing statistics
compare same-repo vs dependency-aware
measure context gain
run dependency type ablations
check whether evaluation actually requires cross-document dependency
```
