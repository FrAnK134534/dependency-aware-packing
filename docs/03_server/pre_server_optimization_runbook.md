# Pre-Server Optimization Runbook

Use this checklist before spending 8-GPU time. The goal is to make sure the
dataset, packing, and smoke-training inputs are already inspectable.

## 1. Training-Grade Tokenizer Packing

Generate train and validation packed data with the target model tokenizer:

```bash
PACK_SPLITS="train validation" MAX_DOCS_PER_REPO=300 \
bash scripts/server/run_packing_only_experiment.sh \
  configs/datasets/python50_repos.tsv \
  data/processed/python50_qwen \
  8192 \
  Qwen/Qwen2.5-Coder-7B \
  random,bm25,datasculpt_lite,dependency_aware_v2_strong_first
```

Check:

```text
data/processed/python50_qwen/packed/train_8192/summary.csv
data/processed/python50_qwen/packed/validation_8192/summary.csv
```

## 2. Manual Edge Review

Sample dependency edges for human inspection:

```bash
python scripts/data/sample_dependency_edges.py \
  --documents data/processed/python50_qwen/splits/train_docs.jsonl \
  --edges data/processed/python50_qwen/splits/train_edges.jsonl \
  --output data/processed/python50_qwen/review/train_edge_review.csv \
  --sample-size 100
```

Review the `manual_reasonable` and `manual_note` columns. Strong relations
such as import, source-test, docs-code, README-code, config-script, and
example-code should be judged separately from weak same-directory edges.

## 3. Cap Sensitivity

Run:

```bash
CAP_VALUES="100 300 500" bash scripts/server/run_cap_sensitivity.sh \
  configs/datasets/python50_repos.tsv \
  data/processed/cap_sensitivity \
  8192 \
  simple \
  random,bm25,datasculpt_lite,dependency_aware_v2_strong_first
```

Main output:

```text
data/processed/cap_sensitivity/cap_sensitivity_summary.csv
```

The main method should keep its advantage in weighted edge coverage and strong
order dependency across caps.

## 4. Dependency-Sensitive Validation Set

Build paired records for later context-gain evaluation:

```bash
python scripts/evaluation/build_dependency_validation.py \
  --documents data/processed/python50_qwen/splits/validation_docs.jsonl \
  --edges data/processed/python50_qwen/splits/validation_edges.jsonl \
  --output data/processed/python50_qwen/eval/dependency_validation.jsonl \
  --tokenizer Qwen/Qwen2.5-Coder-7B \
  --max-examples-per-relation 200
```

Each record contains:

```text
context_without_dependency
context_with_dependency
target_text
```

Later evaluation computes:

```text
context_gain = loss(target_text | context_without_dependency)
             - loss(target_text | context_with_dependency)
```

After a model or adapter is trained, score context gain:

```bash
python scripts/evaluation/score_dependency_validation.py \
  --model Qwen/Qwen2.5-Coder-7B \
  --adapter outputs/training/qwen7b_depaware_smoke/final_adapter \
  --input data/processed/python50_qwen/eval/dependency_validation.jsonl \
  --output outputs/evaluation/qwen7b_depaware_context_gain.jsonl \
  --bf16
```

## 5. 8-GPU QLoRA Smoke Run

After server dependencies are installed, run a short smoke job:

```bash
MODEL=Qwen/Qwen2.5-Coder-7B \
TRAIN_FILE=data/processed/python50_qwen/packed/train_8192/dependency_aware_v2_strong_first_8192.jsonl \
VALIDATION_FILE=data/processed/python50_qwen/packed/validation_8192/dependency_aware_v2_strong_first_8192.jsonl \
OUTPUT_DIR=outputs/training/qwen7b_depaware_smoke \
MAX_STEPS=100 \
bash scripts/server/run_7b_8k_qlora.sh
```

Smoke success criteria:

```text
8 processes launch
no immediate CUDA out-of-memory
training loss is logged
validation loss is logged if VALIDATION_FILE is provided
adapter checkpoint is saved under OUTPUT_DIR
```
