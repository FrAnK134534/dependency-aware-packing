# Server Scripts

This folder is reserved for scripts that run on the 8-GPU NVLink server.

The current repository supports packing generation, edge review, validation-set
construction, and an initial QLoRA smoke launcher. Confirm the server CUDA stack
before running model training.

Recommended future files:

```text
prepare_env.sh
run_7b_8k_lora.sh
run_eval_suite.sh
collect_run_summary.py
```

Current starter scripts:

```text
build_dataset_pipeline.sh
run_cap_sensitivity.sh
run_packing_only_experiment.sh
run_high_precision_freeze.sh
run_packing_matrix.sh
run_7b_8k_qlora.sh
```

Run the 50-repository packing-only experiment with the lightweight tokenizer:

```bash
MAX_DOCS_PER_REPO=300 \
bash scripts/server/run_packing_only_experiment.sh \
  configs/datasets/python50_repos.tsv \
  data/processed/python50 \
  8192 \
  simple
```

Before model training, rerun packing with the target model tokenizer:

```bash
PACK_SPLITS="train validation" MAX_DOCS_PER_REPO=300 \
bash scripts/server/run_packing_only_experiment.sh \
  configs/datasets/python50_repos.tsv \
  data/processed/python50_qwen \
  8192 \
  Qwen/Qwen2.5-Coder-7B \
  random,bm25,datasculpt_lite,dependency_aware_v2_strong_first
```

For the current main thesis setting, regenerate high-precision dependency edges
and target-tokenizer packing with:

```bash
PACK_SPLITS="train validation" bash scripts/server/run_high_precision_freeze.sh \
  data/processed/repo_main_v1 \
  8192 \
  Qwen/Qwen2.5-Coder-7B
```

Use `LOCAL_FILES_ONLY=1` if the model/tokenizer is already cached and the
server has no outbound network access.

Run cap sensitivity:

```bash
CAP_VALUES="100 300 500" bash scripts/server/run_cap_sensitivity.sh \
  configs/datasets/python50_repos.tsv \
  data/processed/cap_sensitivity \
  8192 \
  simple \
  random,bm25,datasculpt_lite,dependency_aware_v2_strong_first
```

Run a small 8-GPU QLoRA smoke test after packed data is ready:

```bash
MODEL=Qwen/Qwen2.5-Coder-7B \
TRAIN_FILE=data/processed/python50_qwen/packed/train_8192/dependency_aware_v2_strong_first_8192.jsonl \
VALIDATION_FILE=data/processed/python50_qwen/packed/validation_8192/dependency_aware_v2_strong_first_8192.jsonl \
OUTPUT_DIR=outputs/training/qwen7b_depaware_smoke \
MAX_STEPS=100 \
bash scripts/server/run_7b_8k_qlora.sh
```

For the high-precision main method, use files under
`data/processed/repo_main_v1/packed/*_8192_high_precision/`.

Build and score dependency-sensitive validation data after training:

```bash
python scripts/evaluation/build_dependency_validation.py \
  --documents data/processed/python50_qwen/splits/validation_docs.jsonl \
  --edges data/processed/python50_qwen/splits/validation_edges.jsonl \
  --output data/processed/python50_qwen/eval/dependency_validation.jsonl \
  --tokenizer Qwen/Qwen2.5-Coder-7B

python scripts/evaluation/score_dependency_validation.py \
  --model Qwen/Qwen2.5-Coder-7B \
  --adapter outputs/training/qwen7b_depaware_smoke/final_adapter \
  --input data/processed/python50_qwen/eval/dependency_validation.jsonl \
  --output outputs/evaluation/qwen7b_depaware_context_gain.jsonl \
  --bf16
```

Every server run should save:

```text
git commit hash
config file
dataset manifest
packing method
model path
training log
evaluation results
GPU memory and throughput summary
```
