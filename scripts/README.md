# Scripts

Scripts are workflow entry points. They should orchestrate reusable code in
`src/dapacking/` and keep command-line behavior stable.

## Top-Level CLIs

```text
run_packing.py          Run one packing method.
run_packing_matrix.py   Run several packing methods and write summary.csv.
summarize_packing.py    Summarize existing packed JSONL files.
```

## Workflow Directories

```text
data/                   Corpus, dependency edge, split, review, and dataset
                        card builders.
analysis/               Packing fairness and cap-sensitivity summaries.
baselines/              External baseline adapters, especially DataSculpt.
evaluation/             Dependency validation and context-gain scoring.
server/                 Server-facing pipeline and QLoRA launch scripts.
training/               Model training entry points.
```

## Current Main Server Path

```bash
MAX_DOCS_PER_REPO=300 \
bash scripts/server/build_dataset_pipeline.sh \
  data/raw/python50_repos_local_manifest.txt \
  data/processed/repo_main_v1

PACK_SPLITS="train validation" \
bash scripts/server/run_high_precision_freeze.sh \
  data/processed/repo_main_v1 \
  8192 \
  Qwen/Qwen2.5-Coder-7B

MODEL=Qwen/Qwen2.5-Coder-7B \
TRAIN_FILE=data/processed/repo_main_v1/packed/train_8192_high_precision/dependency_aware_high_precision_only_8192.jsonl \
VALIDATION_FILE=data/processed/repo_main_v1/packed/validation_8192_high_precision/dependency_aware_high_precision_only_8192.jsonl \
OUTPUT_DIR=outputs/training/qwen7b_8k_high_precision_smoke \
MAX_STEPS=100 \
bash scripts/server/run_7b_8k_qlora.sh
```

## Boundary

If a script grows complex, move the reusable logic into `src/dapacking/` and
leave the script as a thin CLI wrapper.
