# Pre-Training Freeze Protocol

Use this protocol before launching formal 8-GPU LoRA/QLoRA runs.

## 1. Freeze Dataset

Build the main repository dataset:

```bash
MAX_DOCS_PER_REPO=300 \
  bash scripts/server/build_dataset_pipeline.sh \
  data/raw/python50_local_manifest.txt \
  data/processed/repo_main_v1
```

For mixed generalization data, first build external documents and then merge:

```bash
python scripts/data/build_external_corpus.py \
  --manifest configs/datasets/external_manifest.example.tsv \
  --output data/processed/mixed_generalization_v1/external_documents.jsonl

python scripts/data/build_mixed_corpus.py \
  --input data/processed/repo_main_v1/documents.jsonl \
  --input data/processed/mixed_generalization_v1/external_documents.jsonl \
  --output data/processed/mixed_generalization_v1/documents.jsonl
```

Write a dataset card:

```bash
python scripts/data/write_dataset_card.py \
  --documents data/processed/repo_main_v1/documents.jsonl \
  --edges data/processed/repo_main_v1/dependency_edges.jsonl \
  --split-dir data/processed/repo_main_v1/splits \
  --name repo-main-v1 \
  --source-manifest data/raw/python50_local_manifest.txt \
  --output data/processed/repo_main_v1/DATASET_CARD.md
```

## 2. Audit Dependency Edges

Sample balanced strong edges:

```bash
python scripts/data/sample_dependency_edges.py \
  --documents data/processed/repo_main_v1/splits/train_docs.jsonl \
  --edges data/processed/repo_main_v1/splits/train_edges.jsonl \
  --output data/processed/repo_main_v1/review/train_edges_balanced.csv \
  --strong-only \
  --per-relation 30
```

After manual annotation, summarize quality:

```bash
python scripts/data/summarize_edge_review.py \
  --input data/processed/repo_main_v1/review/train_edges_balanced_annotated.csv \
  --output-csv data/processed/repo_main_v1/review/train_edge_review_summary.csv \
  --output-md data/processed/repo_main_v1/review/train_edge_review_report.md
```

## 3. Build Context-Gain Validation

Build positive reviewed dependency records:

```bash
python scripts/evaluation/build_dependency_validation.py \
  --documents data/processed/repo_main_v1/splits/validation_docs.jsonl \
  --edges data/processed/repo_main_v1/splits/validation_edges.jsonl \
  --review-annotations data/processed/repo_main_v1/review/validation_edges_annotated.csv \
  --allowed-review-labels yes,partial \
  --min-review-confidence 0.6 \
  --output data/processed/repo_main_v1/dependency_validation.jsonl
```

Build anti-bias controls:

```bash
python scripts/evaluation/build_context_gain_controls.py \
  --documents data/processed/repo_main_v1/splits/validation_docs.jsonl \
  --edges data/processed/repo_main_v1/splits/validation_edges.jsonl \
  --output data/processed/repo_main_v1/context_gain_controls.jsonl
```

## 4. Run Packing Matrix

Use the target tokenizer before training:

```bash
bash scripts/server/run_packing_only_experiment.sh \
  configs/datasets/python50_repos.tsv \
  data/processed/repo_main_v1 \
  8192 \
  Qwen/Qwen2.5-Coder-7B
```

Check fairness gates:

```bash
python scripts/analysis/check_packing_fairness.py \
  --summary data/processed/repo_main_v1/packed/train_8192/summary.csv \
  --method dependency_aware_v2_strong_first \
  --output data/processed/repo_main_v1/packed/train_8192/fairness_check.csv
```

## 5. Go / No-Go Criteria

Proceed to 8-GPU smoke run only if:

```text
unit tests pass
dataset card exists
edge review report exists
dependency validation exists
context-gain controls exist
target-tokenizer packing summary exists
dependency-aware utilization is within 0.05 of strong baselines
dependency-aware weighted strong-edge coverage is higher than BM25/DataSculpt-lite
```

Proceed to formal training only after the 8-GPU smoke run succeeds.
