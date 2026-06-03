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

For a standalone technical-doc/web generalization set:

```bash
FETCH_URLS=1 \
  bash scripts/data/build_external_dataset_pipeline.sh \
  configs/datasets/tech_web_seed_manifest.tsv \
  data/processed/tech_web_generalization_v1 \
  tech-web-generalization-v1
```

The external pipeline defaults to `GROUP_KEY=document_id`, so sections from the
same web page or document do not leak across splits.

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

Optionally create assistant-assisted suggestions first:

```bash
python scripts/data/assistant_annotate_edge_review.py \
  --input data/processed/repo_main_v1/review/train_edges_balanced.csv \
  --documents data/processed/repo_main_v1/splits/train_docs.jsonl \
  --output data/processed/repo_main_v1/review/train_edges_balanced_assistant.csv
```

These are suggestions, not independent human labels. Use them to speed up the
manual audit, then confirm labels before treating precision as paper-quality.

Build relation reliability from the confirmed manual labels. This command
intentionally ignores assistant labels:

```bash
python scripts/data/build_relation_reliability.py \
  --input data/processed/repo_main_v1/review/train_edges_balanced_annotated.csv \
  --output data/processed/repo_main_v1/review/relation_reliability.yaml
```

Freeze the high-precision edge files used by the main dependency-aware setting:

```bash
python scripts/data/filter_dependency_edges.py \
  --input data/processed/repo_main_v1/splits/train_edges.jsonl \
  --relation-config configs/relations/main_high_precision.yaml \
  --output data/processed/repo_main_v1/splits/train_edges_high_precision.jsonl

python scripts/data/filter_dependency_edges.py \
  --input data/processed/repo_main_v1/splits/validation_edges.jsonl \
  --relation-config configs/relations/main_high_precision.yaml \
  --output data/processed/repo_main_v1/splits/validation_edges_high_precision.jsonl
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

Before the target-tokenizer run, use a fast local pass to verify the
high-precision method and order ablations:

```bash
python scripts/run_packing_matrix.py \
  --input data/processed/repo_main_v1/splits/validation_docs.jsonl \
  --edges data/processed/repo_main_v1/splits/validation_edges_high_precision.jsonl \
  --relation-config configs/relations/main_high_precision.yaml \
  --output-dir data/processed/repo_main_v1/packed/validation_8192_high_precision \
  --methods dependency_aware_high_precision_only,dependency_aware_high_precision_random_order,dependency_aware_high_precision_reverse_order,dependency_aware_v2_strong_first,datasculpt_lite,bm25 \
  --max-tokens 8192 \
  --tokenizer simple \
  --summary data/processed/repo_main_v1/packed/validation_8192_high_precision/summary.csv
```

To run the original DataSculpt pipeline as an external baseline:

```bash
bash scripts/baselines/run_datasculpt_original.sh \
  data/processed/repo_main_v1/splits/train_docs.jsonl \
  data/processed/repo_main_v1/splits/train_edges.jsonl \
  data/processed/repo_main_v1/packed/train_8192_datasculpt_original \
  /Users/frank/Documents/课题组/datasculpt/DataSculpt \
  8192
```

This baseline requires the DataSculpt environment, embedding model, Ray, and GPU
resources. It is stronger than `datasculpt_lite`, but also more expensive and
should be reported as an external pipeline baseline.

Check fairness gates:

```bash
python scripts/analysis/check_packing_fairness.py \
  --summary data/processed/repo_main_v1/packed/train_8192/summary.csv \
  --method dependency_aware_high_precision_only \
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
high-precision dependency-aware utilization is within 0.05 of strong baselines
high-precision dependency-aware weighted strong-edge coverage is higher than BM25/DataSculpt-lite
same-docs random/reverse order ablations are generated for order-sensitivity checks
```

Proceed to formal training only after the 8-GPU smoke run succeeds.
