# Data Directory

This directory is a local workspace for raw and generated data. Large data is
normally not committed to git.

## Layout

```text
examples/       Tiny JSONL fixtures used for smoke tests and examples.
raw/            Cloned repositories, local manifests, downloaded inputs.
processed/      Generated documents, dependency edges, splits, packed windows,
                reviews, dataset cards, and validation records.
```

## Current Main Dataset

The current local main dataset is:

```text
data/processed/repo_main_v1/
```

Expected contents:

```text
documents.jsonl
dependency_edges.jsonl
splits/
DATASET_CARD.md
review/
packed/
```

High-precision edge files:

```text
splits/train_edges_high_precision.jsonl
splits/validation_edges_high_precision.jsonl
```

These can be regenerated with:

```bash
bash scripts/server/run_high_precision_freeze.sh \
  data/processed/repo_main_v1 \
  8192 \
  Qwen/Qwen2.5-Coder-7B
```

## Sharing Rule

When sharing only the GitHub repository, assume this directory is incomplete.
Server runners should rebuild data from manifests or receive a separate data
archive.
