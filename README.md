# Dependency-Aware Packing for Low-Resource Long-Context Adaptation

This project is a master's-thesis-oriented research scaffold for studying
dependency-aware data packing in low-resource long-context adaptation.

The first target scenario is code repositories. Instead of packing documents
only by length, semantic similarity, or retrieval relevance, the project builds
long-context training samples that preserve learnable cross-file dependencies:
imports, source-test relations, README-to-code relations, config-to-script
relations, and same-module structure.

## Research Scope

Recommended thesis title:

> Structure-Dependency-Aware Data Packing for Low-Resource Long-Context
> Adaptation in Code Repository Scenarios

The initial version focuses on:

- code repository documents;
- 4K/8K context windows;
- structure-based dependency scores;
- packing statistics before expensive model training;
- lightweight baselines for controlled comparison.

## Project Layout

```text
configs/                     Experiment and packing configs
data/
  raw/                       Raw datasets, ignored by git
  processed/                 Generated jsonl files, ignored by git
  examples/                  Tiny example data for smoke tests
docs/                        Thesis notes and experiment records
experiments/notebooks/       Analysis notebooks
outputs/                     Generated outputs, ignored by git
scripts/                     CLI entry points
src/dapacking/               Python package
tests/                       Basic tests
dependency_aware_packing_experiment_plan.md
```

## Data Format

Input documents are JSONL records:

```json
{"docid": "repo_a:README.md", "content": "...", "metadata": {"repo": "repo_a", "path": "README.md", "language": "markdown"}}
```

Packed samples are JSONL records:

```json
{
  "sample_id": "dependency_aware_000001",
  "method": "dependency_aware",
  "docids": ["repo_a:README.md", "repo_a:src/model.py"],
  "content": "...",
  "stats": {
    "tokens": 4096,
    "num_docs": 2,
    "dependency_score": 0.8,
    "token_utilization": 1.0,
    "truncation_rate": 0.0
  }
}
```

## Quick Start

Install dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run a smoke packing job:

```bash
python scripts/run_packing.py \
  --input data/examples/code_docs.jsonl \
  --output outputs/example_dependency_aware.jsonl \
  --method dependency_aware \
  --max-tokens 512
```

Compare methods:

```bash
python scripts/run_packing.py --input data/examples/code_docs.jsonl --output outputs/random.jsonl --method random --max-tokens 512
python scripts/run_packing.py --input data/examples/code_docs.jsonl --output outputs/length_aware.jsonl --method length_aware --max-tokens 512
python scripts/run_packing.py --input data/examples/code_docs.jsonl --output outputs/same_repo.jsonl --method same_repo --max-tokens 512
python scripts/run_packing.py --input data/examples/code_docs.jsonl --output outputs/dependency.jsonl --method dependency_aware --max-tokens 512
```

## Initial Baselines

- `random`: randomly shuffles documents and fills context windows.
- `length_aware`: first-fit decreasing by token length.
- `same_repo`: packs documents from the same repository.
- `dependency_aware`: greedily maximizes structural dependency edges while
  controlling token utilization and truncation.

BM25 and semantic DataSculpt-lite baselines are planned as the next milestone.

## Thesis Milestones

1. Build a stable packing pipeline and statistics dashboard.
2. Add BM25 and semantic baselines.
3. Validate structural dependency edges with sampled loss-reduction analysis.
4. Run pilot QLoRA adaptation on 1.5B/3B models at 4K context.
5. Expand to 8K and code long-context benchmarks if pilot results are stable.
