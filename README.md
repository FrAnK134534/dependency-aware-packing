# Dependency-Aware Packing for Long-Context Adaptation

This project is a research scaffold for studying dependency-aware data packing
in long-context adaptation.

The main question is:

> Under a fixed model, context length, and training token budget, can better
> packing strategies help a language model learn to use long-context
> dependencies more effectively?

The first paper-quality target scenario is multi-source code repository
context. Instead of packing documents only by length, same-repository
membership, lexical retrieval, or semantic similarity, the project builds
long-context training samples that preserve learnable dependencies:
README-to-code, docs-to-implementation, config-to-script, source-to-test,
issue-to-file, commit-to-file, and API-doc-to-usage relations.

## Research Scope

Working thesis title:

> Dependency-Aware Data Packing for Efficient Long-Context Adaptation

The current version focuses on:

- multi-source code repository context;
- 8K main experiments, with 16K or 13B extensions when stable;
- dependency-aware packing scores;
- packing statistics before expensive model training;
- controlled 8-GPU NVLink training comparisons.

## Project Layout

```text
AGENTS.md                    Agent and experiment-runner project guide
configs/                     Experiment and packing configs
  packing/                   Packing configs
  training/                  7B/8K LoRA and QLoRA templates
  evaluation/                Evaluation suite templates
data/
  raw/                       Raw datasets, ignored by git
  processed/                 Generated jsonl files, ignored by git
  examples/                  Tiny example data for smoke tests
docs/
  00_overview/               Advisor report, design rationale, thesis scope
  01_design/                 Macro experiment design
  02_metrics/                Metric definitions
  03_server/                 8-GPU NVLink deployment plan
  archive/                   Older plans kept for reference
experiments/                 Run manifests, logs, and notebooks
outputs/                     Generated outputs, ignored by git
scripts/                     CLI entry points
  server/                    Future server launch scripts
src/dapacking/               Python package
tests/                       Basic tests
```

Start reading from [docs/README.md](docs/README.md).

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
python scripts/run_packing.py --input data/examples/code_docs.jsonl --output outputs/bm25.jsonl --method bm25 --max-tokens 512
python scripts/run_packing.py --input data/examples/code_docs.jsonl --output outputs/dependency.jsonl --method dependency_aware --max-tokens 512
```

Summarize generated packing files:

```bash
python scripts/summarize_packing.py \
  outputs/random.jsonl \
  outputs/length_aware.jsonl \
  outputs/same_repo.jsonl \
  outputs/bm25.jsonl \
  outputs/dependency.jsonl \
  --output outputs/packing_summary.csv
```

Build a local repository corpus:

```bash
python scripts/data/build_repo_corpus.py \
  --manifest data/raw/repo_manifest.txt \
  --output data/processed/documents.jsonl

python scripts/data/build_dependency_edges.py \
  --input data/processed/documents.jsonl \
  --output data/processed/dependency_edges.jsonl

python scripts/data/split_by_repo.py \
  --documents data/processed/documents.jsonl \
  --edges data/processed/dependency_edges.jsonl \
  --output-dir data/processed/splits
```

Generate a packing matrix for one split:

```bash
python scripts/run_packing_matrix.py \
  --input data/processed/splits/train_docs.jsonl \
  --output-dir data/processed/packed/train_8192 \
  --max-tokens 8192 \
  --edges data/processed/splits/train_edges.jsonl \
  --summary data/processed/packed/train_8192/summary.csv
```

## Current Baselines

- `random`: randomly shuffles documents and fills context windows.
- `length_aware`: first-fit decreasing by token length.
- `same_repo`: packs documents from the same repository.
- `bm25`: uses lexical retrieval from an anchor document to fill each window.
- `semantic`: uses a dependency-free TF-IDF cosine index as an early semantic
  similarity baseline.
- `datasculpt_lite`: uses TF-IDF similarity plus lightweight token-fit,
  repository-integrity, and redundancy-penalty terms.
- `dependency_aware`: greedily maximizes structural dependency edges while
  controlling token utilization and truncation.

## Research Milestones

1. Build a stable packing pipeline and statistics dashboard.
2. Add structure-reranking variants on top of BM25 and semantic retrieval.
3. Build multi-source repository data preprocessing.
4. Run packing-only quality analysis.
5. Run 7B + 8K LoRA/QLoRA experiments on the 8-GPU NVLink server.
6. Evaluate with RepoBench, cross-file completion, context gain, passkey,
   needle, and selected LongBench-style tasks.
