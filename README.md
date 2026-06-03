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

For a more detailed responsibility map, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

```text
AGENTS.md                    Agent and experiment-runner project guide
PROJECT_STRUCTURE.md         Directory responsibilities and ownership boundaries
configs/                     Experiment and packing configs
  datasets/                  Repo lists and external manifests
  packing/                   Packing configs
  relations/                 Dependency relation allowlists and reliability
  training/                  7B/8K LoRA and QLoRA templates
  evaluation/                Evaluation suite templates
data/
  raw/                       Raw datasets, ignored by git
  processed/                 Generated jsonl files, ignored by git
  examples/                  Tiny example data for smoke tests
docs/
  00_overview/               Project report, thesis scope, evaluation dossier
  01_design/                 Macro experiment design
  02_metrics/                Metric definitions
  03_server/                 8-GPU NVLink deployment plan
  05_training/               Training controls and checklists
  archive/                   Older plans kept for reference
experiments/                 Run manifests, logs, and notebooks
outputs/                     Generated outputs, ignored by git
scripts/                     CLI entry points
  data/                      Corpus, edge, split, and review builders
  evaluation/                Context-gain validation builders/scorers
  server/                    Server launch scripts
  training/                  QLoRA training entry point
src/dapacking/               Python package
tests/                       Basic tests
```

Start reading from [docs/README.md](docs/README.md) or
[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

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
python scripts/data/clone_repo_manifest.py \
  --input configs/datasets/python50_repos.tsv \
  --repo-dir data/raw/repos \
  --output-manifest data/raw/python50_repos_local_manifest.txt

python scripts/data/build_repo_corpus.py \
  --manifest data/raw/python50_repos_local_manifest.txt \
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
  --tokenizer simple \
  --edges data/processed/splits/train_edges.jsonl \
  --summary data/processed/packed/train_8192/summary.csv
```

Or run the full 50-repository packing-only pipeline:

```bash
bash scripts/server/run_packing_only_experiment.sh \
  configs/datasets/python50_repos.tsv \
  data/processed/python50 \
  8192 \
  simple
```

For large repository sets, `run_packing_only_experiment.sh` defaults to
`MAX_DOCS_PER_REPO=300` so a single large repository does not dominate the
pilot. Override it explicitly if needed:

```bash
MAX_DOCS_PER_REPO=500 bash scripts/server/run_packing_only_experiment.sh
```

Generate both train and validation packed splits before smoke training:

```bash
PACK_SPLITS="train validation" MAX_DOCS_PER_REPO=300 \
bash scripts/server/run_packing_only_experiment.sh \
  configs/datasets/python50_repos.tsv \
  data/processed/python50_qwen \
  8192 \
  Qwen/Qwen2.5-Coder-7B \
  random,bm25,datasculpt_lite,dependency_aware_v2_strong_first
```

For the current high-precision main setting, regenerate filtered edges and
target-tokenizer packing with:

```bash
PACK_SPLITS="train validation" \
bash scripts/server/run_high_precision_freeze.sh \
  data/processed/repo_main_v1 \
  8192 \
  Qwen/Qwen2.5-Coder-7B
```

Sample dependency edges for manual review:

```bash
python scripts/data/sample_dependency_edges.py \
  --documents data/processed/python50_qwen/splits/train_docs.jsonl \
  --edges data/processed/python50_qwen/splits/train_edges.jsonl \
  --output data/processed/python50_qwen/review/train_edge_review.csv \
  --sample-size 100
```

Build dependency-sensitive validation examples:

```bash
python scripts/evaluation/build_dependency_validation.py \
  --documents data/processed/python50_qwen/splits/validation_docs.jsonl \
  --edges data/processed/python50_qwen/splits/validation_edges.jsonl \
  --output data/processed/python50_qwen/eval/dependency_validation.jsonl \
  --tokenizer Qwen/Qwen2.5-Coder-7B
```

Score context gain after training:

```bash
python scripts/evaluation/score_dependency_validation.py \
  --model Qwen/Qwen2.5-Coder-7B \
  --adapter outputs/training/qwen7b_depaware_smoke/final_adapter \
  --input data/processed/python50_qwen/eval/dependency_validation.jsonl \
  --output outputs/evaluation/qwen7b_depaware_context_gain.jsonl \
  --bf16
```

Run cap sensitivity before committing to a server-scale cap:

```bash
CAP_VALUES="100 300 500" bash scripts/server/run_cap_sensitivity.sh
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
- `dependency_aware_v2_token_fit`: first packs dependency-linked documents,
  then fills remaining context budget with same-repo, low-redundancy documents.
- `dependency_aware_v2_strong_first`: first packs explicit structural edges,
  then weak same-directory edges, then token-fit fillers.
- `dependency_aware_high_precision_only`: main thesis setting that uses only
  high-precision, auditable relation labels from
  `configs/relations/main_high_precision.yaml`, then fills unused budget with
  same-repo/collection token-fit candidates.
- `dependency_aware_high_precision_random_order`: uses the same selected
  document sets as high-precision packing, but shuffles order inside each
  window.
- `dependency_aware_high_precision_reverse_order`: uses the same selected
  document sets as high-precision packing, but reverses order inside each
  window.
- `dependency_aware_no_same_directory`: ablation that removes directory
  co-location as a dependency signal.
- `dependency_aware_no_same_repo`: ablation that removes repository membership
  as a dependency-score bonus.
- `dependency_aware_strong_edges_only`: ablation that uses only explicit
  relations such as imports, source-test, docs-code, config-script, README-code,
  and example-code links.

For training-grade packing, pass the target model tokenizer:

```bash
python scripts/run_packing_matrix.py \
  --input data/processed/splits/train_docs.jsonl \
  --output-dir data/processed/packed/train_8192_qwen_tokens \
  --max-tokens 8192 \
  --tokenizer Qwen/Qwen2.5-Coder-7B \
  --edges data/processed/splits/train_edges.jsonl \
  --summary data/processed/packed/train_8192_qwen_tokens/summary.csv
```

Use `--tokenizer-local-files-only` on servers where the tokenizer has already
been cached or downloaded manually.

## Research Milestones

1. Build a stable packing pipeline and statistics dashboard.
2. Add structure-reranking variants on top of BM25 and semantic retrieval.
3. Build multi-source repository data preprocessing.
4. Run packing-only quality analysis.
5. Run 7B + 8K LoRA/QLoRA experiments on the 8-GPU NVLink server.
6. Evaluate with RepoBench, cross-file completion, context gain, passkey,
   needle, and selected LongBench-style tasks.
