# AGENTS.md

This file guides future coding agents and experiment runners working on this
repository. The project is a research codebase for studying dependency-aware
data packing for long-context adaptation.

## 1. Project Goal

The central research question is:

> Under a fixed model, context length, and training token budget, can better
> packing strategies help a language model learn to use long-context
> dependencies more effectively?

The main method is **Dependency-Aware Packing**. Instead of packing documents
only by length, repository/topic membership, lexical retrieval, or semantic
similarity, the project constructs long-context training samples that contain
explicit, learnable dependencies.

The initial primary data scenario is **multi-source code repository context**:

- source code;
- tests;
- README files;
- docs;
- config files;
- examples;
- issue or PR descriptions;
- commit messages;
- benchmark logs;
- API usage examples.

The method should remain general enough to extend to technical documents later,
but the first paper-quality experimental loop should stay centered on code
repository and software-engineering context.

The repository also contains a first implementation of manifest-driven
non-code collection for papers, web pages, technical documents, and plain text.
Use it as a generalization path, while keeping the main thesis claim anchored
in dependency-aware packing rather than broad web crawling.

## 2. What Is the Innovation?

Do not frame the project as "training a larger model on 8 GPUs." The 8-GPU
NVLink server is infrastructure, not the contribution.

The contribution is:

1. **A dependency-aware packing objective.**
   The long-context window should contain useful cross-document dependencies,
   not only similar or nearby text.

2. **A lightweight dependency model for multi-source repository context.**
   Examples:
   - README -> source code;
   - docs -> implementation;
   - config -> training or run script;
   - source file -> test file;
   - API document -> usage example;
   - issue/PR description -> changed file;
   - commit message -> changed file;
   - benchmark log -> implementation.

3. **A controlled experimental framework.**
   Keep model, context length, token budget, optimizer, LoRA/QLoRA setup, and
   evaluation protocol fixed. Change only the packing method.

## 3. Current Repository Layout

```text
configs/                     Packing and experiment configs
  experiment_matrix.yaml     High-level experiment matrix
  packing/                   Packing configs
  training/                  7B/8K LoRA and QLoRA templates
  evaluation/                Evaluation suite templates
data/
  examples/                  Tiny smoke-test data
  raw/                       Raw datasets, gitignored
  processed/                 Generated datasets, gitignored
docs/                        Research plans and reports
  00_overview/               Advisor report, design rationale, thesis scope
  01_design/                 Macro experiment design
  02_metrics/                Metric definitions
  03_server/                 8-GPU NVLink deployment plan
  archive/                   Older low-resource plan
experiments/notebooks/       Analysis notebooks
experiments/logs/            Small run summaries
outputs/                     Generated outputs, gitignored
scripts/
  data/                      Corpus, dependency edge, external corpus, and repo split builders
  run_packing.py             Generate packed JSONL files
  run_packing_matrix.py      Generate multiple packing baselines
  summarize_packing.py       Summarize packing outputs
  server/                    Future server launch scripts
src/dapacking/
  bm25.py                    Lightweight BM25 implementation
  collectors/                Manifest-driven PDF/HTML/Markdown/Text collectors
  dependency.py              Structural dependency rules
  documents.py               Data classes
  io.py                      JSONL readers/writers
  packers.py                 Packing methods
  stats.py                   Packing summary metrics
  tokenization.py            Lightweight token counter
tests/                       Unit tests
```

## 4. Implemented Packing Methods

Current methods:

- `random`: shuffle documents and fill windows.
- `length_aware`: sort by length and fill windows.
- `same_repo`: group documents by repository or external collection.
- `bm25`: use lexical retrieval from an anchor document.
- `semantic`: use a lightweight TF-IDF cosine index as an early semantic
  baseline.
- `datasculpt_lite`: combine TF-IDF coherence, token-fit efficiency,
  repository integrity, and redundancy penalty.
- `dependency_aware`: greedily add documents with structural dependency edges.
- `dependency_aware_v2_token_fit`: first add dependency-linked documents, then
  fill remaining budget using same-repo token-fit candidates.
- `dependency_aware_v2_strong_first`: first add explicit structural edges, then
  weak directory/repository edges, then token-fit candidates.
- `dependency_aware_no_same_directory`: ablation removing directory co-location.
- `dependency_aware_no_same_repo`: ablation removing the same-repository score
  bonus.
- `dependency_aware_strong_edges_only`: ablation keeping only explicit
  structural relations and excluding `same_directory` / `same_repo`.

Planned methods:

- `bm25_structure_rerank`: retrieve with BM25, rerank by dependency score.
- `semantic_structure_rerank`: retrieve with embeddings, rerank by dependency
  score.

## 5. Data Format

Input JSONL document format:

```json
{
  "docid": "repo_a:src/model.py",
  "content": "...",
  "metadata": {
    "repo": "repo_a",
    "path": "src/model.py",
    "language": "python",
    "source_type": "source"
  }
}
```

Packed JSONL sample format:

```json
{
  "sample_id": "dependency_aware_000001",
  "method": "dependency_aware",
  "docids": ["repo_a:README.md", "repo_a:src/model.py"],
  "content": "...",
  "stats": {
    "tokens": 8192,
    "num_docs": 2,
    "dependency_score": 0.7,
    "token_utilization": 1.0,
    "truncation_rate": 0.0
  }
}
```

External/non-code documents should preserve:

```json
{
  "collection": "manuals",
  "document_id": "paper_a",
  "section_id": "0001-method",
  "section_title": "Method",
  "section_index": 1,
  "source_kind": "local_pdf",
  "source_type": "paper_section",
  "url": "https://example.org/paper.html",
  "license": "CC-BY"
}
```

When adding new data sources, preserve metadata. Do not throw away repo, path,
language, source type, collection, document ID, URL, issue ID, commit ID, or
license information.

## 6. Core Metrics to Implement and Preserve

### 6.1 Packing Quality Metrics

For a packed sample `S_i` and maximum context length `L`:

- `token_utilization = tokens(S_i) / L`
- `avg_docs_per_window = mean_i(|S_i|)`
- `doc_truncation_rate = truncated_docs / candidate_docs`
- `token_truncation_rate = truncated_tokens / original_candidate_tokens`

Dependency metrics:

```text
dependency_density(S_i)
= sum_{a,b in S_i, a != b} w(a -> b) / (|S_i| * (|S_i| - 1))
```

Order-aware dependency:

```text
order_dependency(S_i)
= sum_{j=2..n} max_{k<j} w(d_k -> d_j) / (n - 1)
```

Edge coverage:

```text
edge_coverage
= covered_dependency_edges / all_candidate_dependency_edges
```

Weighted edge coverage:

```text
weighted_edge_coverage
= sum_{covered edges} w(edge) / sum_{all candidate edges} w(edge)
```

Semantic and redundancy metrics:

- `semantic_similarity`: mean pairwise token-Jaccard similarity in the summary
  statistics; semantic/DataSculpt-lite packers use a TF-IDF cosine index for
  candidate selection.
- `redundant_pair_rate`: fraction of document pairs whose token-Jaccard
  similarity exceeds a high threshold.
- `strong_edge_coverage` / `weighted_strong_edge_coverage`: coverage of edges
  containing at least one explicit relation such as import, source-test,
  docs-code, README-code, config-script, example-code, hyperlink, citation,
  API-doc-usage, definition-usage, or equation/figure reference.
- `weak_edge_coverage` / `weighted_weak_edge_coverage`: coverage of edges made
  only from `same_directory`, `same_repo`, `same_document`,
  `same_collection`, `section_neighbor`, and/or `same_domain`.

Tokenizer policy:

- Use `--tokenizer simple` for fast local smoke tests.
- Before training, rerun packing with the target model tokenizer, for example
  `--tokenizer Qwen/Qwen2.5-Coder-7B`.
- Use `--tokenizer-local-files-only` when the server has no outbound network
  access and the tokenizer is already cached.

Pre-training freeze additions:

- Write a dataset card with `scripts/data/write_dataset_card.py`.
- Sample balanced edge-review CSVs with
  `scripts/data/sample_dependency_edges.py --per-relation`.
- Summarize annotated edge reviews with `scripts/data/summarize_edge_review.py`.
- Build context-gain controls with
  `scripts/evaluation/build_context_gain_controls.py`.
- Check matched-utilization fairness with
  `scripts/analysis/check_packing_fairness.py`.

### 6.2 Post-Training Metrics

Main metrics:

- long-context validation loss;
- dependency-sensitive validation loss;
- context gain:

```text
context_gain(A -> B) = loss(B alone) - loss(A + B)
```

Code/repository metrics:

- RepoBench-R: Recall@k, MRR@k, nDCG@k, Hit@k.
- RepoBench-C: Exact Match, Edit Similarity, CodeBLEU, Identifier F1.
- RepoBench-P: retrieval + completion pipeline metrics.

Probe metrics:

- passkey retrieval accuracy;
- needle-in-a-haystack exact retrieval accuracy;
- accuracy by context length and insertion depth.

LongBench-style metrics:

- QA: F1 / EM;
- summarization: ROUGE-L;
- classification or multiple choice: Accuracy;
- code tasks: EM / EditSim / pass rate where applicable.

## 7. Experimental Design

Always keep controlled variables explicit.

Main comparison:

```text
G1 Random Packing
G2 Length-Aware Packing
G3 Same-Repo / Same-Topic Packing
G4 BM25 Packing
G5 Semantic / DataSculpt-Lite Packing
G6 Dependency-Aware Packing
```

Critical comparisons:

```text
Same-Repo vs Dependency-Aware
BM25 vs Dependency-Aware
Semantic/DataSculpt-Lite vs Dependency-Aware
```

Important ablations:

```text
Dependency-Aware without import relation
Dependency-Aware without source-test relation
Dependency-Aware without README/docs relation
Dependency-Aware without config-script relation
Dependency-Aware without same-repo weak prior
BM25 + structure reranking
Semantic + structure reranking
```

If the dependency-aware method only beats random packing, the result is weak.
The method must be compared against same-repo, BM25, and semantic baselines.

## 8. 8-GPU NVLink Server Deployment Expectations

The target server is a single node with 8 GPUs and NVLink. Future server scripts
should assume distributed training, but should not hard-code one cluster layout.

Recommended initial server stack:

```text
Python 3.10+
PyTorch with CUDA support
Transformers
Datasets
Accelerate
PEFT
bitsandbytes, if QLoRA is used
FlashAttention, if supported by the GPU/CUDA stack
DeepSpeed or FSDP
```

Recommended training progression:

```text
Stage 1: 7B + 8K + LoRA/QLoRA, small token budget smoke run
Stage 2: 7B + 8K, full baseline comparison
Stage 3: 7B + 16K or 13B + 8K, selected methods only
Stage 4: ablations and case studies
```

Do not begin with full-parameter fine-tuning. The project requires many
baseline and ablation runs, so LoRA/QLoRA is the practical default.

Server runs must log:

```text
git commit
packing method
dataset version
model name/path
context length
token budget
effective batch size
micro batch size
gradient accumulation
number of GPUs
precision
LoRA/QLoRA config
optimizer and learning rate
tokens/sec
peak memory
training hours
validation loss
evaluation metrics
```

## 9. Commands for Current Local Code

Run tests:

```bash
PYTHONPATH=src python -m pytest -q
```

Generate packed examples:

```bash
python scripts/run_packing.py \
  --input data/examples/code_docs.jsonl \
  --output outputs/dependency.jsonl \
  --method dependency_aware \
  --max-tokens 512
```

Generate several baselines:

```bash
python scripts/run_packing.py --input data/examples/code_docs.jsonl --output outputs/random.jsonl --method random --max-tokens 512
python scripts/run_packing.py --input data/examples/code_docs.jsonl --output outputs/length_aware.jsonl --method length_aware --max-tokens 512
python scripts/run_packing.py --input data/examples/code_docs.jsonl --output outputs/same_repo.jsonl --method same_repo --max-tokens 512
python scripts/run_packing.py --input data/examples/code_docs.jsonl --output outputs/bm25.jsonl --method bm25 --max-tokens 512
python scripts/run_packing.py --input data/examples/code_docs.jsonl --output outputs/dependency.jsonl --method dependency_aware --max-tokens 512
```

Summarize packing outputs:

```bash
python scripts/summarize_packing.py \
  outputs/random.jsonl \
  outputs/length_aware.jsonl \
  outputs/same_repo.jsonl \
  outputs/bm25.jsonl \
  outputs/dependency.jsonl \
  --output outputs/packing_summary.csv
```

Build external/non-code documents from a manifest:

```bash
python scripts/data/build_external_corpus.py \
  --manifest configs/datasets/external_manifest.example.tsv \
  --output data/processed/external_documents.jsonl
```

Merge repository and external documents:

```bash
python scripts/data/build_mixed_corpus.py \
  --input data/processed/documents.jsonl \
  --input data/processed/external_documents.jsonl \
  --output data/processed/mixed_documents.jsonl
```

Build context-gain validation from reviewed edges:

```bash
python scripts/evaluation/build_dependency_validation.py \
  --documents data/processed/documents.jsonl \
  --edges data/processed/dependency_edges.jsonl \
  --review-annotations data/processed/review/edge_review_annotated.csv \
  --output data/processed/dependency_validation.jsonl
```

Build anti-bias context-gain controls:

```bash
python scripts/evaluation/build_context_gain_controls.py \
  --documents data/processed/splits/validation_docs.jsonl \
  --edges data/processed/splits/validation_edges.jsonl \
  --output data/processed/context_gain_controls.jsonl
```

Write a dataset card:

```bash
python scripts/data/write_dataset_card.py \
  --documents data/processed/documents.jsonl \
  --edges data/processed/dependency_edges.jsonl \
  --split-dir data/processed/splits \
  --name repo-main-v1 \
  --output data/processed/DATASET_CARD.md
```

## 10. Data and Evaluation Safety

Avoid data leakage:

- split by repository or external collection group, not by individual file;
- do not train on held-out evaluation repositories;
- decontaminate against RepoBench, LongBench, SWE-bench, and any custom test
  sets used in the paper;
- keep dataset manifests and license information.

Avoid overclaiming:

- passkey and needle are probes, not the main proof;
- RepoBench and dependency-sensitive validation loss should carry the main
  argument;
- if a metric improves only because token utilization is higher, say so and
  control for it.

## 11. Coding Guidelines

- Prefer simple, inspectable implementations for research code.
- Keep packing algorithms deterministic under a seed.
- Record all generated data paths and configs.
- Add unit tests for new scoring and packing logic.
- Keep generated large data under `data/raw`, `data/processed`, or `outputs`;
  these are gitignored.
- Do not commit datasets, model checkpoints, or large logs.
- Update docs when experiment assumptions change.

## 12. Near-Term TODOs

1. Add structure-aware reranking on top of BM25 and semantic retrieval.
2. Build multi-source repository data preprocessing:
   source, tests, README, docs, configs, examples, issues, commits.
3. Add server-oriented training configs for 7B + 8K LoRA/QLoRA.
4. Add evaluation scripts for RepoBench, passkey, needle, and context gain.
