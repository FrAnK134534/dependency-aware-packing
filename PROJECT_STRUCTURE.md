# Project Structure

This repository is organized as a research pipeline:

```text
dataset manifest
  -> Document JSONL
  -> dependency_edges.jsonl
  -> packed training windows
  -> packing metrics / edge review
  -> QLoRA smoke training
  -> dependency-sensitive evaluation
```

The project should stay centered on dependency-aware packing. The 8-GPU server
is only the execution environment.

## Top-Level Responsibilities

```text
AGENTS.md           Agent and experiment-runner guidance.
README.md           User-facing quick start and main commands.
PROJECT_STRUCTURE.md
                    This file: responsibilities and ownership boundaries.
pyproject.toml      Python package metadata and optional dependency groups.

configs/            Declarative experiment, relation, packing, training, and
                    evaluation settings.
data/               Local raw and processed data. Large/generated files are
                    usually gitignored and must be rebuilt or transferred.
docs/               Research design, method explanation, metrics, server
                    runbooks, and advisor-facing reports.
experiments/        Small logs, run manifests, and notebooks.
outputs/            Local smoke outputs and generated experiment artifacts.
scripts/            Command-line entry points. Scripts orchestrate modules but
                    should not hold core algorithm logic.
src/dapacking/      Core Python package and reusable research logic.
tests/              Unit and regression tests for core behavior.
```

## Config Layer

Use `configs/` for choices that should be explicit and reproducible:

```text
configs/datasets/       Repo lists and external-data manifests.
configs/evaluation/     Evaluation defaults.
configs/packing/        Packing defaults and smoke settings.
configs/relations/      Relation allowlists, noisy relations, and reliability.
configs/training/       LoRA/QLoRA templates.
configs/experiment_matrix.yaml
                        Broad experiment matrix.
configs/pretraining_freeze.yaml
                        Current pre-server freeze gates and required methods.
```

Rule of thumb: if changing a value changes the experimental condition, prefer
putting it in a config file or documenting it in a run directory.

## Data Layer

`data/` is a local workspace, not the paper claim by itself.

```text
data/examples/          Tiny fixtures and smoke examples.
data/raw/               Cloned repos, local manifests, downloaded source data.
data/processed/         Generated Document JSONL, edges, splits, packed windows,
                        dataset cards, reviews, and validation records.
```

The main local dataset version is currently:

```text
data/processed/repo_main_v1/
```

When sharing the project through GitHub, assume `data/raw/`, `data/processed/`,
and `outputs/` are not included. The server runner must rebuild them or receive
them separately.

## Source Package Layer

Core logic belongs in `src/dapacking/`:

```text
documents.py            Document and PackedSample data classes.
corpus.py               Repository corpus construction helpers.
collectors/             Manifest-driven text/Markdown/HTML/PDF collectors.
dependency.py           Relation-level dependency detection rules.
edges.py                DependencyEdge construction and JSONL IO.
relation_config.py      Relation allowlist and reliability config loading.
edge_filter.py          High-precision edge filtering and reweighting.
packers.py              Packing methods and ablations.
stats.py                Packing metric summaries.
validation.py           Dependency-sensitive validation records.
audit.py / review.py    Edge review summaries and review helpers.
tokenization.py         Simple and HuggingFace tokenizer counting policy.
bm25.py / semantic.py   Lightweight retrieval/coherence baselines.
dataset_card.py         Dataset-card generation.
```

Scripts should call these modules instead of duplicating logic.

## Script Layer

Scripts are organized by workflow:

```text
scripts/data/           Build corpora, edges, splits, dataset cards, and review
                        files.
scripts/analysis/       Packing fairness and cap-sensitivity summaries.
scripts/baselines/      External baseline adapters such as DataSculpt-original.
scripts/evaluation/     Build and score context-gain validation data.
scripts/server/         Server-facing pipeline and QLoRA launch scripts.
scripts/training/       Model training entry points.
scripts/run_packing.py  Single-method packing CLI.
scripts/run_packing_matrix.py
                        Multi-method packing CLI.
scripts/summarize_packing.py
                        Summary CLI for existing packed files.
```

## Documentation Layer

Docs are split by audience:

```text
docs/00_overview/       Advisor-facing project reports, thesis scope, and
                        external evaluation dossier.
docs/01_design/         Method formalization and macro experiment design.
docs/02_metrics/        Metric definitions and interpretation.
docs/03_server/         Server deployment, freeze protocol, handoff, and
                        pre-server runbooks.
docs/04_logs/           Small human-readable run logs.
docs/05_training/       Training-control checklist.
docs/archive/           Old plans kept for provenance, not current truth.
```

For a new reader, the recommended order is:

1. `README.md`
2. `PROJECT_STRUCTURE.md`
3. `docs/README.md`
4. `docs/00_overview/project_report.md`
5. `docs/01_design/method_formalization.md`
6. `docs/03_server/server_training_handoff.md`

## Current Main Experimental Path

The current main path is:

```text
configs/datasets/python50_repos.tsv
  -> scripts/server/build_dataset_pipeline.sh
  -> scripts/data/filter_dependency_edges.py
  -> configs/relations/main_high_precision.yaml
  -> scripts/server/run_high_precision_freeze.sh
  -> scripts/server/run_7b_8k_qlora.sh
  -> scripts/evaluation/score_dependency_validation.py
```

The current main method is:

```text
dependency_aware_high_precision_only
```

Critical ablations:

```text
dependency_aware_high_precision_random_order
dependency_aware_high_precision_reverse_order
dependency_aware_v2_strong_first
datasculpt_lite
bm25
same_repo
```

## Boundaries

Keep these boundaries clear:

- Relation rules live in `dependency.py`; relation selection/reliability lives
  in `configs/relations/` and `relation_config.py`.
- Edge filtering/reweighting lives in `edge_filter.py`; scripts only expose it
  as a CLI.
- Packing algorithms live in `packers.py`; scripts only choose inputs, outputs,
  methods, tokenizer, and edge files.
- Metrics live in `stats.py`; do not hide paper-critical metric definitions
  inside ad hoc notebooks.
- Server scripts should launch reproducible workflows, not define new research
  logic.
- Generated data should not be treated as committed source of truth unless the
  corresponding manifest, config, and commit hash are recorded.
