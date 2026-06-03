# dapacking Package

`dapacking` contains reusable research logic for dependency-aware packing.
Command-line scripts should call these modules rather than duplicating their
behavior.

## Module Responsibilities

```text
documents.py            Document and PackedSample data classes.
io.py                   JSONL readers and writers.
corpus.py               Repository document extraction helpers.
dataset_card.py         Dataset-card summaries.

dependency.py           Dependency relation detection rules.
edges.py                DependencyEdge construction and IO.
relation_config.py      Relation allowlists and reliability priors.
edge_filter.py          Filter/reweight edges for high-precision settings.

packers.py              Packing algorithms and ablations.
stats.py                Packing construction metrics.
validation.py           Dependency-sensitive validation records.

bm25.py                 Lexical retrieval baseline.
semantic.py             TF-IDF semantic/coherence baseline.
tokenization.py         Simple and HuggingFace tokenizer counting.

audit.py                Manual edge-review summaries.
review.py               Review sampling/record helpers.
edge_annotation.py      Assistant-assisted review utilities.

collectors/             Manifest-driven external text/HTML/PDF collectors.
```

## Current Main Method

The main thesis method is implemented in `packers.py` as:

```text
dependency_aware_high_precision_only
```

Its relation set is configured by:

```text
configs/relations/main_high_precision.yaml
```

Order ablations:

```text
dependency_aware_high_precision_random_order
dependency_aware_high_precision_reverse_order
```

These ablations reuse selected document sets and only change within-window
document order.

## Development Rule

Keep paper-critical behavior deterministic under a fixed seed. Add tests when
changing relation scoring, edge filtering, packing order, tokenizer counting, or
metric calculation.
