# Tests

Tests cover the reusable research logic under `src/dapacking/`.

## Main Coverage

```text
test_dependency.py          Relation detection rules.
test_edges.py               Dependency edge construction.
test_edge_filter.py         High-precision edge filtering/reweighting.
test_relation_config.py     Relation config loading.
test_packers.py             Packing methods and order ablations.
test_stats.py               Packing summary metrics.
test_validation.py          Dependency-sensitive validation records.
test_audit.py               Manual edge-review summaries.
test_collectors.py          External corpus collectors.
test_datasculpt_baseline.py DataSculpt adapter behavior.
test_tokenization.py        Simple/HF tokenizer policy.
```

## Standard Check

```bash
PYTHONPATH=src python -m pytest -q
python -m compileall -q src scripts
git diff --check
```

Add or update tests whenever changing relation rules, edge filtering, packing
selection/order, tokenizer counting, or metrics.
