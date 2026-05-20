# Pilot Dataset and Packing Report

Date: 2026-05-20

## 1. Pilot Repository Set

This pilot uses 20 real Python repositories cloned under the gitignored
`data/raw/repos/` directory:

attrs, build, click, dateutil, httpcore, iniconfig, installer, itsdangerous,
jsonschema, markupsafe, marshmallow, packaging, pluggy, pyproject-hooks,
requests, starlette, structlog, tqdm, typing_extensions, zipp.

`cpython` was intentionally excluded from this first pilot because it is much
larger than the other repositories and would distort the early sanity check.

## 2. Dataset Build Result

Command:

```bash
bash scripts/server/build_dataset_pipeline.sh data/raw/repo_manifest.txt data/processed/pilot
```

Result:

| Item | Value |
|---|---:|
| repositories | 20 |
| documents | 2072 |
| dependency edges | 32363 |
| train repositories | 16 |
| train documents | 1217 |
| train edges | 15427 |
| validation repositories | 2 |
| validation documents | 662 |
| validation edges | 14081 |
| test repositories | 2 |
| test documents | 193 |
| test edges | 2855 |

Document type distribution:

| Source type | Count |
|---|---:|
| test | 868 |
| docs | 483 |
| source | 320 |
| config | 292 |
| readme | 44 |
| example | 42 |
| script | 23 |

## 3. Dependency Edge Sanity Check

Top dependency relations:

| Relation | Count | Total weight |
|---|---:|---:|
| same_directory + same_repo | 28802 | 10080.70 |
| docs_code_relation + same_repo | 1447 | 1012.90 |
| test_source_relation + same_repo | 715 | 715.00 |
| import_relation + test_source_relation + same_repo | 430 | 860.00 |
| import_relation + same_directory + same_repo | 251 | 338.85 |
| config_script_relation + same_repo | 230 | 138.00 |
| import_relation + same_repo | 160 | 176.00 |
| readme_code_relation + same_repo | 112 | 78.40 |
| example_code_relation + same_repo | 91 | 54.60 |

Judgement:

- The edge file is usable for pilot experiments.
- The strong relations are present: imports, source-test links, docs-code
  links, README-code links, config-script links, and example-code links.
- `same_directory + same_repo` is still the largest relation by count. This is
  acceptable as a weak structural prior, but it should not be treated as the
  main evidence for the thesis claim. In paper experiments, we should report
  results with and without weak directory edges.

## 4. Packing Matrix Result

Command:

```bash
python scripts/run_packing_matrix.py \
  --input data/processed/pilot/splits/train_docs.jsonl \
  --output-dir data/processed/pilot/packed/train_8192_v5 \
  --max-tokens 8192 \
  --edges data/processed/pilot/splits/train_edges.jsonl \
  --summary data/processed/pilot/packed/train_8192_v5/summary.csv
```

All generated samples were checked to be within the 8192-token window.

| Method | Utilization | Truncation | Docs/window | Dep. score | Weighted edge coverage | Semantic sim. | Redundant pair rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 0.7855 | 0.0667 | 4.7539 | 0.0080 | 0.0016 | 0.0301 | 0.0000 |
| length_aware | 0.8630 | 0.0733 | 5.2232 | 0.0090 | 0.0117 | 0.0422 | 0.0032 |
| same_repo | 0.7917 | 0.0672 | 4.7913 | 0.1084 | 0.0317 | 0.0459 | 0.0006 |
| bm25 | 0.9714 | 0.0825 | 5.8792 | 0.1141 | 0.0311 | 0.1095 | 0.0117 |
| semantic | 0.9621 | 0.0817 | 5.8230 | 0.1042 | 0.0315 | 0.0919 | 0.0120 |
| datasculpt_lite | 0.9761 | 0.0829 | 5.9078 | 0.1228 | 0.0421 | 0.0875 | 0.0081 |
| dependency_aware | 0.5845 | 0.0496 | 3.5378 | 0.1884 | 0.1280 | 0.0696 | 0.0018 |

Interpretation:

- BM25, semantic, and DataSculpt-lite fill the window better.
- Dependency-aware packing sacrifices utilization in this first version, but it
  has the strongest dependency score and weighted edge coverage.
- This is a useful contrast for the thesis: semantic similarity and token
  efficiency are not the same thing as structural dependency coverage.
- The next algorithmic improvement should be a hybrid method:
  dependency-aware candidate selection plus token-fit optimization, so the
  method preserves structural gains while closing the utilization gap.

## 5. Current Engineering State

Implemented:

- real-repository corpus builder;
- dependency edge builder;
- repo-level train/validation/test split;
- packing matrix runner;
- random, length-aware, same-repo, BM25, semantic, DataSculpt-lite, and
  dependency-aware packers;
- summary metrics for utilization, truncation, docs/window, dependency score,
  order dependency, edge coverage, weighted edge coverage, semantic similarity,
  redundancy, and same-repo ratio.

The pilot is now ready to be copied to an 8-GPU NVLink server for larger
repository sets and later LoRA/QLoRA training runs.
