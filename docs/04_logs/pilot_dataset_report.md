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

## 6. Dependency-Aware V2 and Ablation Pilot

Command:

```bash
python scripts/run_packing_matrix.py \
  --input data/processed/pilot/splits/train_docs.jsonl \
  --output-dir data/processed/pilot/packed/train_8192_v6_dependency_ablation \
  --methods dependency_aware,dependency_aware_v2_token_fit,dependency_aware_no_same_directory,dependency_aware_no_same_repo,dependency_aware_strong_edges_only \
  --max-tokens 8192 \
  --edges data/processed/pilot/splits/train_edges.jsonl \
  --summary data/processed/pilot/packed/train_8192_v6_dependency_ablation/summary.csv
```

All generated samples were checked to be within the 8192-token window.

| Method | Utilization | Docs/window | Dep. score | Weighted edge coverage | Order dependency | Same-repo pair ratio |
|---|---:|---:|---:|---:|---:|---:|
| dependency_aware | 0.5845 | 3.5378 | 0.1884 | 0.1280 | 0.2960 | 0.5727 |
| dependency_aware_v2_token_fit | 0.8977 | 5.4330 | 0.1694 | 0.1185 | 0.3091 | 0.7232 |
| dependency_aware_no_same_directory | 0.2363 | 1.4301 | 0.0626 | 0.0706 | 0.1464 | 0.1387 |
| dependency_aware_no_same_repo | 0.5845 | 3.5378 | 0.1884 | 0.1280 | 0.2960 | 0.5727 |
| dependency_aware_strong_edges_only | 0.2363 | 1.4301 | 0.0626 | 0.0706 | 0.1464 | 0.1387 |

Interpretation:

- `dependency_aware_v2_token_fit` solves the main engineering weakness of the
  first dependency-aware method: utilization increases from 0.5845 to 0.8977.
- The v2 method keeps most of the structural advantage: weighted edge coverage
  decreases only from 0.1280 to 0.1185, while order dependency slightly
  improves.
- Removing `same_directory` sharply reduces utilization and edge coverage. This
  confirms that directory co-location is currently a major weak structural
  prior. It should be reported as an ablation, not hidden.
- Removing only `same_repo` has no visible effect in this pilot because
  same-repo-only edges are below the current dependency threshold; most useful
  weak edges are actually `same_directory + same_repo`.
- `strong_edges_only` is conservative and useful for analysis, but too sparse
  as a standalone packing strategy. It is better used as an ablation or as the
  first stage of a hybrid method.

Recommended paper-facing method:

```text
Dependency-Aware V2 = explicit/weak dependency selection + token-fit completion.
```

This method gives a clearer thesis contribution than the original version:
it preserves dependency structure while making the packed training data dense
enough for practical long-context adaptation.

## 7. Strong-First V2 and Strong/Weak Edge Metrics

Command:

```bash
python scripts/run_packing_matrix.py \
  --input data/processed/pilot/splits/train_docs.jsonl \
  --output-dir data/processed/pilot/packed/train_8192_v7_strong_first \
  --methods dependency_aware,dependency_aware_v2_token_fit,dependency_aware_v2_strong_first,dependency_aware_strong_edges_only \
  --max-tokens 8192 \
  --tokenizer simple \
  --edges data/processed/pilot/splits/train_edges.jsonl \
  --summary data/processed/pilot/packed/train_8192_v7_strong_first/summary.csv
```

All generated samples were checked to be within the 8192-token window.

| Method | Utilization | Dep. score | Weighted all coverage | Weighted strong coverage | Weighted weak coverage | Strong order dep. | Weak order dep. |
|---|---:|---:|---:|---:|---:|---:|---:|
| dependency_aware | 0.5845 | 0.1884 | 0.1280 | 0.1176 | 0.1349 | 0.1409 | 0.1658 |
| dependency_aware_v2_token_fit | 0.8977 | 0.1694 | 0.1185 | 0.1128 | 0.1222 | 0.1932 | 0.1320 |
| dependency_aware_v2_strong_first | 0.8977 | 0.1682 | 0.1240 | 0.1311 | 0.1193 | 0.2235 | 0.1295 |
| dependency_aware_strong_edges_only | 0.2363 | 0.0626 | 0.0706 | 0.1482 | 0.0194 | 0.1464 | 0.0076 |

Interpretation:

- `dependency_aware_v2_strong_first` keeps the high utilization of v2 while
  improving weighted strong-edge coverage from 0.1128 to 0.1311.
- It also raises strong order dependency from 0.1932 to 0.2235, meaning explicit
  structural relations are more often placed in useful left-to-right order.
- Weak-edge coverage drops slightly, which is expected: the method deliberately
  gives import/test/docs/config/example/README relations priority over
  same-directory relations.
- For paper experiments, `dependency_aware_v2_strong_first` is currently the
  best main method candidate. `dependency_aware_v2_token_fit` remains a useful
  ablation showing what happens without strong-edge priority.

Tokenizer note:

- The pilot above uses `--tokenizer simple` for fast local verification.
- Before server training, regenerate packed data with the target model tokenizer,
  for example `--tokenizer Qwen/Qwen2.5-Coder-7B`.
