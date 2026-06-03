# Configs

This directory contains declarative experiment choices. Core algorithm logic
should stay in `src/dapacking/`; configs should state which dataset, relation
set, method, tokenizer, and training settings are used for a run.

## Files And Directories

```text
experiment_matrix.yaml      Broad method/dataset/metric matrix.
pretraining_freeze.yaml     Current pre-server required methods and gates.

datasets/                   Repo lists and manifest examples.
evaluation/                 Evaluation defaults.
packing/                    Packing defaults.
relations/                  Relation allowlists and reliability priors.
training/                   LoRA/QLoRA templates.
```

## Relation Configs

The current main relation config is:

```text
relations/main_high_precision.yaml
```

It defines the high-precision main graph:

```text
import_relation
test_source_relation
hyperlink_relation
```

Noisy or not-yet-audited relations should stay out of the main graph until edge
review produces enough evidence.

## Editing Rule

When adding a new experimental condition, prefer creating or updating a config
instead of hard-coding values inside scripts. This keeps server runs
reproducible and easier to explain to an advisor.
