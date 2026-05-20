# Experiments

This directory stores lightweight experiment manifests, logs, and analysis
notebooks. Large datasets, checkpoints, and raw logs should stay outside git or
under gitignored output directories.

Suggested structure:

```text
experiments/
  logs/                 Small run summaries
  notebooks/            Analysis notebooks
```

Recommended run summary fields:

```text
run_name
git_commit
date
server
num_gpus
model
context_length
packing_method
token_budget
training_method
train_data_path
validation_data_path
tokens_per_second
peak_memory
validation_loss
evaluation_summary_path
```
