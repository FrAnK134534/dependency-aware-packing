# Server Scripts

This folder is reserved for scripts that run on the 8-GPU NVLink server.

The current repository already supports packing generation and summary locally.
Training and evaluation launchers should be added here once the server stack is
confirmed.

Recommended future files:

```text
prepare_env.sh
run_packing_matrix.sh
run_7b_8k_lora.sh
run_7b_8k_qlora.sh
run_eval_suite.sh
collect_run_summary.py
```

Current starter scripts:

```text
build_dataset_pipeline.sh
run_packing_matrix.sh
run_7b_8k_qlora.sh
```

Every server run should save:

```text
git commit hash
config file
dataset manifest
packing method
model path
training log
evaluation results
GPU memory and throughput summary
```
