# Training Control Checklist

Use this checklist before running LoRA/QLoRA experiments on the 8-GPU NVLink
server. The goal is to make packing method the only meaningful variable.

## 1. Dataset And Packing

Record:

```text
git_commit:
repo_set: configs/datasets/python50_repos.tsv
local_repo_manifest:
processed_dataset_dir:
packed_dataset_dir:
packing_method:
max_context_tokens:
max_docs_per_repo:
tokenizer:
train_documents:
validation_documents:
test_documents:
train_edges:
validation_edges:
test_edges:
edge_review_path:
dependency_validation_path:
```

Required pre-training checks:

- `summary.csv` exists for every packing method.
- No packed sample exceeds `max_context_tokens`.
- `tokenizer` is the target model tokenizer, not `simple`.
- The same train/validation/test split is used for every method.
- The same total training-token budget is used for every method.
- A manual edge-review CSV has been sampled and inspected.
- A dependency-sensitive validation JSONL has been generated from held-out
  repositories.

Recommended training candidates:

```text
random
bm25
datasculpt_lite
dependency_aware_v2_strong_first
```

Optional ablation if compute allows:

```text
dependency_aware_v2_token_fit
dependency_aware_strong_edges_only
```

## 2. Model And Tokenizer

Record:

```text
base_model_name_or_path:
tokenizer_name_or_path:
model_revision_or_commit:
context_length: 8192
precision:
flash_attention:
gradient_checkpointing:
```

Controls:

- Use the same base model for every packing method.
- Use the same tokenizer for packing and training.
- Keep context length fixed.
- Keep special-token handling fixed.

## 3. LoRA / QLoRA

Record:

```text
training_type: lora_or_qlora
lora_rank:
lora_alpha:
lora_dropout:
target_modules:
quantization:
bnb_4bit_compute_dtype:
```

Controls:

- Do not tune LoRA config per packing method.
- Do not change quantization settings per packing method.

## 4. Optimization

Record:

```text
max_steps:
tokens_per_method:
learning_rate:
lr_scheduler:
warmup_ratio_or_steps:
optimizer:
weight_decay:
global_batch_size:
micro_batch_size_per_gpu:
gradient_accumulation_steps:
seed:
```

Controls:

- Equal total training tokens across methods.
- Equal global batch size across methods.
- Equal validation interval and checkpoint interval.
- Same random seed unless explicitly running repeated trials.

## 5. Hardware And Throughput

Record:

```text
gpu_count: 8
gpu_type:
nvlink_available: true
cuda_version:
driver_version:
pytorch_version:
deepspeed_or_accelerate_version:
tokens_per_second:
peak_gpu_memory:
```

Controls:

- Run all compared methods on the same node type.
- Save throughput because packing changes sequence composition and may affect
  training speed.

## 6. Validation And Evaluation

Minimum validation before claiming improvement:

```text
long_context_validation_loss
dependency_sensitive_validation_loss
passkey_accuracy
needle_accuracy
cross_file_completion_or_repobench_subset
```

Evaluation controls:

- Use the same validation/evaluation prompts for every trained model.
- Keep decoding settings fixed.
- Report both final performance and training efficiency.

## 7. Run Naming

Recommended run name:

```text
{date}_{model}_{context}_{packing_method}_{token_budget}_{seed}
```

Example:

```text
2026-05-20_qwen2.5coder7b_8k_depaware_v2strong_100m_seed42
```

## 8. Before Starting The 8-GPU Run

Confirm:

- The repo is clean or the current diff is saved.
- The current commit hash is written into the run directory.
- `summary.csv` has acceptable utilization and strong-edge coverage.
- Cap sensitivity does not remove the dependency-aware advantage.
- A tiny smoke run has completed successfully.
- The first full run is `random` or another baseline, not only the proposed
  method.
