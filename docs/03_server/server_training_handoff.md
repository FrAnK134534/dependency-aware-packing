# Server Training Handoff

这份文档用于把当前项目交给导师或服务器实验执行者。它的目标不是重新解释整个课题，而是说明：

1. 当前版本已经实现了什么；
2. 服务器上应该先跑哪些命令；
3. 训练后应该如何比较结果；
4. 哪些结论目前还不能提前声称。

## 1. 当前项目主线

论文主线建议收缩为：

```text
High-Precision Dependency-Aware Packing for Long-Context Adaptation in Code Repositories
```

核心问题是：

```text
固定模型、tokenizer、context length、训练 token budget 和 LoRA/QLoRA 设置，
只改变 packing strategy，观察模型是否更好地学习跨文档依赖。
```

当前最适合进入服务器 smoke run 的方法是：

```text
dependency_aware_high_precision_only
```

它只把高精度、可审计的 dependency relation 用作主依赖信号：

```text
import_relation
test_source_relation
hyperlink_relation
```

同时保留两个顺序消融：

```text
dependency_aware_high_precision_random_order
dependency_aware_high_precision_reverse_order
```

这两个方法复用同一组被选中文档，只改变窗口内部顺序，用来判断收益来自“依赖文档共现”还是“依赖顺序”。

## 2. 当前代码状态

已实现的关键模块：

```text
src/dapacking/dependency.py          dependency relation 规则
src/dapacking/edges.py               dependency_edges.jsonl 构建
src/dapacking/relation_config.py     relation config / reliability 加载
src/dapacking/edge_filter.py         dependency edge 过滤和重加权
src/dapacking/packers.py             packing baselines 和 dependency-aware 方法
src/dapacking/stats.py               packing 指标统计
src/dapacking/validation.py          dependency-sensitive validation 支持
```

服务器相关入口：

```text
scripts/server/build_dataset_pipeline.sh
scripts/server/run_packing_only_experiment.sh
scripts/server/run_high_precision_freeze.sh
scripts/server/run_7b_8k_qlora.sh
scripts/training/train_causal_lm_qlora.py
scripts/evaluation/build_dependency_validation.py
scripts/evaluation/build_context_gain_controls.py
scripts/evaluation/score_dependency_validation.py
```

relation 配置：

```text
configs/relations/main_high_precision.yaml
```

## 3. 本地已完成的验证

本地代码检查已通过：

```bash
PYTHONPATH=src python -m pytest -q
python -m compileall -q src scripts
git diff --check
```

最近一次结果：

```text
48 tests passed
compileall passed
git diff --check passed
```

本地 `repo_main_v1` 数据已存在：

```text
data/processed/repo_main_v1/documents.jsonl
data/processed/repo_main_v1/dependency_edges.jsonl
data/processed/repo_main_v1/splits/
data/processed/repo_main_v1/DATASET_CARD.md
```

注意：`data/processed/` 通常不进 git。如果导师只从 GitHub 拉代码，需要在服务器上重建数据；如果要复用本地数据，需要单独同步 `data/raw/` 和 `data/processed/repo_main_v1/`。

本地 simple-tokenizer high-precision 验证已生成：

```text
data/processed/repo_main_v1/splits/train_edges_high_precision.jsonl
data/processed/repo_main_v1/splits/validation_edges_high_precision.jsonl
data/processed/repo_main_v1/packed/validation_8192_high_precision/summary.csv
```

validation split 上的核心 offline 观察：

```text
dependency_aware_high_precision_only:
  avg_token_utilization: 0.8801
  weighted_strong_edge_coverage: 0.3372
  avg_order_dependency: 0.2293

dependency_aware_high_precision_random_order:
  weighted_strong_edge_coverage: 0.3372
  avg_order_dependency: 0.0770

dependency_aware_high_precision_reverse_order:
  weighted_strong_edge_coverage: 0.3372
  avg_order_dependency: 0.0050

datasculpt_lite:
  avg_token_utilization: 0.9253
  weighted_strong_edge_coverage: 0.1313
  avg_order_dependency: 0.0233

bm25:
  avg_token_utilization: 0.9253
  weighted_strong_edge_coverage: 0.1529
  avg_order_dependency: 0.0369
```

解释：high-precision 方法的窗口没有 BM25/DataSculpt-lite 那么满，但 dependency coverage 和 order dependency 明显更高；random/reverse order 的同文档集合消融说明顺序本身是重要变量。

## 4. 服务器环境建议

建议环境：

```text
Python >= 3.10
PyTorch + CUDA
transformers
datasets
accelerate
peft
bitsandbytes
flash-attn, if supported
```

安装项目：

```bash
python -m pip install -e ".[dev,tokenizer,collectors]"
python -m pip install torch transformers datasets accelerate peft bitsandbytes
```

如果服务器无法联网，需要提前缓存：

```text
Qwen/Qwen2.5-Coder-7B
Qwen/Qwen2.5-Coder-7B tokenizer
```

离线时给脚本加：

```bash
LOCAL_FILES_ONLY=1
```

## 5. 服务器执行顺序

### Step 0: 基础检查

```bash
git rev-parse HEAD
PYTHONPATH=src python -m pytest -q
python -m compileall -q src scripts
```

记录当前 commit hash，所有训练结果都要写入对应 run 目录。

### Step 1: 构建或同步数据

如果服务器上重新构建：

```bash
python scripts/data/clone_repo_manifest.py \
  --input configs/datasets/python50_repos.tsv \
  --repo-dir data/raw/repos \
  --output-manifest data/raw/python50_repos_local_manifest.txt

MAX_DOCS_PER_REPO=300 \
bash scripts/server/build_dataset_pipeline.sh \
  data/raw/python50_repos_local_manifest.txt \
  data/processed/repo_main_v1
```

如果复用本地生成数据，直接同步：

```text
data/raw/repos/
data/raw/python50_repos_local_manifest.txt
data/processed/repo_main_v1/
```

### Step 2: 用目标 tokenizer 重新生成 high-precision packing

正式训练前不要使用 simple tokenizer 的 packing 结果。使用：

```bash
PACK_SPLITS="train validation" \
bash scripts/server/run_high_precision_freeze.sh \
  data/processed/repo_main_v1 \
  8192 \
  Qwen/Qwen2.5-Coder-7B
```

如果 tokenizer 已缓存但不能联网：

```bash
LOCAL_FILES_ONLY=1 PACK_SPLITS="train validation" \
bash scripts/server/run_high_precision_freeze.sh \
  data/processed/repo_main_v1 \
  8192 \
  Qwen/Qwen2.5-Coder-7B
```

输出：

```text
data/processed/repo_main_v1/splits/train_edges_high_precision.jsonl
data/processed/repo_main_v1/splits/validation_edges_high_precision.jsonl
data/processed/repo_main_v1/packed/train_8192_high_precision/summary.csv
data/processed/repo_main_v1/packed/validation_8192_high_precision/summary.csv
```

### Step 3: 人工 edge review

优先审查 high-precision edge：

```bash
python scripts/data/sample_dependency_edges.py \
  --documents data/processed/repo_main_v1/splits/validation_docs.jsonl \
  --edges data/processed/repo_main_v1/splits/validation_edges_high_precision.jsonl \
  --output data/processed/repo_main_v1/review/validation_edges_high_precision_balanced.csv \
  --strong-only \
  --per-relation 30
```

人工标注后生成 reliability：

```bash
python scripts/data/build_relation_reliability.py \
  --input data/processed/repo_main_v1/review/validation_edges_high_precision_balanced_annotated.csv \
  --output data/processed/repo_main_v1/review/relation_reliability.yaml
```

注意：assistant-assisted review 只能作为预筛选建议，不能当正式人工标签。

### Step 4: 构建 dependency-sensitive validation

```bash
python scripts/evaluation/build_dependency_validation.py \
  --documents data/processed/repo_main_v1/splits/validation_docs.jsonl \
  --edges data/processed/repo_main_v1/splits/validation_edges_high_precision.jsonl \
  --review-annotations data/processed/repo_main_v1/review/validation_edges_high_precision_balanced_annotated.csv \
  --allowed-review-labels yes,partial \
  --min-review-confidence 0.6 \
  --output data/processed/repo_main_v1/eval/dependency_validation.jsonl
```

同时保留负控：

```bash
python scripts/evaluation/build_context_gain_controls.py \
  --documents data/processed/repo_main_v1/splits/validation_docs.jsonl \
  --edges data/processed/repo_main_v1/splits/validation_edges_high_precision.jsonl \
  --output data/processed/repo_main_v1/eval/context_gain_controls.jsonl
```

## 6. 训练矩阵建议

先做 smoke run，不要直接全矩阵正式训练。

### Smoke Run

最小 3 个方法：

```text
same_repo
datasculpt_lite
dependency_aware_high_precision_only
```

每个方法 100 steps 左右，只验证：

```text
能否启动
显存是否稳定
loss 是否下降
validation 是否能跑
checkpoint 是否能保存
```

示例：

```bash
MODEL=Qwen/Qwen2.5-Coder-7B \
TRAIN_FILE=data/processed/repo_main_v1/packed/train_8192_high_precision/dependency_aware_high_precision_only_8192.jsonl \
VALIDATION_FILE=data/processed/repo_main_v1/packed/validation_8192_high_precision/dependency_aware_high_precision_only_8192.jsonl \
OUTPUT_DIR=outputs/training/qwen7b_8k_high_precision_smoke \
RUN_NAME=qwen7b_8k_high_precision_smoke \
MAX_STEPS=100 \
MAX_TRAIN_SAMPLES=512 \
MAX_VALIDATION_SAMPLES=128 \
bash scripts/server/run_7b_8k_qlora.sh
```

### Main 8K Run

如果 smoke run 正常，第一轮主实验建议：

```text
same_repo
bm25
datasculpt_lite
dependency_aware_high_precision_only
dependency_aware_high_precision_random_order
```

如果算力允许，再加入：

```text
dependency_aware_high_precision_reverse_order
dependency_aware_v2_strong_first
DataSculpt-original
```

正式训练必须固定：

```text
base model
tokenizer
context length
LoRA/QLoRA config
learning rate
effective batch size
active token budget
train/validation split
seed
```

## 7. 训练后评测

先跑 dependency-sensitive context gain：

```bash
python scripts/evaluation/score_dependency_validation.py \
  --model Qwen/Qwen2.5-Coder-7B \
  --adapter outputs/training/qwen7b_8k_high_precision_smoke/final_adapter \
  --input data/processed/repo_main_v1/eval/dependency_validation.jsonl \
  --output outputs/evaluation/qwen7b_8k_high_precision_context_gain.jsonl \
  --bf16
```

然后比较：

```text
reviewed dependency edge context gain
same-group non-edge context gain
random cross-group context gain
long-context validation loss
RepoBench / cross-file retrieval or completion
needle/passkey sanity check
```

## 8. 如何判断结果

比较时不要只看 dependency score。主表至少要同时报告：

```text
token utilization
truncation rate
weighted strong edge coverage
avg order dependency
redundancy
validation loss
context gain
RepoBench or cross-file score
```

理想结果：

```text
dependency_aware_high_precision_only
  > datasculpt_lite / bm25
  > same_repo
```

同时：

```text
dependency order > random order > reverse order
```

如果 high-precision offline 指标更好，但训练后没有明显提升，也仍然可以写成硕士论文中的机制分析：

```text
dependency-aware packing improves structural training signal,
but transfer to broad downstream benchmarks depends on edge quality,
token budget, and model adaptation scale.
```

## 9. 当前不能提前声称的结论

当前版本还不能声称：

```text
已经证明模型长上下文能力提升
已经证明优于 DataSculpt-original
已经完成人工 edge precision audit
已经完成 Qwen tokenizer 正式 packing freeze
已经完成 8-GPU training result
```

当前可以声称：

```text
代码和本地 pipeline 已经跑通
repo_main_v1 数据集和 split 已构建
high-precision dependency graph 已实现
same-docs order ablation 已实现
simple-tokenizer offline 结果显示 dependency order 信号明显强于 random/reverse order
服务器训练入口和评测入口已准备好
```

## 10. 交付前检查

把项目交给导师前建议确认：

```bash
PYTHONPATH=src python -m pytest -q
python -m compileall -q src scripts
git diff --check
git status --short
```

如果通过 GitHub 交付，记住：

```text
代码、配置、文档会进入 git；
data/raw、data/processed、outputs 通常不会进入 git；
大数据和模型 checkpoint 需要单独传输或在服务器重建。
```
