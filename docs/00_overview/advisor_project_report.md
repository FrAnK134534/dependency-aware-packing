# 项目设想与导师汇报：Dependency-Aware Packing for Long-Context Adaptation

## 1. 一句话概括

我想研究的问题是：

> **在长上下文适配中，训练样本不应该只是被拼长，而应该被组织成包含“可学习上下文依赖”的长上下文窗口。**

现有很多 packing 方法主要考虑：

```text
窗口是否被填满；
文档是否同主题；
BM25 是否检索相关；
embedding 是否语义相似；
文档是否完整、是否截断少。
```

这些目标都重要，但它们没有直接回答一个问题：

> **A 放在 B 前面，是否真的能帮助模型理解或预测 B？**

因此，我希望提出一个新的数据构造目标：

> **Dependency-Aware Packing：以“上下文依赖/上下文增益”为目标来组织长上下文训练样本。**

这个方法不是单纯面向代码仓库的规则方法，而是一个更一般的 packing 框架：

```text
有显式结构时：利用结构依赖，例如代码仓库、论文引用、网页链接；
没有显式结构时：利用隐式依赖，例如 loss reduction / context gain。
```

代码仓库多源上下文会作为第一阶段主实验场景，因为它依赖关系清晰、可解释、容易构造 controlled experiment。后续再用弱结构或隐式依赖数据做泛化验证。

---

## 2. 为什么做 Packing 策略优化

长上下文模型能接收 8K、16K 甚至更长输入，但这不等于模型真的学会了使用长上下文。

训练时如果只是把无关内容拼成长文本，模型看到的是长序列，但不一定能学到跨文档关系。

例如下面两个训练窗口长度可能一样：

```text
窗口 A：
随机网页 1 + 随机代码文件 + 随机论文段落 + 随机日志

窗口 B：
README + config + model.py + train.py + test_model.py
```

窗口 A 只是“长”。  
窗口 B 中的内容存在明确关系：

```text
README 说明项目；
config 影响 train.py；
model.py 定义模型；
train.py 调用 model.py；
test_model.py 验证 model.py。
```

模型在窗口 B 中更可能学到：

```text
前文如何帮助后文；
配置如何对应脚本；
定义如何被调用；
说明文档如何映射到实现；
测试如何反映代码预期行为。
```

所以这个课题的核心不是“把上下文做长”，而是：

> **让长上下文窗口里的内容更有学习价值。**

---

## 3. 与已有方法的关系

这个方向可以和 DataSculpt、BM25 packing、semantic packing 等已有方法形成清晰关系。

### 3.1 现有方法主要优化什么

常见 packing 策略可以概括为：

| 方法 | 主要目标 | 局限 |
|---|---|---|
| Random Packing | 随机拼接，作为最弱 baseline | 内容关系弱 |
| Length-Aware Packing | 尽量填满窗口 | 只关注容量，不关注学习价值 |
| Same-Repo / Same-Topic Packing | 粗粒度相关 | 同仓库/同主题不等于有依赖 |
| BM25 Packing | 关键词相关 | 词面相关不等于上下文帮助 |
| Semantic Packing | embedding 语义相似 | 语义相似不等于依赖 |
| DataSculpt-style Packing | 语义相关、完整性、同质性、效率 | 没有显式优化上下文增益 |

DataSculpt 的价值在于它证明了：

> 长上下文训练数据不能简单随机拼接，需要系统的数据组织策略。

我的工作可以借鉴它的 coarse-to-fine 思路：

```text
先用语义/BM25/结构信息召回候选集合；
再在候选集合内做更细粒度的 packing 优化。
```

但我的目标函数不同：

> DataSculpt 主要关注相似性、完整性、效率；  
> 我进一步关注一个窗口内部是否存在可学习的上下文依赖。

### 3.2 我的核心区别

已有方法通常问：

```text
这些文档像不像？
这些文档是否来自同一主题？
这些文档能否填满窗口？
```

我的方法问：

```text
A 是否能帮助理解或预测 B？
```

这就是 Dependency-Aware Packing 的核心。

---

## 4. Dependency 的定义：显式依赖与隐式依赖

为了避免方法只局限于代码仓库，我把 dependency 分成两类。

### 4.1 显式结构依赖

当数据有清晰结构时，可以用结构关系作为低成本、可解释的 dependency signal。

典型场景包括：

```text
代码仓库：import、source-test、README-code、config-script、issue-file、commit-file
论文数据：citation、section hierarchy、background-method、method-experiment
网页数据：hyperlink、same domain、navigation hierarchy、anchor text
技术文档：definition-example、API doc-usage example、tutorial-code
```

以代码仓库为例：

| 依赖关系 | 含义 |
|---|---|
| source -> test | 测试文件验证源码行为 |
| README/docs -> source | 文档解释或说明代码 |
| config -> script | 配置文件影响运行或训练脚本 |
| API doc -> usage example | API 文档帮助理解调用方式 |
| issue/PR -> changed file | issue 或 PR 描述修改需求 |
| commit message -> changed file | commit message 解释修改原因 |
| source -> source | import / call / same module |

这些关系可以组成一个 dependency graph：

```text
节点：文档、文件、段落、代码片段
边：A -> B，表示 A 可能帮助理解或预测 B
权重：依赖强度
```

例如：

```text
import relation: 1.0
source-test relation: 0.9
config-script relation: 0.7
README-code relation: 0.6
docs-code relation: 0.6
same-directory: 0.25
same-repo: 0.1
```

这里 same-repo 只能作为弱信号，因为它太粗。如果权重过高，方法会退化成 same-repo packing。

### 4.2 隐式上下文依赖

对于很多普通文本数据，未必有明确结构边。

这时可以使用更通用的定义：

```text
dependency(A -> B) = loss(B alone) - loss(A + B)
```

含义是：

> 如果把 A 放在 B 前面后，模型预测 B 的 loss 降低，说明 A 对 B 有上下文帮助。

这也可以叫：

```text
context gain(A -> B)
```

这个定义更通用，适用于：

```text
技术文档；
论文段落；
Wikipedia / linked pages；
multi-hop QA support documents；
普通多文档集合。
```

问题是全量两两计算太贵，所以我会采用两阶段策略：

```text
第一步：用 BM25 / embedding / topic / metadata 召回 top-k 候选；
第二步：只对候选 pair 计算 loss reduction / context gain；
第三步：用这个依赖分数指导 packing。
```

这样方法就不是代码仓库专用，而是：

> 有结构时用显式结构依赖；没有结构时用模型增益估计隐式依赖。

---

## 5. 方法框架

整体方法可以抽象成三步。

### 5.1 候选召回

先缩小搜索空间，避免全局两两比较。

候选来源可以是：

```text
same repo / same topic
BM25 检索
embedding similarity
metadata relation
section / hyperlink / citation relation
```

这一步不直接决定 packing，只是找可能相关的候选。

### 5.2 Dependency Scoring

对候选文档对计算 dependency score。

显式场景：

```text
dependency_score(A -> B)
= structure_relation_score(A -> B)
```

隐式场景：

```text
dependency_score(A -> B)
= loss(B alone) - loss(A + B)
```

混合场景：

```text
dependency_score(A -> B)
= alpha * structure_score
+ beta * semantic_score
+ gamma * loss_reduction_score
```

第一版不一定要全部做完，可以按阶段推进：

```text
v1：结构依赖
v2：结构依赖 + BM25/semantic candidate
v3：结构/semantic candidate + loss-reduction reranking
```

### 5.3 Dependency-Aware Packing

给定一个 context window，贪心加入最有依赖价值的文档。

基本流程：

```text
1. 选择一个 anchor document
2. 找候选文档
3. 计算候选与当前窗口的 dependency score
4. 优先加入对窗口已有内容存在强依赖的文档
5. 控制 max tokens、截断率、冗余度
6. 输出 packed training sample
```

打分形式可以是：

```text
score(candidate, window)
= dependency_score(candidate, window)
+ capacity_bonus
- truncation_penalty
- redundancy_penalty
```

其中 dependency 是主目标，capacity 和 integrity 是约束项。

---

## 6. 数据集构建

数据集构建分成两个层次：主实验数据和辅助泛化数据。

### 6.1 主实验：代码仓库多源上下文

主实验建议使用代码仓库多源数据，而不是只使用纯代码文件。

原因是：

```text
真实软件理解不只依赖 source code；
README、docs、config、tests、examples、issues、commits 都是上下文；
多源材料之间有清晰依赖关系；
更适合验证 dependency-aware packing。
```

建议收集：

```text
source code
tests
README
docs
config files
examples
scripts
issue / PR descriptions
commit messages
benchmark logs
API usage examples
```

第一阶段最小可行数据：

```text
source code
tests
README
docs
config
examples/scripts
```

issue、PR、commit、benchmark logs 可以作为第二阶段增强。

统一 JSONL 格式：

```json
{
  "docid": "repo_a:src/train.py",
  "content": "...",
  "metadata": {
    "repo": "repo_a",
    "path": "src/train.py",
    "language": "python",
    "source_type": "source",
    "license": "mit"
  }
}
```

依赖边文件：

```json
{
  "repo": "repo_a",
  "source_docid": "repo_a:config/train.yaml",
  "target_docid": "repo_a:src/train.py",
  "relation": "config_script",
  "weight": 0.7
}
```

数据 split 必须按 repo 划分：

```text
train repos
validation repos
test repos
```

不能按文件随机切分，否则同一个 repo 的信息会同时出现在训练和测试里，造成数据泄漏。

### 6.2 辅助实验：弱结构或隐式依赖数据

为了回应“是否只能用于代码仓库”的问题，可以增加一个辅助场景。

可选数据：

```text
技术文档：API docs、tutorials、usage examples
论文数据：section、citation、Qasper-style QA
Wikipedia / multi-hop QA：linked pages、support documents
```

这一部分不一定一开始做完整大规模训练，可以先做：

```text
packing quality analysis
context gain 抽样分析
小规模训练或验证
```

它的作用是证明：

> dependency-aware objective 不依赖代码结构本身；代码结构只是显式依赖的一种实现。

### 6.3 数据处理流程

整体流程：

```text
1. 收集 repo 或文档集合
2. 过滤 license 不清晰、太小、太乱的数据
3. 提取 source_type、repo、path、language 等 metadata
4. 对长文件进行切分
5. 构建 dependency_edges.jsonl
6. 按 repo 或文档集合划分 train/val/test
7. 对每个 split 生成不同 packing 版本
8. 统计 packing 指标
9. 将 packed JSONL 用于训练和评测
```

---

## 7. 实验设计

实验分为四层。

### 7.1 Packing 数据质量实验

不训练模型，先比较不同 packing 方法生成的数据。

比较方法：

```text
Random Packing
Length-Aware Packing
Same-Repo / Same-Topic Packing
BM25 Packing
Semantic / DataSculpt-lite Packing
Dependency-Aware Packing
Hybrid: Semantic + Dependency Packing
```

指标：

```text
token utilization
truncation rate
avg docs per window
order-aware dependency score
dependency edge coverage
semantic similarity
redundancy
document integrity
```

这一层要证明：

> dependency-aware packing 不是简单填满窗口，而是提高了窗口内部的依赖密度。

### 7.2 主训练实验

在 8 卡 NVLink 服务器上训练。

主设置：

```text
model: 7B
context length: 8K
training method: LoRA / QLoRA
token budget: 50M / 100M / 200M
hardware: single-node 8-GPU NVLink
```

控制变量：

```text
base model 相同
tokenizer 相同
context length 相同
training token budget 相同
optimizer 相同
learning rate 相同
LoRA/QLoRA config 相同
effective batch size 相同
validation set 相同
evaluation protocol 相同
```

唯一主要变量：

```text
packing method
```

### 7.3 消融实验

消融用于证明提升来自 dependency modeling，而不是规则堆叠。

建议消融：

```text
Dependency-Aware without import/source-source relation
Dependency-Aware without source-test relation
Dependency-Aware without README/docs relation
Dependency-Aware without config-script relation
Dependency-Aware without same-repo weak prior
BM25 vs BM25 + structure reranking
Semantic vs Semantic + dependency reranking
Only-code vs multi-source context
Explicit dependency vs implicit loss-reduction dependency
```

最关键对比：

```text
Same-Repo vs Dependency-Aware
DataSculpt-lite vs Dependency-Aware
DataSculpt-lite vs Hybrid Dependency + Semantic
```

### 7.4 泛化验证实验

为了避免方法看起来只适用于代码仓库，需要有一个弱结构或隐式依赖场景。

可以先做轻量版本：

```text
选择技术文档或论文段落数据；
用 BM25/embedding 召回候选；
计算 loss-reduction dependency；
比较 semantic packing 和 loss-reduction dependency packing；
报告 packing statistics 和 context gain。
```

如果资源允许，再做小规模训练。

---

## 8. 如何比较不同 Packing 策略优劣

比较分为数据层面和模型层面。

### 8.1 数据层面

理想结果：

```text
Dependency-Aware:
  dependency score 更高；
  edge coverage 更高；
  token utilization 接近 baseline；
  truncation rate 可控；
  redundancy 不高；
  semantic similarity 不失控。
```

这说明方法构造出了更高依赖密度的训练样本，而且没有严重牺牲窗口利用率和完整性。

### 8.2 模型层面

训练后评测：

```text
long-context validation loss
dependency-sensitive validation loss
context gain
RepoBench / cross-file retrieval
cross-file completion
passkey retrieval
needle-in-a-haystack
LongBench subset
```

其中最贴合本课题的是：

```text
context_gain(A -> B) = loss(B alone) - loss(A + B)
```

如果 dependency-aware 训练出来的模型在真实依赖边上 context gain 更高，就说明它更会利用上下文依赖。

---

## 9. 实验代码设计

当前项目会按模块化方式组织，便于在本地开发和服务器运行。

### 9.1 数据模块

目标：

```text
把原始 repo / 文档集合转换成统一 JSONL；
构建 dependency_edges.jsonl；
按 repo 或文档集合做 train/val/test split。
```

计划脚本：

```text
scripts/data/build_repo_corpus.py
scripts/data/extract_repo_metadata.py
scripts/data/build_dependency_edges.py
scripts/data/split_by_repo.py
```

### 9.2 Packing 模块

已有基础：

```text
random
length_aware
same_repo
bm25
dependency_aware
```

后续补充：

```text
semantic
datasculpt_lite
bm25_structure_rerank
semantic_dependency_rerank
loss_reduction_dependency
```

统一入口：

```bash
python scripts/run_packing.py \
  --input data/processed/train_docs.jsonl \
  --output data/processed/train_8k_dependency.jsonl \
  --method dependency_aware \
  --max-tokens 8192
```

### 9.3 指标统计模块

已有：

```text
scripts/summarize_packing.py
```

后续扩展：

```text
dependency edge coverage
order-aware dependency score
same-repo ratio
same-directory ratio
semantic similarity
redundancy
```

输出：

```text
outputs/packing_summary.csv
```

### 9.4 训练模块

目标：

```text
固定模型、token budget、context length；
只替换 packed training data；
在 8 卡服务器上跑 LoRA/QLoRA。
```

配置模板：

```text
configs/training/7b_8k_lora.yaml
configs/training/7b_8k_qlora.yaml
```

服务器脚本规划：

```text
scripts/server/run_7b_8k_lora.sh
scripts/server/run_7b_8k_qlora.sh
scripts/server/run_packing_matrix.sh
```

### 9.5 评测模块

评测配置：

```text
configs/evaluation/default.yaml
```

计划支持：

```text
RepoBench-R / C / P
cross-file completion
context gain
passkey
needle
LongBench subset
```

---

## 10. 8 卡 NVLink 部署运行设计

服务器部署目标不是一次性训练最大模型，而是建立稳定可复现实验流程。

### 10.1 推荐环境

```text
Python 3.10+
PyTorch + CUDA
transformers
datasets
accelerate
peft
bitsandbytes, 如果使用 QLoRA
flash-attn, 如果服务器 CUDA 支持
deepspeed 或 FSDP
wandb / tensorboard
```

### 10.2 运行阶段

第一阶段：环境与 packing smoke test。

```bash
PYTHONPATH=src python -m pytest -q

python scripts/run_packing.py \
  --config configs/packing/default.yaml
```

第二阶段：packing matrix。

```text
为 random / length-aware / same-repo / BM25 / DataSculpt-lite /
Dependency-Aware 生成同样 token budget 的训练数据。
```

第三阶段：7B + 8K 小预算训练。

```text
先跑 random 与 dependency-aware；
确认显存、吞吐、loss curve 正常。
```

第四阶段：完整 baseline。

```text
固定 7B + 8K + token budget；
跑所有 packing 方法；
统一评测。
```

第五阶段：扩展实验。

```text
7B + 16K；
13B + 8K；
消融；
Hybrid dependency + semantic。
```

### 10.3 每次实验必须记录

```text
git commit
packing method
dataset version
model path
context length
token budget
LoRA/QLoRA config
effective batch size
number of GPUs
precision
tokens/sec
peak memory
training hours
validation loss
evaluation results
```

这样后续论文表格可以直接从实验日志整理。

---

## 11. 预期论文产出

论文可以形成以下结果。

### 11.1 方法贡献

```text
提出 Dependency-Aware Packing 框架；
将 packing 目标从相似性/窗口利用率扩展到上下文依赖；
区分显式结构依赖和隐式模型增益依赖。
```

### 11.2 指标贡献

```text
order-aware dependency score
dependency edge coverage
weighted edge coverage
context gain
dependency-sensitive validation loss
```

### 11.3 实验贡献

```text
代码仓库多源上下文主实验；
弱结构/隐式依赖辅助实验；
DataSculpt-lite、BM25、semantic、same-repo 等强 baseline；
8 卡 NVLink 下系统训练和评测；
消融与 case study。
```

---

## 12. 给导师汇报时的简短说法

可以这样概括：

> 我现在想做的是长上下文训练数据 packing 策略优化。现有方法多按长度、检索相关或语义相似来拼接文档，但这些目标不能保证窗口内部存在真正可学习的上下文依赖。我提出 Dependency-Aware Packing，把“一个文档是否能帮助理解或预测另一个文档”作为 packing 目标。  
>
> 在有显式结构的场景，例如代码仓库，我会用 README、docs、config、source、test、issue、commit 之间的结构关系构建依赖图；在没有显式结构的普通文本中，我会用 BM25/embedding 召回候选，再用 loss reduction 或 context gain 估计隐式依赖。  
>
> 实验上，我会先比较不同 packing 方法生成的数据质量，再在 8 卡 NVLink 服务器上固定模型、context length 和 token budget，只改变 packing 方法，比较训练后长上下文表现。最终通过 packing 指标、context gain、RepoBench、cross-file completion、needle/passkey 和消融实验来验证依赖感知 packing 是否比 Random、BM25、Semantic/DataSculpt-lite 更有效。

---

## 13. 当前项目状态

当前仓库已经具备：

```text
基础 packing 框架；
random / length-aware / same-repo / BM25 / dependency-aware baseline；
packing summary 脚本；
8 卡训练配置模板；
评测配置模板；
指标定义文档；
服务器部署规划文档；
导师汇报文档。
```

下一步应做：

```text
1. 构建多源代码仓库数据预处理脚本；
2. 加入 dependency_edges.jsonl 构建流程；
3. 实现 DataSculpt-lite / semantic baseline；
4. 扩展 packing 指标统计；
5. 在服务器上跑 7B + 8K smoke training；
6. 开始完整 baseline 实验。
```
