# 实验设计方案：低资源长上下文适配中的 Dependency-Aware Packing

## 0. 一句话概括

本项目研究的问题是：

> 在双 RTX 4080 这类低资源环境下，已有大语言模型进行长上下文适配时，训练数据的 packing 方式是否会显著影响模型的长上下文能力？如果会，能否设计一种比随机拼接、长度优先拼接、BM25/检索拼接、语义相似拼接更有效的 **Dependency-Aware Packing** 方法？

本项目不是单纯比较几种 packing 方法，而是提出一种新的数据构造目标：

> 不只是把“相似文档”放在同一个窗口里，而是把“存在可学习依赖关系”的文档放在同一个长上下文窗口里。

最终目标是在低资源训练条件下，用可复现的小规模实验验证：

1. packing 策略确实会影响长上下文适配效果；
2. 只依赖语义相似或检索相关并不充分；
3. dependency-aware packing 能在固定训练预算下更有效提升模型的长上下文利用能力。

---

## 1. 研究背景与动机

长上下文能力是当前大语言模型的重要能力之一。已有工作通常从三条线推进：

### 1.1 模型/训练策略线

例如 LongQLoRA、LongLoRA、PoSE 等，关注如何用较低训练成本把已有模型扩展到更长上下文。

它们回答的问题是：

> 如何让模型低成本适配更长的 context length？

这类方法通常使用 QLoRA、位置插值、稀疏注意力、position id 变换等手段降低显存和训练成本。

### 1.2 数据 packing / 数据组织线

例如 DataSculpt、SPLiCe、ICLM 等，关注如何把原始文档组织成长上下文训练样本。

它们回答的问题是：

> 哪些文档应该被放进同一个 context window？

DataSculpt 主要考虑语义相关性、窗口内文档数量、文档完整性和计算效率；SPLiCe 使用 BM25、Contriever 或代码 repo 结构把相关文档组织到同一个长上下文样本中；ICLM 使用 kNN/retrieval 关系排列文档。

### 1.3 数据质量 / 长程依赖线

例如 LADM、EntropyLong 等，关注如何判断一个长上下文样本是否真的包含可学习的长程依赖。

它们回答的问题是：

> 一个长上下文样本内部是否真的有值得模型学习的 long-range dependency？

这类方法通常使用 attention、loss reduction、entropy/uncertainty 等信号衡量长上下文数据质量。

---

## 2. 本项目的切入点

现有方法已经证明：

- 长上下文训练不是简单把文档拼长；
- 语义相关、结构相关的数据组织方式可以优于随机拼接；
- 低资源条件下可以通过 QLoRA 等方法进行长上下文适配。

但仍有一个问题没有被充分系统研究：

> 在低资源训练预算下，context window 中的 token 非常宝贵。此时，packing 不应该只追求“填满窗口”或“语义相似”，而应该优先构造存在可学习依赖关系的训练样本。

因此，本项目关注：

> **低资源长上下文适配中的 dependency-aware data packing。**

---

## 3. 核心研究问题

本项目围绕三个问题展开：

### RQ1：低资源长上下文适配中，packing 策略是否会影响模型效果？

在相同模型、相同训练 token 数、相同训练步数、相同 context length 和相同优化配置下，只改变 packing 策略，观察模型在长上下文任务上的表现变化。

### RQ2：现有 packing 方法的不足是什么？

随机 packing 只追求简单拼接；长度优先 packing 只追求容量利用率；BM25/SPLiCe-style packing 主要追求检索相关；DataSculpt-lite 主要追求语义相似和窗口完整性。

这些方法都不直接回答：

> 文档 A 放在文档 B 前面，是否真的能帮助模型理解或预测 B？

### RQ3：Dependency-Aware Packing 是否能带来更好的长上下文适配效率？

如果 dependency-aware packing 能在同样训练预算下带来更好的 passkey retrieval、needle-in-a-haystack、long QA、repo-level code completion 或 validation loss，那么说明数据组织方式不仅影响大规模训练，也影响低资源适配。

---

## 4. 核心创新点

### 创新点 1：从 Similarity-Aware Packing 转向 Dependency-Aware Packing

已有方法多关注：

- 文档是否语义相似；
- 文档是否由检索方法召回；
- 文档是否属于同一个 repo / topic；
- 文档是否能填满 context window。

本项目进一步关注：

> 文档之间是否存在可学习的上下文依赖。

也就是说，两个文档不只是“相关”，而是一个文档能帮助模型理解、预测或解释另一个文档。

示例：

普通语义相似 packing 可能构造：

```text
CUDA 博客 + Attention 博客 + GPU 博客 + Transformer 博客
```

Dependency-aware packing 更希望构造：

```text
FlashAttention 原理说明 + CUDA kernel 实现 + benchmark 结果 + 优化讨论
```

后者更像一个有结构的知识链，更适合训练模型在长上下文中建立跨文档联系。

---

### 创新点 2：提出 Dependency Score，用于衡量文档间上下文增益

本项目不只用 embedding cosine similarity 衡量文档关系，而是引入 dependency score。

可以设计三个层次：

#### 2.1 结构依赖分数

适合代码 repo 或结构化技术文档。

示例：

```text
README -> source file
source file -> test file
config -> training script
imported file -> importing file
function definition -> function call
background section -> method section
method section -> experiment section
```

这些关系不是普通语义相似，而是结构性依赖。

#### 2.2 Loss-reduction dependency

定义为：

```text
dependency(A -> B) = loss(B alone) - loss(A + B)
```

含义：

> 如果把 A 放在 B 前面后，模型预测 B 的 loss 降低，说明 A 对 B 有上下文帮助。

这个定义比 embedding 相似度更接近“训练有用性”。

为了降低成本，不对所有文档两两计算，只对 embedding/BM25 召回的 top-k 候选计算。

#### 2.3 Attention / uncertainty dependency

借鉴 LADM / EntropyLong：

如果加入远距离上下文后，模型对后文 token 的预测不确定性降低，或者产生稳定、有意义的跨文档 attention，则说明该样本存在长程依赖。

第一阶段可以先实现结构依赖 + loss-reduction dependency，attention/uncertainty 作为后续增强。

---

### 创新点 3：在低资源长上下文适配中系统验证 packing 策略

DataSculpt 等工作在大规模资源下验证了数据组织的重要性，但本项目关注低资源场景：

```text
硬件：双 RTX 4080
模型：已有 7B 模型的 QLoRA 长上下文适配
context length：优先 8K，后续尝试 12K/16K
训练步数：500-1000 steps 起步
训练数据：固定 token budget
唯一变量：packing 方法
```

这使实验更可复现，也更贴近普通实验室或学生项目的实际条件。

---

## 5. 方法设计：Dependency-Aware Packing

### 5.1 输入与输出

输入是一批原始文档：

```json
{"docid": "doc_001", "content": "...", "metadata": {...}}
{"docid": "doc_002", "content": "...", "metadata": {...}}
```

输出是一批固定长度的长上下文训练样本：

```json
{
  "sample_id": "dep_000001",
  "method": "dependency_aware",
  "docids": ["doc_001", "doc_017", "doc_023"],
  "content": "...拼接后的文本...",
  "stats": {
    "tokens": 8192,
    "num_docs": 3,
    "semantic_score": 0.72,
    "dependency_score": 0.61,
    "truncation_rate": 0.01
  }
}
```

所有 packing 方法最终都输出相同格式，以便公平训练。

---

### 5.2 方法整体流程

```text
Step 1：文档预处理
- 清洗文本
- 切分过长文档
- 统计 token 长度
- 提取 metadata，例如 repo、文件路径、标题、章节、引用等

Step 2：候选文档召回
- 对每个 anchor document，使用 BM25 / dense embedding / structure relation 找 top-k 候选文档

Step 3：依赖关系打分
- 对候选文档对计算 dependency score
- 可包含结构依赖、loss reduction、attention/uncertainty 信号

Step 4：构建文档依赖图
- 节点是文档
- 边表示 A -> B 的依赖强度

Step 5：Dependency-aware greedy packing
- 从 anchor document 出发
- 优先加入与当前窗口存在强依赖边的文档
- 同时控制容量、截断、冗余和语义一致性

Step 6：输出固定长度训练样本
- 每条样本长度为 4K / 8K
- 记录 docids 和统计指标
```

---

### 5.3 打分函数设计

为了兼容 Markdown/飞书，这里用纯文本表达：

```text
Score(d, w) =
semantic_score(d, w)
+ dependency_score(d, w)
+ structure_score(d, w)
+ capacity_score(d, w)
- truncation_penalty(d, w)
- redundancy_penalty(d, w)
```

各项含义：

```text
semantic_score:
当前文档 d 和窗口 w 已有内容是否属于相近主题。

dependency_score:
当前文档 d 和窗口 w 中已有文档是否存在上下文增益关系。

structure_score:
当前文档 d 和窗口 w 中已有文档是否存在结构关系，例如同 repo、同目录、import 依赖、README-code 关系等。

capacity_score:
当前文档 d 放入窗口后是否能较好利用剩余 context 长度。

truncation_penalty:
当前文档 d 放入窗口后是否会被截断，尤其是否会截断重要部分。

redundancy_penalty:
当前文档 d 是否与窗口已有内容高度重复。
```

---

### 5.4 和 DataSculpt 的区别

DataSculpt 的核心是：

```text
语义相关性 + 同质性 + 文档完整性 + 计算效率
```

本项目的区别是：

```text
不仅考虑语义相似，还显式考虑“文档之间是否存在可学习依赖关系”。
```

对比：

| 方法 | 核心目标 | 局限 |
|---|---|---|
| Random Packing | 随机拼接 | 内容混乱，噪声大 |
| Length-Aware Packing | 尽量填满窗口 | 只考虑长度，不考虑学习价值 |
| SPLiCe / BM25 Packing | 检索相关文档 | 关键词相关不一定有上下文依赖 |
| DataSculpt-lite | 语义相似 + 贪心装箱 | 相似不等于有训练增益 |
| Dependency-Aware Packing | 最大化可学习文档依赖 | 本项目方法 |

---

## 6. 实验设计

### 6.1 总体原则

核心控制变量：

```text
模型相同
训练数据来源相同
训练 token 数相同
context length 相同
训练步数相同
LoRA/QLoRA 配置相同
优化器配置相同
唯一变化：packing 方法
```

这样才能说明性能差异来自 packing 策略。

---

## 7. 数据选择

### 7.1 推荐主数据：代码 repo 数据

优先建议选择代码 repo 数据，因为 dependency 最清楚，容易定义和解释。

可用关系包括：

```text
README -> source file
source file -> test file
config -> training script
imported file -> importing file
function definition -> function call
same repo
same directory
same module
```

优势：

- dependency 可解释；
- 与 SPLiCe 的 repo structured packing 可形成自然对比；
- 评测可以用 RepoBench / LCC / 代码补全任务；
- 适合体现“依赖关系不是简单语义相似”。

### 7.2 备选数据：技术文档 / arXiv / Wikipedia

如果代码数据处理成本过高，可以使用技术文档或论文摘要。

可用关系包括：

```text
background -> method
definition -> example
method -> experiment
paper -> cited paper
survey -> original paper
```

评测可以使用：

```text
passkey retrieval
mini needle-in-a-haystack
long QA
summarization
Qasper / GovReport / LongBench 子集
```

---

## 8. Baseline 设计

### Baseline 1：Random Packing

随机打乱文档，然后顺序拼入窗口。

作用：

> 最弱 baseline，用于证明简单拼接不够。

---

### Baseline 2：Length-Aware Packing

只考虑长度，使用 first-fit / best-fit，尽量提高窗口 token 利用率。

作用：

> 证明“填满窗口”不等于“高质量 packing”。

---

### Baseline 3：BM25 / SPLiCe-style Packing

以一个 anchor document 为中心，用 BM25 或检索方法召回相关文档并拼接。

作用：

> 代表 retrieval-based packing。

---

### Baseline 4：DataSculpt-lite Semantic Packing

轻量实现 DataSculpt 思路：

```text
1. 对文档做 embedding
2. 使用 FAISS/KMeans 聚类
3. 在 cluster 内部按长度排序
4. 使用语义相似度 + 剩余容量 + 截断惩罚进行贪心 packing
```

作用：

> 代表 semantic similarity based packing。

---

### Proposed：Dependency-Aware Packing

本项目提出的方法：

```text
1. 先召回候选相关文档
2. 计算依赖分数
3. 构建文档依赖图
4. 在固定 context window 内最大化依赖边覆盖
5. 同时控制容量、截断和冗余
```

作用：

> 证明 dependency-aware objective 优于单纯长度、检索相关或语义相似。

---

## 9. 训练设置

### 9.1 主训练框架

优先使用 LongQLoRA 或 LongLoRA-QLoRA。

推荐先使用 LongQLoRA，因为它更贴近低资源长上下文适配场景。

### 9.2 推荐配置

| 项目 | 推荐设置 |
|---|---|
| GPU | 2×RTX 4080 16GB |
| Base model | 7B 级已有模型 |
| 训练方式 | QLoRA / LongQLoRA |
| context length | 8K |
| precision | 4-bit |
| micro batch | 1 |
| gradient accumulation | 根据显存调整 |
| training steps | pilot 100-300 steps；main 500-1000 steps |
| token budget | pilot 10M-20M；main 50M-100M |
| optimizer | 与 LongQLoRA 默认设置保持一致 |
| LoRA rank | 固定，例如 r=8 或 r=16 |
| 目标 | 比较不同 packing 方法对长上下文适配的影响 |

### 9.3 训练组别

| 组别 | 数据 |
|---|---|
| G1 | Random Packing |
| G2 | Length-Aware Packing |
| G3 | BM25 / SPLiCe-style Packing |
| G4 | DataSculpt-lite Semantic Packing |
| G5 | Dependency-Aware Packing |

每组训练配置完全相同，只改变训练数据。

---

## 10. 评测设计

### 10.1 数据构造质量指标

训练前先比较不同 packing 方法生成的数据本身。

| 指标 | 说明 |
|---|---|
| token utilization | 窗口是否被充分利用 |
| avg num docs | 平均每个窗口包含多少文档 |
| truncation rate | 文档被截断比例 |
| semantic similarity | 窗口内部语义一致性 |
| embedding variance | 窗口内主题是否分散 |
| dependency score | 是否存在可学习依赖 |
| redundancy score | 是否重复堆叠相似内容 |
| repo/topic consistency | 是否保持结构一致 |

预期：

```text
Dependency-Aware Packing 的 dependency score 应该最高；
token utilization 不应明显低于其他方法；
truncation rate 应保持可控；
semantic similarity 不一定最高，但应该足够稳定。
```

---

### 10.2 训练过程指标

| 指标 | 说明 |
|---|---|
| training loss | 训练收敛速度 |
| validation loss | 泛化能力 |
| long-context validation loss | 长上下文建模能力 |
| tokens/sec | 训练效率 |
| peak memory | 低资源可行性 |
| loss curve stability | 训练是否稳定 |

关键观察：

```text
在相同 token budget 下，Dependency-Aware Packing 是否收敛更快，或者 long-context validation loss 更低。
```

---

### 10.3 长上下文能力评测

#### 必做轻量评测

```text
1. Passkey Retrieval
2. Mini Needle-in-a-Haystack
```

这两个任务实现简单，适合快速验证长上下文检索能力。

#### 代码方向评测

如果主数据选择代码 repo：

```text
1. RepoBench 子集
2. LCC 小样本
3. 跨文件代码补全
4. README/config/source/test 文件关系补全
```

#### 通用文本方向评测

如果主数据选择技术文档 / arXiv / Wikipedia：

```text
1. LongBench 子集
2. Qasper
3. GovReport
4. Multi-document QA
5. Summarization
```

---

## 11. 消融实验

为了证明方法不是“多个分数随便相加”，需要做消融。

| 消融实验 | 目的 |
|---|---|
| 去掉 dependency_score | 验证依赖建模是否有效 |
| 去掉 semantic_score | 验证语义一致性是否必要 |
| 去掉 structure_score | 验证结构关系是否有效 |
| 去掉 redundancy_penalty | 验证去重是否有用 |
| 只用 dependency_score | 看 dependency 是否可以单独发挥作用 |
| post-filter vs online packing | 比较“先生成再筛选”和“packing 阶段直接优化” |
| 4K vs 8K vs 12K | 看方法是否随 context length 变化 |
| 10M vs 50M vs 100M tokens | 看不同训练预算下方法是否稳定 |

最关键的消融：

```text
DataSculpt-lite
vs
DataSculpt-lite + dependency_score
```

如果这个对比有提升，就能直接说明 dependency-aware 目标有贡献。

---

## 12. 预期结果与解释

### 12.1 预期结果

理想情况下：

```text
Dependency-Aware Packing 在 passkey retrieval、needle、RepoBench、long QA 等长上下文任务上优于 baseline；
在普通 validation loss 上不一定全面领先；
训练效率不应明显低于 DataSculpt-lite / BM25 baseline；
数据统计上 dependency score 更高，截断率和冗余率可控。
```

### 12.2 合理解释

如果结果符合预期，可以解释为：

> 在低资源长上下文适配中，训练 token 预算有限，因此 context window 的组织质量非常关键。单纯语义相似或检索相关不能保证窗口内部存在有用的长程依赖，而 dependency-aware packing 能更有效地构造具有上下文增益的训练样本。

### 12.3 可能出现的负结果

如果 dependency-aware packing 没有明显提升，需要检查：

```text
1. dependency score 是否定义过弱；
2. 训练 token 是否太少，模型没有学到差异；
3. 评测任务是否无法体现跨文档依赖；
4. packing 输出是否和 DataSculpt-lite 差异不大；
5. dependency 计算是否引入了噪声。
```

---

## 13. 最小可行实验版本

为了尽快判断方向是否可行，建议先做最小闭环。

### 13.1 Mini 实验设置

| 项目 | 设置 |
|---|---|
| 数据 | 代码 repo 小数据集 |
| 模型 | 7B QLoRA 或更小模型 |
| context length | 4K 或 8K |
| 训练步数 | 100-300 steps |
| packing 方法 | Random / BM25 / DataSculpt-lite / Dependency-Aware |
| 评测 | validation loss + passkey + RepoBench 小子集 |
| 目标 | 看是否有趋势 |

### 13.2 Mini 实验判断标准

如果出现以下现象，说明值得继续：

```text
Dependency-aware 的数据统计明显不同；
dependency score 明显高；
截断率不高；
训练 loss 不崩；
long-context mini evaluation 有提升趋势。
```

---

## 14. 完整实验版本

如果 mini 实验有效，再扩大到主实验。

### 14.1 主实验设置

| 项目 | 设置 |
|---|---|
| 数据 | 代码 repo / 技术文档 |
| 模型 | 7B QLoRA |
| context length | 8K |
| token budget | 50M-100M |
| steps | 500-1000 |
| baselines | Random / Length-Aware / BM25 / DataSculpt-lite / Dependency-Aware |
| 评测 | passkey, needle, RepoBench/LongBench 子集, validation loss |
| 消融 | dependency_score, structure_score, semantic_score |

---

## 15. Storyline：向导师汇报时可以这样讲

### 15.1 开场

长上下文能力不仅取决于模型结构和位置编码，也取决于训练时 context window 内部的数据组织方式。DataSculpt、SPLiCe 等工作已经说明，随机拼接并不是最优的长上下文训练数据构造方式。

但现有方法大多关注语义相似、检索相关或窗口利用率。对于低资源长上下文适配来说，训练 token 和显存预算都很有限，因此每个 context window 中是否包含“可学习的跨文档依赖”更加关键。

### 15.2 问题定义

本项目研究：

> 在双 RTX 4080 这种低资源环境下，如何通过更好的 packing 数据，让已有模型更有效地适配长上下文？

### 15.3 方法

我们提出 Dependency-Aware Packing。它不只是把相似文档放在一起，而是在构造窗口时显式考虑文档之间的上下文依赖关系。

具体来说，我们先用 BM25/embedding/结构信息召回候选文档，再用结构关系、loss reduction 或 attention/uncertainty 信号估计依赖强度，最后在固定 context 长度内最大化依赖边覆盖，同时控制截断和冗余。

### 15.4 实验

实验使用 LongQLoRA/LongLoRA-QLoRA 作为低资源长上下文适配训练框架，在双 RTX 4080 上进行 7B 模型 8K context 的小规模训练。

对比方法包括：

```text
Random Packing
Length-Aware Packing
BM25 / SPLiCe-style Packing
DataSculpt-lite Semantic Packing
Dependency-Aware Packing
```

所有组保持相同模型、相同训练 token、相同训练步数和相同训练配置，只改变 packing 方法。

### 15.5 预期贡献

本项目预期贡献包括：

```text
1. 提出 dependency-aware packing 目标，从“相似文档拼接”推进到“可学习依赖关系构造”。
2. 设计低成本 dependency score，用于指导长上下文训练样本构造。
3. 在低资源 GPU 环境中系统验证 packing 策略对长上下文适配的影响。
4. 提供可复现的小规模实验框架，为普通实验室研究长上下文数据构造提供参考。
```

---

## 16. 时间规划

### 第 1 阶段：数据与 packing 实现

时间：1-2 周

任务：

```text
1. 选定数据集
2. 实现统一 jsonl 格式
3. 实现 Random / Length-Aware / BM25 / DataSculpt-lite
4. 实现 Dependency-Aware Packing 初版
5. 输出 packing 统计指标
```

### 第 2 阶段：小规模训练验证

时间：1-2 周

任务：

```text
1. 跑通 LongQLoRA 或 LongLoRA-QLoRA
2. 先用 4K context 做 100-300 steps pilot
3. 比较不同 packing 的 loss 曲线和简单 passkey 结果
```

### 第 3 阶段：主实验

时间：2-4 周

任务：

```text
1. 扩展到 8K context
2. 增加 token budget 和训练 steps
3. 跑完整 baseline
4. 做 needle / RepoBench / LongBench 子集评测
```

### 第 4 阶段：消融和写作

时间：1-2 周

任务：

```text
1. 做 dependency_score / semantic_score / structure_score 消融
2. 做 case study
3. 整理论文故事和实验表格
4. 写汇报或论文初稿
```

---

## 17. 风险与备选方案

### 风险 1：双 4080 跑 7B-8K 仍然困难

备选：

```text
1. 先降到 4K context
2. 使用更小模型，例如 3B / 1.5B
3. 降低训练步数，只做 pilot
4. 使用 gradient checkpointing、QLoRA、ZeRO
```

### 风险 2：Dependency score 计算成本高

备选：

```text
1. 只对 top-k 候选计算
2. 先使用结构依赖分数
3. 后续再加入 loss reduction
4. attention/uncertainty 分数作为增强实验
```

### 风险 3：训练效果差异不明显

备选：

```text
1. 加强评测任务的长程依赖属性
2. 使用代码 repo 数据而不是普通网页数据
3. 增加 token budget
4. 做更多数据统计和 case study
5. 对比 DataSculpt-lite + dependency_score 的直接提升
```

### 风险 4：方法看起来像多个模块拼接

解决：

```text
必须强调核心目标是 dependency-aware packing。
LADM / EntropyLong / DataSculpt / SPLiCe 都只是参考或 baseline。
论文贡献要落在新的 packing objective 和构造算法上。
```

---

## 18. 最终建议

最推荐的第一版实验不要做得过大。

建议最小闭环是：

```text
数据：代码 repo 小数据集
模型：7B QLoRA 或 3B QLoRA
context：4K -> 8K
方法：Random / BM25 / DataSculpt-lite / Dependency-Aware
训练：LongQLoRA 小步数
评测：validation loss + passkey + RepoBench 小子集 + packing statistics
```

如果这条线跑通，再扩展到：

```text
Length-Aware baseline
LongAlign-style loss weighting
LiteLong-style topic/BM25 baseline
attention/uncertainty dependency score
8K/12K context
更多 LongBench 子集
```

---

## 19. 最后总结

本项目的核心不是“比较 packing 方法”，而是提出一种新的 packing 目标：

> 在低资源长上下文适配中，context window 不应该只是被填满，也不应该只是由语义相似文档组成，而应该尽可能包含存在可学习依赖关系的文档组合。

因此，本项目的完整 storyline 是：

```text
背景：长上下文适配需要高质量长序列训练数据。
问题：低资源条件下 token budget 有限，随机/语义相似 packing 可能浪费上下文窗口。
方法：提出 Dependency-Aware Packing，显式优化文档间上下文依赖。
实验：在双 RTX 4080 上使用 LongQLoRA/LongLoRA 进行 7B 模型 8K context 适配，对比多种 packing。
结论：更好的 packing 可以提升低资源长上下文适配效率，dependency-aware 目标比单纯相似度更有价值。
```
