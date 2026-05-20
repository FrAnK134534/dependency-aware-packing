# 宏观实验设计：依赖感知 Packing 策略优化

## 1. 总体判断

本课题仍然建议以 **packing 策略优化** 作为主线。

导师提供单节点 8 卡且有 NVLink 的环境后，课题定位可以从“低资源可行性验证”升级为：

> 在较充分但仍可控的训练资源下，系统研究不同数据 packing 策略对长上下文适配效果的影响。

也就是说，8 卡环境不是论文的核心创新点，而是让实验更完整、更有说服力。论文真正要回答的问题仍然是：

> 长上下文适配中，什么样的训练样本组织方式更能让模型学会利用上下文依赖？

本课题的核心方法是 **Dependency-Aware Packing**：在构造长上下文训练样本时，不只考虑长度、同仓库、检索相关或语义相似，而是显式考虑样本内部是否存在可学习的上下文依赖关系。

---

## 2. 论文核心问题

当前长上下文训练和适配面临一个重要问题：

```text
模型能接收更长输入，不代表训练数据真的教会模型如何利用长上下文。
```

如果一个 8K 或 16K 的训练窗口只是由多个弱相关甚至无关文档拼成，模型虽然看到了很长的序列，但不一定能学习到跨文档、跨文件、跨段落的信息关联。

因此，本课题研究：

> 在固定模型、固定训练预算、固定 context length 下，packing 策略是否会显著影响长上下文适配效果？

进一步研究：

> dependency-aware packing 是否比 random、length-aware、BM25、semantic/DataSculpt-lite 等方法更有效？

---

## 3. 方法定位

本课题不是做新的模型结构，也不是提出新的 LoRA/QLoRA 微调算法。

本课题关注的是 **训练数据组织方式**：

```text
给定一批原始文档或文件，如何把它们组织成更有学习价值的长上下文训练样本？
```

现有常见 packing 目标包括：

```text
Random Packing：随机拼接
Length-Aware Packing：尽量填满窗口
BM25 Packing：词面相关
Semantic Packing：语义相似
DataSculpt-like Packing：多目标数据组织，例如相关性、完整性、效率
```

本课题提出的重点是：

```text
Dependency-Aware Packing：优先拼接存在上下文依赖的内容。
```

这种依赖可以来自代码仓库，也可以来自多源技术材料。

例如：

```text
README -> source code
config -> training script
source file -> test file
API document -> usage example
issue -> changed file
commit message -> changed file
method section -> experiment section
definition -> example
```

核心思想是：

> 不是把“相似内容”放在一起，而是把“一个内容能帮助理解另一个内容”的材料放在一起。

---

## 4. 数据范围设计

第一版建议以 **代码仓库多源上下文** 为主场景。

这里的“代码仓库”不等于只使用代码文件，而是包括围绕代码项目的多种材料：

```text
source code
tests
README
docs
config files
examples
issue / PR description
commit message
benchmark logs
API usage
```

这样做有三个好处：

1. 依赖关系清晰，可解释性强。
2. 评测可以使用 RepoBench、跨文件补全等任务。
3. 相比纯代码文件，更接近真实开发场景中的长上下文使用方式。

后续如果时间允许，可以增加第二类数据，例如技术文档或论文材料，但不建议第一阶段就扩展到通用网页或 Wikipedia。

---

## 5. 实验总体分层

建议把实验分成三层，而不是一开始直接训练大模型。

### 5.1 第一层：Packing 数据质量实验

这一层不训练模型，只比较不同 packing 方法生成的数据本身。

目标是证明：

```text
不同 packing 方法确实构造了不同类型的长上下文样本。
Dependency-aware packing 能提高窗口内依赖关系密度。
```

比较方法：

```text
Random Packing
Length-Aware Packing
Same-Repo / Same-Topic Packing
BM25 Packing
Semantic / DataSculpt-lite Packing
Dependency-Aware Packing
```

主要指标：

```text
token utilization
truncation rate
avg number of documents per window
dependency score
dependency edge coverage
same-repo ratio
semantic similarity
redundancy score
document integrity
```

这一层实验是论文中“数据构造有效性”的基础。

### 5.2 第二层：主训练实验

这一层使用 8 卡 NVLink 环境进行长上下文适配训练。

建议主线配置：

```text
模型规模：7B 作为主实验
context length：8K 作为主实验
训练方式：LoRA / QLoRA / LongLoRA-style
训练预算：固定 token budget，例如 50M、100M 或 200M tokens
变量：只改变 packing 方法
```

如果主实验稳定，再扩展：

```text
13B 模型
16K context
更大 token budget
```

需要强调：

> LoRA/QLoRA 只是为了降低训练成本，让多组 packing 对比能完成；论文创新点仍然是 packing 策略。

### 5.3 第三层：评测与机制分析

评测不应只依赖 needle 或 passkey 这种合成任务。

建议分成三类：

```text
代码仓库任务：RepoBench、跨文件补全、跨文件检索
通用长上下文探针：passkey retrieval、needle-in-a-haystack
训练过程指标：validation loss、long-context validation loss、loss curve
```

其中，论文主结论应优先依赖真实或半真实的跨文件、跨文档任务。

---

## 6. 主要实验组

主实验建议保留以下组别：

| 组别 | 方法 | 作用 |
|---|---|---|
| G1 | Random Packing | 最弱基线 |
| G2 | Length-Aware Packing | 控制窗口利用率 |
| G3 | Same-Repo / Same-Topic Packing | 控制粗粒度相关性 |
| G4 | BM25 Packing | 控制词面检索相关性 |
| G5 | Semantic / DataSculpt-lite Packing | 控制语义相似性 |
| G6 | Dependency-Aware Packing | 本课题方法 |

最关键对比：

```text
Same-Repo vs Dependency-Aware
BM25 vs Dependency-Aware
Semantic/DataSculpt-lite vs Dependency-Aware
```

如果 dependency-aware 只比 random 好，不够有说服力。必须证明它相对更强 baseline 仍然有价值。

---

## 7. 消融实验设计

为了证明方法不是多个规则随便相加，需要做消融。

建议保留以下消融：

```text
Dependency-Aware without import relation
Dependency-Aware without source-test relation
Dependency-Aware without README/docs relation
Dependency-Aware without config-script relation
Dependency-Aware without same-repo weak prior
BM25 + structure reranking
Semantic + structure reranking
```

这些消融可以回答：

```text
哪类依赖最有效？
结构依赖是否只是 same-repo 的替代？
结构依赖能否作为 BM25/semantic 的 reranking 信号？
```

---

## 8. 8 卡 NVLink 环境下的训练策略

8 卡 NVLink 的优势是：

```text
显存总量更大
卡间通信更快
更适合分布式训练
可以支持更长 context、更大 batch、更大模型
```

但不建议一开始就做全参数微调。原因是本课题需要跑多组 baseline 和消融，全参微调会让实验成本过高。

建议优先路线：

```text
第一阶段：7B + 8K + LoRA/QLoRA
第二阶段：7B + 16K 或 13B + 8K
第三阶段：选择最有说服力的设置做补充实验
```

训练框架可考虑：

```text
DeepSpeed ZeRO
FSDP
FlashAttention
gradient checkpointing
LoRA/QLoRA
sequence parallelism, 如果进入 16K/32K
```

训练时必须记录：

```text
tokens/sec
GPU memory
training time
loss curve
effective batch size
gradient accumulation
context length
token budget
```

这些指标可以支撑论文中的效率分析。

---

## 9. 论文结果应该产出什么

最终论文最好至少包含以下结果。

### 9.1 数据统计表

展示不同 packing 方法生成的数据差异：

```text
token utilization
truncation rate
dependency score
dependency edge coverage
redundancy
avg docs per sample
```

### 9.2 主结果表

展示不同 packing 方法训练后的模型效果：

```text
validation loss
long-context validation loss
RepoBench-R
RepoBench-C
RepoBench-P
needle/passkey
LongBench subset
```

### 9.3 消融表

展示不同依赖类型的贡献。

### 9.4 训练效率表

展示不同方法的训练成本：

```text
tokens/sec
peak memory
training hours
```

### 9.5 曲线和案例分析

建议包含：

```text
validation loss curve
不同 context length 下的效果曲线
dependency-aware packing 示例
失败案例分析
```

---

## 10. 论文贡献写法

论文贡献可以写成三点：

1. **提出 dependency-aware packing 目标。**

   从语义相似、词面相关和长度利用率，转向显式建模训练窗口内部的可学习上下文依赖。

2. **设计面向多源代码仓库上下文的依赖建模方法。**

   利用代码、README、docs、config、tests、issue、commit 等材料之间的结构关系构造长上下文样本。

3. **在 8 卡 NVLink 环境下系统验证 packing 策略对长上下文适配的影响。**

   通过数据统计、主训练实验、消融实验和 case study，分析 dependency-aware packing 的有效性和适用边界。

---

## 11. 推荐推进顺序

建议按照以下顺序推进：

```text
1. 完成 dependency-aware packing v1
2. 补齐 BM25、semantic/DataSculpt-lite baseline
3. 构建多源代码仓库数据集
4. 跑 packing 数据统计，确认不同方法差异明显
5. 在 7B + 8K 上跑小 token budget 试验
6. 固定训练预算，跑完整 baseline
7. 做消融和 case study
8. 根据结果决定是否扩展到 16K 或 13B
```

不要一开始就把所有变量都打开。论文最怕实验矩阵过大但每组都不够扎实。

---

## 12. 当前阶段的结论

当前阶段可以明确：

```text
主线仍然是 packing 策略优化。
8 卡 NVLink 让实验从“小规模可行性”升级到“系统验证”。
论文创新点不是训练更大的模型，而是提出并验证 dependency-aware packing。
数据可以不局限于代码文件，但建议围绕代码仓库多源上下文展开。
```

一句话总结：

> 本课题应研究“如何构造更有依赖关系的长上下文训练样本”，而不是单纯研究“如何把模型训得更长”。
