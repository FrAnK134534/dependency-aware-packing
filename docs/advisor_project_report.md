# 项目设想与汇报：依赖感知长上下文 Packing

## 1. 我想做什么

我现在想做的不是单纯训练一个更大的模型，也不是提出新的模型结构，而是研究一个更基础但很关键的问题：

> **长上下文训练时，应该把什么内容放进同一个上下文窗口里？**

现在很多长上下文模型可以支持 8K、16K 甚至更长输入。但模型能看这么长，并不代表它真的学会了利用长上下文。

如果训练时只是把很多无关文档拼在一起，模型虽然看到了很长的序列，但这些序列里没有清楚的上下文关系，模型不一定能学到“前面的内容如何帮助理解后面的内容”。

所以我想研究：

> 在固定模型、固定训练预算、固定 context length 的情况下，**不同 packing 策略是否会影响模型的长上下文能力**？

进一步地，我想提出一种新的 packing 思路：

> **Dependency-Aware Packing：把存在依赖关系的材料放进同一个长上下文窗口，而不是只把相似或同主题材料放在一起。**

---

## 2. 什么是 Packing

Packing 可以理解为：

> 把多个较短的文件、文档或片段，拼成一个固定长度的训练样本。

例如模型训练时需要 8192 tokens 的输入窗口。如果一个文件只有 1000 tokens，单独训练会浪费很多空间，所以通常会把多个文件拼起来。

常见做法包括：

```text
Random Packing：随机拼接
Length-Aware Packing：尽量把窗口填满
Same-Repo Packing：把同一个仓库的文件放一起
BM25 Packing：把关键词相关的文件放一起
Semantic Packing：把语义相似的文件放一起
```

这些方法有用，但它们都有一个共同问题：

> 它们不直接判断“一个文件是否真的能帮助理解另一个文件”。

这正是我想改进的地方。

---

## 3. 我的核心想法

我的核心想法是：

> 长上下文窗口不应该只是“长”，而应该包含有学习价值的上下文依赖。

例如在一个代码项目中：

```text
README.md          说明项目怎么用
config/train.yaml  保存训练配置
src/model.py       定义模型结构
src/train.py       调用模型并读取配置
tests/test_model.py 测试模型行为
```

这些文件之间是有关系的。把它们放在同一个上下文窗口中，模型更可能学到：

```text
文档如何对应代码；
配置如何影响脚本；
函数和类如何被调用；
测试如何反映代码预期行为；
跨文件信息如何关联。
```

相比之下，如果只是随机拼接几个文件，即使窗口被填满了，也不一定有这种学习价值。

因此，我的项目核心是：

> 用依赖关系指导长上下文训练数据的构造。

---

## 4. 数据不只包含代码

虽然我会以代码仓库作为主要场景，但上下文并不只包含代码文件。

我希望使用的是 **代码仓库多源上下文**，包括：

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
API usage examples
```

这样做的好处是：

1. 这些材料之间有明确依赖关系，便于定义方法。
2. 它更接近真实开发场景，因为开发者理解代码时也会看 README、文档、配置、测试和 issue。
3. 可以用 RepoBench、跨文件补全等任务评估模型是否真的学会利用跨文件上下文。

---

## 5. 什么叫“依赖关系”

这里的依赖关系不是只指代码 import，也包括更广义的上下文帮助关系。

例如：

| 依赖关系 | 含义 |
|---|---|
| README -> code | README 说明代码如何使用 |
| docs -> implementation | 文档解释实现逻辑 |
| config -> script | 配置文件影响脚本运行 |
| source -> test | 测试文件验证源码行为 |
| API doc -> usage example | API 文档帮助理解调用方式 |
| issue -> changed file | issue 描述需要修改的文件 |
| commit message -> changed file | commit message 解释文件修改原因 |

我会给这些关系设计一个轻量的 dependency score：

```text
dependency_score(A -> B)
```

它表示：

> A 放在 B 前面，是否可能帮助模型理解或预测 B。

第一版会优先使用可解释规则，而不是一开始训练复杂打分模型。这样方法更可控，也更容易向论文读者解释。

---

## 6. 和已有方法相比，我的不同点

已有 packing 方法通常关注：

```text
窗口是否填满；
文档是否来自同一仓库；
关键词是否相似；
embedding 语义是否接近。
```

我的方法关注：

```text
窗口内部是否存在可学习的上下文依赖。
```

举例来说：

```text
BM25 可能认为两个文件相关，因为它们都出现 model、train、loss 等词。
Semantic packing 可能认为两个文件相关，因为它们主题相似。
Dependency-aware packing 会进一步问：一个文件是否定义、解释、配置或测试了另一个文件？
```

所以我的核心区别是：

> 从“相似内容拼接”转向“依赖关系驱动的样本构造”。

---

## 7. 8 卡 NVLink 服务器怎么用

导师提供的单节点 8 卡 NVLink 环境非常有价值，但我不会把论文重点写成“我用了 8 张卡训练模型”。

8 卡环境的作用是：

```text
支持更稳定的长上下文训练；
支持 7B 模型作为主实验；
支持 8K 甚至 16K context；
支持多组 baseline 和消融实验；
让实验结果更系统、更有说服力。
```

我建议的训练路线是：

```text
第一阶段：7B + 8K + LoRA/QLoRA，小规模 token budget 试跑
第二阶段：7B + 8K，完整比较不同 packing 方法
第三阶段：7B + 16K 或 13B + 8K，只选择关键方法做扩展
第四阶段：做消融实验和案例分析
```

这里 LoRA/QLoRA 不是创新点，它只是节省显存和训练成本的工具。因为我需要跑很多组对比实验，如果每组都做全参数微调，成本会太高。

---

## 8. 实验整体流程

实验会分三层进行。

### 第一层：先比较 packing 数据本身

这一层不训练模型，只比较不同方法构造出来的数据有什么区别。

我会比较：

```text
Random Packing
Length-Aware Packing
Same-Repo Packing
BM25 Packing
Semantic / DataSculpt-lite Packing
Dependency-Aware Packing
```

主要看：

```text
窗口是否被充分利用；
是否大量截断文档；
每个窗口平均有多少文档；
窗口内部依赖关系强不强；
依赖边覆盖率高不高；
语义是否一致；
是否有大量重复内容。
```

这一层的目的是证明：

> 我的 packing 方法确实构造出了依赖关系更密集的训练样本。

### 第二层：固定训练条件，只改变 packing 方法

这一层是主实验。

在同样的条件下训练模型：

```text
同一个 base model；
同一个 context length；
同一个训练 token budget；
同一个 LoRA/QLoRA 配置；
同一个 optimizer；
同样的训练步数；
只改变 packing 方法。
```

这样如果模型效果不同，就更有理由说明差异来自数据组织方式，而不是训练设置不同。

### 第三层：评估模型是否更会用长上下文

训练后会评估：

```text
RepoBench / 跨文件补全；
跨文件检索；
long-context validation loss；
dependency-sensitive validation loss；
passkey retrieval；
needle-in-a-haystack；
LongBench 子集。
```

其中主结论会优先依赖跨文件任务和 validation loss，passkey/needle 主要作为长上下文探针。

---

## 9. 如何量化 packing 好不好

我会从数据层面量化 packing 质量。

### 9.1 Token Utilization

衡量窗口是否被充分利用：

```text
token_utilization = 实际 token 数 / 最大窗口 token 数
```

它不能单独代表好坏，因为随机 packing 也可能填得很满。

### 9.2 Truncation Rate

衡量是否大量截断文档：

```text
token_truncation_rate = 被截断 token 数 / 原始候选 token 数
```

截断太多说明样本完整性差。

### 9.3 Avg Docs per Window

衡量每个窗口平均包含多少文档：

```text
avg_docs_per_window = 每个窗口文档数的平均值
```

文档太少，跨文档训练信号不足；文档太多，可能噪声较大。

### 9.4 Dependency Score

衡量窗口内部依赖关系强不强。

我会构建依赖图：

```text
A -> B 表示 A 可能帮助理解 B
```

然后计算窗口中依赖边的平均强度。

更重要的是顺序依赖：

```text
前面的文档是否能帮助后面的文档？
```

这和语言模型的因果训练方式更一致。

### 9.5 Dependency Edge Coverage

衡量全局依赖关系中，有多少被 packing 放进了同一个训练窗口。

如果 dependency-aware packing 覆盖了更多依赖边，说明它更有效地把相关上下文组织到一起。

### 9.6 Semantic Similarity

衡量窗口内材料语义是否过于分散。

它不是越高越好，但不能太低。我的方法不一定追求最高语义相似，而是追求更高依赖密度。

### 9.7 Redundancy

衡量是否重复堆叠相似内容。

如果一个窗口里都是高度重复的文件，模型看到的信息量其实不高。

---

## 10. 如何量化训练后的长上下文能力

训练后主要从以下方面评估。

### 10.1 跨文件补全

例如给模型 `README + config + model.py + train.py` 的上下文，让它补全某一段代码。

指标可以包括：

```text
Exact Match
Edit Similarity
CodeBLEU
Identifier F1
```

### 10.2 跨文件检索

给定一个目标位置，让模型或检索模块找出最相关的跨文件上下文。

指标：

```text
Recall@k
MRR@k
Hit@k
nDCG@k
```

### 10.3 Long-Context Validation Loss

在 held-out 的长上下文验证集上计算 loss。

如果 dependency-aware 训练出的模型 loss 更低，说明它在类似长上下文分布上建模更好。

### 10.4 Context Gain

这是最贴合我方法的指标。

对于一条依赖关系：

```text
A -> B
```

比较：

```text
loss(B alone)
loss(A + B)
```

定义：

```text
context_gain(A -> B) = loss(B alone) - loss(A + B)
```

如果一个模型真的学会利用依赖上下文，那么加入 A 后预测 B 的 loss 应该下降更多。

### 10.5 Passkey 和 Needle

这类任务是在长文本中插入一条关键信息，然后问模型能否找回来。

它们可以测试模型的长距离检索能力，但比较合成，所以只作为辅助评测。

---

## 11. 最关键的对比和消融

为了证明我的方法不是简单把同仓库文件放一起，最关键对比是：

```text
Same-Repo Packing vs Dependency-Aware Packing
```

为了证明不是词面相关就够了，要比较：

```text
BM25 Packing vs Dependency-Aware Packing
```

为了证明不是语义相似就够了，要比较：

```text
Semantic/DataSculpt-lite Packing vs Dependency-Aware Packing
```

消融实验包括：

```text
去掉 import relation；
去掉 source-test relation；
去掉 README/docs relation；
去掉 config-script relation；
去掉 same-repo 弱信号；
BM25 + structure reranking；
Semantic + structure reranking。
```

这些实验可以回答：

```text
哪类依赖最有用？
结构依赖是否只是 same-repo 的替代？
结构依赖能否增强 BM25 或 semantic packing？
```

---

## 12. 预期论文产出

最终论文可以产出以下内容：

1. 一种 dependency-aware packing 方法。
2. 一套多源代码仓库上下文的数据构造流程。
3. 一组 packing 质量指标。
4. 8 卡 NVLink 环境下的长上下文适配对比实验。
5. 消融实验和案例分析。

论文中可以包含：

```text
不同 packing 方法的数据统计表；
不同 packing 方法的主评测结果表；
不同依赖类型的消融表；
训练效率表；
validation loss 曲线；
context length 分组结果；
dependency-aware packing 成功和失败案例。
```

---

## 13. 项目的最终一句话

这个项目的重点不是：

> 我能把模型训到多长。

而是：

> 在长上下文训练中，如何把有限的上下文窗口组织得更有学习价值。

更具体地说：

> 通过 dependency-aware packing，把存在依赖关系的多源材料放在一起，让模型在训练中更好地学习跨文档、跨文件、跨段落的长上下文利用能力。
