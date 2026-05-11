# 硕士课题实验方案：代码仓库场景下的结构依赖感知长上下文数据 Packing

## 0. 题目定位

推荐论文题目：

> **面向低资源长上下文适配的代码仓库结构依赖感知数据 Packing 方法研究**

英文可写为：

> **Structure-Dependency-Aware Data Packing for Low-Resource Long-Context Adaptation in Code Repository Scenarios**

本课题不再追求“所有文本场景通用”的 dependency-aware packing，而是收窄到一个更适合硕士论文完成的方向：

> 在代码仓库数据中，利用文件之间天然存在的结构依赖关系，构造更适合长上下文适配训练的数据样本，并验证其相比随机、长度优先、同仓库、BM25 和语义相似 packing 是否更有效。

---

## 1. 一句话概括

在双 RTX 4080 等低资源环境下，长上下文适配训练的 token budget 有限。对于代码仓库数据，简单随机拼接、按长度填满窗口、同仓库拼接或 BM25 检索相关拼接，都不一定能保证窗口内部存在可学习的跨文件依赖。

本课题提出一种 **结构依赖感知的 Dependency-Aware Packing** 方法，优先把存在 import、source-test、README-code、config-script、same-module 等结构关系的文件放入同一个长上下文训练窗口，从而提高有限训练预算下的长上下文学习效率。

---

## 2. 研究边界

### 2.1 本课题要做什么

本课题聚焦：

```text
数据场景：代码仓库数据
核心关系：结构依赖关系
训练目标：低资源长上下文适配
主要模型：1.5B / 3B 起步，7B 作为增强实验
上下文长度：4K pilot，8K main
主要方法：结构依赖感知 packing
主要评测：packing statistics + validation loss + 长上下文探针 + RepoBench 子集
```

### 2.2 本课题暂时不做什么

为了保证硕士课题可控，第一版不把以下内容作为主线：

```text
不做通用网页 / Wikipedia / arXiv 全场景覆盖
不把 attention dependency 作为主方法
不把 uncertainty dependency 作为主方法
不对所有文档两两计算 loss-reduction dependency
不一开始追求 16K / 32K context
不一开始使用 7B-8K 作为唯一闭环
```

其中，loss-reduction 可以作为后续抽样分析：

```text
dependency(A -> B) = loss(B alone) - loss(A + B)
```

但它不作为第一版 packing 算法的主要依赖信号。

---

## 3. 核心研究问题

### RQ1：代码仓库长上下文适配中，packing 策略是否会影响模型效果？

在相同模型、相同训练 token 数、相同 context length、相同 LoRA/QLoRA 配置下，只改变 packing 方法，比较不同策略对长上下文任务和代码跨文件任务的影响。

### RQ2：同仓库、BM25 或语义相似 packing 是否足够？

代码仓库内部文件可能属于同一项目，但并不是所有文件之间都有直接依赖。BM25 检索能找到关键词相关文件，语义相似 packing 能找到主题接近文件，但这些关系不一定等价于“一个文件能帮助理解或预测另一个文件”。

因此需要验证：

```text
same-repo packing 是否只是粗粒度相关；
BM25 packing 是否只捕捉词面相关；
semantic packing 是否只捕捉主题相关；
dependency-aware packing 是否能带来额外收益。
```

### RQ3：结构依赖感知 packing 能否在固定训练预算下提升长上下文适配效率？

如果 dependency-aware packing 在 RepoBench、跨文件补全、long-context validation loss 或 needle/passkey 探针上优于 baseline，则说明数据组织方式不仅影响大规模训练，也影响低资源长上下文适配。

---

## 4. 主要贡献点

本课题预期贡献控制在硕士论文可完成范围内：

1. **提出面向代码仓库的结构依赖感知 packing 目标。**

   与随机、长度优先、同仓库、BM25 或语义相似 packing 不同，本方法显式考虑代码仓库中文件之间的结构关系。

2. **设计轻量、可解释的结构依赖分数。**

   依赖分数来自 import、source-test、README-code、config-script、same-directory、same-repo 等关系，不需要昂贵的全量 pairwise loss 计算。

3. **构建低资源长上下文适配实验框架。**

   在固定模型、token budget、context length 和训练配置下比较 packing 方法。

4. **通过统计指标、训练结果、消融实验和 case study 分析结构依赖的作用。**

   即使模型指标提升幅度有限，也能通过 packing statistics 和抽样 loss-reduction 分析支撑结论。

---

## 5. 方法设计

### 5.1 输入与输出

输入为代码仓库文档 JSONL：

```json
{
  "docid": "repo_a:src/model.py",
  "content": "...",
  "metadata": {
    "repo": "repo_a",
    "path": "src/model.py",
    "language": "python"
  }
}
```

输出为固定长度的 packed training sample：

```json
{
  "sample_id": "dependency_aware_000001",
  "method": "dependency_aware",
  "docids": [
    "repo_a:README.md",
    "repo_a:config/train.yaml",
    "repo_a:src/model.py",
    "repo_a:src/train.py"
  ],
  "content": "...",
  "stats": {
    "tokens": 4096,
    "num_docs": 4,
    "dependency_score": 0.72,
    "token_utilization": 0.95,
    "truncation_rate": 0.0
  }
}
```

---

## 6. 结构依赖分数

第一版 dependency score 使用可解释结构关系：

```text
dep(A -> B) =
  w1 * import_relation(A, B)
+ w2 * test_source_relation(A, B)
+ w3 * readme_code_relation(A, B)
+ w4 * config_script_relation(A, B)
+ w5 * same_directory(A, B)
+ w6 * same_repo(A, B)
```

推荐默认权重：

| 关系 | 含义 | 默认权重 |
|---|---|---:|
| import_relation | B import / require A | 1.0 |
| test_source_relation | B 是 A 的测试或显式引用 A | 0.9 |
| readme_code_relation | README / 文档说明代码文件 | 0.6 |
| config_script_relation | 配置文件服务于脚本或训练入口 | 0.5 |
| same_directory | 同目录或同模块 | 0.25 |
| same_repo | 同仓库弱先验 | 0.1 |

这里的核心思想是：

> 结构依赖不是简单语义相似，而是一个文件对另一个文件存在上下文增益的先验。

---

## 7. Dependency-Aware Packing 算法

整体流程：

```text
Step 1：读取代码仓库文档
Step 2：提取 repo、path、language、filename、directory 等 metadata
Step 3：根据结构规则计算候选依赖边
Step 4：从 anchor file 出发构造窗口
Step 5：优先加入与窗口内文件存在强依赖边的候选文件
Step 6：控制 max_tokens、truncation、token utilization
Step 7：输出 packed JSONL 和统计指标
```

贪心选择目标：

```text
Score(candidate, window) =
  max dependency_score(existing_doc -> candidate)
+ capacity_bonus
- truncation_penalty
```

第一版约束：

```text
只有 dependency_score > 0 的文件才能被加入 dependency-aware 窗口；
不为了填满窗口而加入完全无结构关系的文件；
same_repo 只是弱信号，需要通过 same-repo baseline 控制。
```

---

## 8. Baseline 设计

### Baseline 1：Random Packing

随机打乱文档后顺序装入窗口。

作用：

```text
证明简单拼接不是合理长上下文训练数据构造方式。
```

### Baseline 2：Length-Aware Packing

按长度优先装箱，尽量提高 token utilization。

作用：

```text
控制“窗口是否被填满”这一因素。
```

### Baseline 3：Same-Repo Packing

只把同一个 repo 内的文件尽量放在一起。

作用：

```text
这是本课题最关键 baseline，用于排除“提升只是因为同仓库”的解释。
```

### Baseline 4：BM25 Packing

以 anchor document 为查询，用 BM25 找词面相关文件并装入窗口。

作用：

```text
代表 SPLiCe-style / retrieval-based packing 的轻量版本。
```

### Baseline 5：Semantic / DataSculpt-Lite Packing

后续加入 embedding 或轻量 sentence embedding，将语义相似文件聚在一起。

作用：

```text
代表 similarity-aware packing，用于验证“相似不等于依赖”。
```

### Proposed：Structure-Dependency-Aware Packing

优先覆盖结构依赖边。

作用：

```text
验证显式建模代码结构依赖是否优于随机、长度、同仓库、检索和语义相似。
```

---

## 9. 实验设计

### 9.1 控制变量

所有训练组保持：

```text
base model 相同
training token budget 相同
context length 相同
训练 steps 相同
LoRA/QLoRA 配置相同
optimizer 配置相同
训练数据来源相同
```

唯一主要变量：

```text
packing method
```

### 9.2 Mini Pilot

第一阶段目标不是追求最终指标，而是跑通闭环。

| 项目 | 推荐设置 |
|---|---|
| 数据 | 小规模代码 repo 数据 |
| 模型 | 1.5B / 3B |
| context length | 4K |
| 训练方式 | QLoRA |
| steps | 100-300 |
| 方法 | random / length-aware / same-repo / BM25 / dependency-aware |
| 评测 | packing statistics + validation loss + passkey/needle 小样本 |

判断是否继续：

```text
dependency-aware 的 dependency_score 明显更高；
token utilization 不低得离谱；
truncation rate 可控；
训练 loss 不崩；
至少一个长上下文或代码任务出现正向趋势。
```

### 9.3 Main Experiment

| 项目 | 推荐设置 |
|---|---|
| 数据 | 中等规模代码 repo 数据 |
| 模型 | 3B 为主，7B 可选 |
| context length | 4K / 8K |
| token budget | 10M-50M 起步 |
| steps | 500-1000 |
| 方法 | random / length-aware / same-repo / BM25 / semantic / dependency-aware |
| 评测 | validation loss + passkey/needle + RepoBench 子集 + case study |

---

## 10. 评测指标

### 10.1 Packing 质量指标

训练前必须先比较 packed 数据本身：

| 指标 | 说明 |
|---|---|
| total tokens | 总 token 数是否一致 |
| token utilization | 窗口利用率 |
| avg num docs | 每个窗口平均文档数 |
| truncation rate | 截断比例 |
| dependency score | 窗口内部结构依赖强度 |
| same-repo ratio | 同仓库比例 |
| same-directory ratio | 同目录/同模块比例 |
| redundancy score | 重复堆叠程度 |

预期：

```text
dependency-aware 的 dependency_score 应显著高于 baseline；
token utilization 可以略低，但不能显著浪费窗口；
truncation rate 应可控；
same-repo baseline 和 dependency-aware 必须分开比较。
```

### 10.2 训练过程指标

| 指标 | 说明 |
|---|---|
| training loss | 收敛速度 |
| validation loss | 泛化能力 |
| long-context validation loss | 长上下文建模能力 |
| tokens/sec | 训练效率 |
| peak memory | 显存压力 |
| loss stability | 是否稳定 |

### 10.3 长上下文能力评测

轻量必做：

```text
passkey retrieval
mini needle-in-a-haystack
```

代码方向主评测：

```text
RepoBench 子集
跨文件代码补全
README/config/source/test 关系补全
```

注意：

```text
passkey 和 needle 只能说明长上下文检索能力；
论文主结论应更多依赖代码跨文件任务和 validation loss。
```

---

## 11. 消融实验

硕士论文建议做少而关键的消融：

| 消融 | 目的 |
|---|---|
| 去掉 same_repo 弱信号 | 验证方法不是只靠同仓库 |
| 去掉 import_relation | 验证 import 依赖贡献 |
| 去掉 test_source_relation | 验证 source-test 关系贡献 |
| dependency-aware vs same-repo | 排除同仓库因素 |
| dependency-aware vs BM25 | 排除词面检索因素 |
| dependency-aware vs semantic | 排除语义相似因素 |

最关键对比：

```text
same-repo packing
vs
dependency-aware packing within repo
```

如果这个对比有提升，论文说服力会明显增强。

---

## 12. 抽样 Loss-Reduction 分析

虽然第一版算法不依赖全量 loss-reduction，但可以抽样验证结构依赖是否真的有上下文增益。

定义：

```text
gain(A -> B) = loss(B alone) - loss(A + B)
```

实验方式：

```text
随机抽样 dependency edges；
随机抽样 same-repo non-edge pairs；
随机抽样 cross-repo pairs；
比较三类 pair 的平均 gain。
```

预期：

```text
dependency edge 的 gain > same-repo non-edge > random cross-repo
```

这个分析可以作为论文中“结构依赖分数合理性”的证据。

---

## 13. 低资源训练建议

双 RTX 4080 场景下建议：

```text
第一闭环：1.5B / 3B + 4K + QLoRA
稳定后：3B + 8K
增强实验：7B + 4K 或 8K
```

不要一开始把 7B-8K 作为唯一目标。更稳妥路线：

```text
先证明 packing 数据分布不同；
再证明 pilot training 不崩；
再扩大 context 和模型规模。
```

---

## 14. 时间规划

### 第 1 阶段：Packing Pipeline

时间：1-2 周

任务：

```text
实现统一 JSONL 数据格式；
实现 random / length-aware / same-repo / BM25；
实现 structure-dependency-aware packing；
输出 packing summary 表格。
```

### 第 2 阶段：数据集构建

时间：1-2 周

任务：

```text
收集代码 repo 数据；
提取 repo/path/language metadata；
切分过长文件；
构造 train/validation split；
检查数据泄漏。
```

### 第 3 阶段：Pilot Training

时间：2-3 周

任务：

```text
跑通 1.5B/3B QLoRA；
使用 4K context；
比较 3-5 个 packing 方法；
记录 loss、显存、tokens/sec。
```

### 第 4 阶段：Main Experiments

时间：3-5 周

任务：

```text
扩展到 8K；
加入 semantic baseline；
补充 RepoBench / needle / passkey；
做关键消融。
```

### 第 5 阶段：论文写作

时间：2-4 周

任务：

```text
整理方法章节；
整理实验表格；
写 case study；
讨论失败案例和局限性。
```

---

## 15. 风险与备选方案

### 风险 1：7B-8K 训练资源不足

备选：

```text
使用 1.5B/3B 作为主实验；
7B 只做补充；
先做 4K，再做 8K；
减少 token budget，保留严格控制变量。
```

### 风险 2：Dependency-Aware 提升不明显

排查：

```text
结构依赖边是否太弱；
same-repo baseline 是否已经很强；
评测任务是否没有考察跨文件依赖；
packing 输出是否与 BM25/semantic 太相似；
训练步数是否太少。
```

备选：

```text
加强跨文件补全评测；
增加 source-test / import 关系样本；
做抽样 loss-reduction 分析；
把重点转向数据构造质量和机制分析。
```

### 风险 3：方法被认为只是 Same-Repo Packing

解决：

```text
必须加入 same-repo baseline；
必须做 within-repo dependency-aware 对比；
必须展示 case study：同仓库但无依赖 vs 同仓库且有结构依赖。
```

### 风险 4：BM25 或 Semantic Baseline 表现更好

这不是绝对失败。可以分析：

```text
BM25 是否在代码中天然捕捉 import/name overlap；
dependency-aware 是否需要与 BM25 结合；
结构依赖是否更适合某些任务，例如跨文件补全，而不是通用 retrieval。
```

后续方法可以变为：

```text
BM25 candidate retrieval + structural dependency reranking
```

---

## 16. 论文故事线

可以这样讲：

```text
背景：
长上下文适配不仅依赖模型结构，也依赖训练时长序列样本的组织方式。

问题：
在低资源环境下，训练 token 有限。代码仓库数据中，简单把同仓库或相似文件拼在一起，不一定能构造有效的跨文件学习信号。

方法：
提出结构依赖感知 packing，利用 import、source-test、README-code、config-script 等关系构造长上下文训练样本。

实验：
在固定训练预算下，对比 random、length-aware、same-repo、BM25、semantic 和 dependency-aware packing。

结论：
结构依赖感知的数据组织方式可以更有效地利用有限上下文窗口，并在代码长上下文任务中带来更稳定的收益。
```

---

## 17. 当前项目实现状态

当前仓库已经具备第一版工程基础：

```text
random packing
length-aware packing
same-repo packing
BM25 packing
dependency-aware packing
packing summary script
示例代码仓库数据
基础单元测试
```

下一步建议：

```text
1. 加入 semantic/DataSculpt-lite baseline；
2. 增加 same-repo ratio、same-directory ratio 等统计指标；
3. 构建真实代码 repo 数据预处理脚本；
4. 接入 tokenizer，替换当前轻量 token counter；
5. 开始 1.5B/3B + 4K 的 pilot training。
```

---

## 18. 最终建议

本课题作为硕士论文是可行的，但必须坚持收窄：

```text
不要做通用 dependency-aware packing；
不要一开始追求大模型和超长上下文；
不要把 attention/uncertainty/loss-reduction 都塞进主方法；
要把代码仓库结构依赖这条线做扎实。
```

最稳的论文主线是：

> 在代码仓库低资源长上下文适配中，结构依赖感知 packing 能否比随机、长度、同仓库、检索和语义相似 packing 更有效地构造训练样本？

只要 packing 统计、关键 baseline、pilot training 和 case study 做扎实，这个方向足够支撑一篇硕士课题级别论文。
