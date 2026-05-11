# 简明实验方案：代码仓库结构依赖感知 Packing

## 1. 论文题目

**面向低资源长上下文适配的代码仓库结构依赖感知数据 Packing 方法研究**

英文题目：

**Structure-Dependency-Aware Data Packing for Low-Resource Long-Context Adaptation in Code Repository Scenarios**

## 2. 核心问题

低资源条件下，长上下文适配训练的 token budget 有限。对于代码仓库数据，简单随机拼接、按长度填满窗口、同仓库拼接或 BM25 检索相关拼接，都不一定能保证训练样本内部存在真正有用的跨文件依赖。

本课题研究：

> 能否利用代码仓库中的结构依赖关系，构造更有效的长上下文训练样本？

## 3. 核心想法

普通 packing 关注：

```text
是否能填满窗口
是否属于同一仓库
是否词面相关
是否语义相似
```

本课题关注：

```text
一个文件是否能帮助模型理解或预测另一个文件
```

因此提出 **Structure-Dependency-Aware Packing**，优先把存在结构依赖的文件放入同一个长上下文窗口。

## 4. 结构依赖定义

第一版只使用轻量、可解释的代码仓库结构关系：

| 关系 | 示例 |
|---|---|
| import relation | `train.py` import `model.py` |
| source-test relation | `test_model.py` 测试 `model.py` |
| README-code relation | `README.md` 说明 `src/model.py` |
| config-script relation | `train.yaml` 服务于 `train.py` |
| same directory/module | 同目录或同模块文件 |
| same repo | 同仓库弱先验 |

第一版不把 attention、uncertainty 或全量 loss-reduction 作为主方法，避免实验过重。

## 5. 方法流程

```text
1. 收集代码仓库文档
2. 统一为 JSONL 格式，保留 repo/path/language metadata
3. 计算文件间结构依赖分数
4. 从 anchor file 出发贪心构造窗口
5. 优先加入与窗口内文件存在结构依赖的文件
6. 控制 max_tokens、截断率和窗口利用率
7. 输出 packed training samples 和统计指标
```

## 6. 对比方法

| 方法 | 作用 |
|---|---|
| Random Packing | 最弱 baseline |
| Length-Aware Packing | 控制窗口利用率 |
| Same-Repo Packing | 排除“只是同仓库有效”的解释 |
| BM25 Packing | 代表检索相关 packing |
| Semantic Packing | 代表语义相似 packing |
| Dependency-Aware Packing | 本课题方法 |

其中最关键对比是：

```text
Same-Repo Packing vs Dependency-Aware Packing
```

如果 dependency-aware 在同仓库基础上仍有提升，说明结构依赖本身有贡献。

## 7. 实验设置

推荐路线：

```text
第一阶段：1.5B / 3B + 4K context + QLoRA
第二阶段：3B + 8K context
增强实验：7B + 4K/8K，视显存情况决定
```

训练时保持：

```text
模型相同
训练 token 数相同
context length 相同
训练 steps 相同
LoRA/QLoRA 配置相同
唯一主要变量是 packing 方法
```

## 8. 评测指标

### Packing 质量指标

```text
token utilization
avg num docs
truncation rate
dependency score
same-repo ratio
same-directory ratio
```

### 模型效果指标

```text
training loss
validation loss
long-context validation loss
passkey retrieval
mini needle-in-a-haystack
RepoBench 子集
跨文件代码补全
```

## 9. 消融实验

建议保留少而关键的消融：

```text
去掉 import relation
去掉 source-test relation
去掉 same-repo 弱信号
dependency-aware vs same-repo
dependency-aware vs BM25
dependency-aware vs semantic
```

## 10. 预期贡献

1. 提出代码仓库场景下的结构依赖感知 packing 目标。
2. 设计轻量、可解释、低成本的结构依赖分数。
3. 构建低资源长上下文适配的 packing 对比实验框架。
4. 验证结构依赖是否能提升跨文件代码理解和长上下文适配效果。

## 11. 主要风险

| 风险 | 应对 |
|---|---|
| 7B-8K 显存不足 | 先做 1.5B/3B + 4K |
| 提升不明显 | 加强跨文件任务和 case study |
| 被认为只是 same-repo packing | 必须加入 same-repo baseline |
| BM25 更强 | 分析 BM25 是否捕捉了命名/import overlap，可做 BM25 + structure rerank |

## 12. 当前下一步

当前仓库已经实现：

```text
random packing
length-aware packing
same-repo packing
BM25 packing
dependency-aware packing
packing summary script
```

下一步建议：

```text
1. 增加 semantic/DataSculpt-lite baseline
2. 增加 same-repo ratio、same-directory ratio 等统计指标
3. 构建真实代码仓库数据预处理脚本
4. 接入真实 tokenizer
5. 开始 1.5B/3B + 4K pilot training
```
