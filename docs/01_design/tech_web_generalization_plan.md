# 技术文档与网页泛化实验计划

## 定位

代码仓库仍是主实验场景。技术文档和网页扩展用于回答一个更宽的问题：

> dependency-aware packing 是否只适用于代码，还是也能处理显式链接、API 使用、定义-使用、引用等非代码依赖？

因此，非代码数据不替代 `repo-main-v1`，而是作为 `tech-web-generalization-v1` 的泛化证据。

## 数据来源

第一版采用 manifest-driven 采集，不做无限制爬虫。入口文件：

```bash
configs/datasets/tech_web_seed_manifest.tsv
```

支持的来源包括：

```text
url_html
local_html
local_markdown
local_pdf
local_text
```

URL 抓取必须显式打开：

```bash
FETCH_URLS=1 \
  bash scripts/data/build_external_dataset_pipeline.sh \
  configs/datasets/tech_web_seed_manifest.tsv \
  data/processed/tech_web_generalization_v1 \
  tech-web-generalization-v1
```

默认按 `document_id` 划分 train/validation/test，避免同一网页或同一文档的不同 section 泄漏到不同 split。

## Dependency 规则

非代码强依赖主要来自显式证据：

```text
hyperlink_relation: 页面或文档显式链接到目标 URL/path
citation_relation: 论文或技术文档显式引用 DOI、标题、bibkey、URL
definition_usage_relation: 定义段落中的术语被后续段落显式使用
api_doc_usage_relation: API 文档中的 API 名在 tutorial/example 中显式调用
equation_or_figure_reference_relation: 正文显式引用 figure/table/equation
```

弱依赖只作为背景 prior：

```text
same_document
same_collection
section_neighbor
same_domain
```

## 实验口径

主论文结论应优先基于代码仓库数据。技术文档/网页部分建议放在泛化实验或 case study：

1. 先跑 packing construction metrics。
2. 人工审查每类非代码 strong edge。
3. 如果 edge quality 足够，再构建 context-gain validation。
4. 不把 toy external smoke 数据作为正式结论。

## 与 DataSculpt 的关系

非代码数据尤其适合加入 DataSculpt 原版 baseline，因为 DataSculpt 的强项是语义聚类与密度采样。公平比较时：

```text
DataSculpt-original: 语义聚类/密度/贪心窗口
DataSculpt-lite: 当前仓库内的轻量语义近似 baseline
Dependency-aware: 显式依赖优先，再做 token-fit
```

如果 DataSculpt-original 在 semantic similarity 上更好，而 dependency-aware 在 strong edge coverage 和 context gain 上更好，这会形成很清晰的论文故事。
