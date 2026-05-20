# Metrics Definition

This document defines the quantitative metrics used to compare packing methods
and trained models.

## 1. Packing Quality Metrics

Let a packed sample be:

```text
S_i = [d_1, d_2, ..., d_n]
```

Let the maximum context length be:

```text
L = max_context_tokens
```

Let the actual token count be:

```text
T_i = tokens(S_i)
```

### 1.1 Token Utilization

Measures how much of the context window is used.

```text
token_utilization_i = T_i / L
```

Report:

```text
mean token utilization
median token utilization
percentage of samples below 80% utilization
```

This is a control metric, not the main objective. A random or length-aware
method can fill windows well while still producing weak training samples.

### 1.2 Truncation Rate

Measures whether packing destroys document integrity.

Document-level truncation:

```text
doc_truncation_rate = truncated_docs / candidate_docs
```

Token-level truncation:

```text
token_truncation_rate = truncated_tokens / original_candidate_tokens
```

Token-level truncation should be the main reported version.

### 1.3 Average Documents per Window

Measures how many documents are included in each training sample.

```text
avg_docs_per_window = mean_i(|S_i|)
```

Too few documents may not provide cross-document learning signals. Too many
small documents may introduce noise.

### 1.4 Dependency Score

Build a dependency graph:

```text
G = (D, E)
```

Each edge has weight:

```text
w(A -> B)
```

The edge means that document `A` may help understand or predict document `B`.

Unordered dependency density:

```text
dependency_density(S_i)
= sum_{a,b in S_i, a != b} w(a -> b) / (|S_i| * (|S_i| - 1))
```

Order-aware dependency score:

```text
order_dependency(S_i)
= sum_{j=2..n} max_{k<j} w(d_k -> d_j) / (n - 1)
```

The order-aware version should be treated as the primary dependency metric
because language-model training is causal: previous tokens help predict later
tokens.

### 1.5 Dependency Edge Coverage

Measures how many global dependency edges are placed into the same training
window.

Unweighted:

```text
edge_coverage = covered_dependency_edges / all_candidate_dependency_edges
```

Weighted:

```text
weighted_edge_coverage
= sum_{covered edges} w(edge) / sum_{all candidate edges} w(edge)
```

Use the weighted version as the main metric when edges have different
importance.

For the current code-repository setting, report strong and weak coverage
separately:

```text
strong edges = edges containing import/source-test/docs-code/README-code/config-script/example-code
weak edges   = edges containing only same_directory and/or same_repo
```

The summary CSV therefore includes:

```text
strong_edge_coverage
weighted_strong_edge_coverage
weak_edge_coverage
weighted_weak_edge_coverage
avg_strong_order_dependency
avg_weak_order_dependency
```

This split is important because same-directory relations are useful but weak.
The thesis claim should mainly rely on explicit strong edges and use weak edges
as a controlled structural prior.

### 1.6 Semantic Similarity

Measures whether packed windows remain semantically coherent.

For document embeddings `e_a` and `e_b`:

```text
semantic_similarity(S_i)
= average_{a,b in S_i, a != b} cosine(e_a, e_b)
```

Dependency-aware packing does not need the highest semantic similarity, but it
should not create incoherent windows.

### 1.7 Redundancy

Measures repeated or near-duplicate content.

Embedding-based redundant pair rate:

```text
redundant_pair_rate(S_i)
= count(cosine(e_a, e_b) > threshold) / all_pairs
```

Token Jaccard alternative:

```text
jaccard(a,b) = |tokens(a) intersect tokens(b)| / |tokens(a) union tokens(b)|
```

Recommended thresholds:

```text
embedding cosine: 0.90 or 0.95
token Jaccard: 0.80
```

## 2. Post-Training Metrics

### 2.1 Long-Context Validation Loss

Evaluate the trained model on held-out packed long-context samples.

```text
val_loss = average negative log likelihood over held-out long-context tokens
```

Use the same validation set for all packing methods.

### 2.2 Dependency-Sensitive Context Gain

For a dependency edge:

```text
A -> B
```

Compute:

```text
context_gain(A -> B) = loss(B alone) - loss(A + B)
```

If the model benefits from dependency context, adding `A` before `B` should
reduce the loss on `B`.

Report:

```text
mean context gain for dependency edges
mean context gain for same-repo non-edge pairs
mean context gain for random cross-repo pairs
```

The expected pattern is:

```text
dependency edges > same-repo non-edges > random cross-repo pairs
```

### 2.3 RepoBench / Cross-File Retrieval

For repository-level retrieval tasks, report:

```text
Recall@k
MRR@k
nDCG@k
Hit@k
```

These metrics quantify whether relevant cross-file context can be found.

### 2.4 Cross-File Completion

For code completion with cross-file context, report:

```text
Exact Match
Edit Similarity
CodeBLEU
Identifier F1
```

Exact Match is strict. Edit Similarity and CodeBLEU are often more informative
for longer completions.

### 2.5 Passkey Retrieval

Insert a key-value fact into a long context and ask the model to retrieve it.

Main metric:

```text
accuracy = correct_answers / total_questions
```

Report by:

```text
context length
insertion depth
number of distractors
```

### 2.6 Needle-in-a-Haystack

Place a target fact inside a long distractor context.

Report:

```text
exact retrieval accuracy
substring match accuracy
F1, if answer text is not a simple key
```

Single-needle tasks can saturate, so multi-needle or distractor variants are
preferred for stronger models.

### 2.7 LongBench-Style Tasks

Use a small, targeted subset rather than the full benchmark at first.

Possible metrics:

```text
QA: F1 / EM
summarization: ROUGE-L
classification or multiple choice: Accuracy
code: EM / EditSim / pass rate
```

LongBench-style tasks should be supporting evidence. The main argument should
come from repository-level tasks and dependency-sensitive validation.
