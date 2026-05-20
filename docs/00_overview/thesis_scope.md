# Thesis Scope

## Working Title

**Dependency-Aware Data Packing for Efficient Long-Context Adaptation**

Chinese title:

**面向长上下文适配的依赖感知数据 Packing 方法研究**

## Core Claim

Long-context adaptation is not only a model-side problem. Even if a model can
accept 8K or 16K tokens, it may not learn useful long-context behavior if the
training windows are filled with weakly related or redundant text.

This thesis studies whether dependency-aware packing can construct more useful
long-context training samples under a fixed training budget.

## Current Scope

The first paper-quality setting is multi-source code repository context:

```text
source code
tests
README
docs
config files
examples
issue / PR descriptions
commit messages
benchmark logs
API usage examples
```

The method is allowed to extend to technical documents later, but the main
experimental story should stay centered on repository-level and
software-engineering context.

## Main Research Questions

1. Do different packing strategies lead to measurably different long-context
   adaptation outcomes under the same model and training budget?
2. Is lexical or semantic similarity sufficient, or does explicit dependency
   modeling provide additional value?
3. Which dependency types contribute most to cross-document and cross-file
   long-context ability?

## Must-Have Baselines

```text
Random Packing
Length-Aware Packing
Same-Repo / Same-Topic Packing
BM25 Packing
Semantic / DataSculpt-Lite Packing
Dependency-Aware Packing
```

## Must-Have Evidence

The thesis should provide evidence from four angles:

```text
1. Packing statistics
2. Training and validation loss
3. Long-context and repository-level evaluations
4. Ablations and case studies
```

## Main Risk

The largest scientific risk is that dependency-aware packing is interpreted as
only same-repository or same-topic packing. The experiment must include
same-repo/same-topic baselines and within-repo dependency comparisons.
