# Thesis Scope

## Recommended Title

Structure-Dependency-Aware Data Packing for Low-Resource Long-Context
Adaptation in Code Repository Scenarios

## Core Claim

Under a fixed low-resource training budget, long-context adaptation benefits
from packing samples that contain learnable cross-file dependencies, not only
documents that are semantically similar or easy to fit into a context window.

## Minimal Closed Loop

```text
Data: small code repository corpus
Packing: random / length-aware / same-repo / dependency-aware
Context: 4K first, 8K after pilot
Model: 1.5B or 3B QLoRA first, 7B optional
Evaluation: packing statistics + validation loss + passkey/needle + RepoBench subset
```

## First-Version Dependency Definition

The first version uses structural dependency edges:

- imported file -> importing file;
- source file -> test file;
- README/documentation -> source file;
- config file -> script file;
- same directory or same module;
- same repository as a weak prior.

Loss-reduction dependency is reserved for sampled validation and analysis, not
for the first production packing algorithm.

## Must-Have Baselines

- random packing;
- length-aware packing;
- same-repo random packing;
- BM25/SPLiCe-style packing;
- semantic/DataSculpt-lite packing;
- dependency-aware packing;
- dependency-aware without structure score as ablation.

## Main Risk

The largest scientific risk is that dependency-aware packing is interpreted as
only same-repository packing. The experiment must include same-repo baselines
and within-repo comparisons.
