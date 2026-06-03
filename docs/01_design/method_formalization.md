# Dependency-Aware Packing: Method Formalization

## 1. Objects

The corpus is a set of documents:

```text
D = {d_1, d_2, ..., d_n}
```

Each document has content and metadata:

```text
d = (docid, content, metadata)
metadata = repo / collection / path / source_type / document_id / section_id / license / ...
```

The dependency graph is:

```text
G = (D, E)
```

Each edge is directed:

```text
e = (A -> B, relation, weight, evidence)
```

The interpretation is:

```text
A may help a causal language model predict or understand B.
```

## 2. Strong And Weak Edges

Strong edges require explicit evidence:

```text
import_relation
test_source_relation
readme_code_relation
docs_code_relation
config_script_relation
example_code_relation
api_doc_usage_relation
hyperlink_relation
citation_relation
definition_usage_relation
equation_or_figure_reference_relation
```

Weak edges encode proximity only:

```text
same_repo
same_directory
same_document
same_collection
same_domain
section_neighbor
```

The thesis claim should mainly rely on strong-edge coverage and context gain.
Weak edges are useful priors, but not sufficient evidence by themselves.

## 3. High-Precision Main Graph

The main thesis setting is deliberately narrower than the full graph. It uses
only relations that are expected to have high precision before large-scale
training:

```text
import_relation
test_source_relation
hyperlink_relation
```

`config_script_relation` is treated as optional after audit. README/docs/example
relations are still generated for full-graph and appendix experiments, but they
are not part of the primary claim until human review shows enough precision.

Each relation has an audit-derived reliability prior:

```text
q_r in [0, 1]
```

The effective edge weight is:

```text
w_hat(e) = sum_{r in labels(e) and r is allowed} DEFAULT_WEIGHTS[r] * q_r
```

The default reliability values live in
`configs/relations/main_high_precision.yaml`; after manual edge review,
`scripts/data/build_relation_reliability.py` can convert relation-level review
rates into an updated reliability YAML.

## 4. Packing Objective

For a packed sample:

```text
S = [d_1, d_2, ..., d_k]
```

and context length limit:

```text
tokens(S) <= L
```

Dependency-aware packing aims to maximize order-aware dependency utility:

```text
order_dependency(S)
= sum_{j=2..k} max_{i<j} w_hat(d_i -> d_j) / (k - 1)
```

under practical controls:

```text
high token utilization
low truncation
low redundancy
same fixed tokenizer
same fixed token budget
```

The current main implementation is `dependency_aware_high_precision_only`:

1. use only high-precision allowed relations for dependency scoring;
2. choose anchors with high-precision outgoing dependency signal;
3. fill dependency-linked candidates first;
4. use same-repo/collection token-fit candidates only as fillers, so windows do
   not become underutilized.

The older full-graph implementation remains available as
`dependency_aware_v2_strong_first`:

1. choose anchors with strong outgoing dependency signal;
2. fill dependency-linked candidates first;
3. then use token-fit candidates from the same repo/collection to avoid
   underfilled windows.

## 5. Order Ablation

To separate "putting related documents together" from "putting dependencies in
causal order", the project includes same-docs order ablations:

```text
dependency_aware_high_precision_only
dependency_aware_high_precision_random_order
dependency_aware_high_precision_reverse_order
```

The random/reverse variants reuse the same selected document sets as the
high-precision method and only reorder documents inside each window. If ordering
matters, dependency order should show higher order-aware dependency and stronger
context gain than random or reverse order.

## 6. Anti-Bias Design

Dependency metrics naturally favor dependency-aware packing, so they are not
enough for the final claim.

The experiment must also report:

```text
same_group_non_edge context gain
random_cross_group context gain
matched-utilization packing comparison
semantic similarity
redundancy
long-context validation loss
RepoBench or cross-file retrieval/completion
```

Expected context-gain ordering:

```text
reviewed dependency edge > same_group_non_edge > random_cross_group
```

If this ordering does not appear, the dependency model should be treated as
unvalidated even if packing edge coverage is high.

## 7. Ablations

Required ablations:

```text
dependency_aware_high_precision_only
dependency_aware_high_precision_random_order
dependency_aware_high_precision_reverse_order
dependency_aware_v2_strong_first
dependency_aware_v2_token_fit
dependency_aware_strong_edges_only
dependency_aware_no_same_directory
dependency_aware_no_same_repo
```

Relation-level ablations should be added if compute allows:

```text
without import
without source-test
without README/docs-code
without config-script
without non-code relations
```

These ablations answer whether the method works because of explicit dependency
evidence, weak grouping, or just higher token utilization.
