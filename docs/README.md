# Documentation Map

This folder is organized around the research workflow described in
`AGENTS.md`.

## Reading Order

For repository structure and module responsibilities, read
[../PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md).

1. [Project report](00_overview/project_report.md)
   - Chinese, plain-language explanation for advisor meetings.
   - Use this when explaining what the project is doing and why packing is the
     research target.

2. [Thesis scope](00_overview/thesis_scope.md)
   - Concise thesis positioning after upgrading the target environment to an
     8-GPU NVLink node.

3. [Macro experiment design](01_design/macro_experiment_design.md)
   - High-level experimental design: baselines, ablations, 8K/16K training, and
     paper outputs.

4. [Method formalization](01_design/method_formalization.md)
   - Formal definitions for documents, dependency edges, strong/weak relations,
     the packing objective, anti-bias controls, and ablations.

5. [Metrics definition](02_metrics/metrics_definition.md)
   - Quantifies packing quality and post-training long-context performance.

6. [Server deployment plan](03_server/server_deployment_plan.md)
   - How the project should be deployed and run on an 8-GPU NVLink server.

7. [Pre-server optimization runbook](03_server/pre_server_optimization_runbook.md)
   - Concrete commands for real-tokenizer packing, edge review, cap
     sensitivity, dependency-sensitive validation data, and QLoRA smoke runs.

8. [Pre-training freeze protocol](03_server/pretraining_freeze_protocol.md)
   - Go/no-go procedure before spending 8-GPU time on formal training.

9. [Server training handoff](03_server/server_training_handoff.md)
    - Practical handoff document for an advisor or server runner: current
      status, target-tokenizer packing, QLoRA smoke run, and post-training
      evaluation commands.

## Archive

[archive/legacy_low_resource_experiment_plan.md](archive/legacy_low_resource_experiment_plan.md)
keeps the older low-resource thesis plan for historical context. It should not
be treated as the current main plan.

[archive/legacy_design_rationale.md](archive/legacy_design_rationale.md)
keeps the older plain-language design rationale. It has been superseded by
`00_overview/project_report.md` and `01_design/method_formalization.md`.

## Current Main Line

The current paper direction is:

> Dependency-aware packing for efficient long-context adaptation.

The first paper-quality scenario is multi-source code repository context:

```text
source code
tests
README
docs
config files
examples
issues / PRs
commit messages
benchmark logs
API usage examples
```

The key experimental rule is:

```text
Fix model, context length, token budget, optimizer, and LoRA/QLoRA settings.
Only change the packing method.
```
