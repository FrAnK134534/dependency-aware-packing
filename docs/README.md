# Documentation Map

This folder is organized around the research workflow described in
`AGENTS.md`.

## Reading Order

1. [Project report for advisor](00_overview/advisor_project_report.md)
   - Chinese, plain-language explanation for advisor meetings.
   - Use this when explaining what the project is doing and why packing is the
     research target.

2. [Design rationale](00_overview/design_rationale.md)
   - Explains packing, dependency-aware packing, LoRA/QLoRA, and context
     windows in accessible language.

3. [Thesis scope](00_overview/thesis_scope.md)
   - Concise thesis positioning after upgrading the target environment to an
     8-GPU NVLink node.

4. [Macro experiment design](01_design/macro_experiment_design.md)
   - High-level experimental design: baselines, ablations, 8K/16K training, and
     paper outputs.

5. [Metrics definition](02_metrics/metrics_definition.md)
   - Quantifies packing quality and post-training long-context performance.

6. [Server deployment plan](03_server/server_deployment_plan.md)
   - How the project should be deployed and run on an 8-GPU NVLink server.

## Archive

[archive/legacy_low_resource_experiment_plan.md](archive/legacy_low_resource_experiment_plan.md)
keeps the older low-resource thesis plan for historical context. It should not
be treated as the current main plan.

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
