# Release Documentation

This folder documents the paper-facing artifact for **Safety Beyond Refusal**
and its three-axis **Prosocial Readiness Bench**:

- `DATA_CARD.md`: dataset contents, collection process, validation, and safety policy.
- `MODEL_REGISTRY.md`: frozen matched 24-system coverage and the 81-target Part
  1 registry, distinguishing 75 reportable targets, three operationally
  unavailable execution subjects, and three pre-execution unavailable targets,
  while preserving explicit n=12, n=96, and n=384 scopes.
- `REPRODUCIBILITY.md`: commands for tests, validation, table generation, figures, and paper builds.
- `COMPUTE.md`: hosted execution controls, rate limits, and legacy local compute
  context.
- `LICENSES_AND_TERMS.md`: repository license, upstream Part 0 source licenses, and supplement release policy.
- `research-proposal.pdf`: historical, superseded May 2026 proposal; it is not
  the current execution protocol or evidence contract.
- `research-proposal-metadata.json`: deposit metadata carrying the same superseded-proposal warning.

## Anonymous-review boundary

The anonymous supplement supports validation of the sealed sanitized result
graph and regeneration of paper-facing tables, figures, and checks. It does not
include the private prompts, responses, reasoning, authenticated routes,
manifests, or journals needed to independently regenerate that result graph.
Croissant metadata is packaged without a dataset URL during anonymous review;
a real public landing page and hosted validation are external release steps.
