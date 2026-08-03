# Release Documentation

This folder documents the paper-facing artifact for **Safety Beyond Refusal**
and its three-axis **Prosocial Readiness Bench**:

- `DATA_CARD.md`: dataset contents, collection process, validation, and safety policy.
- `MODEL_REGISTRY.md`: definitive primary scopes (22 Part 0 routes, 75 Part 1
  routes, and 19 Part 2 routes), exact exclusions without substitution, the
  fixed Part 0 judge, and the separate role, sensitivity, and local-control
  panels.
- `REPRODUCIBILITY.md`: commands for tests, validation, table generation, figures, and paper builds.
- `COMPUTE.md`: hosted execution controls, rate limits, and legacy local compute
  context.
- `LICENSES_AND_TERMS.md`: repository license, upstream Part 0 source licenses, and supplement release policy.
- `research-proposal.pdf`: historical, superseded May 2026 proposal; it is not
  the current execution protocol or evidence contract.
- `research-proposal-metadata.json`: deposit metadata carrying the same superseded-proposal warning.

## Anonymous-review boundary

The anonymous supplement supports validation of the definitive sanitized
aggregate graph and regeneration of paper-facing tables, figures, and checks.
It does not
include the private prompts, responses, reasoning, authenticated routes,
manifests, or journals needed to independently regenerate that result graph.
Croissant metadata is packaged without a dataset URL during anonymous review;
a real public landing page and hosted validation are external release steps.
