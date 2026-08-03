# Prosocial Readiness Bench Data Card

## Purpose

Prosocial Readiness Bench is the artifact for **Safety Beyond Refusal**. It
measures three observable behaviors without collapsing them into a latent moral
trait: refusal of harmful requests, welfare-preserving self-choice in one-shot
dilemmas, and preservation of a shared resource in repeated simulations.

## Executed data

The current deadline collection has three distinct scopes:

| Part | Actually executed | Evidence status |
| --- | --- | --- |
| Part 0 | 24 matched systems; 24 English source roots × 3 response-language instructions | Exploratory. No benign controls and no completed human validation of the fixed judge. |
| Part 1 | 75 routes × balanced 96 roots; 2 routes × balanced 12 roots; one separate GLM 5.1 route × 384 roots | Exploratory. The draft bank lacks independent content approval; the three sample-size scopes are not pooled. |
| Part 2 | 24 matched systems × 8 independent common-seed trajectories | Corrected engine, but below the intended 12-trajectory promotion threshold and without parameter sensitivity. |

`experiments/sota_cross_axis_panel.json` preserves the larger intended deadline
configuration of 48 Part 0 roots per condition and 12 Part 2 trajectories. The
private execution manifests are authoritative for the lower CLI limits that
were actually completed: 24 roots and eight trajectories. Metadata and reports
must never substitute intended counts for observed counts.

The April 13-model local pilot is historical provenance. Its legacy Part 0
labels, legacy Part 2 estimates, and dependent cross-part correlations are
withdrawn and are not inputs to current result generation.

## Records and transformations

Hosted calls retain private, append-only, hash-chained journals. They record
exact target and response model identity, request settings, completion IDs,
usage, retry decisions, parser status, and hashes. Part 0 additionally retains
the visible subject response and fixed-judge result. Private records may contain
harmful prompts or unsafe model output.

`analysis.build_final_results` validates complete private manifests and emits a
new immutable directory containing only:

- `final_results.json`, a self-hashed text-free result and evidence-status graph;
- `part0_model_rates.csv`;
- `part1_model_rates.csv`;
- `part2_model_metrics.csv`;
- `cross_axis_spearman.csv` only if every cross-axis gate passes;
- generated paper rows, macros, and figures bound by the result graph.

The final builder retains every scheduled invalid or unclear outcome in the
appropriate denominator. It uses prompt roots for Part 0 uncertainty,
game-domain stratified root resampling for Part 1, and independent trajectories
for Part 2.

## Release policy

The anonymous supplement distributes reviewed code, tests, documentation, and
sanitized final aggregates only. It excludes:

- `.env` files, API keys, credentials, and private endpoints;
- harmful prompts and source prompt banks;
- visible responses, reasoning, and raw provider payloads;
- private manifests, attempt ledgers, and journals;
- incomplete, interrupted, identity-mismatched, or hash-invalid runs;
- deprecated legacy evidence and withdrawn derived outputs.

Croissant metadata is generated only after the sanitized final-results
directory exists and passes schema, self-hash, privacy, output-hash, and exact
coverage checks. An old Croissant file is not evidence that these checks pass.

## Limitations

- Part 0 inputs are archived English requests with requested response-language
  instructions, not recovered translated prompts. The task has no benign
  controls and one generation per root-condition cell.
- The Part 0 judge has not completed quantitative human validation.
- The Part 1 draft bank lacks independent content approval. The two 12-root
  routes, 75 96-root routes, and single 384-root route have different support.
- Part 2 uses homogeneous five-agent populations, 12 steps, one parameter
  setting, no communication or memory, and eight trajectories per system.
- Related routes are not independent samples of developers or model families.
- The benchmark evaluates artificial task outputs, not intent, moral status,
  altruism, or general deployment safety.

## Responsible use

Use the artifact to reproduce task-specific aggregates, inspect evidence gates,
and compare behavior within the exact executed designs. Do not use it to train
harmful-compliance systems, release raw unsafe text, claim vendor-wide rankings,
or certify deployment safety.
