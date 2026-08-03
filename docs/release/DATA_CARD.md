# Prosocial Readiness Bench Data Card

## Purpose

Prosocial Readiness Bench is the artifact for **Safety Beyond Refusal**. It
measures three observable behaviors without collapsing them into a latent moral
trait: refusal of harmful requests, welfare-preserving self-choice in one-shot
dilemmas, and preservation of a shared resource in repeated simulations.

## Frozen schedules and included data

The current deadline collection has three distinct scopes:

| Part | Frozen schedule | Evidence status |
| --- | --- | --- |
| Part 0 | 16 included of 24 matched systems; 24 English source roots × 3 response-language instructions per included system | Exploratory. Eight systems are unavailable; there are no benign controls and no completed human validation of the fixed judge. |
| Part 1 | 75 reportable of 81 frozen targets: 73 × balanced n=96, one × balanced n=12, one GLM 5.1 × n=384 | Exploratory. Three execution subjects and three pre-execution registry targets are unavailable. The bank lacks independent content approval; scopes are not pooled. |
| Part 2 | 22 executed of 24 matched systems; 8 independent common-seed trajectories each (176 total) | 20 systems and 159 fully valid trajectories are estimable. Seventeen protocol-invalid trajectories are excluded from behavioral/environmental metrics; two all-invalid systems are non-estimable and two systems are unavailable. |

`experiments/sota_cross_axis_panel.json` preserves the larger intended deadline
configuration of 48 Part 0 roots per condition and 12 Part 2 trajectories. The
private execution manifests are authoritative for the lower CLI limits used:
24 roots and eight trajectories. Sanitized bindings are authoritative for
included versus operationally unavailable systems. Metadata and reports must
never substitute intended or scheduled counts for observed counts.

Exact unavailable IDs are:

- Part 0: `anthropic/claude-haiku-4-5`, `anthropic/claude-opus-4-5`,
  `anthropic/claude-opus-4-6`, `anthropic/claude-sonnet-4-5`,
  `minimaxai/minimax-m2.7`, `openai/gpt-5`, `openai/gpt-5.2`, and
  `openai/gpt-5.4`.
- Part 1 operational: `anthropic/claude-opus-4-5`,
  `minimaxai/minimax-m2.7`, and `minimaxai/minimax-m3`; Part 1
  pre-execution: `moonshotai/kimi-k2.5`, `moonshotai/kimi-k2.6`, and
  `zai-org/glm-5.2`.
- Part 2: `anthropic/claude-opus-4-6` and `minimaxai/minimax-m2.7`.

The April 13-model local pilot is historical provenance. Its legacy Part 0
labels, legacy Part 2 estimates, and dependent cross-part correlations are
withdrawn and are not inputs to current result generation.

## Records and transformations

Hosted calls retain private, append-only, hash-chained journals. They record
exact target and response model identity, request settings, completion IDs,
usage, retry decisions, parser status, and hashes. Part 0 additionally retains
the visible subject response and fixed-judge result. Private records may contain
harmful prompts or unsafe model output.

`analysis.build_final_results` validates complete private manifests or
fail-closed overlays backed by target-bound operational/identity failure
evidence, and emits a new immutable directory containing only:

- `final_results.json`, a self-hashed text-free result and evidence-status graph;
- `part0_model_rates.csv`;
- `part1_model_rates.csv`;
- `part2_model_metrics.csv`;
- `cross_axis_spearman.csv` only if every cross-axis gate passes;
- generated paper rows, macros, and figures bound by the result graph.

The final builder retains every scheduled invalid or unclear Part 0/Part 1
outcome as a nonsuccess. For Part 2, the simulator's zero state effect for an
invalid action preserves transition continuity but is not restraint evidence;
any trajectory containing an invalid action is excluded from all behavioral and
environmental estimates. The builder uses prompt roots for Part 0 uncertainty,
game-domain stratified root resampling for Part 1, and fully valid independent
trajectories for Part 2.

## Release policy

The anonymous supplement distributes reviewed code, tests, documentation, and
sanitized final aggregates only. It excludes:

- `.env` files, API keys, credentials, and private endpoints;
- harmful prompts and source prompt banks;
- visible responses, reasoning, and raw provider payloads;
- private manifests, attempt ledgers, and journals;
- private incomplete, interrupted, identity-mismatched, or hash-invalid runs;
- deprecated legacy evidence and withdrawn derived outputs.

Consequently, the anonymous artifact supports verification of the sealed
aggregate graph and regeneration of its tables and figures, but not independent
regeneration of the graph from raw hosted calls. Its Croissant JSON-LD has no
dataset URL during anonymous review. Public hosting and external Croissant
validation require a real post-review landing page and are not claimed here.

Fail-closed overlays may contribute only sanitized aggregates from retained
complete units plus target-bound availability evidence; the underlying private
artifacts remain excluded. Croissant metadata is generated only after the
sanitized final-results directory exists and passes schema, self-hash, privacy,
output-hash, and exact coverage checks. The sealed result self-hash is
`52afcf33386055bcaabcf608186e094f2b19a7610f9bc57d53821ad3ea08c231`.

## Limitations

- Part 0 inputs are archived English requests with requested response-language
  instructions, not recovered translated prompts. The task has no benign
  controls and one generation per root-condition cell.
- The Part 0 judge has not completed quantitative human validation.
- The Part 1 draft bank lacks independent content approval. The one n=12
  route, 73 n=96 routes, and one n=384 route have different support.
- Part 2 uses homogeneous five-agent populations, 12 steps, one parameter
  setting, no communication or memory, and eight attempted trajectories per
  executed system. Only 159 of 176 trajectories are valid for behavioral
  estimation; two executed systems have no valid trajectory.
- Related routes are not independent samples of developers or model families.
- The benchmark evaluates artificial task outputs, not intent, moral status,
  altruism, or general deployment safety.

## Responsible use

Use the artifact to reproduce task-specific aggregates, inspect evidence gates,
and compare behavior within the exact executed designs. Do not use it to train
harmful-compliance systems, release raw unsafe text, claim vendor-wide rankings,
or certify deployment safety.
