# Prosocial Readiness Bench Data Card

## Purpose

Prosocial Readiness Bench is the artifact for **Safety Beyond Refusal**. It
measures three observable behaviors without collapsing them into a latent moral
trait: refusal of harmful requests, welfare-preserving self-choice in one-shot
dilemmas, and preservation of a shared resource in repeated simulations.

## Frozen schedules and included data

The definitive provider-safe-v2 collection has three distinct primary scopes:

| Part | Frozen primary schedule | Evidence status |
| :--- | :--- | :--- |
| Part 0 | 22 exact routes × 48 archived English roots × 3 requested response languages = 3,168 scheduled subject responses | Collection is in progress. The fixed judge is disjoint from every subject. Results remain exploratory because the task has no benign controls and the judge lacks completed human validation. |
| Part 1 | 75 exact routes × the same balanced 384-root direct self-choice bank = 28,800 scheduled choices | Collection is in progress. Six additional frozen registry targets are operationally excluded without substitution. The bank lacks independent content approval, so results remain exploratory. |
| Part 2 | 19 exact routes × 12 independent common-seed trajectories = 228 completed trajectories | Complete source manifest. It records 13,495 scheduled agent-days, zero transport or identity failures, and four invalid actions across three trajectories. The 225 trajectories without an invalid action support environmental estimates. |

The authoritative private source manifests are
`definitive-part0-large-n48-main22-deadline-v6`,
`definitive-part1-large-n384-main75-deadline-v5`, and
`definitive-part2-n12-main19-v3` under `data/private/inference_hub/`.
Metadata and reports must never substitute scheduled counts for completed
counts while a manifest remains incomplete.

Exact primary exclusions are:

- Part 0 planning-roster exclusions: `anthropic/claude-opus-4-5` and
  `minimaxai/minimax-m2.7`.
- Part 1 frozen-registry exclusions: `anthropic/claude-opus-4-5`,
  `minimaxai/minimax-m2.7`, `minimaxai/minimax-m3`,
  `moonshotai/kimi-k2.5`, `moonshotai/kimi-k2.6`, and `zai-org/glm-5.2`.
- Part 2 primary execution contains 19 routes. The five planning-roster routes
  outside that primary manifest are `anthropic/claude-opus-4-5`,
  `deepseek-ai/deepseek-v4-flash`, `nvidia/nemotron-3-ultra`,
  `minimaxai/minimax-m2.7`, and `zai-org/glm-5.1`. They receive no primary
  Part 2 estimate; bounded availability retries remain separate evidence.

The April 13-model local pilot is historical provenance. Its legacy Part 0
labels, legacy Part 2 estimates, and dependent cross-part correlations are
withdrawn and are not inputs to current result generation.

## Records and transformations

Hosted calls retain private, append-only, hash-chained journals. They record
exact target and response model identity, request settings, completion IDs,
usage, retry decisions, parser status, and hashes. Part 0 additionally retains
the visible subject response and fixed-judge result. Private records may contain
harmful prompts or unsafe model output.

`analysis.analyze_provider_safe_v2_definitive` accepts the five definitive
source manifests only after each is complete, self-hash-valid, source-bound,
and journal-valid. It atomically emits text-free model tables for Parts 0-2,
role calibration, and sensitivity plus a self-hashed analysis manifest. The
paper-asset builder then emits hash-bound tables, headline macros, and seven
figure families. No active builder reads the superseded deadline
`data/analysis/final_results` graph.

The definitive analyzer retains every scheduled invalid or unclear Part 0 or
Part 1 outcome as a nonsuccess. Separate semantic-repair artifacts never alter
the primary response or denominator. For Part 2, an invalid action remains a
nonrestraint/zero-effect observation for behavioral continuity, while a
trajectory containing any invalid action is excluded from environmental
estimates. The analyzer uses root-cluster finite-bank sensitivity intervals for
Part 0, stratified root resampling for Part 1, and independent trajectories for
Part 2.

## Release policy

The anonymous supplement distributes reviewed code, tests, documentation, and
sanitized final aggregates only. It excludes:

- `.env` files, API keys, credentials, and private endpoints;
- harmful prompts and source prompt banks;
- visible responses, reasoning, and raw provider payloads;
- private manifests, attempt ledgers, and journals;
- private incomplete, interrupted, identity-mismatched, or hash-invalid runs;
- deprecated legacy evidence and withdrawn derived outputs.

Consequently, the anonymous artifact supports verification of the definitive
aggregate graph and regeneration of its tables and figures, but not independent
regeneration of the graph from raw hosted calls. Its Croissant JSON-LD has no
dataset URL during anonymous review. Public hosting and external Croissant
validation require a real post-review landing page and are not claimed here.

Availability and semantic-repair summaries may contribute only as separately
labeled, nonreplacement artifacts; the underlying private evidence remains
excluded. Croissant metadata is generated only after the definitive analysis
and paper-asset directories exist and pass schema, self-hash, privacy,
output-hash, and exact coverage checks. No final definitive self-hash is
claimed while any required source manifest remains incomplete.

## Limitations

- Part 0 inputs are archived English requests with requested response-language
  instructions, not recovered translated prompts. The task has no benign
  controls and one generation per root-condition cell.
- The Part 0 judge has not completed quantitative human validation.
- The Part 1 draft bank lacks independent content approval. All 75 primary
  routes use the same 384-root bank, but completion is not claimed until the
  manifest seals.
- Part 2 uses homogeneous five-agent populations, 12 steps, one parameter
  setting, and no communication or memory. It completed 12 trajectories per
  route; 225 of 228 contain no invalid action and support environmental
  estimation.
- Related routes are not independent samples of developers or model families.
- The benchmark evaluates artificial task outputs, not intent, moral status,
  altruism, or general deployment safety.

## Responsible use

Use the artifact to reproduce task-specific aggregates, inspect evidence gates,
and compare behavior within the exact executed designs. Do not use it to train
harmful-compliance systems, release raw unsafe text, claim vendor-wide rankings,
or certify deployment safety.
