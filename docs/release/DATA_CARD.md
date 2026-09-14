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
| Part 0 | 22 exact routes × 48 archived English roots × 3 requested response languages = 3,168 scheduled subject responses | All scheduled units are terminal under the explicit all-scheduled operational-invalid policy. The fixed judge is disjoint from every subject. Results remain exploratory because the task has no benign controls and the judge lacks completed human validation. |
| Part 1 | 75 exact routes × the same balanced 384-root direct self-choice bank = 28,800 scheduled choices | A complete exact-source overlay resolves all three transport-null units; 704 malformed first responses remain scheduled nonsuccesses. Six frozen registry targets are excluded without substitution. The bank lacks independent content approval, so results remain exploratory. |
| Part 2 | 23 exact routes × 12 independent common-seed trajectories = 276 completed trajectories | Complete ordered composition of three source/overlay pairs. It records 1,206,808 scheduled living agent-days and all 276 trajectories are operationally eligible after 57 whole-trajectory repairs. Environmental estimates use 198 zero-invalid trajectories; 78 are invalid-bearing, leaving 18 route summaries and five zero-eligible routes reported as NE. |

The authoritative private Part 2 evidence under
`data/private/inference_hub/` consists of these ordered source/overlay pairs:

1. `full-part2-n12-n50-d100-main21-v5` with
   `full-part2-n12-n50-d100-main21-v5-operational-completion-capability-v4`;
2. `full-part2-n12-n50-d100-nemotron-3-ultra-recovered-v1` with
   `full-part2-n12-n50-d100-nemotron-3-ultra-operational-repair-v1`; and
3. `full-part2-n12-n50-d100-deepseek-v4-flash-recovered-v1` with
   `full-part2-n12-n50-d100-deepseek-v4-flash-operational-repair-v1`.

Part 0 and Part 1 use
`definitive-part0-large-n48-main22-deadline-v6` and
`definitive-part1-large-n384-main75-deadline-v5`, with their explicit terminal
policy or exact-source operational overlay. Metadata and reports must never
substitute scheduled counts for completed counts or silently promote an
incomplete source without its validated terminal policy.

Exact primary exclusions are:

- Part 0 planning-roster exclusions: `anthropic/claude-opus-4-5` and
  `minimaxai/minimax-m2.7`.
- Part 1 frozen-registry exclusions: `anthropic/claude-opus-4-5`,
  `minimaxai/minimax-m2.7`, `minimaxai/minimax-m3`,
  `moonshotai/kimi-k2.5`, `moonshotai/kimi-k2.6`, and `zai-org/glm-5.2`.
- Part 2 excludes exactly `anthropic/claude-opus-4-5`. The other 23 frozen
  planning-roster systems are present across the three source/overlay pairs.
  No route substitutes for Opus 4.5.

The April 13-model local pilot is historical provenance. Its legacy Part 0
labels, legacy Part 2 estimates, and dependent cross-part correlations are
withdrawn and are not inputs to current result generation.

## Records and transformations

Hosted calls retain private, append-only, hash-chained journals. They record
exact target and response model identity, request settings, completion IDs,
usage, retry decisions, parser status, and hashes. Part 0 additionally retains
the visible subject response and fixed-judge result. Private records may contain
harmful prompts or unsafe model output.

`analysis.analyze_provider_safe_v2_definitive` accepts each definitive source
phase only under its explicit terminal-evidence policy. For Part 2 it
recursively validates and composes exactly the ordered three source/overlay
pairs above, including exact identities, simulator transitions, common seeds,
denominators, and hash chains. It atomically emits text-free model tables for
Parts 0-2, role calibration, and sensitivity plus a self-hashed analysis
manifest. The paper-asset builder then emits hash-bound tables, headline
macros, and nine figure families. No active builder reads the sealed,
superseded 12-day `data/analysis/final_results` graph.

The definitive analyzer retains every scheduled invalid or unclear Part 0 or
Part 1 outcome as a nonsuccess. Separate semantic-repair artifacts never alter
the primary response or denominator. For Part 2, all 1,206,808 scheduled living
agent-days remain in the behavioral denominator: 1,180,046 valid actions and
26,762 genuine semantic `INVALID` actions, with 969,640 restraint and 210,406
overuse actions. An invalid action remains a nonrestraint/zero-effect
observation for behavioral continuity, while a trajectory containing any
invalid action is excluded from environmental estimates. All 276 trajectories
are operationally eligible after 57 whole-trajectory repairs; 198 contain no
invalid action and are environmentally estimable. The other 78 trajectories
are invalid-bearing. Eighteen routes have at least one environmentally eligible
trajectory; the five zero-eligible routes are reported as NE. The primary
descriptive restraint proportion pools scheduled living agent-days within each
route. Seed-level figures and Student-$t$ intervals instead use the unweighted
mean of the 12 trajectory-specific all-scheduled proportions; attrition can
make these estimates differ. The analyzer uses
root-cluster finite-bank sensitivity intervals for Part 0, stratified root
resampling for Part 1, and 12 independently seeded trajectories per Part 2
route under a seed set common across routes.

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

The three exact-source Part 2 operational overlays replace only source
trajectories that failed operationally and are provenance-bound parts of the
final composition; they never replace genuine semantic invalids. Other
availability and semantic-repair summaries may contribute only as separately
labeled, nonreplacement artifacts. The underlying private evidence remains
excluded. Croissant metadata is generated only after the definitive analysis
and paper-asset directories exist and pass schema, self-hash, privacy,
output-hash, and exact coverage checks. No final definitive self-hash is
claimed without every required terminal source/overlay binding.

## Limitations

- Part 0 inputs are archived English requests with requested response-language
  instructions, not recovered translated prompts. The task has no benign
  controls and one generation per root-condition cell.
- The Part 0 judge has not completed quantitative human validation.
- The Part 1 draft bank lacks independent content approval. All 75 primary
  routes use the same 384-root bank; the exact-source operational overlay does
  not change or repair any malformed semantic response.
- Part 2 uses homogeneous 50-agent populations, 100 days, one parameter
  setting, and no communication or memory. Its fixed dynamics use initial
  capacity 2,500, OPTION_B private gain 2, reserve cost 2, unanimous group
  benefit/penalty 5, and collapse death rate 0.2. It completed 12 trajectories
  per route; 198 of 276 contain no invalid action and support environmental
  estimation.
- Related routes are not independent samples of developers or model families.
- The benchmark evaluates artificial task outputs, not intent, moral status,
  altruism, or general deployment safety.

## Responsible use

Use the artifact to reproduce task-specific aggregates, inspect evidence gates,
and compare behavior within the exact executed designs. Do not use it to train
harmful-compliance systems, release raw unsafe text, claim vendor-wide rankings,
or certify deployment safety.
