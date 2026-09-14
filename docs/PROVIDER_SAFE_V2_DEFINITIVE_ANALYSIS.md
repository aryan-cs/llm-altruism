# Definitive provider-safe-v2 descriptive adapter

`analysis/analyze_provider_safe_v2_definitive.py` is the fail-closed bridge from
the terminal InferenceHub source phases and their exact-source operational
overlays to machine-readable descriptive tables. It does not edit the paper,
create human labels, or authorize a paper claim.

## Accepted inputs

The command requires terminal evidence for each of:

- the 22-model, 48-root-per-language Part 0 panel;
- the 75-model, 384-trial Part 1 panel;
- the ordered three-pair Part 2 composition: the 21-route main source and its
  complete cascading v4 overlay, then the one-route Nemotron Ultra source and
  overlay, then the one-route DeepSeek V4 Flash source and overlay;
- the six-sentinel Part 1 role-calibration panel; and
- the five-compatible-sentinel, 16-cell, two-common-seed deadline-sensitivity panel.

Every input must satisfy its explicit terminal-evidence policy, have a valid
self-hash, and bind the exact frozen runner and policy. The Part 0 source is
accepted only with the explicit all-scheduled operational-invalid policy; the
Part 1 and sensitivity sources require their complete exact-source operational
overlays. A conservative provider-safe-v2 policy is accepted. Accelerated
definitive outputs must also bind the exact current launcher source and its
self-consistent policy hash. Part 0 and Part 1 use their phase-specific
deadline runners, while role calibration and sensitivity bind their respective
exploratory runners and overlays. A launcher or overlay in the wrong campaign
class is rejected.
For the deadline run only, Part 1 may instead bind
`inference_hub_part1_deadline_accelerated.py`, whose still-bounded shared policy
uses global concurrency 24, provider concurrency 4, 12 global starts/second,
and 2.5 starts/second per provider. Its local executor may exceed the global
network ceiling solely to prevent tasks waiting on the one-request-per-route
semaphore from starving unrelated routes; the network limiter remains the
binding ceiling. This exception is not accepted for any other phase.
Part 0 may likewise use `inference_hub_part0_deadline_retry.py`, which is
bounded at global concurrency 16 and provider concurrency 3 and records every
same-payload HTTP-400 retry in its existing append-only attempt ledger. This
operational retry exception is accepted only for Part 0. Its local executor
may exceed the network ceiling solely to prevent tasks waiting on the stricter
one-request-per-upstream-provider lock from starving unrelated providers; the
shared network limiter remains the binding outer ceiling.
The deadline sensitivity matrix may bind
`inference_hub_sensitivity_deadline_accelerated.py`, capped at global
concurrency 24, provider concurrency 3, 12 global starts/second, and 2.5
starts/second per provider. It is accepted only for the sensitivity phase and
does not alter the frozen two-seed design or invalid-action policy.
The original-scale Part 2 contract is validated independently for all three
source/overlay pairs: 12 common seeds, 50 agents, 100 days, initial capacity
2,500, OPTION_B private gain 2, reserve cost 2, unanimous group
benefit/penalty 5, and collapse death rate 0.2. The ordered composition must
contain exactly 23 routes and 276 trajectories, declare only
`anthropic/claude-opus-4-5` as excluded without substitution, and prove 57
whole-trajectory repairs. Private journal references must remain inside each
run's `private/` directory, have mode `0600`, and match the full JSONL hash
chain, record count, tail hash, and file hash. Sanitized Part 2 artifacts must
remain inside `sanitized/` and match both their file hash and sealed payload
hash.

The fixed judge is checked against every subject at three levels: target ID,
exact route, and `(upstream_provider, model)` identity. Reserved judges must be
non-dispatchable in their subject runner. Any failed check aborts before an
output directory is created.

## Invalid-outcome policy

Primary rates use every scheduled unit or scheduled agent-day. A malformed
first attempt therefore remains in the denominator and cannot disappear through
complete-case filtering. Explicit repair markers, if a future bound artifact
contains them, are counted in `repaired_invalid_count`; they do not erase the
corresponding `first_attempt_invalid_count`. Current provider-safe-v2 runners do
not semantically retry, so that repair count should be zero.

Part 2 contracts remain separate. The 100-day table composes only the ordered,
source-bound three-pair panel. It retains all 1,206,808 scheduled living
agent-days in the behavioral denominator: 1,180,046 valid actions, 26,762
genuine semantic `INVALID` actions, 969,640 restraint actions, and 210,406
overuse actions. `restraint_rate_all_scheduled` is the primary descriptive
proportion that pools those agent-days within a route. The separate
`mean_trajectory_restraint_rate_all_scheduled` gives each of the 12 seed
trajectories equal weight for trajectory-level Student-$t$ inference; attrition
can make the two estimates differ. All 276 trajectories are operationally eligible after the 57
whole-trajectory repairs; only the 198 zero-invalid trajectories support
environmental estimates. The other 78 trajectories are invalid-bearing; 18
routes have at least one environmentally eligible trajectory, while five
zero-eligible routes are reported as NE for AURC, AUPC, reserve nondepletion,
and population retention. The deadline-sensitivity table separately preserves
its preregistered exploratory rule that malformed decisions are
nonrestraint/zero-effect observations, while identity or transport failure
blocks operational eligibility. The two are not pooled.

## Sensitivity verification

The adapter requires exactly 25 deadline-exploratory main effects (five exact
compatible sentinels by five factors), valid global-Holm metadata, and p-values
in `[0,1]`. It recomputes the complete main-effect and Holm rows from the hash-bound 160
trajectory records and rejects any difference. The call-order diagnostic must
be explicitly outside the Holm-25 family. The revised design excludes the one
exact route that cannot accept the common `top_p` control and substitutes no route.

## Run

```bash
uv run python -m analysis.analyze_provider_safe_v2_definitive \
  --part0 data/private/inference_hub/definitive-part0-large-n48-main22-deadline-v6 \
  --part0-terminal-policy all-scheduled-operational-invalid-v1 \
  --part1 data/private/inference_hub/definitive-part1-large-n384-main75-deadline-v5 \
  --part1-operational-repair data/private/inference_hub/definitive-part1-large-n384-main75-deadline-v5-operational-repair-v1 \
  --part2-source-overlay data/private/inference_hub/full-part2-n12-n50-d100-main21-v5 data/private/inference_hub/full-part2-n12-n50-d100-main21-v5-operational-completion-capability-v4 \
  --part2-source-overlay data/private/inference_hub/full-part2-n12-n50-d100-nemotron-3-ultra-recovered-v1 data/private/inference_hub/full-part2-n12-n50-d100-nemotron-3-ultra-operational-repair-v1 \
  --part2-source-overlay data/private/inference_hub/full-part2-n12-n50-d100-deepseek-v4-flash-recovered-v1 data/private/inference_hub/full-part2-n12-n50-d100-deepseek-v4-flash-operational-repair-v1 \
  --part2-declared-exclusion anthropic/claude-opus-4-5 \
  --role-calibration data/private/inference_hub/definitive-part1-role-calibration-v3 \
  --sensitivity data/private/inference_hub/definitive-part2-sensitivity-deadline-fast-v9 \
  --sensitivity-operational-repair data/private/inference_hub/definitive-part2-sensitivity-deadline-fast-v9-operational-repair-v1 \
  --output-dir data/processed/provider-safe-v2-definitive-analysis
```

Running this before every phase has the required terminal source/overlay
evidence is expected to fail. The adapter also refuses to overwrite an existing
output directory.

## Outputs

Each table is written as CSV and JSONL:

- `part0_models`
- `part1_models`
- `part2_models`
- `role_calibration_model_frames`
- `sensitivity_models`
- `sensitivity_main_effects`

`figure_aggregates.json` contains text-free model/language, model/game/domain,
trajectory, model/frame, and sensitivity-effect rows. `analysis_manifest.json`
binds every input manifest and the ordered three-pair Part 2 composition,
records judge-disjointness results and output row counts, and permanently
states:

```json
{
  "human_labels_generated": false,
  "exploratory_only": true,
  "confirmatory_or_paper_promotion_permitted": false
}
```

These outputs are evidence for reviewer-facing robustness and qualification,
not permission to replace the original paper's thesis or promote exploratory
results to confirmatory claims.

## Verification

```bash
UV_CACHE_DIR=/tmp/llm-altruism-uv-cache \
  uv run pytest -q tests/test_analyze_provider_safe_v2_definitive.py
```

The full test fixture uses the production matrix sizes and performs the actual
deadline-sensitivity Holm analysis. It also checks incomplete-manifest,
self-hash-tamper, and judge-overlap failures.
