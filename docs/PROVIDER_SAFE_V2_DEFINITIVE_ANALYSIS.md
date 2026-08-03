# Definitive provider-safe-v2 descriptive adapter

`analysis/analyze_provider_safe_v2_definitive.py` is the fail-closed bridge from
the five new InferenceHub campaigns to machine-readable descriptive tables. It
does not edit the paper, create human labels, or authorize a paper claim.

## Accepted inputs

The command requires one manifest (or run directory) for each of:

- the 22-model, 48-root-per-language Part 0 panel;
- the 75-model, 384-trial Part 1 panel;
- the 19-model, 12-trajectory Part 2 matched panel;
- the six-sentinel Part 1 role-calibration panel; and
- the six-sentinel, 16-cell, two-common-seed deadline-sensitivity panel.

Every manifest must be `complete: true`, have a completion timestamp and a
valid self-hash, and bind `inference_hub_provider_safe_v2.py`. A conservative
provider-safe-v2 policy is accepted. Accelerated definitive outputs must also
bind the exact current launcher source and its self-consistent policy hash:
main Part 0--2 require `inference_hub_main_accelerated.py` with provider
concurrency 2 and 1.5 starts/second, while role calibration and sensitivity
require `inference_hub_exploratory_accelerated.py` with provider concurrency 3
and 2 starts/second. A launcher in the wrong campaign class is rejected.
For the deadline run only, Part 1 may instead bind
`inference_hub_part1_deadline_accelerated.py`, whose still-bounded shared policy
uses global concurrency 24, provider concurrency 4, 12 global starts/second,
and 2.5 starts/second per provider. This exception is not accepted for any
other phase.
Part 0 may likewise use `inference_hub_part0_deadline_retry.py`, which is
bounded at global concurrency 16 and provider concurrency 3 and records every
same-payload HTTP-400 retry in its existing append-only attempt ledger. This
operational retry exception is accepted only for Part 0.
Private journal references must remain inside the run's
`private/` directory, have mode `0600`, and match the full JSONL hash chain,
record count, tail hash, and file hash. Sanitized Part 2 artifacts must remain
inside `sanitized/` and match both their file hash and sealed payload hash.

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

Part 2 contracts remain separate. The matched-panel table reports its stored
trajectory metrics under that runner's contract. The deadline-sensitivity
table separately preserves its preregistered exploratory rule that malformed
decisions are nonrestraint/zero-effect observations, while identity or
transport failure blocks operational eligibility. The two are not pooled.

## Sensitivity verification

The adapter requires exactly 30 deadline-exploratory main effects (six sentinels
by five factors), valid global-Holm metadata, and p-values in `[0,1]`. It
recomputes the complete main-effect and Holm rows from the hash-bound 192
trajectory records and rejects any difference. The call-order diagnostic must
be explicitly outside the Holm-30 family.

## Run

```bash
uv run python -m analysis.analyze_provider_safe_v2_definitive \
  --part0 data/private/inference_hub/definitive-part0-large-n48-main22-deadline-v4 \
  --part1 data/private/inference_hub/definitive-part1-large-n384-main75-deadline-v3 \
  --part2 data/private/inference_hub/definitive-part2-n12-main19-v3 \
  --role-calibration data/private/inference_hub/definitive-part1-role-calibration-v2 \
  --sensitivity data/private/inference_hub/definitive-part2-sensitivity-deadline-v3 \
  --output-dir data/processed/provider-safe-v2-definitive-analysis
```

Running this before all five manifests are complete is expected to fail. The
adapter also refuses to overwrite an existing output directory.

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
binds the five input manifests, records judge-disjointness results and output
row counts, and permanently states:

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
