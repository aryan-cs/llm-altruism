# Budgeted Confirmatory Campaign

`experiments/confirmatory_campaign.py` is the strict orchestration layer for
the three-part InferenceHub campaign. It is isolated from the April pilot and
from `experiments/campaign.py`. The scientific contract is frozen in
`docs/CONFIRMATORY_PROTOCOL.md`.

## Mandatory inputs

Planning is side-effect free and fail-closed. It requires:

1. the exact ordered `current_sota` and `historical` evaluated cohorts plus the
   one-target `judge_only` cohort from an authenticated InferenceHub census;
2. `verification_status=verified` evidence for every exact route and the Part 0
   judge, including matching catalog responses and a structured smoke result;
3. a hash-pinned all-target evidence bundle from
   `experiments.misc.inference_hub_discovery verify-cohorts`;
4. a genuinely approved Part 0 multilingual registry and Part 1 384-root bank;
5. the corrected Part 2 prompt and N=10, horizon=30, capacity=150 contract;
6. the frozen request/token ledger; and
7. a clean 40-character Git commit plus source and dependency bundle hashes.

Display labels in `agents/agent_config.registry.json` are candidates only. They
cannot be executed or reported until authenticated evidence replaces them.
Duplicate IDs/routes, missing families, stale evidence, silent substitutions,
or identity mismatch are fatal planning errors.

The judge is resolved only from `judge_only`; it is never added to the
evaluated target union. Planning and resume hard-fail if its target ID,
provider+route, or upstream-provider+model matches any evaluated target. The
endpoint-evidence bundle must cover the evaluated cohorts followed by
`judge_only`, so the distinct judge route receives the same verification and
discovery-ledger attestation as the evaluated routes.

The required discovery command supplies both `--catalog-output` and
`--attempt-ledger`. The resulting schema-v2 bundle must contain an explicit
decision for every authenticated catalog route and an atomically written
reservation/outcome row for every smoke POST, including failures. Campaign
planning imports that full ledger rather than reconstructing successful calls.

Route evidence must be fresh when the immutable campaign is created. Each
native runner then accepts the exact hash-pinned campaign route after the
168-hour discovery window, while continuing to require byte-identical registry
identity and returned-model identity. This prevents a long campaign from aging
into an unrecoverable state without permitting route substitution or a new
unattested freeze.

## One-stage matrix

The final campaign uses one fixed stage. There is no variance-pilot/baseline
continuation and no extractor route.

For each verified target it schedules:

- Part 0: six smoke subject cells plus their six judge calls, followed by 1,752
  production subject calls and 1,752 judge calls;
- Part 1: one smoke root in each of 12 game-domain cells, followed by 384
  production calls;
- Part 2: 12 smoke agent-days, followed by 24 trajectories of at most 300
  agent-day calls each.

For 30 routes this is 333,750 successful POSTs including the per-route route
verification call. Attrition may reduce realized Part 2 calls, but planning
budgets the no-collapse maximum. One complete stored-response Part 0 rejudge
and a 10% retry reserve fit under the immutable 430,000-attempt ceiling.

The plan uses all 484 harmful prompt roots because their worst-case
language-specific half-width is about 4.4 points; subsampling to 150 would widen
it to about 8 points for only a modest operational saving. Twenty-four commons
trajectories similarly keep a run-SD-0.15 t half-width near 6.3 points.

The separately gated resolution-V sensitivity stage is not included in the
333,750-call primary matrix or its 430,000-attempt ceiling. Its revised
six-sentinel, 16-cell, twelve-common-seed design has a no-collapse maximum of
3,240,000 successful agent-day POSTs, a separate 10% transport ceiling of
3,564,000 attempts, and 103,680,000 maximum scheduled output tokens. It may not
run until a dedicated manifest freezes those counts and the conservative
input-byte token bound. Twelve seeds are the smallest even count above the
mathematical floor: 4,096 sign patterns give minimum two-sided p `0.000488`, so
the first hypothesis can pass a 30-test Holm threshold; six seeds could not.

## Durable budget ledger

The manifest stores a self-hashed, role-specific budget. Before every physical
InferenceHub POST, the caller atomically reserves the request hash, a
one-token-per-UTF-8-byte input upper bound, and the full role output-token cap in a
shared ledger. The reservation is fsynced before dispatch; a crash may
overcount but cannot hide a request. Resume replays the ledger and rejects any
mismatch. No request is dispatched if its conservative reservation would
exceed a role cap, 430,000 attempts, or 1.5 billion conservatively reserved
tokens.

Each native attempt record also retains the exact complete-request hash returned
by its pre-dispatch reservation. The final data lock requires the non-discovery
ledger hashes and native smoke/production attempt hashes to match as an exact
multiset, so retries, failures, omissions, and duplicate accounting are all
fail-closed.

Provider-reported usage remains in each response audit for reconciliation.
Missing or internally inconsistent usage makes the route incomplete. The
frozen schedule and role output caps must fit before the first scientific
request; live conservative reservations enforce the combined token ceiling.

## Smoke dependencies

Every scientific unit names a matching same-target, same-part smoke. Part 0
smoke exercises subject and judge. Part 1 smoke exercises direct structured
X/Y parsing across all 12 game-domain cells. Part 2 smoke uses four agents for
three days and exercises state feedback, direct structured actions, transition
checks, and collapse behavior. Smoke artifacts carry hash-bound exclusion
markers and cannot enter analysis.

A missing, failed, or incomplete smoke marks dependent units
`blocked_smoke`. It does not authorize a substitute route or a different model
from the same family.

## Execution order and failures

Target/part/wave order is SHA-256 block-randomized. Waves interleave routes so
one provider or model is not confounded with an entire early or late calendar
period. Commands use argv arrays with `shell=False`; private logs, attempts,
raw responses, and manifests use mode 0600 and atomic fsync-backed replacement.

Only network failures, timeouts, HTTP 408/429, and 5xx failures without a
retained response are retryable. Each retry uses identical bytes and at most
two retries. Truncation, malformed structured output, and semantic invalidity
are retained as outcomes and never retried. Identity mismatch is fatal.

Any retry-exhausted job or exact-identity failure quarantines all remaining
jobs for the same target and part. The failed job and each blocked job remain
durable manifest state. The campaign is incomplete, and restarting that
target-part requires a separately frozen campaign. Stopping never reads
scientific labels or summaries.

## Completion and locking

A target is complete only when all three scientific parts pass native artifact
verification. The primary fixed-panel data lock requires every planned
scientific job and therefore refuses incomplete routes or scientific
exclusions. Failures remain in the campaign coverage report, but no
confirmatory estimator is released from an incomplete campaign.

The data lock revalidates route evidence, approved inputs, prompt and source
hashes, the clean commit, every request/response and attempt record, ledger
totals, exact Part 0/1 coverage, all 24 Part 2 trajectories per completed
target, transition replay, and smoke exclusions. Outcomes remain private until
this structural lock is written.

## Dry run

The production CLI remains intentionally unusable while any registry target is
`unverified` or either human-approved input is absent. Once those inputs exist,
the dry run must print exact routes, exclusions, role counts, worst-case calls,
token bounds, source hashes, and the immutable plan hash without creating a
campaign directory or making a provider request.

```bash
python -m experiments.confirmatory_campaign \
  --campaign-id confirmatory-budgeted-v1 \
  --judge-target-id judge.nvidia-evals-nemotron-3-30b-a3b \
  --part0-registry /absolute/private/part0-registry.json \
  --part0-registry-sha256 <sha256> \
  --part1-bank /absolute/private/part1-bank.json \
  --part1-bank-sha256 <sha256> \
  --endpoint-evidence /absolute/private/all-target-evidence.json \
  --endpoint-evidence-sha256 <sha256> \
  --dry-run
```

Remove `--dry-run` only after the printed plan matches the preregistration.
Resume accepts only the campaign ID and revalidates every pinned byte and
ledger entry before continuing.

Target shards may execute in parallel only when a master merge validator
proves disjoint route membership, identical frozen inputs, complete smoke
lineage, and exact ledger addition. Until that validator exists, a shard is not
a publication-complete campaign.
