# Paper completion checkpoint

Updated: 2026-09-13 (America/New_York)

## North star

- The exact paper title is **Safety Beyond Refusal**.
- This is an expanded, corrected resubmission of the original three-axis paper,
  not a benchmark-audit replacement. The central question remains whether
  harmful-request refusal, welfare-preserving dyadic choice, and repeated
  commons restraint expose complementary observable behavior.
- Reviewer-driven changes are broader model coverage, larger independent-unit
  counts, judge--subject separation, corrected task contracts, stronger
  statistics, clearer validity gates, and reproducible artifacts.
- No axis is a latent altruism score. No cross-axis association is promoted
  while Part 0 lacks human label validation and Part 1 lacks independent prompt
  approval.

## Current source state

- Branch: `master` (no additional branch or tag).
- Latest pushed repository commit before this terminal release:
  `22a06df`.
- Prior definitive paper and release-artifact commit: `5cd54a0` (superseded by
  the completed 100-day Part 2 evidence).
- Final supplemental retry and repair-provenance commit preceding this update:
  `9272c67`.
- Important preceding commits:
  - `d11c739`: deterministic, privacy-safe clean-room supplement;
  - `b94e65a`: deterministic paper headline macros and six asset families;
  - `ba28b5f`: role-invalid repair and availability-snapshot merge support;
  - `a75e6b9`: bounded two-seed deadline sensitivity and Part 1 invalid repair.
- The exact title remains in
  `docs/conference_submission/conference_submission.tex`.
- The concession-heavy draft was deleted and rewritten from a blank manuscript
  around the original three-stage scientific question. The single tracked
  generated review-artifact path is
  `docs/conference_submission/conference_submission.pdf`; it has been rebuilt
  in place from the 100-day release source. There is no second working-draft
  PDF.
- `.env` is ignored, mode `0600`, and never enters Git or release artifacts.

## Terminal definitive evidence and release status

All campaign roots below are private and excluded from Git and the anonymous
supplement. Every runner used append-only, fsync-backed, hash-chained journals
and fail-closed resume. All Part 2 collection and repair processes have ended.
The fixed Part 0 judge is
`judge.nvidia-evals-nemotron-3-30b-a3b`; it is disjoint from every subject at
target ID, exact route, and upstream provider/model identity.

### Primary main scope

- Part 0: `data/private/inference_hub/definitive-part0-large-n48-main22-deadline-v6`
  - 22 exact systems;
  - 48 archived English harmful-request roots crossed with three requested
    response languages (144 scheduled subject responses/system);
  - one fixed disjoint judge for every subject response;
  - a bounded retry lane uses global concurrency 16 and provider concurrency 3
    and retries identical HTTP-400 payloads within the eight-attempt budget;
    every failed attempt and the retry policy remain manifest-bound. Its local
    executor has 64 slots so work waiting on the stricter process-local
    one-request-per-upstream-provider lock cannot starve unrelated providers;
    this does not raise the global or provider network ceilings. The campaign
    has terminalized all 3,168 scheduled units: 3,127 visible subject
    responses, 73 semantic-invalid responses, 41 exhausted exact-route subject
    calls, and 16 visible responses whose two judge batches exhausted the
    eight-attempt connection-retry budget. The 57 operational nonsuccesses are
    retained in the all-scheduled denominator but are never interpreted as
    refusal, compliance, or semantic model behavior. The fixed judge output is
    descriptive pending the prespecified human-validation gate. The
    superseded v3/v4/v5 partial runs remain preserved but are not paper inputs.
- Part 1: `data/private/inference_hub/definitive-part1-large-n384-main75-deadline-v5`
  - 75 exact routes, each on the same balanced 384-root self-choice bank;
  - malformed first responses remain all-scheduled nonsuccesses;
  - six unavailable targets remain explicit rather than substituted;
  - a source-bound deadline lane uses bounded global concurrency 24,
    provider concurrency 4, 12 global starts/second, and 2.5 starts/second per
    provider. The local executor has 256 slots so tasks waiting on the strict
    one-request-per-exact-route semaphore cannot starve unrelated routes; it
    does not raise any network or provider ceiling. The superseded v2/v3/v4
    partial runs remain preserved but are not paper inputs;
  - all 28,800 scheduled roots are terminal. The source retained 28,797
    visible responses, including 704 genuine format-invalid responses, plus
    three transport-null rows. A separate immutable, source-bound operational
    overlay repaired exactly those three rows, leaving 28,800 visible effective
    responses, 28,096 format-valid responses, and all 704 genuine invalids in
    the all-scheduled denominator. No semantic response was regenerated.
- Historical 12-day Part 2 evidence (sealed, non-paper):
  `data/private/inference_hub/definitive-part2-n12-main19-v3` remains immutable
  with 19 routes, 228 trajectories, five agents, 12 days, 13,495 realized
  agent-days, four genuine semantic-invalid actions, and 225 environmentally
  estimable trajectories. The sealed
  `data/analysis/final_results/final_results.json` and this 12-day source are
  historical provenance only. They must never be altered, merged into, or
  substituted for the definitive 100-day paper evidence.
- Definitive original-scale Part 2 evidence (complete):
  `experiments/sota_cross_axis_part2_100day_panel.json` freezes 23 observed
  exact routes from the 24-target planning roster, 12 independent common seeds,
  50 agents, 100 days, initial capacity 2,500, OPTION_B private gain 2, reserve
  cost 2, unanimous group benefit/penalty 5, collapse death rate 0.2, and the
  `INVALID`-as-zero-effect/nonrestraint policy. Exactly
  `anthropic/claude-opus-4-5` is excluded, without substitution.
- The paper composition is the following ordered set of immutable source and
  complete whole-trajectory operational-overlay pairs:
  1. 21-route main source
     `data/private/inference_hub/full-part2-n12-n50-d100-main21-v5` with final
     cascading overlay
     `data/private/inference_hub/full-part2-n12-n50-d100-main21-v5-operational-completion-capability-v4`;
  2. Nemotron 3 Ultra singleton source
     `data/private/inference_hub/full-part2-n12-n50-d100-nemotron-3-ultra-recovered-v1`
     with `data/private/inference_hub/full-part2-n12-n50-d100-nemotron-3-ultra-operational-repair-v1`;
  3. DeepSeek V4 Flash singleton source
     `data/private/inference_hub/full-part2-n12-n50-d100-deepseek-v4-flash-recovered-v1`
     with `data/private/inference_hub/full-part2-n12-n50-d100-deepseek-v4-flash-operational-repair-v1`.
- The v4 overlay completed at `2026-09-13T20:10:51.961115Z`. Its terminal
  manifest evidence SHA-256 is
  `8b2034df599ec7a6e2abbdf5964127bb0987fd1f95372be7343e06baef715079`.
  It binds the immutable main source and immutable v3 parent, inherits all 37
  valid parent repairs, reruns only the one unresolved trajectory from day 1,
  and records one cascading success with zero unresolved trajectories. The
  frozen v4 runner SHA-256 remains
  `3d18d7a0ab11cc43e5648c60abc24b243e18e5195437cc345cd7479e43301dec`.
- The external credential file at
  `/Users/aryagupta/Desktop/cdo-better-gos/.env` supplied three configured
  NVIDIA InferenceHub account slots. They were opened through the runner's
  held-file-descriptor interface and blind-qualified without reading values
  into logs or source. Two slots authenticated and identity-matched the exact
  route; the third failed catalog authentication and was quarantined. The two
  qualified slots were used under separate high-latency-v2 rate limiters and
  locked round-robin dispatch. No API key value was printed, persisted in
  provenance, committed, or included in a release artifact.
- Formal production validation passed with this read-only command:

  ```bash
  .venv/bin/python -m analysis.validate_inference_hub_part2_operational_overlays \
    --pair data/private/inference_hub/full-part2-n12-n50-d100-main21-v5/private/manifest.json data/private/inference_hub/full-part2-n12-n50-d100-main21-v5-operational-completion-capability-v4/private/manifest.json \
    --pair data/private/inference_hub/full-part2-n12-n50-d100-nemotron-3-ultra-recovered-v1/private/manifest.json data/private/inference_hub/full-part2-n12-n50-d100-nemotron-3-ultra-operational-repair-v1/private/manifest.json \
    --pair data/private/inference_hub/full-part2-n12-n50-d100-deepseek-v4-flash-recovered-v1/private/manifest.json data/private/inference_hub/full-part2-n12-n50-d100-deepseek-v4-flash-operational-repair-v1/private/manifest.json
  ```

  The validator held a consistent read-only snapshot; checked exact identities,
  all manifest/artifact/journal hashes, attempt bindings, simulator transitions,
  source replacement rules, denominators, and sanitized aggregates; and replayed
  all sources and repair rounds through the frozen simulator. It passed exactly
  three pairs, 23 routes, 276 trajectories, and the same 12 common seeds. It
  found 57 source operational-failure trajectories resolved across 85 repair
  rounds and all 276 trajectories operationally eligible. No report artifact
  was emitted; the formal result was stdout-only.
- The validator reported panel ID
  `sota_cross_axis_part2_corrected_original_scale_100d_v1`, base seed 20260802,
  and common environment seeds 945353965, 674434863, 373620026, 161949049,
  305447851, 1694051603, 827330313, 526445251, 1853673051, 1941677986,
  1517370141, and 597333924. It confirms that the cascading main overlay binds
  its credential pool and all execution settings. The older singleton overlays
  do not uniformly manifest-bind a configured attempt ceiling or the
  nonsemantic local scheduling settings (`trajectory_workers`,
  `participant_workers`, and initial exponential backoff); their journals
  nevertheless expose and pass the eight-attempt dispatch ceiling. This
  limitation is retained explicitly rather than silently inferred away.
- The validated all-scheduled behavioral denominator is 1,206,808 living
  agent-days: 1,180,046 valid actions and 26,762 genuine semantic `INVALID`
  actions. The valid actions divide into 969,640 restraint and 210,406 overuse.
  Exactly 198/276 trajectories contain zero semantic invalids and are eligible
  for environmental metrics. The all-scheduled restraint rate is
  `0.8034749521050573`; the valid-action restraint rate is
  `0.8216967813119149`.
- The earlier main repair roots are
  `data/private/inference_hub/full-part2-n12-n50-d100-main21-v5-operational-repair-v1`,
  `data/private/inference_hub/full-part2-n12-n50-d100-main21-v5-operational-repair-multikey-v2`,
  and
  `data/private/inference_hub/full-part2-n12-n50-d100-main21-v5-operational-repair-multikey-v3`.
  They are immutable diagnostic, non-paper evidence. V3 remains terminal at
  37/38 successful repairs with the sole unresolved Gemini 3.5 Flash trajectory
  caused by one failed credential-pool position. None of v1--v3 may be resumed,
  merged directly, or substituted for the complete v4 overlay. The main,
  Nemotron, and DeepSeek source campaigns are also terminal and must not be
  restarted.

The evidence-collection phase is closed. Part 2 used the manifest-bound
high-latency-v2 policy independently per qualified account: global concurrency
60, provider concurrency 10, global 12 starts/second, and provider 2.5
starts/second. Parts 0 and 1 used their separately source-bound bounded deadline
policies.

### Role calibration (complete, identity-safe, operationally repaired)

- Path: `data/private/inference_hub/definitive-part1-role-calibration-v3`.
- Six frozen sentinels × 96 roots × three distinct frames (advice, observer
  evaluation, prediction) × four counterbalances = 6,912 requests.
- Frames remain separate estimands and are never pooled into self-choice.
- The exploratory accelerated policy is source-bound at global concurrency 12,
  provider concurrency 3, global 8 starts/second, and provider 2 starts/second.
- The preserved v2 campaign retained all 6,912 rows but correctly remained
  incomplete after five DeepSeek V4 Pro responses named a different served
  route. The v3 runner records such a 200-response identity drift as a failed
  attempt and retries it within the eight-attempt budget before retaining any
  primary row; v2 is excluded from all analysis.
- V3 is COMPLETE with 6,912 visible responses and zero identity mismatches.
  Four original transport-null rows were retried through an immutable,
  hash-bound operational overlay; all four exact retries succeeded. The 375
  visible format-invalid model responses were not retried and remain in the
  frozen all-scheduled denominator. Original raw journals remain byte-identical.
- The definitive analyzer validates overlay source, retry lineage, route,
  trial, prompt, request, original-record, payload, and response-identity
  bindings before merging the four operational replacements.

### Part 2 deadline sensitivity (complete, operationally repaired)

- Path: `data/private/inference_hub/definitive-part2-sensitivity-deadline-fast-v9`.
- Five exact compatible sentinels × 16 resolution-V cells × two common
  environment seeds = 160 trajectories, with 14,400 as the execution ceiling;
  agents that die stop producing later calls.
- Five factors vary: capacity per initial agent, depletion units, collapse death
  rate, society size, and horizon.
- Invalid visible actions are retained as nonrestraint/zero-effect observations;
  transport or identity failure blocks operational eligibility.
- The panel is explicitly deadline-exploratory and underpowered. Twenty-five
  sentinel-by-factor Holm rows document the prespecified family; they do not
  authorize confirmatory robustness claims.
- Four source trajectories were operationally ineligible after six transport
  failures and zero identity mismatches. A complete fresh 12-trajectory exact-
  contract subset supplied four immutable whole-trajectory replacements from
  day 1. The effective panel has all 160 trajectories, zero transport or
  identity failures, and no semantic-invalid trajectory retry. Across the 25
  prespecified effects, the largest absolute high-minus-low AURC contrast is
  0.2881 and none is significant after global Holm adjustment.
- The revised v2 design excludes only the frozen Claude Sonnet 4.6 exact route,
  which cannot accept the common `top_p` control. It substitutes no route or
  model and reuses no incomplete v8 evidence. Its frozen shared policy used
  global concurrency 24, provider concurrency 3, 12 global starts/second, and
  2.5 starts/second per provider. The writer used 15 campaign workers to
  balance network throughput against local fsync pressure;
  earlier partial runs remain preserved but are not paper inputs.

## Availability and invalid retries

- Three fresh compatibility rounds were executed. The latest same-policy
  visible-content snapshot again selected 77 of 84 SOTA target families.
  MiniMax M2.7 again returned target-bound HTTP 503 and Kimi K2.5 target-bound
  HTTP 529 on their exact minimal routes; both remain unavailable without
  silent substitution and may be retried again later.
- A fourth low-rate generic exact-candidate sweep was retained separately as
  `sota-visible-compatibility-retry-evidence-20260803-v4.json`. It is
  intentionally not a registry input because unrelated structured-schema
  checks left the generic bundle incomplete. Its exact minimal calls still
  provide supplemental availability evidence: MiniMax M2.7 returned HTTP 500
  and Kimi K2.5 again returned HTTP 529. Neither route is promoted or
  substituted.
- The 2026-08-04 bounded heartbeat re-probe recovered both formerly unavailable
  exact routes: `nvidia/minimaxai/minimax-m2.7` and
  `nvidia/moonshotai/kimi-k2.5` each returned an identity-matched structured
  smoke response under the frozen temperature-0, seed-20260801, 8-token
  contract. Evidence is retained at
  `heartbeat-minimax-m27-20260804-v1.json` and
  `heartbeat-kimi-k25-20260804-v1.json`. These smoke recoveries do not alter a
  frozen primary panel, substitute a model, or promote partial evidence into
  the paper; complete supplemental fixed schedules remain separate work.
- Supplemental availability runs are private, separate, and never merged into
  the primary paper evidence:
  - Part 0 Opus 4.5 fresh full replicate:
    `data/private/inference_hub/definitive-part0-unavailable-retry-opus45-v5`;
  - Part 1 Opus 4.5, MiniMax M3, Kimi K2.6, and GLM 5.2:
    `data/private/inference_hub/definitive-part1-unavailable-retry-n384-v2`;
  - Part 2 Opus 4.5, DeepSeek V4 Flash, Nemotron 3 Ultra, and GLM 5.1:
    `data/private/inference_hub/definitive-part2-availability-retry-n12-v1`.
- The Part 2 supplemental schedule is a complete, historical 12-day diagnostic
  only: 48/48 trajectories, 2,824 realized agent-days, zero transport failures,
  zero identity mismatches, and five genuine semantic-invalid actions retained.
  Its Opus 4.5, DeepSeek V4 Flash, Nemotron 3 Ultra, and GLM 5.1 estimates do
  not enter or alter the 100-day paper panel; in particular, they do not erase
  the sole primary Opus 4.5 exclusion.
- The four-route Part 1 source terminalized all 1,536 roots with 1,529 visible
  responses, seven transport-null rows, zero genuine format-invalid responses,
  and zero identity mismatches. A separate immutable operational overlay at
  `definitive-part1-unavailable-retry-n384-v2-operational-repair-v2` recovered
  all seven exact rows without modifying the source or regenerating any
  semantic response. Effective welfare-preserving rates are Opus 4.5 99.2%,
  MiniMax M3 0.0%, Kimi K2.6 0.0%, and GLM 5.2 0.8%.
- The fresh full Opus 4.5 Part 0 replicate terminalized all 144 units with 121
  refusals, two compliance outcomes, 12 genuine semantic invalids, and nine
  exact-route transport failures retained as operational invalids. Judge
  failures and identity mismatches are both zero. The fail-closed combined
  supplemental analyzer is not run because this manifest correctly remains
  non-COMPLETE after the nine transport failures; the isolated row never enters
  the primary paper evidence.
- `experiments.misc.inference_hub_part1_semantic_invalid_repair` and
  `experiments.misc.inference_hub_part1_role_semantic_invalid_repair` operate
  only on COMPLETE source manifests. They preserve every primary row and
  denominator, use up to eight periodic rounds spaced by 30 seconds, and emit
  separate text-free repair evidence. Run them immediately after their source
  campaigns complete.
- `analysis.analyze_availability_retry_panels` validates and reports the three
  supplemental panels separately; it never replaces or merges primary rows.

## Completed local controls and current release state

- Four pinned local Hugging Face controls completed the full 384-root Part 1
  bank (1,536 real generations) under
  `data/private/local_hf/part1-large-n-20260802-v3-hardened`.
- They are reported as separate exploratory execution-scale controls, never as
  hosted-route substitutes.
- `analysis.analyze_provider_safe_v2_definitive` accepts the terminal Part 0
  policy, the Part 1 and sensitivity exact-source operational overlays, the
  complete role-calibration source, and exactly the ordered three-pair Part 2
  composition above. It produces per-model tables while preserving every
  first-attempt invalid denominator.
- `analysis.build_provider_safe_v2_paper_assets` produces nine vector-PDF and
  300-dpi PNG figure families, including current-route refusal bars, the
  75-route cooperation rank line, a matched 22-route task profile, and the
  original red/green route-profile idiom; full model-row LaTeX tables; and deterministic
  headline macros including Part 2 AUPC, reserve-nondepletion, and
  population-retention summaries. Figures use the original submission palette
  and a Times-compatible NeurIPS-template font. Row-based model comparisons
  group provider families contiguously with newer frozen route versions above
  older ones; global outcome order is retained only where a figure's analytical
  purpose is the rank distribution. It validates the two-seed
  sensitivity contract and generates no cross-axis composite.
- The terminal definitive analysis is provenance-bound to the ordered 100-day
  composition above. Its evidence SHA-256 is
  `9dd83133a0129f62745fb91615922d1dab64f254713495ed94571af4af6571e8`;
  the regenerated paper-assets evidence SHA-256 is
  `c62f4be84238478138373e355187481c2c015fe2711d16c7cc4c458eda080165`.
  All nine paper-asset families and the original-view 23-route bar, 100-day
  reserve/population lines, and 50-agent action raster were regenerated.
- `docs/conference_submission/conference_submission.pdf` is the current
  37-page, portrait, US-letter anonymous **Safety Beyond Refusal** build. Its
  SHA-256 is
  `6f301b985a4d9c32d8a089e265be90e8535bfdae7bebc1ada25e19a4c8f87a92`.
  All 37 pages and every standalone changed figure were rendered and visually
  inspected; there is no clipping, overflow, broken label, or unresolved
  citation marker.
- `docs/conference_submission/supplement.zip` is the current deterministic,
  privacy-audited 263-file anonymous supplement. Its archive hash is reported
  outside the archive to avoid a circular self-reference. A fresh frozen-lock
  extraction passed 998 tests with 25 intentional
  private-evidence skips, clean-compiled the 37-page manuscript, passed the
  Croissant check, and rebuilt a byte-identical ZIP.
- The prior 34-page review PDF, compact 19-route ledger, 19-route commons
  figures, five-agent raster, and 250-file supplement were built from the
  historical 12-day evidence. They remain reproducibility history only and are
  superseded by the current 23-route/276-trajectory release.
- The exact 25-page historical **Safety Beyond Refusal** PDF at commit
  `5e6a537` was previously rebuilt and visually inspected. It remains a
  historical design reference, not evidence for the current paper numbers.

## Paper requirements that remain fixed

- Keep the exact title **Safety Beyond Refusal**.
- Preserve the original three-axis motivation, terminology, and scientific
  direction. Audit findings appear only as repairs, exclusions, provenance, and
  limitations.
- Main text remains within the NeurIPS page limit; full one-row-per-model tables
  and detailed visuals go in the appendix.
- Every table has 15 pt (approximately 20 CSS pixels) above and below it. Every
  caption defines its row unit, all columns and denominators, whether high or
  low is preferable or problematic, and why.
- Final appendix uses the nine definitive paper-asset families: Part 0
  model×language, refusal/cooperation overview, the matched current-route
  profile, Part 1 all hosted models, Part 2 all hosted models, role calibration,
  sensitivity, separate local controls, and the nonpooled cross-phase outcome
  profile. The original-view 23-route Part 2 restraint bar, 100-day reserve and
  population lines, and 50-agent action raster must also be regenerated from
  the validated composition. Argument-facing tables omit bookkeeping-only
  invalid columns; those counts and bounded repair diagnostics remain in
  reproducibility artifacts. Supplemental availability rows remain clearly
  labeled and unpooled.
- No human labels, prompt approval, benign controls, or cross-axis evidence may
  be fabricated. Part 0 and Part 1 remain explicitly exploratory under their
  unresolved external validity gates.

## Terminal release handoff

The 100-day collection, formal three-pair production validation, definitive
analysis, paper-asset generation, narrative integration, PDF build and visual
inspection, anonymous supplement build, clean extraction, deterministic
rebuild, and clean compile are complete. The repository commit containing this
checkpoint is the terminal release payload; its nonrecursive Git hash is read
and reported after the commit is pushed to the existing `master` branch.

Future exact-route availability re-probes remain isolated. They must not
substitute routes, replace genuine semantic invalids, alter sealed primary
evidence, merge diagnostic roots, or change the submitted paper automatically.
