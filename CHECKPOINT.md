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
- Latest pushed repository commit before this update: `a0859ac`.
- Definitive paper and release-artifact commit: `5cd54a0`.
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
  around the original three-stage scientific question. The single current
  review artifact is `docs/conference_submission/conference_submission.pdf`;
  there is no second working-draft PDF.
- `.env` is ignored, mode `0600`, and never enters Git or release artifacts.

## Active definitive campaigns

All paths below are private and excluded from Git and the anonymous supplement.
Every runner uses append-only, fsync-backed, hash-chained journals and supports
fail-closed resume. The fixed Part 0 judge is
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
- Part 2: `data/private/inference_hub/definitive-part2-n12-main19-v3`
  - 19 exact systems, 12 independent common-seed trajectories/system;
  - five agents, 12 days, corrected prompt--engine incentives;
  - invalid actions retain zero state effect and count as nonrestraint in the
    all-scheduled behavioral rate; affected trajectories are excluded only
    from environmental outcomes;
  - host process resumed with the exact frozen arguments and an eight-hour
    wrapper after the original one-hour shell timeout;
  - COMPLETE with 228 trajectories, 13,495 realized agent-days, four genuine
    semantic-invalid actions, 225 environmentally estimable trajectories, and
    zero operational failures.
- A fresh corrected original-scale Part 2 replication is now required before
  the paper is final. `experiments/sota_cross_axis_part2_100day_panel.json`
  freezes the historical 50-agent, 100-day, capacity-2,500, private-gain-2,
  reserve-cost-2, collapse-death-rate-0.2 environment with 12 independent
  common-seed trajectories per exact route. The active resumable production
  journals are `data/private/inference_hub/full-part2-n12-n50-d100-main21-v5`;
  all 21 routes have now terminalized all 252 trajectories. The source manifest
  records 1,115,745 scheduled agent-days, 1,114,883 received responses, 862
  exhausted transport failures, zero identity mismatches, and 214/252
  operationally eligible trajectories. It correctly remains non-COMPLETE while
  38 trajectories require operational repair. This campaign remains separate
  from the sealed 12-day evidence until the repair is complete and validated.
  Its exact source command was:

  ```bash
  uv run python -m experiments.misc.inference_hub_part2_panel \
    --panel-config experiments/sota_cross_axis_part2_100day_panel.json \
    --output-dir data/private/inference_hub/full-part2-n12-n50-d100-main21-v5 \
    --target openai/gpt-3.5-turbo --target openai/gpt-4o \
    --target openai/gpt-4.1 --target openai/gpt-5 --target openai/gpt-5.2 \
    --target openai/gpt-5.4 --target openai/gpt-oss-20b \
    --target anthropic/claude-haiku-4-5 \
    --target anthropic/claude-sonnet-4-5 \
    --target anthropic/claude-sonnet-4-6 \
    --target anthropic/claude-opus-4-6 \
    --target google/gemini-2.5-flash --target google/gemini-2.5-pro \
    --target google/gemini-3.1-pro-preview --target google/gemini-3.5-flash \
    --target meta/llama-3.3-70b-instruct \
    --target qwen/qwen3.5-35b-a3b --target qwen/qwen3.6-27b \
    --target nvidia/nemotron-3-super-v3 --target minimaxai/minimax-m2.7 \
    --target zai-org/glm-5.1 --trajectory-workers 48 \
    --participant-workers 16 --max-attempts 8 \
    --initial-backoff-seconds 1 --timeout-seconds 900 \
    --rate-profile accelerated_original_scale_high_latency_v2
  ```

  Nemotron 3 Ultra and DeepSeek V4 Flash terminal sources remain immutable at
  `data/private/inference_hub/full-part2-n12-n50-d100-nemotron-3-ultra-recovered-v1`
  and `data/private/inference_hub/full-part2-n12-n50-d100-deepseek-v4-flash-recovered-v1`.
  Their complete operational overlays remain at the correspondingly named
  `-operational-repair-v1` roots; never restart either terminal source.

  The v3 main repair at
  `data/private/inference_hub/full-part2-n12-n50-d100-main21-v5-operational-repair-multikey-v3`
  is immutable terminal diagnostic evidence and must never be resumed or used
  directly as the paper overlay. It exhausted all eight rounds with 37/38
  successful full-trajectory repairs. Its sole unresolved item is exact route
  `gcp/google/gemini-3.5-flash`, trajectory/seed index 1, common environment seed
  674434863. Across rounds 3--8, 9,915/30,000 semantic units received HTTP 401,
  matching one failed position in the former blind three-account rotation. A
  secret-free live qualification check found two accounts that authenticate,
  list, and identity-match this exact route and one account that fails catalog
  authentication. This is credential-pool operational failure, not model
  behavior; v3 lacks per-attempt account provenance and cannot be repaired in
  place.

  The next paper candidate is a fresh child overlay at
  `data/private/inference_hub/full-part2-n12-n50-d100-main21-v5-operational-completion-capability-v4`.
  Its new runner is
  `experiments.misc.inference_hub_part2_cascading_operational_repair`. It binds
  the immutable original source plus the immutable incomplete v3 parent,
  inherits exactly 37 successful parent replacements, and reruns only the one
  unresolved trajectory from day 1. Before launch, its focused tests, pinned
  implementation hash, recursive validator, non-secret per-slot credential
  commitments, canonical endpoint binding, fresh resume preflight, account-slot
  and global-dispatch provenance, independent v2 limiter scopes, and fail-closed
  post-preflight 401/403 quarantine must all pass. Once those gates are recorded
  here, launch only this exact command (append `--resume` only for this v4 root
  after all resume bindings pass):

  ```bash
  .venv/bin/python -m experiments.misc.inference_hub_part2_cascading_operational_repair \
    --source-manifest data/private/inference_hub/full-part2-n12-n50-d100-main21-v5/private/manifest.json \
    --parent-overlay-manifest data/private/inference_hub/full-part2-n12-n50-d100-main21-v5-operational-repair-multikey-v3/private/manifest.json \
    --output-dir data/private/inference_hub/full-part2-n12-n50-d100-main21-v5-operational-completion-capability-v4 \
    --maximum-rounds 8 --trajectory-workers 1 --participant-workers 30 \
    --max-attempts 8 --initial-backoff-seconds 1 --timeout-seconds 900 \
    --rate-profile accelerated_original_scale_high_latency_v2 \
    --credential-env-file /Users/aryagupta/Desktop/cdo-better-gos/.env \
    --expected-api-key-count 3
  ```

  The v4 throughput setting is a deliberate three-account completion override:
  the local executor exposes 30 participant slots, while the runtime cap is
  recomputed as 10 per currently qualified account (30/20/10 workers for
  three/two/one qualified accounts). Each account retains its own file-backed
  high-latency-v2 limiter at 10 in flight and 2.5 starts/second, so aggregate
  throughput never borrows another account's rate budget. Clients are built
  directly from held-fd credential reads and never through process-global API
  key mutation.

  The prelaunch runner is frozen at SHA-256
  `3d18d7a0ab11cc43e5648c60abc24b243e18e5195437cc345cd7479e43301dec`.
  The validator pins that exact digest. A secret-free configuration check saw
  exactly three ordered, distinct credential commitments, three distinct
  limiter scopes, and the canonical InferenceHub endpoint; credential values
  were neither printed nor persisted. The runner/validator/analyzer gate passes 123
  focused tests, including simultaneous pool construction, exact round-robin
  allocation, post-preflight 401/403 quarantine, bootstrap reprobe, nofollow
  credential/preflight/journal reads, exact journal references, retained-state
  causal timestamps, and a physical-reservation crash followed by a complete
  resume and recursive validation. The four downstream analysis/asset suites
  pass 99 tests with one intentional skip. Direct recursive validation of the
  immutable v3 parent passes with 37 successful replacements, exactly one
  unresolved trajectory, and 58 populated parent round journals.

  A failed/crashed initial qualification may be retried without `--resume`
  only when the v4 tree is the exact uncommitted bootstrap shape. A lone sealed
  `preflight-000.json` is treated only as a cursor/account crash binding: all
  three accounts are freshly re-probed and that ledger is atomically replaced
  before any manifest seal or experiment dispatch. Once `manifest.json`
  exists, only the exact `--resume` command is permitted.

  The earlier single-account `...operational-repair-v1`, pooled
  `...operational-repair-multikey-v2`, and terminal pooled v3 roots are preserved
  as incomplete diagnostic evidence and must not be resumed or merged into the
  paper. Never launch v4 while its implementation/test/hash gate above is
  incomplete.

Part 2 uses the manifest-bound high-latency v2 policy independently per account:
global concurrency 60, provider concurrency 10, global 12 starts/second, and
provider 2.5 starts/second. Parts 0
and 1 use the separately source-bound bounded deadline policies above. Primary
campaigns have priority over supplemental retries.

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
  model and reuses no incomplete v8 evidence. The active
  shared policy is global concurrency 24, provider concurrency 3, 12 global
  starts/second, and 2.5 starts/second per provider. The active writer uses 15
  campaign workers to balance network throughput against local fsync pressure;
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
- The Part 2 supplemental schedule is COMPLETE: 48/48 trajectories, 2,824
  realized agent-days, zero transport failures, zero identity mismatches, and
  five genuine semantic-invalid actions retained. Opus 4.5 produced 100.0%
  restraint and 1.000 mean AURC; DeepSeek V4 Flash 50.5% and 0.249; Nemotron 3
  Ultra 63.8% and 0.299; and GLM 5.1 73.8% and 0.535.
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

## Completed local and release controls

- Four pinned local Hugging Face controls completed the full 384-root Part 1
  bank (1,536 real generations) under
  `data/private/local_hf/part1-large-n-20260802-v3-hardened`.
- They are reported as separate exploratory execution-scale controls, never as
  hosted-route substitutes.
- `analysis.analyze_provider_safe_v2_definitive` accepts only five COMPLETE,
  source-bound primary/robustness manifests and produces per-model tables while
  preserving first-attempt invalid denominators.
- `analysis.build_provider_safe_v2_paper_assets` produces nine vector-PDF and
  300-dpi PNG figure families, including current-route refusal bars, the
  75-route cooperation rank line, a matched 19-route task profile, and the
  original red/green route-profile idiom; full model-row LaTeX tables; and deterministic
  headline macros including Part 2 AUPC, reserve-nondepletion, and
  population-retention summaries. Figures use the original submission palette
  and a Times-compatible NeurIPS-template font. Row-based model comparisons
  group provider families contiguously with newer frozen route versions above
  older ones; global outcome order is retained only where a figure's analytical
  purpose is the rank distribution. It validates the two-seed
  sensitivity contract and generates no cross-axis composite.
- The definitive supplement pipeline at commit `99b56d6` validates all current
  paper assets from their hash-bound manifest, emits portable Croissant 1.1
  metadata, rebuilds from a clean extraction, and rejects stale, partial,
  private, or tampered inputs. Private/live outputs remain excluded.
- The definitive analysis and current figure families are current. Croissant 1.1
  metadata validates against the hash-bound analysis and paper-assets
  manifests, and the anonymous supplement rebuilt deterministically with 250
  files.
- The final review PDF has the exact title, anonymous author block, nine-page
  main body, and 34 portrait pages including references, appendices, and the
  NeurIPS checklist. The former 11-page table dump is replaced by one compact
  19-route matched ledger; the full 22-, 75-, and 19-route tables, intervals,
  exact target IDs, language cells, role calibration, sensitivity results, and
  local controls remain in the supplement.
- The exact 25-page historical `Safety Beyond Refusal` PDF at commit `5e6a537`
  was rebuilt and every page was visually inspected. Its two vertical model
  bars and two commons time-series plots were traced to
  `data/graphs/paper_visuals.py` and recreated for the current panel by
  `analysis/build_original_view_figures.py`. The new Part 0 bar covers all 22
  current routes with harmful-root sensitivity intervals; the Part 2 bar and
  reserve/population lines cover all 19 current routes and 12 common-seed
  trajectories per route. All 228 journal replays pass hash-chain, identity,
  transition, endpoint, AURC, AUPC, and public-aggregate reconciliation.
  Providers are contiguous and newest frozen route versions appear first.
  The original agent-day action raster is also restored for prespecified seed
  index 0 across all 19 current routes: it contains five agent subrows per
  route, 12 day columns, 935 restraint cells, 180 overuse cells, and 25 gray
  post-attrition cells, with no invalid response in that displayed trajectory.
  Numeric bar labels sit beyond their upper whiskers, and all five plots use
  Times-compatible typography and the original paper palette.
- Bundled Tectonic compiled the PDF successfully; all changed figures, their
  paper pages, including the restored raster page, were visually inspected with no
  clipping, element overlap, landscape pages, or unreadable glyphs. The clean
  extracted 250-file anonymous supplement independently compiles the same
  34-page portrait paper. The focused supplement/format suite is 28 passed;
  all conference-format checks pass. A repository-wide run reached 47\% with
  no failures before the 300-second command window expired.

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
- Final appendix uses seven updated visual families: Part 0 model×language,
  Part 1 all hosted models, Part 2 all hosted models, role calibration,
  sensitivity, separate local controls, and a nonpooled red/green cross-phase
  route profile. Argument-facing tables omit bookkeeping-only invalid columns;
  those counts and bounded repair diagnostics remain in reproducibility
  artifacts. Supplemental availability rows remain clearly labeled and unpooled.
- No human labels, prompt approval, benign controls, or cross-axis evidence may
  be fabricated. Part 0 and Part 1 remain explicitly exploratory under their
  unresolved external validity gates.

## Exact next steps

1. Finish and verify the new v4 cascading runner/validator gate, pin its exact
   implementation hash, then launch or safely resume only the v4 child with the
   exact command above. Never resume v3.
2. Validate source and all three complete overlays end to end: route identities,
   hash chains, attempt bindings, simulator transitions, matched denominators,
   trajectory completeness, and sanitized aggregates.
3. Regenerate every provenance-bound Part 2 analysis artifact, table, bar/line/
   raster figure, and manuscript claim from the effective 100-day evidence.
4. Compile and visually inspect `conference_submission.pdf`, rebuild and clean-
   compile `supplement.zip`, run focused and full relevant tests, then commit and
   push `master`. Keep the title exactly **Safety Beyond Refusal**.
5. Keep future exact-route availability re-probes isolated. Do not substitute
   routes, replace genuine semantic invalids, alter sealed primary evidence, or
   change the submitted paper automatically.
