# Paper completion checkpoint

Updated: 2026-08-03 (America/Los_Angeles)

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
- Latest pushed source commit at this checkpoint: `17fc8bb`.
- Important preceding commits:
  - `d11c739`: deterministic, privacy-safe clean-room supplement;
  - `b94e65a`: deterministic paper headline macros and six asset families;
  - `ba28b5f`: role-invalid repair and availability-snapshot merge support;
  - `a75e6b9`: bounded two-seed deadline sensitivity and Part 1 invalid repair.
- The exact title remains in
  `docs/conference_submission/conference_submission.tex`.
- `.env` is ignored, mode `0600`, and never enters Git or release artifacts.

## Active definitive campaigns

All paths below are private and excluded from Git and the anonymous supplement.
Every runner uses append-only, fsync-backed, hash-chained journals and supports
fail-closed resume. The fixed Part 0 judge is
`judge.nvidia-evals-nemotron-3-30b-a3b`; it is disjoint from every subject at
target ID, exact route, and upstream provider/model identity.

### Primary main scope (currently running)

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
    this does not raise the global or provider network ceilings. The
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
    partial runs remain preserved but are not paper inputs.
- Part 2: `data/private/inference_hub/definitive-part2-n12-main19-v3`
  - 19 exact systems, 12 independent common-seed trajectories/system;
  - five agents, 12 days, corrected prompt--engine incentives;
  - invalid actions retain zero state effect for continuity but invalidate the
    affected matched-panel trajectory for behavioral/environmental estimates;
  - host process resumed with the exact frozen arguments and an eight-hour
    wrapper after the original one-hour shell timeout.

Part 2 retains the main accelerated policy at global concurrency 12, provider
concurrency 2, global 8 starts/second, and provider 1.5 starts/second. Parts 0
and 1 use the separately source-bound bounded deadline policies above. Primary
campaigns have priority over supplemental retries.

### Role calibration (currently running)

- Path: `data/private/inference_hub/definitive-part1-role-calibration-v2`.
- Six frozen sentinels × 96 roots × three distinct frames (advice, observer
  evaluation, prediction) × four counterbalances = 6,912 requests.
- Frames remain separate estimands and are never pooled into self-choice.
- The exploratory accelerated policy is source-bound at global concurrency 12,
  provider concurrency 3, global 8 starts/second, and provider 2 starts/second.
- Role calibration and sensitivity now run concurrently under distinct bounded
  limiter scopes; role semantic-invalid repair still waits for the role source
  manifest to become COMPLETE.

### Part 2 deadline sensitivity (active, resumed)

- Path: `data/private/inference_hub/definitive-part2-sensitivity-deadline-fast-v8`.
- Six sentinels × 16 resolution-V cells × two common environment seeds = 192
  trajectories and 17,280 scheduled agent-day POSTs.
- Five factors vary: capacity per initial agent, depletion units, collapse death
  rate, society size, and horizon.
- Invalid visible actions are retained as nonrestraint/zero-effect observations;
  transport or identity failure blocks operational eligibility.
- The panel is explicitly deadline-exploratory and underpowered. Thirty
  sentinel-by-factor Holm rows document the prespecified family; they do not
  authorize confirmatory robustness claims.
- A fresh source-bound accelerated writer replaced the stalled conservative
  partial run, which remains preserved but is not a paper input. The active
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
- Supplemental availability runs are private, separate, and currently paused so
  primary campaigns retain capacity:
  - Part 0 Opus 4.5:
    `data/private/inference_hub/definitive-part0-unavailable-retry-opus45-v2`;
  - Part 1 Opus 4.5, MiniMax M3, Kimi K2.6, and GLM 5.2:
    `data/private/inference_hub/definitive-part1-unavailable-retry-n384-v2`;
  - Part 2 Opus 4.5, DeepSeek V4 Flash, Nemotron 3 Ultra, and GLM 5.1:
    `data/private/inference_hub/definitive-part2-availability-retry-n12-v1`.
- Resume supplemental runs only with their original frozen worker arguments,
  after primary main capacity is released.
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
- `analysis.build_provider_safe_v2_paper_assets` produces six vector-PDF and
  300-dpi PNG figure families, full model-row LaTeX tables, and 66 deterministic
  headline macros including Part 2 AUPC, reserve-nondepletion, and
  population-retention summaries. It validates the two-seed sensitivity
  contract and generates no cross-axis composite.
- The supplement at commit `c84a841` has 210 payload files plus manifest,
  rebuilds byte-identically, and passes 781 clean-extraction tests with 16
  intentional skips plus Croissant, privacy, anonymity, and manifest checks.
  It exact-allowlists the availability-retry analyzer and current checkpoint;
  private/live outputs remain excluded.

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
- Final appendix uses six updated visual families: Part 0 model×language,
  Part 1 all hosted models, Part 2 all hosted models, role calibration,
  sensitivity, and separate local controls. Supplemental retry tables remain
  clearly labeled and unpooled.
- No human labels, prompt approval, benign controls, or cross-axis evidence may
  be fabricated. Part 0 and Part 1 remain explicitly exploratory under their
  unresolved external validity gates.

## Exact next steps

1. Let primary Part 0/1/2 and role calibration reach COMPLETE manifests; if an
   eight-hour host wrapper expires, resume with byte-identical arguments.
2. Run the two separate semantic-invalid repair campaigns and retain original
   denominators.
3. Resume and complete deadline sensitivity, then resume the three availability
   retry panels; periodically re-probe still-unavailable MiniMax M2.7 and Kimi
   K2.5.
4. Run `analysis.analyze_provider_safe_v2_definitive`, build the six figure
   families, six within-panel tables, nonpooled model-by-phase matrix, and
   headline macros, then run the separate availability-retry and bounded
   semantic-repair analyzers.
5. Replace stale n=24/n=96/n=8 prose and artifacts in the manuscript with only
   the completed generated n=48/n=384/n=12 evidence. Keep role and sensitivity
   exploratory and keep cross-axis output gated.
6. Compile with bundled Tectonic, render every PDF page, verify exact title,
   anonymity, main-body page limit, table spacing/captions, figure legibility,
   citations, and numeric consistency.
7. Run the full repository suite, strict validation, deterministic supplement
   rebuild, three context-fresh paper audits, update this checkpoint with final
   hashes/counts, commit, and push `master`.
