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
- Latest pushed source commit at this checkpoint: `fd44ae8`.
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
    partial runs remain preserved but are not paper inputs.
- Part 2: `data/private/inference_hub/definitive-part2-n12-main19-v3`
  - 19 exact systems, 12 independent common-seed trajectories/system;
  - five agents, 12 days, corrected prompt--engine incentives;
  - invalid actions retain zero state effect and count as nonrestraint in the
    all-scheduled behavioral rate; affected trajectories are excluded only
    from environmental outcomes;
  - host process resumed with the exact frozen arguments and an eight-hour
    wrapper after the original one-hour shell timeout.

Part 2 retains the main accelerated policy at global concurrency 12, provider
concurrency 2, global 8 starts/second, and provider 1.5 starts/second. Parts 0
and 1 use the separately source-bound bounded deadline policies above. Primary
campaigns have priority over supplemental retries.

### Role calibration (clean identity-safe rerun currently running)

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
- `analysis.build_provider_safe_v2_paper_assets` produces seven vector-PDF and
  300-dpi PNG figure families, including restored model bars and the original
  red/green route-profile idiom; full model-row LaTeX tables; and deterministic
  headline macros including Part 2 AUPC, reserve-nondepletion, and
  population-retention summaries. Figures use the original submission palette
  and a Times-compatible NeurIPS-template font. It validates the two-seed
  sensitivity contract and generates no cross-axis composite.
- The definitive supplement pipeline at commit `99b56d6` validates all current
  paper assets from their hash-bound manifest, emits portable Croissant 1.1
  metadata, rebuilds from a clean extraction, and rejects stale, partial,
  private, or tampered inputs. Private/live outputs remain excluded.

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

1. Let primary Part 1 and role calibration reach COMPLETE manifests and finish
   the active Part 2 sensitivity schedule; if a host wrapper expires, resume
   with byte-identical arguments. Part 0 and nominal Part 2 are terminal.
2. Run the two separate semantic-invalid repair campaigns and retain original
   denominators.
3. Resume and complete deadline sensitivity, then resume the three availability
   retry panels; periodically re-probe still-unavailable MiniMax M2.7 and Kimi
   K2.5.
4. The automatic completion watcher now accepts only the explicit
   all-scheduled Part 0 terminal policy, then runs
   `analysis.analyze_provider_safe_v2_definitive` and builds the seven figure
   families, six within-panel tables, nonpooled model-by-phase matrix, and
   headline macros, then run the separate availability-retry and bounded
   semantic-repair analyzers.
5. Replace any residual superseded deadline-scope prose and artifacts in the
   manuscript with only completed generated 48-root, 384-root, and
   12-trajectory evidence. Keep role and sensitivity exploratory and keep
   cross-axis output gated.
6. Compile with bundled Tectonic, render every PDF page, verify exact title,
   anonymity, main-body page limit, table spacing/captions, figure legibility,
   citations, and numeric consistency.
7. Run the full repository suite, strict validation, deterministic supplement
   rebuild, three context-fresh paper audits, update this checkpoint with final
   hashes/counts, commit, and push `master`.
