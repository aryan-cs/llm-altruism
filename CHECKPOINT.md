# Paper completion checkpoint

Updated: 2026-08-02 (America/Los_Angeles)

## North star

- The submission is **Safety Beyond Refusal**, a corrected successor to the
  original Prosocial Readiness Bench manuscript at Git commit `9cdefa4`.
- The scientific question remains whether multilingual safety refusal,
  cooperation in one-shot social dilemmas, and restraint in repeated commons
  environments provide complementary observable evidence about model behavior.
- The August audit is not the paper's thesis. Its valid findings are incorporated
  as protocol repairs, provenance controls, exclusions of unsupported legacy
  results, and explicit limitations.
- No result is promoted from an incomplete run, an unavailable route is never
  silently substituted, and a subject model never judges its own output.

## Frozen corrected experiments

### Matched cross-axis panel

- `experiments/sota_cross_axis_panel.json` freezes 24 subject systems spanning
  OpenAI GPT-3.5/4/5 and GPT-OSS, Anthropic Claude Haiku/Sonnet/Opus, Google
  Gemini 2.5/3.x, Meta Llama, DeepSeek, Qwen, MiniMax, GLM, and NVIDIA
  Nemotron families.
- The fixed judge is `judge.nvidia-evals-nemotron-3-30b-a3b`. It is outside the
  subject roster. Selection rejects overlap by target ID, exact route, and
  provider plus upstream model identity.
- The panel file records the intended promotion design (Part 0: 48 roots per
  condition; Part 2: 12 trajectories). The deadline execution is smaller and
  must always be reported separately: Part 0 uses 24 roots per condition and
  Part 2 uses eight trajectories per model.

### Part 0: response-only safety refusal

- Live artifact:
  `data/private/inference_hub/part0-sota-panel-v2-n24`.
- Design: 24 matched systems, 24 archived English harmful-request roots, and
  three explicit response-language instructions (English, Chinese, Russian),
  for 72 cells per system and 1,728 subject calls.
- The inputs are English requests with reconstructed response-language
  conditions. They are not retained multilingual input translations and must
  never be described that way.
- The fixed disjoint judge scores only the visible final response, in batches of
  eight. Judge failures remain explicitly unjudged and are not converted into
  refusals or compliance labels.
- The deadline artifact has no benign controls and no completed blinded human
  validation. It may support exploratory response-only findings, but not the
  validated Part 0 promotion or cross-axis gate.
- The older `part0-sota-panel-v1` partial run is preserved and excluded.

### Part 1: balanced one-shot cooperation

- Main live artifact:
  `data/private/inference_hub/part1-sota-balanced-main75-v1-n96`.
- Main design: 75 routes, 96 roots per route, with exactly eight roots in each
  of 12 game-by-domain strata.
- Slow-route live artifact:
  `data/private/inference_hub/part1-sota-balanced-slow2-v2-n12`.
- Slow-route design: MiniMax M3 and Qwen3.5 397B, 12 roots per route, with one
  root in each of the same 12 strata. Earlier n=96 slow-route attempts are
  preserved and excluded.
- Full-bank live artifact:
  `data/private/inference_hub/part1-sota-deadline-glm51-v1`.
- Full-bank design: GLM-5.1 over all 384 balanced roots.
- The target roster contains 81 identities. Seventy-eight are being observed:
  75 at n=96, two at n=12, and GLM-5.1 at n=384. Kimi K2.5, Kimi K2.6, and
  GLM-5.2 are explicitly unavailable, with no substitutions.
- Direct self-choice is the primary estimand. Invalid formats are retained as
  non-cooperation for the executed direct-choice estimand. Advice, observer,
  and prediction roles are distinct probes and are not pooled with self-choice.
- The deterministic balanced bank has not received the planned human approval,
  so hosted Part 1 outcomes are exploratory descriptive results. Unbalanced
  prefix-limited shards and older partial main runs are preserved and excluded.

### Part 2: corrected repeated commons restraint

- Live artifact:
  `data/private/inference_hub/part2-sota-matched-v1-n8`.
- Design: 24 matched systems, eight common-seed independent trajectories per
  system, five agents, 12 days, and resource capacity 50.
- Option A provides private payoff +1 and no reserve loss. Option B provides
  private payoff +2 and removes two reserve units. Unanimous A adds five group
  payoff units and unanimous B removes five.
- Invalid visible outputs are retained as zero-effect `INVALID` actions with no
  semantic retry. Transport failures and identity mismatches remain operational
  failures and prevent a complete evidence lock.
- Primary trajectory-level metrics are restraint rate, reserve nondepletion,
  final reserve, population retention, area under the reserve curve, area under
  the population curve, and cumulative private and group payoffs.
- Continuous metric uncertainty uses trajectory-level t intervals; reserve
  nondepletion uses a Wilson interval. The executed n=8 design does not satisfy
  the intended n=12 cross-axis promotion gate and includes no separate
  sensitivity panel.
- `part2-production-smoke-v1` completed a full production-shaped trajectory and
  verified the repaired dispatch, prompt, and dynamics path before the matched
  launch.

## Execution and provenance controls

- All five production processes run concurrently with a single owner per output
  directory. Resumption is allowed only when manifest, source, input, identity,
  and journal bindings match.
- The shared cross-process limiter enforces durable leases, heartbeats, provider
  and global concurrency and request-rate ceilings, `Retry-After`, and 30-second
  cooldowns for HTTP 429 and every HTTP 5xx response. SDK retries are disabled.
- Private journals reserve attempts before dispatch, append with fsync, and form
  SHA-256 chains. Manifests bind exact routes, selected compatibility profiles,
  source hashes, immutable schedules, request hashes, and raw-response hashes.
- Private prompt, response, reasoning, route, and key material must never enter
  Git, the paper, public aggregates, Croissant metadata, or the supplement.
- Four pinned local Hugging Face controls completed 1,536 Part 1 generations in
  a separate private artifact. They remain exploratory controls and are not
  substitutes for unavailable hosted systems.

## Paper and artifact state

- `docs/conference_submission/conference_submission.tex` has the exact title
  **Safety Beyond Refusal** and the restored three-axis motivation. Unsupported
  legacy Part 0, broken-engine Part 2, cross-part, pooled-role, and latent-trait
  claims and figures have been removed. The main paper now has explicit Results,
  Discussion, and Limitations and Broader Impact sections; detailed legacy-audit
  material is subordinate in the appendix, and the full official NeurIPS
  checklist is included after the appendix.
- `docs/conference_submission/working_draft.pdf` is an ignored link to the
  compiled submission PDF and must be refreshed after every paper-facing change.
- `analysis/build_final_results.py` is the fail-closed paper-results builder. It
  validates complete manifests and their chains and emits only sanitized JSON,
  CSV, publication-table LaTeX, and figure outputs under
  `data/analysis/final_results`. Cross-axis analysis remains withheld unless the
  human-validation and intended sample-size gates pass.
- Release documentation and `analysis/build_croissant_metadata.py` are being
  aligned to the mixed Part 1 sample sizes. The checked-in Croissant file must
  be regenerated only after the final results artifact exists.
- The anonymous supplement builder excludes `.env`, keys, private/raw data,
  deprecated/interrupted runs, journals, and harmful text. A final clean-extract
  build and reproduction test remain mandatory.

## Current execution status

- Active process ownership was verified on 2026-08-02 for all five artifacts:
  Part 0 n=24, Part 1 main n=96, Part 1 slow n=12, Part 1 GLM-5.1 n=384, and
  Part 2 n=8.
- Manifests remain `complete: false` while execution is in progress. Paper-facing
  outcomes stay locked until every planned unit for the retained artifact is
  present and strict validation passes.
- Read-only monitoring has identified fail-closed operational records in the
  active Part 0, Part 1 main, and Part 2 artifacts. They are not behavioral
  outcomes and make those primary manifests ineligible as-is. After their sole
  writers exit, only the affected targets will be rerun over their complete
  frozen schedules. A provenance-bound replacement layer must exclude a primary
  target only when a complete identity- and schedule-matched replacement exists;
  successful targets and all failed attempts remain immutable in the archive.

## Remaining critical path

1. Let the five live jobs finish; launch complete-schedule repair artifacts only
   for affected targets after the corresponding primary writer exits.
2. Validate manifests, source/input hashes, identities, attempt chains, exact
   target coverage, balanced strata, and complete sampling units.
3. Build final sanitized results, intervals, tables, and figures, then verify
   every reported number directly against those outputs.
4. Insert only verified results into the abstract, results, discussion,
   limitations, and conclusion; compile and inspect the full PDF.
5. Regenerate Croissant metadata and the anonymous supplement, reproduce from a
   clean extracted ZIP without a Git object store, and run the full repository
   test suite and strict data validation.
6. Complete check-paper passes two and three, including context-fresh scientific,
   statistical, visual, citation, anonymity, secret, placeholder, and format
   audits.
7. Push each coherent checkpoint to `origin/master`; finish with the exact final
   commit hash.

## External gates

- Blinded two-annotator Part 0 validation and adjudication are not complete.
- The balanced Part 1 bank has not received the planned human approval.
- The deadline Part 2 run has eight rather than 12 trajectories per model and
  does not include the planned sensitivity grid.
- Anonymous reviewer-accessible hosting and official external Croissant
  validation cannot be manufactured inside this repository.
