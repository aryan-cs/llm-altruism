# Paper completion checkpoint

Updated: 2026-08-02 (America/Los_Angeles)

## Current evidence state

- The audited paper is `docs/conference_submission/conference_submission.tex`;
  the rendered submission is `docs/conference_submission/conference_submission.pdf`.
- The legacy artifact contains 56,382 stored rows across 13 Ollama variants, but
  those rows are not independent observations: Part 0 has 99 shared prompt roots
  per model, Part 1 has 48 shared scenario roots per model, and Part 2 has 13
  state-coupled trajectories.
- Part 0 is not a valid result. Its legacy labeler could score rationale as well
  as final response and defaulted failed adjudications to denial. The incomplete
  response-only checkpoint has 1,243 judged rows and 300 changes (24.1%): 275
  legacy-complied to denied and 25 in the other direction. The manuscript
  withdraws legacy refusal rates and every refusal-based correlation. A
  sanitized aggregate checkpoint in `data/analysis` binds these counts to the
  withheld source CSV's SHA-256 without releasing harmful content.
- Part 1 point estimates are reproducible. Paper-facing bands are explicitly
  finite-bank sensitivity intervals from a deterministic 2,000-replicate
  cluster bootstrap over the 48 scenario roots, retaining frames and
  presentations within a root. They do not estimate generation variability or
  a scenario superpopulation. Row-binomial Wilson bounds are diagnostic-only.
- Part 2 point estimates are reproducible from one trajectory per model. No
  run-level interval is claimed. A self-hashed legacy provenance seal binds the
  original CSV/sidecar bytes to the archived divisor-of-five attrition source and
  replays every transition without substituting a current default.
- The 13 Part 1 reconstructed sidecars now use exact unique CSV model identities,
  preserve their original reconstruction provenance, and record the repair
  method plus CSV SHA-256.
- Croissant 1.1/RAI 1.0 metadata declares nine distributions and seven loadable
  record sets. Invalid Part 0 rate tables and all dependent cross-part tables
  are excluded. A stable anonymous reviewer-accessible landing URL and official
  Croissant validation are still required before submission; public access is
  required by camera ready.
- A repository-local ignored `.env` now supplies the live `llm_api_routes`
  virtual key to authenticated jobs without exposing it in Git or logs. The
  corrected `/models` census captures all 214 authorized routes; the live v3
  catalog has raw SHA-256
  `e7fd014302bc5f1caf06f1d5b59ace8fcd76e34fd37f00e995ad09360e79b780`.
  The exhaustive bounded minimal-chat census attempted every route, found 69
  minimally callable routes and 145 rejected routes, and then proved a zero-call
  request-hash resume over all 214 entries. The resume-proof artifact has raw
  SHA-256 `bcd4c141b2bfb28128648105f9086ccf7c87adf1a48b25f58774c019162e46f4`.
  The fixed 31-target registry compatibility sweep probed all 44 exact backend
  candidates and selected 29 routes: 28 subjects plus the independent judge.
  Only Mixtral-8x22B and legacy Gemma-2-9B remain unresolved. Claude routes are
  retained through deterministic maximal-valid control profiles rather than
  being rejected when individually supported controls cannot be combined. The
  private schema-v2 evidence has raw SHA-256
  `9bc458a57c69bf4582f2fbbc614e75144c5427227935b5464a45faa51560a1b9`.
  An outcome-blind taxonomy accounts for all 214 routes, excludes 85 non-subject
  or out-of-panel routes with explicit reasons, and defines 84 exact SOTA
  text-chat identities across 129 backend routes. Live compatibility selected
  81/84 identities; Mixtral-8x22B, GPT-5.4-Pro, and Sonar Deep Research were the
  only unresolved subjects. The roster and compatibility artifacts have raw
  SHA-256 values `d34988cd20cdf02ac04a4bc8e11185ec6e8b048bbac7de72ae0195421304f751`
  and `e6a86e7fd8d84a8f519e51b1d2814f743094e5e0c207a89fb02801a5484dbd77`.
  A 28-route production-schema smoke retained all 28 exact identities with 27
  strict final actions; a route-specific long-reasoning allowance then made
  Kimi valid. The 53 nonduplicate expansion smoke retained 53/53 exact-identity,
  strict-format responses. Two private, resumable large-N jobs are active: 28 x
  384 and 53 x 384 trials. Their draft-bank contract permanently bars
  confirmatory or paper-result promotion. One intermittent DeepSeek response
  identity mismatch is quarantined by the analyzer and will require a separate
  exact-identity rerun; it is not silently pooled.
- Four revision-pinned local Hugging Face controls were loaded from a read-only
  cache and completed real offline greedy generations on CPU: SmolLM2 135M,
  360M, and 1.7B plus Qwen2.5 0.5B. A fresh immutable two-worker private artifact
  recursively fingerprints all 74 exact snapshot assets (11,420,103,695 bytes,
  including nested and hidden assets); raw artifact SHA-256
  `9e0cca6a738bf5c794c33c5da2af02c6a4d6b4423cd34ae7a4ba5d527dab961e`.
  All four returned visible text; one matched the exact `READY` format and three
  were retained as format-noncompliant. The separate resumable 384-root local
  Part 1 scale panel completed all 1,536 real generations with two concurrent
  model workers and batched prompts. Strict final-action format was valid for
  Qwen2.5 0.5B on 384/384 rows, SmolLM2 1.7B on 362/384, SmolLM2 360M on
  27/384, and SmolLM2 135M on 3/384. All prompt, response, per-model file, and
  manifest hashes validate. The private manifest has raw SHA-256
  `59b6b7e9efcf812f23c286be99e7e3348cc220def46a242110ec0726d8954afe`
  and self-hash
  `3b7779ec37bcd0b37367ffe756f4b48ca52e19d71cb4811c9d96a13bc83f74bc`.
  Its contract permanently bars confirmatory or paper-result promotion because
  the deterministic draft bank is not human-approved.
  A deliberately interrupted attempt remains preserved separately rather than
  being overwritten.
  This is callability evidence only, not a paper result or a substitute for a
  hosted model.

## Verified gates

- Strict raw validation: 27 files, 14 pass, 13 warning, 0 fail. The warnings are
  the disclosed one-sided Part 2 rationale/action lexical flags.
- Full repository suite: 643 passed, 1 optional dependency skip on 2026-08-02.
  `tests/conftest.py` selects the noninteractive Agg backend before collection.
- Clean extracted-supplement suite: 559 passed, 2 intentional skips (withheld
  raw Part 0 and one optional dependency), without a Git object store.
- NeurIPS format suite: 6 passed. The official style SHA-256 is
  `c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a`.
- Current local pilot PDF: 29 pages, 1.20 MiB, anonymous metadata, main content
  ending and references beginning on page 9, all pages visually inspected. Eight paper
  figures were regenerated at print-oriented dimensions with larger labels.
  Current PDF SHA-256: `8aac3fc6b69a14f6ebc48f9f3c911d0811adcce1bd2340666d8a7b363f46857f`.
- Current local pilot supplement: 304 files including its manifest, 15.7 MiB; every payload has a
  manifest SHA-256. Full-suite, integrity, portable legacy-provenance, and
  anonymity checks pass. Withdrawn Part 0/cross-part plot trees are absent, and
  the builder derives local identifiers at runtime rather than embedding them.
  Current ZIP SHA-256: `c043194382735869b79f012cac481427e0a7b0eafe9bfd8a551f2bfa4bef76d9`.
- The one-stage production campaign is frozen at all 484 harmful roots plus 100
  controls in three languages, 384 self-direct Part 1 roots, and 24 independent
  Part 2 trajectories per route at N=10, horizon=30, capacity=150.
- One dedicated `judge_only` target, NVIDIA Evals Nemotron 3 30B A3B, is outside
  the evaluated panel and is reused across every subject route. Campaign and
  standalone Part 0 guards reject target-ID, provider+route, or upstream-model
  overlap. The judge remains unverified until its structured smoke succeeds.
- Confirmatory finite-bank/run-level estimators use exactly 5,000 frozen
  bootstrap replicates. Part 1 resamples independently within all 12
  game-by-domain cells, Part 2 uses common paired-seed resampling, reserve
  nondepletion uses a Wilson interval, and cross-part unit resampling is shared
  across systems.
- The base 30-route campaign is 333,750 successful POSTs. A full stored-response
  Part 0 rejudge allowance and 10% transport reserve fit below the immutable
  430,000-attempt and 1.5-billion-token ceilings. Every confirmatory POST makes
  an atomic conservative ledger reservation before dispatch. Discovery retains
  failed reservations and a full catalog census; the final lock requires exact
  request-hash reconciliation between the ledger and native attempt logs.
- Offline local-control smoke: 4/4 exact pinned snapshots loaded and generated
  visible text under Python 3.12.13, PyTorch 2.13.0, and Transformers 4.57.6.
  The registry forbids frontier substitution and paper-result promotion.

## External gates that must not be fabricated or bypassed

1. Authenticated catalog, exhaustive minimal-chat census, fixed-registry
   compatibility, and broad SOTA compatibility are complete. These exploratory
   artifacts do not automatically promote the production registry or bypass
   the frozen confirmatory approvals.
2. Parallel large-N hosted Part 1 execution is in progress over 81 unique
   callable subjects. Until both journals lock and strict analysis passes, no
   model rates may be reported. Identity-mismatched or terminally failed targets
   are quarantined whole and rerun separately; unavailable routes are never
   substituted.
3. The Part 0 registry and Part 1 bank still require genuine language/content
   review and approvals. The judge audit requires two independent qualified
   annotators plus adjudication.
4. Part 2 confirmatory trajectories and the separately gated sensitivity panel
   have not been executed; no confirmatory uncertainty claim exists yet.
5. Anonymous reviewer-accessible code/dataset hosting, a real Croissant landing
   URL, and official Croissant validation remain submission-time requirements.
6. The NeurIPS 2026 E&D full-paper/data/code deadline was May 6, 2026 AoE. If
   no final-form package was submitted by that deadline, this repository cannot
   create a new 2026 submission retroactively; an existing OpenReview package
   must be audited, or the completed work must target a later venue.

## Exact next execution order

1. Finish and lock both active 384-root hosted Part 1 jobs; rerun every
   operationally quarantined target separately without substitution.
2. Validate journals and generate only the nonpromotable aggregate artifact with
   5,000-replicate stratified root bootstrap intervals.
3. Complete genuine Part 0/Part 1 human reviews and the Part 0 two-annotator
   audit before any confirmatory promotion.
4. Execute only genuinely approved confirmatory phases under the frozen budget,
   then rebuild every paper-facing table, figure, manifest, Croissant file, and
   supplement.
5. Repeat strict validation, clean extraction, the final check-paper pass, full
   PDF visual review, anonymity scan, and fresh context-free NeurIPS reviews.
