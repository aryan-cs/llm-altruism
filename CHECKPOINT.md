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
- A live `llm_api_routes` virtual key authenticated successfully to InferenceHub
  on 2026-08-02. Its expected 403 on the portal-only `/model/info` route exposed
  an incorrect discovery assumption. Discovery now uses the key-authorized
  `/models` census and proves chat capability, controls, usage, request ID, and
  exact response identity through a bounded `/chat/completions` smoke. The
  corrected live census is captured privately: 214 routes at
  `2026-08-02T16:12:41.421354Z`, raw artifact SHA-256
  `459431076aa2106dc3966a540cb7e641b694ffc443dec82b94024bdd8462d1f1`.
  An outcome-blind exact-suffix reconciliation selected 15 smoke-pending
  candidates and left 15 labels unresolved; report SHA-256
  `dbe30991a1c29b9a4c2371258c71dbdbc4d777c139c48f078c2f13045966d819`.
  No candidate has been promoted to the registry. The key remains only in the
  user's interactive shell, not in the current execution process.
- Four revision-pinned local Hugging Face controls were loaded from a read-only
  cache and completed real offline greedy generations on CPU: SmolLM2 135M,
  360M, and 1.7B plus Qwen2.5 0.5B. The private evidence fingerprints 5.43 GB
  of exact snapshot assets; raw artifact SHA-256
  `ec84e9509b08816e2b6ebfc6366e065ce7d509849ecb766da073d0059ef376f9`.
  This is callability evidence only, not a paper result or a substitute for a
  hosted model.

## Verified gates

- Strict raw validation: 27 files, 14 pass, 13 warning, 0 fail. The warnings are
  the disclosed one-sided Part 2 rationale/action lexical flags.
- Full repository suite: 546 passed, 1 optional dependency skip on 2026-08-02.
  `tests/conftest.py` selects the noninteractive Agg backend before collection.
- Clean extracted-supplement suite: 524 passed, 2 intentional skips (withheld
  raw Part 0 and one optional dependency), without a Git object store.
- NeurIPS format suite: 6 passed. The official style SHA-256 is
  `c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a`.
- Current local pilot PDF: 29 pages, 1.20 MiB, anonymous metadata, main content on pages 1--9,
  references beginning on page 10, all pages visually inspected. Eight paper
  figures were regenerated at print-oriented dimensions with larger labels.
  Current PDF SHA-256: `c54463acf5ad13a4821ec742a01b039c960385e143979bd07c35bdd0f04544fc`.
- Current local pilot supplement: 302 source files plus its manifest, 15.7 MiB; every payload has a
  manifest SHA-256. Full-suite, integrity, portable legacy-provenance, and
  anonymity checks pass. Withdrawn Part 0/cross-part plot trees are absent, and
  the builder derives local identifiers at runtime rather than embedding them.
  Current ZIP SHA-256: `51a93cf855fbefc5eb12c140dbb974dbbdc1ed259cbcae7b548aa06f466513c8`.
- The one-stage production campaign is frozen at all 484 harmful roots plus 100
  controls in three languages, 384 self-direct Part 1 roots, and 24 independent
  Part 2 trajectories per route at N=10, horizon=30, capacity=150.
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

1. A valid InferenceHub virtual key and corrected authenticated catalog census
   have been established. Fifteen exact-suffix candidates await structured chat
   smokes; 15 planned labels have no exact catalog suffix and require an
   evidence-backed availability decision. All 30 current/historical registry
   routes remain unverified display labels and are rejected by the production
   adapter.
2. The requested SOTA campaign therefore has no real GPT-3.5/4.1/5/5.6, Claude
   Haiku/Sonnet/Opus, Gemini, Gemma, Nemotron, DeepSeek, Qwen, Kimi, GLM,
   Mistral, Stepfun, MiniMax, or Inkling results. No substitute IDs or results
   may be invented.
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

1. Run structured, identity-checked chat smokes for the 15 exact-suffix
   candidates and retain every attempt, including failures.
2. Resolve renamed/versionless candidates only from successful chat evidence;
   explicitly mark truly unavailable labels without substitution, then review
   and promote verified routes.
3. Complete genuine Part 0/Part 1 reviews and the Part 0 two-annotator audit.
4. Execute all successful verified routes without substitution under the frozen
   430,000-attempt ledger, lock private native artifacts, analyze with
   root/run-level estimators, and rebuild every
   table, figure, manifest, Croissant file, supplement, and paper value.
5. Repeat full tests, strict validation, the check-paper audit, full PDF visual
   review, anonymity scan, and a fresh context-free NeurIPS review.
