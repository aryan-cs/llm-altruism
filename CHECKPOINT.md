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
  withdraws legacy refusal rates and every refusal-based correlation.
- Part 1 point estimates are reproducible. Paper-facing 95% intervals use a
  deterministic 2,000-replicate cluster bootstrap over the 48 scenario roots,
  retaining frames and presentations within a root. Row-binomial Wilson bounds
  are diagnostic-only.
- Part 2 point estimates are reproducible from one trajectory per model. No
  run-level interval is claimed. A self-hashed legacy provenance seal binds the
  original CSV/sidecar bytes to the archived divisor-of-five attrition source and
  replays every transition without substituting a current default.
- The 13 Part 1 reconstructed sidecars now use exact unique CSV model identities,
  preserve their original reconstruction provenance, and record the repair
  method plus CSV SHA-256.
- Croissant 1.1/RAI 1.0 metadata declares 13 distributions and 11 loadable record
  sets. A stable anonymous public landing URL is still required before hosting.

## Verified gates

- Strict raw validation: 27 files, 14 pass, 13 warning, 0 fail. The warnings are
  the disclosed one-sided Part 2 rationale/action lexical flags.
- Full repository suite: 502 passed, 1 optional dependency skip on 2026-08-02.
- NeurIPS format suite: 5 passed. The official style SHA-256 is
  `c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a`.
- PDF: 30 pages, 1.7 MiB, anonymous metadata, main content on pages 1--9,
  references beginning on page 9, all pages visually inspected.
- Supplement: integrity test passes, anonymity audit returns no findings.

## External gates that must not be fabricated or bypassed

1. InferenceHub credential and authenticated catalog access are absent. All 30
   current/historical planned routes remain unverified display labels and are
   rejected by the production adapter.
2. The requested SOTA campaign therefore has no real GPT-3.5/4.1/5/5.6, Claude
   Haiku/Sonnet/Opus, Gemini, Gemma, Nemotron, DeepSeek, Qwen, Kimi, GLM,
   Mistral, Stepfun, MiniMax, or Inkling results. No substitute IDs or results
   may be invented.
3. The Part 0 registry and Part 1 bank still require genuine language/content
   review and approvals. The judge audit requires two independent qualified
   annotators plus adjudication.
4. Part 2 needs repeated trajectories and sensitivity runs before confirmatory
   uncertainty claims.
5. The current confirmatory design is request-prohibitive (roughly 6.0--12.0
   million successful calls for 30 routes at the baseline stage before retries)
   and must be reduced before credentialed execution without changing the
   estimand after seeing results.
6. Anonymous code/dataset hosting and a real Croissant landing URL remain
   submission-time requirements.

## Exact next execution order

1. Obtain the real InferenceHub base URL/key and authenticated model catalogs.
2. Replace display labels with exact callable backend routes; run the atomic
   discovery plus structured smoke gate and retain returned model identities.
3. Freeze a request-budgeted confirmatory protocol before scientific calls.
4. Complete genuine Part 0/Part 1 reviews and the Part 0 two-annotator audit.
5. Execute all successful verified routes without substitution, lock private
   native artifacts, analyze with root/run-level estimators, and rebuild every
   table, figure, manifest, Croissant file, supplement, and paper value.
6. Repeat full tests, strict validation, the check-paper audit, full PDF visual
   review, anonymity scan, and a fresh context-free NeurIPS review.
