# Reproducibility

Run commands from the repository root. The release pipeline has a hard boundary
between private execution evidence and public aggregate artifacts.

The public workflow is aggregate-reproducible, not collection-reproducible:
reviewers can verify the sealed graph and regenerate every paper-facing table,
figure, and check from it, but cannot rebuild that graph from the private
prompts, responses, authenticated routes, manifests, or journals excluded from
the supplement.

## Environment and tests

```bash
uv sync
cp .env.example .env   # required only for new provider calls
uv run pytest -q
```

Never place a real key in a command, log, supplement, or tracked file. Existing
validation, final-result generation, Croissant generation, and supplement
packaging do not require provider credentials.

## Executed hosted panels

The deadline artifacts used by the final-result gate are:

- Part 0: `data/private/inference_hub/part0-sota-panel-v2-n24/private/manifest.json`
  with 24 roots per response-language condition.
- Part 1 balanced main shard:
  `data/private/inference_hub/part1-sota-balanced-main75-v1-n96/private/manifest.json`.
- Part 1 balanced slow shard:
  `data/private/inference_hub/part1-sota-balanced-slow2-v2-n12/private/manifest.json`.
- Part 1 separate full-bank route:
  `data/private/inference_hub/part1-sota-deadline-glm51-v1/private/manifest.json`.
- Part 2: `data/private/inference_hub/part2-sota-matched-v1-n8/private/manifest.json`
  with eight independent trajectories per matched system.

These paths are local evidence locations and are not packaged. The names encode
the executed CLI limits. The larger 48-root and 12-trajectory values in
`experiments/sota_cross_axis_panel.json` are intended settings, not completed
counts.

Sealed coverage is Part 0: 16 included and 8 unavailable; Part 1: 75
reportable and 6 unavailable of 81 (73 at n=96, one at n=12, one at n=384;
three operational and three pre-execution unavailable); Part 2: 22 executed and
2 unavailable, with 176 total trajectories. Twenty systems and 159 valid
trajectories are estimable; 17 trajectories are protocol-invalid, and two
all-invalid executed systems are non-estimable. Exact IDs are listed in
`MODEL_REGISTRY.md`.

All hosted callers use the shared cross-process rate limiter and exact
compatibility-selected routes. Part 0 and Part 1 subject calls can progress in
parallel across upstream providers, while provider-specific concurrency remains
bounded. Part 0 judge batches use one fixed non-subject judge. Only transport
failures are retried; malformed semantic outputs are retained.

## Reviewer-driven follow-up protocols

Three versioned follow-up workflows are packaged as reproducibility code, not
silently merged into the sealed deadline results above:

- `analysis/accelerated_part0_human_validation.py` exports blinded, balanced
  packets for two independent annotators and an independent adjudicator. It
  never generates human labels and releases no score before its completion,
  qualification, agreement, and source-integrity gates pass.
- `experiments.misc.inference_hub_part1_role_calibration_v1` freezes a
  six-developer sentinel panel over the 96-root role subset. Advice,
  observer-evaluation, and prediction remain separate root-level estimands.
  The immutable configuration is
  `experiments/part1/role_calibration_panel_v1.json` with SHA-256
  `14012c3db18890dd970e826dcf59498eeba6b411fe0b5f04d2df52cc6c6e92b3`.
- `experiments.misc.inference_hub_part2_sensitivity_v1` preserves both the
  3.24-million-post future confirmatory protocol and the separate 17,280-post
  deadline-bounded exploratory profile. Their results may not be pooled or
  described as interchangeable. The deadline profile is exactly six sentinels
  by 16 resolution-V cells by two common seeds: 32 trajectories per sentinel
  and 192 total. Each sentinel-factor effect therefore has four exact paired
  seed-block sign assignments, and all 30 prespecified effects remain in one
  global Holm family.

The compatibility retry/visible-content probes and provider-safe v1/v2
launchers are also included. Provider-safe v2 round-robins upstream providers.
The separately source-bound deadline exploratory launcher is restricted to the
role calibration and sensitivity workflows and caps each provider at three
in-flight calls and two starts per second. The separately source-bound
main-panel deadline launcher retains provider round-robin scheduling while
capping each provider at two in-flight calls and 1.5 starts per second; the
conservative launcher remains capped at one. Raw prompts, responses,
trajectories, annotation packets, and run-local sanitized files remain outside
the supplement. The packaged
`analysis/analyze_provider_safe_v2_definitive.py` verifies that completed
private manifests are bound to the exact conservative or accelerated launcher
and self-hashed rate-limit policy before emitting text-free definitive
aggregates. The separately packaged primary and role-calibration semantic-
invalid repair runners never overwrite a retained first response or change a
primary denominator; their outputs remain distinct exploratory evidence. The
paper-asset generator accepts only complete, text-free definitive adapters and
fails before publication on changed policy hashes, incomplete matrices, or
denominator mismatches. Fresh availability snapshots can be merged with the
dedicated judge only through the exact allowlisted merge utility and its
identity/compatibility checks. The separately packaged
`analysis/analyze_availability_retry_panels.py` accepts only the frozen Part 0,
Part 1, and Part 2 supplemental availability-retry designs, validates complete
private manifests and journals fail-closed, and emits explicitly exploratory,
nonreplacement, non-cross-axis descriptives. Its exact production-shaped test
and `docs/AVAILABILITY_RETRY_ANALYSIS.md` are included; its private inputs and
all live or generated retry-analysis outputs are not. See
`docs/ACCELERATED_PART0_HUMAN_VALIDATION.md`,
`docs/PART1_ROLE_CALIBRATION_V1.md`, and `docs/PART2_SENSITIVITY_V1.md` for the
exact gates and claim boundaries.

The deadline launchers for the definitive Part 0 and Part 1 matrices are also
included by exact path. They preserve the same prompt, parser, identity,
journal, and scheduled-denominator contracts while using source-bound bounded
parallelism. The Part 0 launcher additionally records bounded same-payload
HTTP-400 retries in the append-only attempt ledger and complete manifest.

## Fail-closed offline finalization

Offline administration runs only after a writer has stopped and its run lock
is free. The Part 0/Part 1 retirement tool validates retained target-bound
transport or identity-failure evidence, performs no network dispatch, and
assigns no behavioral outcome:

```bash
uv run python -m experiments.misc.inference_hub_retire_target \
  --manifest <part0-or-part1-private-manifest> \
  --target-id <exact-study-target-id> \
  --reason '<auditable operational reason>' \
  [--close-all-stale-reservations]
```

Part 0 records the eight unavailable IDs in `MODEL_REGISTRY.md`. Part 1 records
the main-shard MiniMax M2.7 retirement; Claude Opus 4.5 and n=12 MiniMax M3
already had complete target-bound failure evidence and required no invented
terminal result.

Part 2 was finalized without further dispatch by replaying only retained
complete trajectories and recording operational evidence for two targets. The
production invocation verified source bytes from clean Git commit `818414c`:

```bash
mkdir -p /tmp/sbr-source-818414c
git archive 818414c | tar -x -C /tmp/sbr-source-818414c
uv run python -m analysis.finalize_inference_hub_part2_offline \
  --manifest data/private/inference_hub/part2-sota-matched-v1-n8/private/manifest.json \
  --unavailable-target anthropic/claude-opus-4-6 \
  --unavailable-target minimaxai/minimax-m2.7 \
  --source-verification-root /tmp/sbr-source-818414c
```

The finalizer checks every journal chain, frozen input, source hash, and replayed
transition, and fails if replay reaches an unretained unit. Its private output
is not distributed.

## Build sanitized final results

Create the immutable public result directory once:

```bash
uv run python -m analysis.build_final_results \
  --part0-manifest data/private/inference_hub/part0-sota-panel-v2-n24/private/manifest.json \
  --part0-unavailable-target anthropic/claude-haiku-4-5 \
  --part0-unavailable-target anthropic/claude-opus-4-5 \
  --part0-unavailable-target anthropic/claude-opus-4-6 \
  --part0-unavailable-target anthropic/claude-sonnet-4-5 \
  --part0-unavailable-target minimaxai/minimax-m2.7 \
  --part0-unavailable-target openai/gpt-5 \
  --part0-unavailable-target openai/gpt-5.2 \
  --part0-unavailable-target openai/gpt-5.4 \
  --part1-full-manifest data/private/inference_hub/part1-sota-deadline-glm51-v1/private/manifest.json \
  --part1-partial-manifest data/private/inference_hub/part1-sota-balanced-main75-v1-n96/private/manifest.json \
  --part1-partial-manifest data/private/inference_hub/part1-sota-balanced-slow2-v2-n12/private/manifest.json \
  --part1-replacement-manifest data/private/inference_hub/part1-sota-repair-deepseek-v4-flash-v1-n96/private/manifest.json \
  --part1-replacement-manifest data/private/inference_hub/part1-sota-repair-deepseek-v4-pro-v1-n96/private/manifest.json \
  --part1-unavailable-target anthropic/claude-opus-4-5 \
  --part1-unavailable-target minimaxai/minimax-m2.7 \
  --part1-unavailable-target minimaxai/minimax-m3 \
  --part2-manifest data/private/inference_hub/part2-sota-matched-v1-n8/private/manifest.json \
  --part2-unavailable-target anthropic/claude-opus-4-6 \
  --part2-unavailable-target minimaxai/minimax-m2.7 \
  --panel-config experiments/sota_cross_axis_panel.json \
  --output-dir data/analysis/final_results
```

For a primary manifest with a validated target-bound operational or identity
failure, add the affected study ID with the repeatable axis-specific
`--part0-unavailable-target`, `--part1-unavailable-target`, or
`--part2-unavailable-target` option. A complete contract-matched replacement
manifest may instead be supplied with the corresponding repeatable
`--part*-replacement-manifest` option. Never add the three Part 1 targets that
were unavailable in the frozen registry before execution to these flags: they
were not manifest subjects and remain a separate pre-execution-unavailable
count.

The command refuses an existing output directory and fails on incomplete
coverage without validated target-bound unavailability, identity drift, a
broken journal chain, a changed response hash, parser inconsistency, an
ineligible Part 2 trajectory, or any forbidden public text field. It preserves
each included Part 1 target's observed count and never pools the 12-root,
96-root, or 384-root rows. Cross-axis
output is absent unless the exact 24-system overlap and every evidence gate
pass. The two replacement manifests preserve the exact DeepSeek target IDs and
frozen n=96 schedules; they are not model substitutions. The sealed result
self-hash is
`a9f961dd3a52fa082c9ed3ccab5c745c2fae7e2b303d9b1f08b3d0920517ecb1`.

Convert the sealed result graph into the exact, scope-separated values used by
the manuscript:

```bash
uv run python -m analysis.build_paper_headlines \
  --input data/analysis/final_results/final_results.json \
  --output-json data/analysis/final_results/paper_headlines.json \
  --output-tex data/analysis/final_results/paper_headlines.tex

uv run python -m analysis.build_paper_visuals \
  --input data/analysis/final_results/final_results.json \
  --output-dir docs/conference_submission/figures
```

These commands validate the final-results self-hash and privacy contract before
emitting within-axis counts, medians, ranges, and LaTeX macros. It refuses to
pool Part 1 scopes or compute model rankings, family effects, significance
tests, or cross-axis associations.

Build the separately labeled developer-route descriptives used for bounded
within-axis discussion:

```bash
uv run python -m analysis.build_developer_descriptives \
  --input data/analysis/final_results/final_results.json \
  --output data/analysis/final_results/developer_descriptives.json
```

This output groups exact systems alphabetically by upstream provider only when
at least two systems are present. It keeps Part 1 scopes separate and supports
neither rankings nor population-level family, vendor, or causal claims.

## Build Croissant metadata

```bash
uv run python -m analysis.build_croissant_metadata \
  --final-results-dir data/analysis/final_results \
  --output data/analysis/croissant_metadata.json
```

This command no longer catalogs April raw CSVs. It accepts only the self-hashed
`prosocial_readiness_final_sanitized_results` artifact and its hash-bound CSVs.
It checks that included rows plus validated, axis-specific operationally
unavailable IDs reconstruct the frozen 24-system Part 0 panel (16+8), the
78-target Part 1 execution roster (75 reported+3 operationally unavailable),
and the 24-system Part 2 panel (22 executed+2 unavailable). It separately accounts for the three frozen Part 1
registry targets that were unavailable before execution, yielding 81 planned
Part 1 targets without describing those three as observed. It also verifies 24
Part 0 roots per condition, 176 total Part 2 trajectories, 159 valid-trajectory
estimands, scope-separated CSV rows, and public-basename-only provenance. Missing final results, stale hashes,
sensitive fields, private paths, or a changed frozen scope stop metadata
emission.

At anonymous review time, omit the dataset URL; the local content-addressed
Croissant package is the available artifact boundary. Do not invent a
placeholder or identifying URL. At hosting time, supply a real public HTTPS
landing page with `--dataset-url` and rerun the metadata tests and external
validator.

## Build the paper and supplement

```bash
cd docs/conference_submission
pdflatex -interaction=nonstopmode conference_submission.tex
bibtex conference_submission
pdflatex -interaction=nonstopmode conference_submission.tex
pdflatex -interaction=nonstopmode conference_submission.tex
cd ../..
uv run python -m analysis.build_supplement
```

The supplement exactly allowlists the hosted runners, compatibility wrappers,
reviewer-driven follow-up protocols, fail-closed retirement and Part 2
finalization tools, and their required non-secret configs, documentation, and
tests. This includes both immutable semantic-invalid repair runners and tests,
the sensitivity runner and both frozen profiles, the definitive aggregate
adapter, the table/figure/headline-macro generator, the conservative and
deadline launchers, the compatibility merge support, and the Part 2
confirmatory CLI/statistics tests. A strict denylist excludes credentials,
private evidence, harmful
content, raw journals, incomplete private artifacts, every raw Part 1/Part 2
prompt, justification, or reasoning trace, and deprecated legacy tables and
plots. Only the three current paper-facing result figures are packaged. Sealed
sanitized final-result files retain their public `data/analysis/final_results/`
paths in the ZIP and remain bound by `SUPPLEMENT_MANIFEST.json`; no run-local
aggregate is imported implicitly from `data/private/`.
The one sanitized local-HF control aggregate consumed by the paper-assets
generator is included by exact public path; its raw JSONL, model cache, prompts,
and local execution runner remain excluded and it is never pooled with hosted
systems.
New hosted-route or availability-retry utilities do not enter through the
broad source roots: they remain excluded until their exact paths are reviewed
and added to the allowlist. At source freeze `17fc8bb`, the sole admitted
availability-retry surface is the exact analyzer, test, and documentation named
above; no retry run directory or generated retry result is packaged.
The current non-secret project checkpoint from `a949789` is included as
`CHECKPOINT.md`; it documents the frozen workflow but does not admit any path
under `data/private/` or any live/generated retry output.

The builder fixes both ZIP entry timestamps and the manifest creation epoch.
With unchanged inputs, two independent builds are byte-identical, including
`SUPPLEMENT_MANIFEST.json` and the ZIP SHA-256. A clean extraction can rebuild
the same manifest and byte-identical ZIP without access to private data; the
manifest intentionally does not record the number or text of private
anonymization-policy replacements.

## Acceptance gate

A release candidate is ready only when:

- the full test suite passes;
- every selected manifest or target-bound overlay is self-hash-valid and passes
  the applicable complete-unit, failure-evidence, identity, and coverage gates;
- final-result generation succeeds exactly once from the named artifacts and
  produces the recorded self-hash;
- Croissant generation and `--check` succeed against that final directory;
- the anonymous supplement builds and its audit reports no identity or private
  data leak;
- the paper reports Part 0, Part 1, Part 2, and cross-axis evidence at the status
  actually encoded by `final_results.json`.
