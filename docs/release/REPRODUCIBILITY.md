# Reproducibility

Run commands from the repository root. The release pipeline has a hard boundary
between private execution evidence and public aggregate artifacts.

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

All hosted callers use the shared cross-process rate limiter and exact
compatibility-selected routes. Part 0 and Part 1 subject calls can progress in
parallel across upstream providers, while provider-specific concurrency remains
bounded. Part 0 judge batches use one fixed non-subject judge. Only transport
failures are retried; malformed semantic outputs are retained.

## Build sanitized final results

Create the immutable public result directory once:

```bash
uv run python -m analysis.build_final_results \
  --part0-manifest data/private/inference_hub/part0-sota-panel-v2-n24/private/manifest.json \
  --part1-full-manifest data/private/inference_hub/part1-sota-deadline-glm51-v1/private/manifest.json \
  --part1-partial-manifest data/private/inference_hub/part1-sota-balanced-main75-v1-n96/private/manifest.json \
  --part1-partial-manifest data/private/inference_hub/part1-sota-balanced-slow2-v2-n12/private/manifest.json \
  --part2-manifest data/private/inference_hub/part2-sota-matched-v1-n8/private/manifest.json \
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
pass.

Convert the sealed result graph into the exact, scope-separated values used by
the manuscript:

```bash
uv run python -m analysis.build_paper_headlines \
  --input data/analysis/final_results/final_results.json \
  --output-json data/analysis/final_results/paper_headlines.json \
  --output-tex data/analysis/final_results/paper_headlines.tex
```

This command validates the final-results self-hash and privacy contract before
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
unavailable IDs reconstruct the frozen 24-system Part 0 panel, the 78-target
Part 1 execution roster (75 at n=96, two at n=12, one at n=384), and the
24-system Part 2 panel. It separately accounts for the three frozen Part 1
registry targets that were unavailable before execution, yielding 81 planned
Part 1 targets without describing those three as observed. It also verifies 24
Part 0 roots per condition, eight Part 2 trajectories, scope-separated CSV
rows, and public-basename-only provenance. Missing final results, stale hashes,
sensitive fields, private paths, or a changed frozen scope stop metadata
emission.

At anonymous review time, omit the dataset URL. At hosting time, supply a real
public HTTPS landing page with `--dataset-url` and rerun the metadata tests.

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

The supplement allowlists the three hosted runners and their required
non-secret dependencies and tests. A strict denylist excludes credentials,
private evidence, harmful content, raw journals, incomplete runs, and deprecated
legacy artifacts. Completed sanitized aggregate files are remapped under
`data/analysis/` in the ZIP and remain bound by `SUPPLEMENT_MANIFEST.json`.

## Acceptance gate

A release candidate is ready only when:

- the full test suite passes;
- every selected private manifest is complete and self-hash-valid;
- final-result generation succeeds exactly once from the named artifacts;
- Croissant generation and `--check` succeed against that final directory;
- the anonymous supplement builds and its audit reports no identity or private
  data leak;
- the paper reports Part 0, Part 1, Part 2, and cross-axis evidence at the status
  actually encoded by `final_results.json`.
