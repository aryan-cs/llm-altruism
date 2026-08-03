# Provider-safe-v2 paper assets

`analysis/build_provider_safe_v2_paper_assets.py` is an isolated, fail-closed
renderer for the completed sanitized directory written by
`analysis/analyze_provider_safe_v2_definitive.py`. It does not read the private
campaign manifests referenced by that analysis, alter the conference TeX,
create human labels, or authorize paper promotion.

## Input contract

The input directory must contain the self-hashed definitive analysis manifest,
its six JSONL tables, and `figure_aggregates.json`. It also reads the separate,
already sanitized `data/analysis/local_hf_part1_controls.json` aggregate; it
never follows that artifact's provenance paths to private JSONL. The generator
requires the complete production matrix:

- 22 Part 0 routes and exactly three 48-root response-language conditions per
  route (`english`, `chinese`, and `russian`);
- 75 Part 1 routes with 384 scheduled self-choice units each;
- 19 Part 2 routes with 12 trajectories each;
- six role-calibration routes by three separate 384-draw frames; and
- five independent exact compatible sensitivity sentinels by five factors,
  with 16 cells, two common seeds, and one global 25-test Holm family; and
- four fixed local HF Part 1 execution-scale controls with 384 scheduled units
  per model, kept separate from all hosted routes.

Counts, rates, denominators, exact route/model identities, frozen factor
levels, exploratory status, and duplicate/missing matrix cells are all checked
before rendering. First-attempt invalid outcomes remain in the primary
scheduled-unit denominators. Existing output directories are never
overwritten; publication is atomic only after every asset succeeds.

## Run

```bash
UV_CACHE_DIR=/tmp/llm-altruism-uv-cache \
  uv run python -m analysis.build_provider_safe_v2_paper_assets \
  --input-dir data/processed/provider-safe-v2-definitive-analysis \
  --local-controls data/analysis/local_hf_part1_controls.json \
  --output-dir data/processed/provider-safe-v2-paper-assets
```

The command must be run only after the definitive descriptive adapter has
completed successfully. It never follows the private paths recorded for input
provenance in `analysis_manifest.json`.

## Outputs

Each figure is emitted as a vector PDF and a 300-dpi PNG:

- `part0_model_language`: two route-by-language heatmaps for refusal and valid
  output coverage;
- `part1_all_models`: one ordered row per each of the 75 routes, with
  welfare-preserving rate and valid coverage;
- `part2_all_models`: one ordered row per each of the 19 routes, with restraint,
  normalized AURC, and valid coverage;
- `part1_role_calibration`: six routes by three separate frames, with welfare
  preservation and valid coverage; and
- `part2_sensitivity_effects`: six routes by five high-minus-low AURC effects,
  annotated with global Holm status; and
- `part1_local_controls`: a compact, separate four-row plot of welfare
  preservation and format validity by exact local model ID and parameter scale.

Matching LaTeX fragments use exact route/model IDs and contain self-contained
captions defining every row and column, denominator, and direction. Long tables
are divided into display blocks without dropping rows. Every `table*` block has
`\par\addvspace{15pt}` before and after it (approximately 20 CSS pixels at
96 dpi), and no generated fragment uses negative vertical space. The fragments
require the paper's existing `booktabs` and `graphicx` packages.

The local-control table and figure define the model ID, advertised parameter
scale, n=384 denominator, welfare-preserving rate with invalid outputs retained
as nonsuccesses, and format-valid rate. They explicitly label the panel as
exploratory execution-scale controls—not substitutes for hosted routes and not
confirmatory evidence. Local rows are never appended to, compared with, or
pooled into the hosted 75-route panel.

All ordering and direction statements are within-task only. The generator does
not create a cross-axis score, composite, leaderboard, or safety ranking.
Sensitivity signs mean only high-factor minus low-factor normalized AURC;
positive and negative are not automatically good or bad parameter settings.

`paper_assets_manifest.json` hashes every emitted asset and records that the
outputs remain exploratory, generate no human labels, preserve exact route and
model IDs, keep local controls separate from hosted routes, and do not permit
confirmatory or paper promotion.

## Deterministic headline macros

The same validated inputs also produce `paper_headlines.tex`. It contains only
deterministic scalar `\newcommand` definitions; it has no timestamp, model
labels, prose claims, cross-axis composite, ranking, or promotion macro. Macro
names contain TeX command letters only. `Pct` values omit the percent sign so
the consuming prose controls typography.

The macro families are:

- `ProviderSafePartZero...`: exact model, scheduled-response, refusal,
  compliance, unclear, and invalid totals, plus minimum/median/maximum model
  refusal percentages. All rates retain all scheduled responses.
- `ProviderSafePartOne...`: exact model, scheduled-unit, welfare-preserving,
  and invalid totals, plus minimum/median/maximum model welfare percentages.
  Invalid first attempts remain scheduled nonsuccesses.
- `ProviderSafePartTwo...`: model and trajectory totals; operationally eligible
  and ineligible trajectory totals; scheduled, valid, and invalid agent-day
  totals; nonestimable-model count; and model minimum/median/maximum normalized
  AURC and restraint percentage. Valid/invalid macros explicitly count
  scheduled agent-days. AURC summaries use estimable model means only; if no
  model is estimable, the three AURC macros emit `NE` rather than changing the
  denominator or inventing a value.
- `ProviderSafeRoleAdvice...`, `ProviderSafeRoleObserverEvaluation...`, and
  `ProviderSafeRolePrediction...`: separate per-frame model counts and
  minimum/median/maximum welfare and valid-coverage percentages. Frames are
  never pooled.
- `ProviderSafeSensitivity...`: sentinel count, total trajectory count, common
  seed count, effect count, maximum absolute high-minus-low normalized-AURC
  effect, and count with global Holm-adjusted p at most 0.05. These remain
  deadline-exploratory descriptors.

The macro file is hashed as a `latex_macros` asset in
`paper_assets_manifest.json`. The generator intentionally emits no headline
macros for combining Part 0, Part 1, Part 2, role, sensitivity, hosted routes,
or local controls.

## Verification

```bash
UV_CACHE_DIR=/tmp/llm-altruism-uv-cache \
  uv run pytest -q tests/test_build_provider_safe_v2_paper_assets.py
```

The tests construct the full production-shaped sanitized matrix, render all
six PDF/PNG/TeX families plus the deterministic headline macro fragment, verify
that the PDFs contain vector marks rather than
embedded raster charts, validate PNG resolution and output hashes, count all 75
and 19 hosted model rows plus four local-control rows, check the 15-point spacing around every table block, and
exercise missing-row, missing-field, duplicate-ID, factor/Holm, identity,
local-control, manifest-tamper, and no-overwrite failures. The production-shaped
headline test parses every emitted macro and compares all 57 names and values
exactly, including one nonestimable Part 2 model and the frozen sensitivity
seed/Holm counts; a second write must be byte-for-byte identical.
