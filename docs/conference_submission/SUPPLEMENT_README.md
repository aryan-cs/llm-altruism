# Safety Beyond Refusal: Anonymous Supplement

This package accompanies **Safety Beyond Refusal** and the three-axis
**Prosocial Readiness Bench**. It reports three observable behaviors
separately: harmful-request refusal, welfare-preserving self-choice in
one-shot dilemmas, and preservation of a shared resource in repeated
simulations. It does not construct an altruism, morality, deployment-safety,
or cross-axis composite score.

## Definitive design and source status

- **Part 0:** 22 exact model routes are scheduled on 48 archived English
  harmful-request roots crossed with English, Simplified Chinese, and Russian
  response-language instructions (144 responses per route). One fixed judge is
  disjoint from every evaluated subject. First-attempt invalids remain in the
  scheduled-response denominator. Because the bank has no benign controls and
  human judge validation is incomplete, these estimates are exploratory. The
  source manifest was still running at the documentation checkpoint, so the
  schedule is not presented as completed coverage.
- **Part 1:** 75 exact model routes are scheduled on the same balanced
  384-root bank, stratified over 12 game-domain cells. First-attempt malformed
  actions remain in the primary denominator as non-welfare-preserving choices.
  The prompt bank lacks independent content approval, so these estimates are
  exploratory. Four local open-weight controls use the same 384-root design
  and are displayed separately rather than pooled with hosted routes. The
  hosted source manifest was still running at the documentation checkpoint.
- **Part 2:** 19 exact model routes are scheduled for 12 independent,
  common-seed trajectories under the corrected five-agent, 12-step commons
  engine (228 trajectories total). Behavioral restraint retains every
  scheduled agent-day in its denominator. Environmental AURC, AUPC, reserve
  nondepletion, and population retention use only trajectories that are both
  operationally eligible and environmentally estimable; semantic-invalid
  trajectories are enumerated rather than silently imputed. This source
  manifest is complete at 228 trajectories and 13,495 scheduled agent-days;
  four invalid actions occur across three trajectories, leaving 225
  environmentally estimable trajectories.
- **Role calibration and sensitivity:** Six prespecified role routes are scheduled
  under three role frames over 96 roots with four counterbalances per frame
  (384 units per route/frame). A separate
  deadline-exploratory Part 2 design evaluates five exact compatible sentinels
  in 16 fractional-factorial cells under two independent common seeds. The one
  incompatible exact route is excluded without substitution. These artifacts diagnose
  framing and environment sensitivity; they are not confirmatory promotions.
- **Unavailable routes and invalid recovery:** Bounded availability retries and
  semantic-invalid repairs are isolated supplemental artifacts. They never
  substitute a different model, overwrite a primary response, alter a primary
  denominator, or enter a cross-axis score. If an exact route remains
  unavailable, it receives no behavioral estimate.

All reportable counts in the Croissant metadata and paper tables are computed
from the hash-bound definitive CSVs. The builders contain no legacy coverage
constants such as 16/24, mixed Part 1 scopes, or eight-trajectory Part 2 runs.

## Reproduction and validation

From the extracted package root:

```bash
uv sync --frozen
uv run pytest -q
uv run python -m analysis.build_provider_safe_v2_croissant_metadata --check
uv run python -m analysis.build_supplement --require-definitive-artifacts
```

To regenerate the paper-facing tables and figures into a fresh directory while
leaving the sealed packaged assets untouched:

```bash
release_tmp="$(mktemp -d)"
uv run python -m analysis.build_provider_safe_v2_paper_assets \
  --input-dir data/processed/provider-safe-v2-definitive-analysis \
  --local-controls data/analysis/local_hf_part1_controls.json \
  --output-dir "$release_tmp/provider-safe-v2-paper-assets"
```

`analysis.analyze_provider_safe_v2_definitive` is the sole bridge from the five
private campaign manifests to the primary public aggregate graph, and it runs
only after all five are complete. It
validates complete self-hashed manifests and journals, exact response-model
identity, fixed judge disjointness, expected schedules, parser outcomes,
invalid-denominator policy, and source bindings before atomically publishing a
text-free directory. Its public manifest uses portable basenames only and
SHA-256-binds every nonmanifest output.

`analysis.build_provider_safe_v2_paper_assets` accepts only that definitive
manifest and the separately self-hashed local-control aggregate. It emits
one-route-per-row Markdown and LaTeX tables, 20-pixel-equivalent outer table
spacing, directional captions, raster PNG figures, and vector PDF figures. The
supplement packages the PNG, Markdown, and LaTeX assets but omits redundant PDF
copies; the submission PDF is uploaded separately.

`analysis.build_provider_safe_v2_croissant_metadata` validates the complete
analysis and paper-asset inventories, recomputes every SHA-256, recomputes all
coverage counts from the definitive CSVs, rejects private fields and host paths,
and emits Croissant 1.1 JSON-LD with portable relative `contentUrl` values. The
metadata intentionally has no dataset URL during anonymous review.

`analysis.build_supplement --require-definitive-artifacts` fails if the
definitive analysis, paper assets, or Croissant metadata are missing, partial,
tampered, privacy-unsafe, or mutually stale. The same gate validates any
present availability-retry or semantic-repair directory and enforces its
nonreplacement contract. `SUPPLEMENT_MANIFEST.json` then SHA-256-binds every
payload in a deterministic, byte-reproducible ZIP.

## Release boundary

This package contains reviewed runner/finalization code, tests, frozen public
design registries, documentation, text-free definitive aggregates, current
paper tables and raster figures, Croissant metadata, and any completed isolated
retry/repair summaries. It excludes `.env` files, API keys, private manifests,
harmful prompts, visible responses, reasoning, raw journals, interrupted runs,
legacy final-results directories, superseded conference figures, and all
private execution evidence.

The artifact is **aggregate-reproducible, not collection-reproducible**:
reviewers can verify hashes and schemas and regenerate paper-facing assets from
the released aggregates, but cannot recollect model responses or reconstruct
the definitive aggregate graph without the deliberately withheld private
execution evidence.
