# Reproducibility

Run commands from the repository root. The release pipeline has a hard boundary
between private collection evidence and public, text-free aggregate artifacts.
Reviewers can verify the definitive aggregate graph and regenerate its tables,
figures, metadata, and supplement, but cannot recollect private model responses
from the anonymous package.

## Environment and tests

```bash
uv sync --frozen
uv run pytest -q
```

A provider key is needed only for new collection. Never place a real key in a
command, log, tracked file, or supplement. Analysis, figure generation,
Croissant validation, and supplement packaging require no provider credential.

## Definitive private inputs

The fail-closed analyzer accepts exactly these five source runs:

| Phase | Private run directory | Frozen scope |
| :--- | :--- | :--- |
| Part 0 | `data/private/inference_hub/definitive-part0-large-n48-main22-deadline-v6` | 22 exact routes × 48 roots × 3 response languages |
| Part 1 | `data/private/inference_hub/definitive-part1-large-n384-main75-deadline-v5` | 75 exact routes × 384 balanced roots |
| Part 2 | `data/private/inference_hub/definitive-part2-n12-main19-v3` | 19 exact routes × 12 common-seed trajectories |
| Role calibration | `data/private/inference_hub/definitive-part1-role-calibration-v3` | 6 exact routes × 96 roots × 3 frames × 4 counterbalances |
| Part 2 sensitivity | `data/private/inference_hub/definitive-part2-sensitivity-deadline-fast-v8` | 6 exact routes × 16 cells × 2 common seeds |

The Part 2 source manifest is complete at 228 trajectories and 13,495
scheduled agent-days. It records zero transport or identity failures and four
invalid actions across three trajectories; 225 trajectories contain no invalid
action and support environmental estimates. At the 2026-08-03 documentation
checkpoint, Part 0, Part 1, role calibration, and sensitivity were still
running. Their scheduled scopes must not be described as completed coverage.
The frozen sensitivity profile has a 17,280-post scheduled maximum with 32 trajectories per sentinel and 192 total.
Each sentinel-factor contrast has four exact paired seed-block sign assignments;
all 30 prespecified contrasts remain in one global Holm family.

Check source state without reading response text:

```bash
for run in \
  data/private/inference_hub/definitive-part0-large-n48-main22-deadline-v6 \
  data/private/inference_hub/definitive-part1-large-n384-main75-deadline-v5 \
  data/private/inference_hub/definitive-part2-n12-main19-v3 \
  data/private/inference_hub/definitive-part1-role-calibration-v3 \
  data/private/inference_hub/definitive-part2-sensitivity-deadline-fast-v8
do
  jq '{complete, completed_at_utc, summary, evidence_sha256}' \
    "$run/private/manifest.json"
done
```

Every hosted caller uses the shared cross-process rate limiter, exact
compatibility-selected routes, `Retry-After` handling, provider cooldowns,
durable leases, and append-only hash-chained journals. Bounded transport retries
never substitute a route. First-attempt semantic invalids remain in primary
denominators; periodic exact-route repair artifacts remain separate.

## Build definitive aggregates

Run only after all five source manifests report `complete: true`:

```bash
uv run python -m analysis.analyze_provider_safe_v2_definitive \
  --part0 data/private/inference_hub/definitive-part0-large-n48-main22-deadline-v6 \
  --part1 data/private/inference_hub/definitive-part1-large-n384-main75-deadline-v5 \
  --part2 data/private/inference_hub/definitive-part2-n12-main19-v3 \
  --role-calibration data/private/inference_hub/definitive-part1-role-calibration-v3 \
  --sensitivity data/private/inference_hub/definitive-part2-sensitivity-deadline-fast-v8 \
  --output-dir data/processed/provider-safe-v2-definitive-analysis
```

The analyzer refuses incomplete or stale manifests, broken journal chains,
wrong source or rate-policy hashes, response-identity drift, judge overlap,
schedule mismatch, denominator changes, private text fields, and an existing
output directory. Its self-hashed manifest binds CSV and JSONL tables for:

- Part 0 model and language refusal summaries;
- Part 1 model, game, and domain direct-choice summaries;
- Part 2 model and trajectory summaries;
- role-calibration model/frame summaries; and
- Part 2 sensitivity model and prespecified global-Holm effect summaries.

Part 0 uses root-cluster finite-bank sensitivity intervals across the three
languages, Part 1 uses root resampling stratified by the 12 game-domain cells,
and Part 2 uses independent trajectories. The analyzer produces no cross-axis
composite and does not promote exploratory evidence to a latent trait.

## Build paper tables and figures

```bash
uv run python -m analysis.build_provider_safe_v2_paper_assets \
  --input-dir data/processed/provider-safe-v2-definitive-analysis \
  --local-controls data/analysis/local_hf_part1_controls.json \
  --output-dir data/processed/provider-safe-v2-paper-assets
```

The paper-asset builder verifies the definitive manifest and the separately
self-hashed local-control aggregate. It emits one-route-per-row tables,
deterministic headline macros, and seven visual families in vector PDF and
300-dpi PNG. Figures use the original submission palette and a Times-compatible
NeurIPS-template font. Local controls remain separate from hosted routes.

Availability-retry and semantic-invalid-repair analyzers validate their own
complete private sources and publish separate, text-free diagnostics. They may
not replace a primary row, change a denominator, enter a cross-axis score, or
make an unavailable route appear observed. See
`docs/AVAILABILITY_RETRY_ANALYSIS.md`,
`docs/PART1_SEMANTIC_INVALID_REPAIR.md`, and
`docs/PART1_ROLE_SEMANTIC_INVALID_REPAIR.md`.

## Build Croissant metadata

```bash
uv run python -m analysis.build_provider_safe_v2_croissant_metadata
uv run python -m analysis.build_provider_safe_v2_croissant_metadata --check
```

The builder recomputes all coverage counts and SHA-256 hashes from the
definitive CSVs and paper-asset manifest. It emits portable Croissant 1.1 JSON-
LD and rejects private fields, absolute host paths, stale inventories, and
tampering. Anonymous-review metadata intentionally omits a dataset URL; a real
reviewer-accessible HTTPS landing page remains an external release gate.

## Build the paper and anonymous supplement

Compile the submission with a full LaTeX installation or Tectonic, then build
the strict supplement:

```bash
cd docs/conference_submission
pdflatex -interaction=nonstopmode conference_submission.tex
bibtex conference_submission
pdflatex -interaction=nonstopmode conference_submission.tex
pdflatex -interaction=nonstopmode conference_submission.tex
cd ../..

uv run python -m analysis.build_supplement --require-definitive-artifacts
```

The strict supplement builder requires the definitive analysis, paper assets,
and Croissant metadata to be present, complete, mutually current, and privacy-
safe. It validates any included availability or semantic-repair summary under
the same nonreplacement boundary. It packages reviewed code, tests,
documentation, text-free aggregates, PNG/Markdown/LaTeX paper assets, and
portable metadata; it excludes credentials, private routes, prompts,
responses, reasoning, journals, interrupted runs, superseded deadline results,
and redundant figure PDFs. Fixed timestamps and exact input hashes make the ZIP
byte-reproducible.

Verify the package from a clean extraction without a Git object store:

```bash
release_tmp="$(mktemp -d)"
unzip -q docs/conference_submission/supplement.zip -d "$release_tmp/extracted"
cd "$release_tmp/extracted"
uv sync --frozen
uv run pytest -q
uv run python -m analysis.build_provider_safe_v2_croissant_metadata --check
uv run python -m analysis.build_supplement --require-definitive-artifacts \
  --output "$release_tmp/rebuilt.zip"
```

## Acceptance gate

A release candidate is ready only when:

- all five named source manifests are complete, self-hash-valid, and pass exact
  schedule, source, rate-policy, identity, and journal validation;
- definitive aggregate generation succeeds and every reported paper number
  matches a generated table or macro;
- Part 0 judge/subject disjointness passes at target, route, and served-identity
  levels, while missing human validation remains disclosed;
- the paper-asset manifest, Croissant `--check`, strict supplement build, and
  clean-extraction rebuild all pass;
- the full repository suite, strict data validation, citations, anonymity,
  NeurIPS format, and every rendered PDF page pass final review; and
- no scheduled count is presented as observed coverage, no route is silently
  substituted, and no supplemental retry changes primary evidence.
