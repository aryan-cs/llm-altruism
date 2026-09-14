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

The fail-closed analyzer accepts exactly these terminal source policies and
source/overlay bindings:

| Phase | Private run directory or ordered pair | Frozen scope |
| :--- | :--- | :--- |
| Part 0 | `data/private/inference_hub/definitive-part0-large-n48-main22-deadline-v6` with `all-scheduled-operational-invalid-v1` | 22 exact routes × 48 roots × 3 response languages |
| Part 1 | `data/private/inference_hub/definitive-part1-large-n384-main75-deadline-v5` + `data/private/inference_hub/definitive-part1-large-n384-main75-deadline-v5-operational-repair-v1` | 75 exact routes × 384 balanced roots |
| Part 2 main pair | `data/private/inference_hub/full-part2-n12-n50-d100-main21-v5` + `data/private/inference_hub/full-part2-n12-n50-d100-main21-v5-operational-completion-capability-v4` | 21 exact routes × 12 common-seed trajectories |
| Part 2 Nemotron pair | `data/private/inference_hub/full-part2-n12-n50-d100-nemotron-3-ultra-recovered-v1` + `data/private/inference_hub/full-part2-n12-n50-d100-nemotron-3-ultra-operational-repair-v1` | 1 exact route × 12 common-seed trajectories |
| Part 2 DeepSeek pair | `data/private/inference_hub/full-part2-n12-n50-d100-deepseek-v4-flash-recovered-v1` + `data/private/inference_hub/full-part2-n12-n50-d100-deepseek-v4-flash-operational-repair-v1` | 1 exact route × 12 common-seed trajectories |
| Role calibration | `data/private/inference_hub/definitive-part1-role-calibration-v3` | 6 exact routes × 96 roots × 3 frames × 4 counterbalances |
| Part 2 sensitivity | `data/private/inference_hub/definitive-part2-sensitivity-deadline-fast-v9` + `data/private/inference_hub/definitive-part2-sensitivity-deadline-fast-v9-operational-repair-v1` | 5 exact compatible routes × 16 cells × 2 common seeds |

The Part 2 composition is complete at 23 routes and 276 trajectories under the
same 12 common seeds. All 276 trajectories are operationally eligible after 57
whole-trajectory repairs. Its 1,206,808 scheduled living agent-days contain
1,180,046 valid actions and 26,762 genuine semantic `INVALID` actions; the
valid actions divide into 969,640 restraint and 210,406 overuse. The 198
zero-invalid trajectories support environmental estimates; the other 78 are
invalid-bearing. Eighteen routes have an environmental summary, and five
zero-eligible routes are reported as NE. Opus 4.5 is the sole declared
exclusion and has no substitute.

The primary descriptive restraint proportion pools all scheduled living
agent-days within each route. Paper figures with trajectory Student-$t$
intervals instead use the equal-seed-weighted mean of the 12
trajectory-specific all-scheduled proportions. Because attrition changes
trajectory denominators, the two point estimates can differ.

The frozen original-scale contract is 50 agents, 100 days, initial capacity
2,500, OPTION_B private gain 2, reserve cost 2, unanimous group benefit/penalty
5, and collapse death rate 0.2. Part 0, Part 1, and sensitivity use the explicit
terminal policies and overlays shown above; a bare incomplete source is not
silently promoted.
The revised frozen sensitivity profile has a 14,400-post scheduled maximum with 32 trajectories per sentinel and 160 total. The one exact route incompatible with the common `top_p` control is excluded without substitution.
Each sentinel-factor contrast has four exact paired seed-block sign assignments;
all 25 prespecified contrasts remain in one global Holm family.

Check source state without reading response text:

```bash
for run in \
  data/private/inference_hub/definitive-part0-large-n48-main22-deadline-v6 \
  data/private/inference_hub/definitive-part1-large-n384-main75-deadline-v5 \
  data/private/inference_hub/definitive-part1-large-n384-main75-deadline-v5-operational-repair-v1 \
  data/private/inference_hub/full-part2-n12-n50-d100-main21-v5 \
  data/private/inference_hub/full-part2-n12-n50-d100-main21-v5-operational-completion-capability-v4 \
  data/private/inference_hub/full-part2-n12-n50-d100-nemotron-3-ultra-recovered-v1 \
  data/private/inference_hub/full-part2-n12-n50-d100-nemotron-3-ultra-operational-repair-v1 \
  data/private/inference_hub/full-part2-n12-n50-d100-deepseek-v4-flash-recovered-v1 \
  data/private/inference_hub/full-part2-n12-n50-d100-deepseek-v4-flash-operational-repair-v1 \
  data/private/inference_hub/definitive-part1-role-calibration-v3 \
  data/private/inference_hub/definitive-part2-sensitivity-deadline-fast-v9 \
  data/private/inference_hub/definitive-part2-sensitivity-deadline-fast-v9-operational-repair-v1
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

Run only after every terminal source/overlay binding above is present. The
output directory must not already exist:

```bash
uv run python -m analysis.analyze_provider_safe_v2_definitive \
  --part0 data/private/inference_hub/definitive-part0-large-n48-main22-deadline-v6 \
  --part0-terminal-policy all-scheduled-operational-invalid-v1 \
  --part1 data/private/inference_hub/definitive-part1-large-n384-main75-deadline-v5 \
  --part1-operational-repair data/private/inference_hub/definitive-part1-large-n384-main75-deadline-v5-operational-repair-v1 \
  --part2-source-overlay data/private/inference_hub/full-part2-n12-n50-d100-main21-v5 data/private/inference_hub/full-part2-n12-n50-d100-main21-v5-operational-completion-capability-v4 \
  --part2-source-overlay data/private/inference_hub/full-part2-n12-n50-d100-nemotron-3-ultra-recovered-v1 data/private/inference_hub/full-part2-n12-n50-d100-nemotron-3-ultra-operational-repair-v1 \
  --part2-source-overlay data/private/inference_hub/full-part2-n12-n50-d100-deepseek-v4-flash-recovered-v1 data/private/inference_hub/full-part2-n12-n50-d100-deepseek-v4-flash-operational-repair-v1 \
  --part2-declared-exclusion anthropic/claude-opus-4-5 \
  --role-calibration data/private/inference_hub/definitive-part1-role-calibration-v3 \
  --sensitivity data/private/inference_hub/definitive-part2-sensitivity-deadline-fast-v9 \
  --sensitivity-operational-repair data/private/inference_hub/definitive-part2-sensitivity-deadline-fast-v9-operational-repair-v1 \
  --output-dir data/processed/provider-safe-v2-definitive-analysis
```

The analyzer refuses incomplete or stale manifests, broken journal chains,
wrong source or rate-policy hashes, response-identity drift, judge overlap,
schedule mismatch, denominator changes, private text fields, and an existing
output directory. For Part 2 it also rejects a reordered pair, wrong singleton,
non-common seed, changed simulator contract, incomplete trajectory, or missing
sole exclusion. Its self-hashed manifest binds CSV and JSONL tables for:

- Part 0 model and language refusal summaries;
- Part 1 model, game, and domain direct-choice summaries;
- Part 2 model and trajectory summaries;
- role-calibration model/frame summaries; and
- Part 2 sensitivity model and prespecified global-Holm effect summaries.

Part 0 uses root-cluster finite-bank sensitivity intervals across the three
languages, Part 1 uses root resampling stratified by the 12 game-domain cells,
and Part 2 uses 12 independently seeded trajectories per route under a seed set
common across routes. The analyzer produces no cross-axis composite and does
not promote exploratory evidence to a latent trait.

## Build paper tables and figures

```bash
uv run python -m analysis.build_provider_safe_v2_paper_assets \
  --input-dir data/processed/provider-safe-v2-definitive-analysis \
  --local-controls data/analysis/local_hf_part1_controls.json \
  --output-dir data/processed/provider-safe-v2-paper-assets
```

The paper-asset builder verifies the definitive manifest and the separately
self-hashed local-control aggregate. It emits one-route-per-row tables,
deterministic headline macros, and nine visual families in vector PDF and
300-dpi PNG. Figures use the original submission palette and a Times-compatible
NeurIPS-template font. Local controls remain separate from hosted routes.

The three Part 2 operational overlays are part of the validated primary
composition because each replaces only a source trajectory that failed
operationally, reruns the whole trajectory from day 1 on the exact route, and
preserves source lineage. They never regenerate a genuine semantic `INVALID`.
Other availability-retry and semantic-invalid-repair analyzers validate their
own complete private sources and publish separate, text-free diagnostics. They
may not replace a primary row, change a denominator, enter a cross-axis score,
or make an unavailable route appear observed. See
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

- every named terminal source policy and operational overlay is self-hash-valid
  and passes exact schedule, source, rate-policy, identity, journal, and
  transition validation;
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
