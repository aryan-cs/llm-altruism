# Reproducibility

This document records the commands needed to reproduce the graph-independent checks, paper tables, figures, and supplement from a clean clone. Unless a section says otherwise, run commands from the repository root.

## Environment

```bash
uv sync
cp .env.example .env
```

Provider API keys are needed only to rerun experiments. Validation and summary generation use existing CSV files and do not require secrets.

## Test Gate

```bash
uv run pytest -q
```

The exact passing-test count may grow as checks are added; the command must exit
successfully with no failures or errors.

## Metadata, Validation, Tables, Manifest

```bash
uv run python -m analysis.backfill_metadata
uv run python -m analysis.build_legacy_part2_provenance --check
uv run python -m analysis.validation
uv run python -m analysis.summarize_results
uv run python -m analysis.build_manifest
uv run python -m analysis.build_croissant_metadata
```

Use strict validation before freezing paper claims:

```bash
uv run python -m analysis.validation --strict
```

The checked-in April Part 2 sidecars predate an explicit
`collapse_death_rate` field. They are not edited or interpreted through the
current default. `build_legacy_part2_provenance --check` reads the immutable,
hash-bound historical execution source and prompt stored under
`data/raw/part_2/legacy_execution_archive/`, verifies the source's
divisor-of-five collapse rule, binds every CSV and sidecar byte hash, and
replays every recorded population transition. The archive manifest is
self-hashed and records the original paths, exact SHA-256 values, and both
recorded execution commits. This check does not read `.git`, so the same command
works after extracting the anonymous supplement. The deterministic seal is
`data/raw/part_2/legacy_structural_provenance.json`; validation refuses a
missing, stale, substituted, or self-hash-invalid seal.

The immutable execution archive is a release artifact, not a routine build
output. If it must be reconstructed from a full clone with the recorded Git
objects, remove no existing archive and run

```bash
uv run python -m analysis.build_legacy_part2_provenance --materialize-execution-archive
```

The command refuses to overwrite an existing archive and requires both
recorded commits to contain byte-identical source and prompt files.

The Part 1 legacy sidecars were reconstructed from filenames whose slugs lose
slashes, colons, underscores, and dots. The checked-in sidecars carry an
`identity_repair` record that binds the CSV SHA-256 and takes the exact unique
provider/model identity from the CSV rows. Reproduce that minimal repair with
`uv run python -m analysis.backfill_metadata --repair-part1-identities`; the
command preserves the original reconstruction timestamp, commit, and runtime
snapshot and rejects ambiguous CSV identities.

Croissant metadata is deterministic and must be rebuilt after any released
table, report, or manifest changes. The local anonymous artifact intentionally
omits a dataset `url` until a stable anonymous landing page exists. At hosting
time, regenerate with `--dataset-url` set to that real page and preserve the
current metadata-relative `tables/`, `validation/`, and manifest layout.

Outputs:

- `data/analysis/validation/validation_report.json`
- `data/analysis/tables/part0_model_summary.csv`
- `data/analysis/tables/part0_language_robustness.csv`
- `data/analysis/tables/part1_model_summary.csv`
- `data/analysis/tables/part1_dimension_summary.csv`
- `data/analysis/tables/part1_frame_effects.csv`
- `data/analysis/tables/part1_prompt_sensitivity.csv`
- `data/analysis/tables/part1_factor_decomposition.csv`
- `data/analysis/tables/part2_model_summary.csv`
- `data/analysis/tables/part2_run_summary.csv`
- `data/analysis/tables/cross_part_model_summary.csv`
- `data/analysis/tables/cross_part_correlations.csv`
- `data/analysis/run_manifest.jsonl`
- `data/analysis/croissant_metadata.json`
- `data/raw/part_2/legacy_structural_provenance.json`
- `data/raw/part_2/legacy_execution_archive/manifest.json`
- `data/raw/part_2/legacy_execution_archive/part_2.py`
- `data/raw/part_2/legacy_execution_archive/part_2_prompt.json`

The Part 0 model/language summaries and both cross-part tables are retained only
as deprecated forensic outputs. Their rows carry invalid-evidence status, and
the Croissant release plus anonymous supplement exclude them and all dependent
plots. Paper-facing results use only the supported Part 1 and Part 2 tables.

## Figures

Graph generation is intentionally separate from the graph-independent analysis pipeline. Figure scripts live under `data/graphs/`. Use the summary tables and validation report above as the authoritative numerical inputs when checking final figures.

```bash
uv run python data/graphs/part_0_graphs.py --latest
uv run python data/graphs/part_1_graphs.py --latest
uv run python data/graphs/part_2_graphs.py --latest
uv run python data/graphs/cross_part_graphs.py
uv run python data/graphs/paper_visuals.py
uv run python -m analysis.sync_conference_figures
```

The main paper uses supported Part 1/Part 2 plots and paper-specific visual
diagnostics rendered by `data/graphs/paper_visuals.py`. The sync command copies
the exact paper-used PNGs into `docs/conference_submission/figures/`, which is
the directory read by the LaTeX source. Invalid Part 0-dependent plots remain
outside the release for forensic replay only.

## Paper PDF

Build the anonymous conference submission from its own folder:

```bash
cd docs/conference_submission
pdflatex -interaction=nonstopmode conference_submission.tex
bibtex conference_submission
pdflatex -interaction=nonstopmode conference_submission.tex
pdflatex -interaction=nonstopmode conference_submission.tex
```

The upload PDF is `docs/conference_submission/conference_submission.pdf`. The anonymous supplement package should be staged in the same folder as `docs/conference_submission/supplement.zip`.

## Anonymous Supplement

Build the supplement archive from the repository root:

```bash
uv run python -m analysis.build_supplement
```

The output is `docs/conference_submission/supplement.zip`. The package contains executable code, release documentation, tests, derived analysis artifacts, figures, Part 1/Part 2 raw CSVs with metadata sidecars, and the hash-bound historical Part 2 source/prompt archive required for Git-independent legacy verification. Raw Part 0 harmful prompts, prompt-source CSVs, model completions, and author-identifying proposal metadata are excluded by policy; the ZIP includes `SUPPLEMENT_MANIFEST.json` documenting included files and exclusions.

## Exact-Version Confirmatory Campaign

`experiments.campaign` and its Cartesian Part 2 sensitivity CLI are retained
only to replay archived pilot workflows. They are not authorized for new
paper-facing collection. The sole active planner is
`experiments.confirmatory_campaign`, whose fixed one-stage design covers the
complete 24-system current cohort and six-system historical cohort. It rejects
catalog-display-only routes, incomplete approvals, and any attempt to revive
the retired adaptive/two-stage Part 2 design.

First produce the authenticated `/models` census and structured-smoke
evidence described in `docs/release/MODEL_REGISTRY.md`. After genuine Part 0
registry and Part 1 bank approvals exist, inspect the exact immutable plan
without writing a campaign directory or calling a provider:

```bash
uv run python -m experiments.confirmatory_campaign \
  --campaign-id confirmatory-budgeted-v1 \
  --cohort current_sota --cohort historical \
  --judge-target-id <verified-judge-target> \
  --part0-registry /absolute/private/part0-registry.json \
  --part0-registry-sha256 <sha256> \
  --part1-bank /absolute/private/part1-bank.json \
  --part1-bank-sha256 <sha256> \
  --endpoint-evidence /absolute/private/all-target-evidence.json \
  --endpoint-evidence-sha256 <sha256> \
  --dry-run
```

The dry run must report the complete 30-route matrix, source hashes, role and
request budgets, exclusions, and immutable plan hash. Remove `--dry-run` only
after those values match the preregistration. Resume with the same campaign ID;
the runner revalidates pinned inputs, route evidence, smoke dependencies,
native artifacts, and durable ledger reservations before continuing. The
separate post-lock Part 2 sensitivity panel is specified in
`docs/CONFIRMATORY_PROTOCOL.md`; it is not an alternative production CLI or a
source of primary estimates.

## Part 0 Judge Audit

`analysis/rejudge_part0.py` rejudges a legacy CSV from final response text only,
with separate input/output files, durable checkpoints, strict JSON labels, and
per-row provenance. `analysis/judge_audit.py` builds a deterministic blinded
human-audit packet and computes agreement after genuine human labels are
supplied. The complete commands and codebook are in `docs/JUDGE_AUDIT.md`.
The anonymous supplement also includes
`data/analysis/part0_rejudge_audit_checkpoint.json`, a non-harmful aggregate of
the incomplete 1,243-row checkpoint. It binds its counts to the withheld source
CSV by path, size, and SHA-256 while explicitly prohibiting corrected
model-level Part 0 inference.

## Local Artifact Acceptance Criteria

A local pilot artifact passes this checklist only when:

- the full test suite passes,
- validation has no failures,
- the source CSV has a metadata sidecar,
- the source CSV appears in `data/analysis/run_manifest.jsonl`,
- the paper-facing Croissant metadata is present at `data/analysis/croissant_metadata.json`,
- the anonymous supplement builds successfully with `uv run python -m analysis.build_supplement`,
- any validation warnings are either resolved or explicitly discussed.

This local checklist does not authorize submission or confirmatory claims.
Those additionally require all 30 authenticated routes and completed native
artifacts, genuine Part 0/Part 1 approvals and judge audit, a complete fixed
campaign/data lock, and anonymous reviewer-accessible hosting with official
Croissant validation.
