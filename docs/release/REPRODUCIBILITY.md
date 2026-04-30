# Reproducibility

This document records the commands needed to reproduce the graph-independent checks and paper tables from a clean clone.

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

Expected status at the time of this update:

```text
155 passed
```

## Metadata, Validation, Tables, Manifest

```bash
uv run python -m analysis.backfill_metadata
uv run python -m analysis.validation
uv run python -m analysis.summarize_results
uv run python -m analysis.build_manifest
```

Use strict validation before freezing paper claims:

```bash
uv run python -m analysis.validation --strict
```

Outputs:

- `data/analysis/validation/validation_report.json`
- `data/analysis/tables/part0_model_summary.csv`
- `data/analysis/tables/part1_model_summary.csv`
- `data/analysis/tables/part1_dimension_summary.csv`
- `data/analysis/tables/part2_model_summary.csv`
- `data/analysis/tables/cross_part_model_summary.csv`
- `data/analysis/tables/cross_part_correlations.csv`
- `data/analysis/run_manifest.jsonl`
- `data/analysis/croissant_metadata.json`

## Figures

Graph generation is intentionally separate from the graph-independent analysis pipeline. Figure scripts live under `data/graphs/`. Use the summary tables and validation report above as the authoritative numerical inputs when checking final figures.

```bash
uv run python data/graphs/part_0_graphs.py --latest
uv run python data/graphs/part_1_graphs.py --latest
uv run python data/graphs/part_2_graphs.py --latest
uv run python data/graphs/cross_part_graphs.py
```

The main paper uses selected master plots plus readable individual cross-part scatterplots. The graph tree also contains model-family plots and the full set of individual cross-part scatterplots for reviewer inspection.

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

## Acceptance Criteria

A result is paper-ready only when:

- the full test suite passes,
- validation has no failures,
- the source CSV has a metadata sidecar,
- the source CSV appears in `data/analysis/run_manifest.jsonl`,
- the paper-facing Croissant metadata is present at `data/analysis/croissant_metadata.json`,
- any validation warnings are either resolved or explicitly discussed.
