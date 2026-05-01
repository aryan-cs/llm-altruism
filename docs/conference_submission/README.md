# Conference Submission

This folder contains the anonymous NeurIPS 2026 Evaluations & Datasets submission source.

Build from this directory:

```bash
pdflatex -interaction=nonstopmode conference_submission.tex
bibtex conference_submission
pdflatex -interaction=nonstopmode conference_submission.tex
pdflatex -interaction=nonstopmode conference_submission.tex
```

The compiled upload PDF is `conference_submission.pdf`. Stage the anonymous supplement ZIP in this folder as `supplement.zip` so all conference-upload materials live under `docs/conference_submission/`.

Build the supplement from the repository root:

```bash
uv run python -m analysis.build_supplement
```

The supplement includes code, release documentation, tests, derived tables, figures, and Part 1/Part 2 raw CSVs with metadata sidecars. Raw Part 0 harmful prompts, source prompt CSVs, and model completions are excluded by policy.
