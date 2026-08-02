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

The source uses the unmodified official NeurIPS 2026 style with the anonymous Evaluations & Datasets option, `\usepackage[eandd]{neurips_2026}`. The full official checklist lives in `checklist.tex` and is included after the appendices.

After compiling, run the format gate from the repository root:

```bash
python3 -m unittest -v docs.conference_submission.test_format_compliance
```

When `pdftotext` is available, the gate also verifies that the Conclusion finishes by page 9 and References begin no later than page 10. Otherwise, that rendered-PDF check is skipped while the source-level style, checklist, order, and anonymization checks still run.

Build the supplement from the repository root:

```bash
uv run python -m analysis.build_supplement
```

The supplement includes code, release documentation, tests, derived tables, figures, and Part 1/Part 2 raw CSVs with metadata sidecars. Raw Part 0 harmful prompts, source prompt CSVs, and model completions are excluded by policy.
