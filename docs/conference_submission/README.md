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
