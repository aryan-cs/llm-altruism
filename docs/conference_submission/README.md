# Conference Submission

This folder contains the anonymous NeurIPS 2026 Evaluations & Datasets submission source.

Build from this directory:

```bash
pdflatex -interaction=nonstopmode conference_submission.tex
bibtex conference_submission
pdflatex -interaction=nonstopmode conference_submission.tex
pdflatex -interaction=nonstopmode conference_submission.tex
```

The submission upload copy is staged as `submission/paper.pdf`.

