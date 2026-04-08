# ICML Submission Build

This directory contains an anonymous ICML-style submission build for the
current paper bundle.

Template source:

- official ICML 2025 style files from `https://media.icml.cc/Conferences/ICML2025/Styles/icml2025.zip`

Build command from the repository root:

```bash
.venv/bin/python scripts/build_icml_submission.py
```

Generated outputs:

- `paper/icml2025/llm_altruism_icml2025_submission.pdf`
- `paper/icml2025/build/main.tex`
- `paper/icml2025/build/generated_body.tex`
- `paper/icml2025/build/generated_appendix.tex`
- `paper/icml2025/build/main.bbl`
- `paper/icml2025/build/selected_figures.tex`
- `paper/icml2025/build/llm_altruism_icml2025_submission.pdf`

Notes:

- the build uses the current Markdown manuscript and appendix plus
  `paper/references.bib` as the source of truth
- the output is an anonymous submission-style PDF, not a camera-ready version
- the style files are vendored locally in `paper/icml2025/style/` so the build
  does not depend on network access
- `paper/icml2025/build/` is generated scratch space and can be deleted at any
  time; the tracked deliverable is `paper/icml2025/llm_altruism_icml2025_submission.pdf`
