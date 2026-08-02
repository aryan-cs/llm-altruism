# When a Benchmark Fails Its Audit — Anonymous Supplement

This package contains the release-safe evidence for the paper's audit of the
April 2026 Prosocial Cost-Shifting Bench pilot. It is not a validated behavioral
benchmark or leaderboard.

Interpretation is deliberately narrow:

- Part 0 model-level rates are withdrawn. The package contains only the
  sanitized aggregate label-instability checkpoint; raw harmful requests and
  completions are withheld.
- Part 1 supports descriptive fixed-bank action-label summaries.
- Part 2 contains stored `OPTION_A`/`OPTION_B` tokens and mechanically downstream
  state traces from a prompt--engine contract mismatch. It supports contract
  diagnosis and transition replay, not commons-preference measurement.

From the package root, reproduce the released artifacts with:

```bash
uv sync
uv run pytest -q
uv run python -m analysis.build_legacy_part2_provenance --check
uv run python -m analysis.validation --strict
uv run python -m analysis.summarize_results
uv run python -m analysis.build_manifest
uv run python -m analysis.build_croissant_metadata
uv run python data/graphs/paper_visuals.py
uv run python -m analysis.sync_conference_figures
```

The summary command intentionally does not recreate withdrawn Part 0
model-level or Part 0-dependent cross-part tables.

`SUPPLEMENT_MANIFEST.json` binds every packaged path to its SHA-256 and records
all policy exclusions. `docs/release/DATA_CARD.md` and
`docs/release/REPRODUCIBILITY.md` document schemas, provenance, limitations, and
responsible use. Post-pilot hosted-route experiments and local scale controls
remain in the development repository and are intentionally absent here. Some
later protocol and statistical modules are retained because the released pilot
analysis and full regression suite import them; their presence is dependency
closure, not evidence used by the paper.
