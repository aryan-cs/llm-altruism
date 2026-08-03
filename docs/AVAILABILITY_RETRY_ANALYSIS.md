# Availability-retry panel analysis

`analysis/analyze_availability_retry_panels.py` is the fail-closed analyzer for
the three supplemental InferenceHub availability retries. It is deliberately
separate from every primary-result pipeline. Its outputs are exploratory
availability evidence and must never be merged into, substituted for, or used
to rank models across Parts 0, 1, and 2.

The analyzer accepts only this exact design:

- Part 0: Claude Opus 4.5, 48 crossed roots in each of English, Chinese, and
  Russian (144 scheduled units), with the fixed disjoint Nemotron judge.
- Part 1: Claude Opus 4.5, MiniMax M3, Kimi K2.6, and GLM 5.2, each on the
  frozen 384-trial schedule. The judge is reserved and cannot be dispatched by
  this runner.
- Part 2: Claude Opus 4.5, DeepSeek V4 Flash, Nemotron 3 Ultra, and GLM 5.1,
  each with 12 independent trajectories. The judge is reserved and cannot be
  dispatched by this runner.

Before writing anything, it validates all three COMPLETE, self-hashed private
manifests; the current `inference_hub_main_accelerated.py` and
`inference_hub_provider_safe_v2.py` source digests; the self-hashed shared rate
policy (12 global / 2 per provider, 8.0 global / 1.5 per-provider requests per
second); exact model identities and selected routes; fixed-judge disjointness;
and one identical combined availability-retry registry and compatibility
artifact across all panels. It then checks every journal hash chain and
checkpoint, required row and trajectory counts, Part 2 reservation/terminal
bindings, raw response and request hashes, and agreement between journals and
sanitized denominators. A failed check produces no output directory.

Run only after all three manifests are COMPLETE:

```bash
uv run python -m analysis.analyze_availability_retry_panels \
  --part0 data/private/inference_hub/<part0-retry-run> \
  --part1 data/private/inference_hub/<part1-retry-run> \
  --part2 data/private/inference_hub/<part2-retry-run> \
  --output-dir artifacts/availability_retry_analysis_<version>
```

The output directory must not already exist. A successful run atomically
creates an analysis manifest plus separate CSV, JSONL, and LaTeX fragments for
each part. Each table has one row per exact target route, uses all-scheduled
denominators, defines every column, states the within-task safer direction,
reports invalid outcomes, defines `NE`, and places 15 pt (approximately 20 CSS
pixels) above and below the table. Every machine-readable row is explicitly labeled
`availability_retry`, `exploratory_only=true`, `replaces_primary=false`, and
`cross_axis_permitted=false`.

Validate the implementation with:

```bash
uv run pytest -q tests/test_analyze_availability_retry_panels.py
```

The tests build the exact 144-unit, four-by-384-trial, and four-by-12-
trajectory production shapes, including a legitimate Part 2 transport retry.
They also prove fail-closed behavior for incomplete evidence, manifest tamper,
response identity mismatch, judge overlap, wrong row counts, and attempted
output overwrite.
