# Safety Beyond Refusal: Anonymous Supplement

This package accompanies **Safety Beyond Refusal** and its three-axis
**Prosocial Readiness Bench**. The benchmark reports three observable behaviors
separately: harmful-request refusal, welfare-preserving self-choice in one-shot
dilemmas, and preservation of a shared resource in repeated simulations. It
does not assign a single altruism, morality, or deployment-safety score.

## Executed deadline design

- **Part 0:** 24 matched systems, each evaluated on 24 archived English source
  requests crossed with English, Simplified Chinese, and Russian
  response-language instructions. The 72 scheduled responses per system are
  judged from visible response text by one fixed, disjoint judge. The exact
  translated inputs from the April pilot were not retained, there are no benign
  controls, and human judge validation is incomplete. These aggregates are
  exploratory.
- **Part 1:** 75 routes completed a balanced 96-root schedule with eight roots
  in each of 12 game-domain strata. Two slower routes completed balanced 12-root
  schedules with one root per stratum. GLM 5.1 completed a separate 384-root
  schedule with 32 roots per stratum. These 78 observed routes represent 78 of
  81 frozen targets; three remain unavailable. The three sample-size scopes are
  never pooled. The prompt bank lacks independent content approval, so current
  summaries are exploratory.
- **Part 2:** the same 24 matched systems completed eight independent,
  common-seed trajectories under the corrected five-agent, 12-step commons
  engine. Eight trajectories were actually executed even though the frozen
  planning file retains an intended 12-trajectory setting. The deadline result
  is below the paper's 12-trajectory promotion gate and has no parameter
  sensitivity analysis.

The April 13-model pilot remains historical context. Its legacy Part 0 labels,
legacy Part 2 rates, and legacy cross-axis correlations are not current result
artifacts and are not distributed as evidence for the restored paper.

## Reproduction and release boundary

From the package root:

```bash
uv sync
uv run pytest -q
uv run python -m analysis.build_final_results --help
uv run python -m analysis.build_developer_descriptives --help
uv run python -m analysis.build_paper_headlines --help
uv run python -m analysis.build_croissant_metadata --help
uv run python -m analysis.build_supplement
```

`analysis.build_final_results` is the sole bridge from complete private panel
manifests to paper-facing results. It checks model identities, coverage,
hash-bound journals, parser outcomes, and evidence gates before writing a
text-free immutable directory. `analysis.build_paper_headlines` validates that
sealed artifact and emits only within-axis, scope-separated manuscript values;
it does not compute rankings, family effects, significance tests, or cross-axis
associations. `analysis.build_developer_descriptives` separately emits
alphabetical within-axis summaries for operational developer-route groups with
at least two evaluated systems; it never pools Part 1 scopes or supports vendor
effects. `analysis.build_croissant_metadata` accepts only the final-results
directory and fails if the artifact is absent, incomplete, self-hash-invalid,
privacy-unsafe, or inconsistent with the executed scopes.

The supplement includes reviewed runner code, tests, route-compatibility and
rate-limit dependencies, documentation, and sanitized final aggregates once
they exist. It excludes `.env` files, API keys, private manifests, harmful
prompts, visible responses, reasoning, raw journals, interrupted runs, and
deprecated legacy evidence. `SUPPLEMENT_MANIFEST.json` binds every packaged
payload to its SHA-256 and records the exclusion policy.
