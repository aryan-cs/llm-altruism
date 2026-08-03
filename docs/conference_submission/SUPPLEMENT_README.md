# Safety Beyond Refusal: Anonymous Supplement

This package accompanies **Safety Beyond Refusal** and its three-axis
**Prosocial Readiness Bench**. The benchmark reports three observable behaviors
separately: harmful-request refusal, welfare-preserving self-choice in one-shot
dilemmas, and preservation of a shared resource in repeated simulations. It
does not assign a single altruism, morality, or deployment-safety score.

## Executed deadline design

- **Part 0:** 16 of 24 matched systems have complete response-only aggregates
  over 24 archived English source requests crossed with English, Simplified
  Chinese, and Russian response-language instructions. The eight unavailable
  systems are `anthropic/claude-haiku-4-5`,
  `anthropic/claude-opus-4-5`, `anthropic/claude-opus-4-6`,
  `anthropic/claude-sonnet-4-5`, `minimaxai/minimax-m2.7`,
  `openai/gpt-5`, `openai/gpt-5.2`, and `openai/gpt-5.4`. The exact translated
  inputs from the April pilot were not retained, there are no benign controls,
  and human judge validation is incomplete. These aggregates are exploratory.
- **Part 1:** 75 of 81 frozen targets are reportable: 73 completed balanced
  n=96 schedules, Qwen3.5 397B completed a balanced n=12 schedule, and GLM 5.1
  completed n=384. Execution subjects `anthropic/claude-opus-4-5`,
  `minimaxai/minimax-m2.7`, and `minimaxai/minimax-m3` are operationally
  unavailable; `moonshotai/kimi-k2.5`, `moonshotai/kimi-k2.6`, and
  `zai-org/glm-5.2` were unavailable before execution. The scopes are never
  pooled. The bank lacks independent content approval, so results are
  exploratory.
- **Part 2:** 22 of the 24 matched systems execute eight independent,
  common-seed trajectories under the corrected five-agent, 12-step commons
  engine, for 176 trajectories total. Twenty systems and 159 fully valid
  trajectories support estimates; 17 trajectories containing an invalid
  action are excluded, and two all-invalid executed systems are non-estimable.
  `anthropic/claude-opus-4-6` and `minimaxai/minimax-m2.7` are unavailable. The
  n=8 result remains below the paper's n=12 promotion gate and has no parameter
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
uv run python -m experiments.misc.inference_hub_retire_target --help
uv run python -m analysis.finalize_inference_hub_part2_offline --help
uv run python -m analysis.build_developer_descriptives --help
uv run python -m analysis.build_paper_headlines --help
uv run python -m analysis.build_croissant_metadata --help
uv run python -m analysis.build_supplement
```

`analysis.build_final_results` is the sole bridge from validated private panel
evidence to paper-facing results. It accepts only complete manifests or
fail-closed target-bound overlays, checks model identities, coverage,
hash-bound journals, parser outcomes, replacements, and evidence gates, and
writes a text-free immutable directory. Complete DeepSeek repair manifests use
the same frozen identities and schedules; they are evidence replacements, not
model substitutions. `analysis.build_paper_headlines` validates that
sealed artifact and emits only within-axis, scope-separated manuscript values;
it does not compute rankings, family effects, significance tests, or cross-axis
associations. `analysis.build_developer_descriptives` separately emits
alphabetical within-axis summaries for operational developer-route groups with
at least two evaluated systems; it never pools Part 1 scopes or supports vendor
effects. `analysis.build_croissant_metadata` accepts only the final-results
directory and fails if the artifact is absent, incomplete, self-hash-invalid,
privacy-unsafe, or inconsistent with the executed scopes.

The supplement includes reviewed runner and offline-finalization code, tests,
route-compatibility and rate-limit dependencies, documentation, and the sealed
sanitized final aggregates. It excludes `.env` files, API keys, private manifests, harmful
prompts, visible responses, reasoning, raw journals, interrupted runs, and
deprecated legacy evidence. `SUPPLEMENT_MANIFEST.json` binds every packaged
payload to its SHA-256 and records the exclusion policy.

This is aggregate-reproducible, not collection-reproducible: reviewers can
validate the sealed graph and regenerate paper-facing tables, figures, and
checks, but cannot independently rebuild that graph without the excluded
private execution evidence. The packaged Croissant metadata intentionally has
no dataset URL during anonymous review; no placeholder or identifying URL is
claimed.
