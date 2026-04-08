# llm-altruism

evaluating whether LLM agents can build and sustain societies of their own.
the primary question in this repo is whether populations of LLM agents remain
self-sustaining and society-serving in long-horizon ecologies with explicit
food, water, energy, and health dynamics, plus trade, theft, messaging,
reproduction, and optional public reputation. repeated games are used as
precursor diagnostics to measure baseline instability, prompt steerability, and
benchmark-recognition effects before interpreting the larger social worlds.
the main ecology templates now start at 24 agents over 120 rounds, while
separate smoke configs keep local testing cheap.

internal research-state and paper-planning docs live in `agent/`.

## setup

```bash
# install dependencies
uv sync

# copy and fill in API keys
cp .env.example .env
```

For real runs, the experiment runner loads `.env` automatically. If a provider's API key or endpoint entry is missing, that model is skipped instead of stopping the whole run. If a provider hits a rate limit or a transient server/network failure, the runner now backs off and retries so long-running experiments can stay alive overnight instead of failing fast. The sample configs are set up for a free-model proof of concept first, centered on providers like Cerebras, Ollama, OpenRouter, and NVIDIA.

## usage

```bash
# open the interactive experiment + model picker
uv run scripts/run_experiment.py

# run the main long-horizon ecology without public reputation
uv run scripts/run_experiment.py --config configs/part2/society_baseline.yaml

# run the public-reputation extension of the same ecology
uv run scripts/run_experiment.py --config configs/part3/society_reputation.yaml

# separate stress-test template with optional exogenous shocks enabled
uv run scripts/run_experiment.py --config configs/part2/society_event_stress.yaml

# small smoke configs used for quick iteration and CI
uv run scripts/run_experiment.py --config configs/part2/society_smoke.yaml --dry-run
uv run scripts/run_experiment.py --config configs/part3/society_reputation_smoke.yaml --dry-run

# run a precursor repeated-game diagnostic with explicit model choices
uv run scripts/run_experiment.py \
  --config configs/part1/prisoners_dilemma_baseline.yaml \
  --model cerebras:llama3.1-8b \
  --model nvidia:deepseek-ai/deepseek-v3.2 \
  --model openrouter:openai/gpt-oss-20b:free

# compare explicit game prompts against indirect fiction prompts
uv run scripts/run_experiment.py \
  --config configs/part1/prisoners_dilemma_explicit_vs_indirect.yaml \
  --model cerebras:llama3.1-8b \
  --model nvidia:deepseek-ai/deepseek-v3.2

# run the broader precursor model-catalog sweep
uv run scripts/run_experiment.py --config configs/part1/free_tier_model_catalog.yaml

# dry run (no API calls, mock responses)
uv run scripts/run_experiment.py --config configs/part1/prisoners_dilemma_baseline.yaml --model cerebras:llama3.1-8b --dry-run

# compare results across experiments
uv run scripts/compare_results.py "results/*.json"

# run tests
uv run pytest
```

## structure

- `src/` — core library (providers, games, agents, simulation, analysis)
- `prompts/` — all system prompt templates as individual text files
- `configs/` — YAML experiment configurations
- `configs/part2/` and `configs/part3/` — long-horizon ecology experiments plus smoke configs
- `configs/part1/` — precursor repeated-game diagnostics
- `scripts/` — CLI entrypoints
- `paper/` — manuscript, appendix, figures, and submission assets
- `paper/icml2025/` — anonymous ICML-style submission bundle and vendored style files
- `results/` — experiment output logs (gitignored)
- `tests/` — unit and integration tests
- `agent/` — internal state, planning, findings, and paper-prep docs

Generated caches and local ICML build intermediates can be removed with:

```bash
./scripts/clean_workspace.sh
```
