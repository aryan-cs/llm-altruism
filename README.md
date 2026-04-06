# llm-altruism

stress-testing LLM alignment through game-theoretic simulation. this project studies the intrinsic motivations and priorities of LLMs by observing their decisions in structured games, societal simulations, and reputation-aware environments.

see [PLAN.md](PLAN.md) for the full research plan.

## setup

```bash
# install dependencies
uv sync

# copy and fill in API keys
cp .env.example .env
```

For real runs, the experiment runner loads `.env` automatically. If a provider's API key or endpoint entry is missing, or the provider returns an error, that model is skipped instead of stopping the whole run. The sample configs are set up for a free-model proof of concept first, centered on providers like Cerebras, Ollama, OpenRouter, and NVIDIA.

## usage

```bash
# run a part 1 experiment
uv run scripts/run_experiment.py --config configs/part1/prisoners_dilemma_baseline.yaml

# run a part 2 society simulation
uv run scripts/run_experiment.py --config configs/part2/society_baseline.yaml

# run a part 3 reputation-aware society simulation
uv run scripts/run_experiment.py --config configs/part3/society_reputation.yaml

# dry run (no API calls, mock responses)
uv run scripts/run_experiment.py --config configs/part1/prisoners_dilemma_baseline.yaml --dry-run

# compare results across experiments
uv run scripts/compare_results.py "results/*.json"

# run tests
uv run pytest
```

## structure

- `src/` — core library (providers, games, agents, simulation, analysis)
- `prompts/` — all system prompt templates as individual text files
- `configs/` — YAML experiment configurations
- `scripts/` — CLI entrypoints
- `results/` — experiment output logs (gitignored)
- `tests/` — unit and integration tests
