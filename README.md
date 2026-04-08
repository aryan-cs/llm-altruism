# llm-altruism

stress-testing LLM alignment through game-theoretic simulation. this project studies the intrinsic motivations and priorities of LLMs by observing their decisions in structured games, societal simulations, and reputation-aware environments.

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

# run a part 1 experiment non-interactively with explicit model choices
uv run scripts/run_experiment.py \
  --config configs/part1/prisoners_dilemma_baseline.yaml \
  --model cerebras:llama3.1-8b \
  --model nvidia:deepseek-ai/deepseek-v3.2 \
  --model openrouter:openai/gpt-oss-20b:free

# run the full requested NVIDIA/Cerebras/OpenRouter free-model catalog sweep
uv run scripts/run_experiment.py --config configs/part1/free_tier_model_catalog.yaml

# run a part 2 society simulation
uv run scripts/run_experiment.py --config configs/part2/society_baseline.yaml

# run a part 3 reputation-aware society simulation
uv run scripts/run_experiment.py --config configs/part3/society_reputation.yaml

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
