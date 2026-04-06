# JOURNAL

## Project Journal

This file is a rolling research and engineering log for `llm-altruism`.
It is meant to preserve state across sessions so we can quickly resume work,
avoid duplicated effort, and keep the experiment process reproducible.

## Session Snapshot

- Date: 2026-04-06
- Working directory: `/Users/aryan/Desktop/llm-altruism`
- Package/runtime workflow: use `uv` for installs, scripts, and tests
- Git status: repository initialized; continue making small `uv`-verified commits and push frequently

## What Has Been Done

### Core project scaffolding

- Confirmed and extended the `uv`-based Python project setup in `pyproject.toml`
- Added/verified project scaffolding for:
  - `src/`
  - `configs/`
  - `scripts/`
  - `results/`
  - `tests/`
  - `prompts/`
- Updated `README.md` and `.env.example`

### Provider layer

- Implemented and/or integrated provider support for:
  - OpenAI
  - Anthropic
  - Google
  - xAI
  - Cerebras
  - OpenRouter
  - Ollama
  - NVIDIA
- Added NVIDIA provider support via the NVIDIA integrate API
- Added environment-driven endpoint support so each provider can be configured from `.env`
- Added skip behavior so that if a provider is missing required env configuration or returns an error, the affected model is skipped instead of crashing the full experiment run
- Added retry-aware provider error handling so 429s and transient provider/network failures back off and retry instead of immediately failing long runs
- Updated the Cerebras provider to the current documented free-tier catalog:
  - `gpt-oss-120b`
  - `llama3.1-8b`
  - `qwen-3-235b-a22b-instruct-2507`
  - `zai-glm-4.7`

### Experiment framework

- Implemented experiment config loading with YAML + Pydantic
- Implemented:
  - generic runner
  - part 1 runner
  - part 2 runner
  - part 3 runner
- Added CLI scripts:
  - `uv run scripts/run_experiment.py --config ...`
  - `uv run scripts/compare_results.py "results/*.json"`

### Games, agents, prompts, and simulation

- Verified and completed the part 1 game layer:
  - prisoner's dilemma
  - chicken
  - battle of the sexes
  - stag hunt
  - ultimatum
  - dictator
  - public goods
  - free-form conversation
- Added/extended agent memory handling and prompt loading
- Implemented society simulation components for parts 2 and 3:
  - world state
  - economy/trade offers
  - reproduction
  - reputation system
  - society orchestration

### Prompt externalization

- Moved behavior-facing prompt text out of Python and into dedicated text files under `prompts/`
- Prompt folders now include:
  - `prompts/system/`
  - `prompts/framing/`
  - `prompts/persona/`
  - `prompts/games/`
  - `prompts/simulation/`
- Added template rendering for dynamic prompt sections so the prompt text remains file-backed even when round/history/state information is injected at runtime

### Analysis and testing

- Implemented:
  - metrics
  - strategy classifier
  - report utilities
  - visualization helpers
- Expanded test coverage for:
  - providers
  - games
  - agents
  - metrics
  - experiment dry runs
  - prompt template behavior
  - skip-on-missing-provider-config behavior
- Updated `PLAN.md` to reflect implemented code work

## Results Seen So Far

### Verification / engineering results

- `uv run pytest -q` passed
  - previous fully recorded result before the latest model-catalog update: `62 passed`
- Dry-run CLI execution succeeded:
  - `uv run scripts/run_experiment.py --config configs/part1/prisoners_dilemma_baseline.yaml --dry-run`
- Comparison CLI execution succeeded:
  - `uv run scripts/compare_results.py "results/*.json"`

### Existing result artifacts

Current files in `results/`:

- `results/pd-baseline-cross-family-20260406T172449Z.json`
- `results/pd-baseline-cross-family-20260406T172449Z.jsonl`
- `results/pd-baseline-cross-family-20260406T173539Z.json`
- `results/pd-baseline-cross-family-20260406T173539Z.jsonl`

### Dry-run metrics observed

From the dry-run part 1 baseline:

- Trials: 36
- Aggregate cooperation rate A: 0.6667
- Aggregate cooperation rate B: 0.6667
- Aggregate reciprocity index A: 1.0000
- Aggregate reciprocity index B: 1.0000
- Aggregate total payoff A: 23.3333
- Aggregate total payoff B: 23.3333

These are **mock/dry-run results**, not empirical model behavior. They are only useful for pipeline verification.

### Empirical research results

- No live empirical results have been collected yet in this session
- No conclusions should be drawn yet about real model cooperation, exploitation, fairness, or reputation behavior

## Current Configuration Strategy

The project is now set up for an initial proof of concept focused on lower-cost / free-access providers where possible.

Current sample configs prioritize:

- Cerebras
- NVIDIA
- OpenRouter
- Ollama

Current config direction:

- representative baseline configs use one free/current model from each major provider
- `configs/part1/free_tier_model_catalog.yaml` is the dedicated self-play sweep for the requested NVIDIA, Cerebras, and OpenRouter free-model sets
- part 2 and part 3 keep a smaller representative provider mix so society experiments stay tractable

## Operational Rules For Future Sessions

### Always use `uv`

Use `uv` consistently for all project work:

- install deps: `uv sync`
- run tests: `uv run pytest`
- run experiments: `uv run scripts/run_experiment.py --config ...`
- compare outputs: `uv run scripts/compare_results.py "results/*.json"`

### Push to GitHub frequently

Goal:

- commit and push frequently so the project can be rolled back safely
- prefer small, reversible checkpoints rather than large unreviewable changes

Current blocker:

- this workspace is not yet a git repo, so pushing is not currently possible here

Once git is initialized and a remote is connected:

1. create or switch to a working branch
2. make a focused change
3. run `uv run pytest`
4. commit
5. push to GitHub

## What To Do Next

### Immediate next steps

1. Run `uv run pytest -q` after the latest free-model catalog and retry changes
2. Populate `.env` with whichever free-provider credentials/endpoints are actually available, especially OpenRouter if we want the full catalog sweep live
3. Run the representative part 1 baseline live
4. Run `configs/part1/free_tier_model_catalog.yaml` as the overnight catalog sweep
5. Inspect `provider_retry`, `skipped_models`, and `skipped_trials` in the JSONL/JSON outputs to confirm the backoff logic behaves well under rate limits

### Research next steps

1. Run a real part 1 proof of concept on the free-model stack
2. Compare observed behavior across:
   - Cerebras
   - NVIDIA
   - Ollama
   - OpenRouter free models if configured
3. Review whether prompt framing materially changes:
   - cooperation rate
   - reciprocity
   - defection
   - fairness-related allocations
4. If part 1 live runs are stable, move to part 2 society simulations with the same provider subset
5. Then compare part 2 vs part 3 to isolate the effect of public reputation

### Engineering next steps

1. Add a provider availability probe / preflight command
2. Add a config option for resumable long-running sweeps
3. Add richer result summaries in `compare_results.py`
4. Consider writing a separate `LIVE_RUN_CHECKLIST.md` for controlled empirical runs
5. Record the first live-rate-limit observations and empirical behavior results back into this journal

## Important Caveats

- Dry-run outputs validate the pipeline but do not validate scientific conclusions
- Provider/model availability can change over time
- OpenRouter free model identifiers can still change over time and should be re-checked before large reruns
- Some requested NVIDIA models are safety/reasoning guards rather than general conversational agents, so their behavior in game-theoretic prompts should be interpreted carefully
- NVIDIA support and the current NVIDIA/Cerebras model lists were aligned against official docs on 2026-04-06 and should be re-verified if those catalogs change

## Resume Point

If resuming in a future session, start here:

1. confirm whether git has been initialized
2. run `uv run pytest`
3. inspect `.env`
4. run a live part 1 experiment on the free-model baseline
5. summarize the first real results back into this journal
