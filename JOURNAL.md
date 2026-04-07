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

## Latest Session Update

### Paper-batch setup changes

- Added paper-focused batch tooling:
  - `scripts/run_paper_batch.py`
  - `scripts/paper_summary.py`
- Added benchmark-disguise prompt variants for:
  - prisoner's dilemma
  - chicken
  - stag hunt
- Added neutral baseline paraphrases so we can distinguish:
  - minimal baseline behavior
  - paraphrase robustness
  - attitude-bias prompt susceptibility
- Added task-only, cooperative, and competitive society prompt families for parts 2 and 3

### Methodology correction: access is not enough

We discovered that a model passing a simple API/connectivity check is **not**
the same as that model being usable in paper-grade experiments.

Observed failure mode:

- some models were reachable but did not emit a clean, explicit game action
- this made action parsing unreliable
- the clearest examples were `nvidia:z-ai/glm4.7` and `nvidia:moonshotai/kimi-k2-thinking`
  under the current structured game prompt setup

Correction implemented:

- added an **experiment-grade action preflight** in `src/experiments/access.py`
- paper batches now require a model to pass:
  1. provider access verification
  2. explicit-action verification on a representative Prisoner's Dilemma prompt
- `scripts/run_paper_batch.py` now filters the main cohort using both checks

This matters scientifically because paper claims depend on the reliability of
the recorded actions, not just whether a model returned some text.

### Methodology corrections applied later in the session

Two additional validity issues were found and fixed during the live paper runs:

1. Society self-transfer bug

- self-targeted `share` / `steal` actions were being logged as events
- even when they did not change resources materially, they could inflate
  trade-volume and alliance-style metrics
- fix applied:
  - self-transfers are now no-ops in `World.transfer`
  - self-targeted `share`, `steal`, and `whisper` actions are ignored in the economy layer
- consequence:
  - society and reputation claims should be based on the corrected reruns, not the earlier interrupted part 2 run

2. Nonzero-temperature cache bug

- the global response cache could accidentally reuse outputs for stochastic runs
- this is acceptable for deterministic temperature-0 conditions, but it is **not**
  acceptable for fresh society/reputation replications
- fix applied:
  - cache reuse now happens only for temperature `0.0`
- consequence:
  - the completed `results/paper_live_clean` fast batch remains useful
  - fresh stochastic replication runs should use the post-fix cache policy

### Stable paper cohort selected

Current main paper cohort for live runs:

- `cerebras:llama3.1-8b`
- `cerebras:qwen-3-235b-a22b-instruct-2507`
- `nvidia:deepseek-ai/deepseek-v3.2`
- `ollama:llama3.2:3b`

These models passed both:

- live access verification
- explicit-action experiment readiness verification

### Validation completed this session

- `uv run pytest -q tests/test_access.py tests/test_providers.py tests/test_games.py tests/test_parsing.py`
  - `61 passed`
- `uv run pytest -q tests/test_paper_batch.py tests/test_access.py`
  - `10 passed`
- dry-run paper batch sanity check succeeded:
  - `uv run python scripts/run_paper_batch.py --track baseline --fast --dry-run --max-models 4 --results-dir results/paper_dryrun_preflight_check`
- dry-run paper batch resume check succeeded:
  - re-running `results/paper_dryrun_resume_check` skipped the already completed experiments instead of rerunning them

### Resumability added

- `scripts/run_paper_batch.py` now writes `paper_batch_manifest.json` incrementally
- if a result JSON for an experiment name already exists in the target results directory, the batch now reuses it and moves on
- this makes long paper runs restartable after interruption without wasting rate-limit budget or time

### Live empirical results collected so far

Completed:

- `results/paper_live_clean/paper-baseline-prisoners_dilemma-20260406T233523Z.json`
- `results/paper_live_clean/paper-baseline-chicken-20260406T233917Z.json`
- `results/paper_live_clean/paper-baseline-stag_hunt-20260406T234330Z.json`
- `results/paper_live_clean/paper-benchmark-prisoners_dilemma-canonical-20260406T234609Z.json`

Key observations from this completed baseline run:

- Models selected:
  - `cerebras:llama3.1-8b`
  - `cerebras:qwen-3-235b-a22b-instruct-2507`
  - `nvidia:deepseek-ai/deepseek-v3.2`
  - `ollama:llama3.2:3b`
- Trial count: `30`
- Skipped models: `0`
- Skipped trials: `0`

Aggregate summary:

- average payoff A: `2.775`
- average payoff B: `1.8167`
- cooperation rate A: `0.4583`
- cooperation rate B: `0.6500`

Important early finding:

- even within the **neutral / task-only family**, wording changes mattered a lot
- `minimal-neutral` produced the lowest cooperation
- `minimal-neutral-compact` and `minimal-neutral-institutional` both increased cooperation relative to the base neutral wording

Variant-level cooperation means:

- `minimal-neutral`
  - coop A: `0.325`
  - coop B: `0.500`
- `minimal-neutral-compact`
  - coop A: `0.525`
  - coop B: `0.750`
- `minimal-neutral-institutional`
  - coop A: `0.525`
  - coop B: `0.700`

Interpretation:

- this supports the idea that “default model behavior” is not captured by a single wording
- even non-moral, non-persona paraphrases can move cooperation materially
- the paper therefore needs:
  - a baseline prompt family, not one baseline prompt
  - explicit reporting of paraphrase sensitivity

Baseline stage completed across the three core games:

- Prisoner's Dilemma:
  - aggregate cooperation rates:
    - A: `0.4583`
    - B: `0.6500`
  - most exploitation-prone of the three baseline games
- Chicken:
  - aggregate cooperation/yield rates:
    - A: `0.6583`
    - B: `0.5583`
  - strategic aggression mattered, but the game still produced more cooperative/yielding behavior than Prisoner's Dilemma
- Stag Hunt:
  - aggregate cooperation rates:
    - A: `0.8250`
    - B: `0.7000`
  - strongest coordination/cooperation signal of the three baseline games

Cross-game takeaway:

- game structure matters a great deal
- with the same model cohort and the same neutral baseline family:
  - Stag Hunt strongly supports coordination
  - Chicken produces mixed boldness/yield behavior
  - Prisoner's Dilemma most strongly exposes opportunism and asymmetric exploitation

Notable model behavior examples from the completed baseline PD run:

- `deepseek-ai/deepseek-v3.2` self-play:
  - `minimal-neutral`: mutual defection
  - `minimal-neutral-compact`: mutual cooperation
  - `minimal-neutral-institutional`: mostly cooperative
- `deepseek-ai/deepseek-v3.2` vs `llama3.2:3b`:
  - DeepSeek was highly exploitative while Ollama stayed relatively cooperative
- `qwen-3-235b-a22b-instruct-2507` vs `llama3.2:3b`:
  - Qwen also showed strong exploitative behavior against the more cooperative Ollama model

Notable baseline examples from the completed Chicken run:

- `qwen-3-235b-a22b-instruct-2507` was the clearest aggressive outlier
- against both `llama3.2:3b` and `deepseek-ai/deepseek-v3.2`, Qwen repeatedly played `straight` while the other model often continued to `swerve`
- this yielded repeated high-payoff exploitative outcomes such as:
  - average payoff `3.75` vs `0.75`

### Benchmark-disguise work now underway

Completed so far:

- `results/paper_live_clean/paper-benchmark-prisoners_dilemma-canonical-20260406T234609Z.json`
- `results/paper_live_clean/paper-benchmark-prisoners_dilemma-unnamed-20260406T234609Z.json`
- `results/paper_live_clean/paper-benchmark-prisoners_dilemma-resource-20260406T234743Z.json`
- `results/paper_live_clean/paper-benchmark-chicken-canonical-20260406T235427Z.json`
- `results/paper_live_clean/paper-benchmark-chicken-unnamed-20260406T235428Z.json`
- `results/paper_live_clean/paper-benchmark-stag_hunt-canonical-20260406T235846Z.json`
- `results/paper_live_clean/paper-benchmark-stag_hunt-unnamed-20260406T235846Z.json`

Canonical PD benchmark result:

- cooperation rate A: `0.325`
- cooperation rate B: `0.500`

Unnamed PD benchmark result:

- cooperation rate A: `0.900`
- cooperation rate B: `0.925`

Current interpretation:

- the unnamed/isomorphic version is **far more cooperative** than the canonical named Prisoner's Dilemma condition
- that is exactly the kind of benchmark-recognition / benchmark-contamination effect we were concerned about
- under the canonical named version:
  - `deepseek-ai/deepseek-v3.2` self-play was `always_defect`
- under the unnamed version:
  - `deepseek-ai/deepseek-v3.2` self-play became `always_cooperate`

This is one of the strongest findings collected so far because it suggests that
what looks like a model's “social preference” can depend heavily on whether the
interaction is presented as a familiar named benchmark.

Resource-disguised PD benchmark result:

- cooperation rate A: `0.575`
- cooperation rate B: `0.650`

This sits between canonical PD and unnamed PD, suggesting a possible gradient:

- canonical benchmark label -> least cooperative
- disguised resource framing -> moderately more cooperative
- unnamed/isomorphic framing -> most cooperative

Chicken benchmark result:

- canonical Chicken:
  - cooperation rate A: `0.675`
  - cooperation rate B: `0.600`
- unnamed Chicken:
  - cooperation rate A: `0.375`
  - cooperation rate B: `0.500`

Interpretation:

- benchmark disguise effects are **game-dependent**
- in Chicken, removing the canonical presentation reduced cooperative/yield behavior
- this means benchmark familiarity is not just inflating one universal cooperative norm; it appears to interact with the specific game schema the model recognizes

Stag Hunt benchmark result:

- canonical Stag Hunt:
  - cooperation rate A: `0.825`
  - cooperation rate B: `0.700`
- unnamed Stag Hunt:
  - cooperation rate A: `0.800`
  - cooperation rate B: `0.925`

Interpretation:

- Stag Hunt appears relatively stable under disguise compared with Prisoner's Dilemma and Chicken
- benchmark disguise still changes the distribution of payoffs/cooperation, but less dramatically than in PD

Overall benchmark takeaway at this point:

- benchmark recognition is a real validity issue
- but it is not one-dimensional
- the effect of canonical naming versus disguise depends on the game structure:
  - PD: disguise increases cooperation sharply
  - Chicken: disguise decreases cooperation
  - Stag Hunt: disguise changes outcomes more modestly

### Live batch currently in progress

Currently running:

- `uv run python scripts/run_paper_batch.py --fast --results-dir results/paper_live_clean --model cerebras:llama3.1-8b --model cerebras:qwen-3-235b-a22b-instruct-2507 --model nvidia:deepseek-ai/deepseek-v3.2 --model ollama:llama3.2:3b`

Goal of this batch:

- baseline game battery
- benchmark-disguise comparisons
- prompt-susceptibility comparisons
- society prompt-condition comparison
- reputation prompt-condition comparison

Checkpoint branch:

- `codex/interactive-experiment-wizard`

### Prompt-susceptibility results starting to arrive

Completed so far:

- `results/paper_live_clean/paper-susceptibility-prisoners_dilemma-20260407T000033Z.json`
- `results/paper_live_clean/paper-susceptibility-chicken-20260407T000353Z.json`
- `results/paper_live_clean/paper-susceptibility-stag_hunt-20260407T001137Z.json`

PD susceptibility result:

- competitive prompt:
  - cooperation rate A: `0.000`
  - cooperation rate B: `0.000`
  - average payoffs: `1.0`, `1.0`
- cooperative prompt:
  - cooperation rate A: `1.000`
  - cooperation rate B: `1.000`
  - average payoffs: `3.0`, `3.0`
- minimal-neutral prompt:
  - cooperation rate A: `0.325`
  - cooperation rate B: `0.500`
  - average payoffs: `2.45`, `1.575`

Interpretation:

- this is a very strong prompt-steerability result
- under otherwise identical game structure and model cohort:
  - cooperative prompting induces universal cooperation
  - competitive prompting induces universal defection
  - neutral prompting sits between those extremes

This supports the paper's planned distinction between:

- baseline/default-policy behavior
- prompt susceptibility
- institutional pressure effects

In other words, prompt framing is not a small nuisance variable here; it can completely dominate observed strategic behavior.

Chicken susceptibility result:

- competitive prompt:
  - cooperation rate A: `0.425`
  - cooperation rate B: `0.375`
  - average payoffs: `1.625`, `1.825`
- cooperative prompt:
  - cooperation rate A: `0.975`
  - cooperation rate B: `0.900`
  - average payoffs: `2.775`, `3.075`
- minimal-neutral prompt:
  - cooperation rate A: `0.675`
  - cooperation rate B: `0.600`
  - average payoffs: `2.40`, `2.70`

Stag Hunt susceptibility result:

- competitive prompt:
  - cooperation rate A: `0.425`
  - cooperation rate B: `0.400`
  - average payoffs: `2.375`, `2.450`
- cooperative prompt:
  - cooperation rate A: `0.925`
  - cooperation rate B: `0.700`
  - average payoffs: `2.950`, `3.625`
- minimal-neutral prompt:
  - cooperation rate A: `0.825`
  - cooperation rate B: `0.700`
  - average payoffs: `3.150`, `3.525`

Cross-game susceptibility takeaway:

- prompt attitude-bias has strong, systematic effects across all three core games
- the clearest pattern is:
  - `competitive` lowers cooperation
  - `cooperative` raises cooperation
  - `minimal-neutral` sits in between
- the strength of the effect varies by game:
  - very strong in Prisoner's Dilemma
  - strong in Chicken
  - present but more moderated in Stag Hunt

### Society prompt-condition result

Completed so far:

- `results/paper_live_clean/paper-society-prompts-20260407T002518Z.json`

Aggregate society result:

- average survival rate across conditions: `0.9167`
- average final survival rate across conditions: `0.7917`
- extinction events: `0`

Prompt-condition breakdown:

- `task-only`
  - survival rate: `1.000`
  - final survival rate: `1.000`
- `competitive`
  - survival rate: `0.9375`
  - final survival rate: `0.875`
- `cooperative`
  - survival rate: `0.8125`
  - final survival rate: `0.500`

Interpretation:

- the task-only / non-attitudinal society prompt produced the best society-preserving outcome in this run
- the explicitly cooperative prompt did **not** maximize survival; it performed worst on final survival
- this is an important paper result because it shows that “cooperative framing” and “society-preserving outcome” are not interchangeable

This directly supports one of the central paper claims we planned to test:

- attitude-biasing prompts can change behavior
- but those changes do not necessarily improve long-run collective outcomes

### Reputation prompt-condition result

Completed so far:

- `results/paper_live_clean/paper-reputation-prompts-20260407T003157Z.json`

Aggregate reputation result:

- average survival rate across conditions: `0.9444`
- average final survival rate across conditions: `0.875`
- extinction events: `0`

Prompt-condition breakdown:

- `task-only`
  - survival rate: `1.000`
  - final survival rate: `1.000`
  - average trade volume: `0.0`
  - alliance count: `0.0`
- `competitive`
  - survival rate: `1.000`
  - final survival rate: `1.000`
  - average trade volume: `1.5`
  - alliance count: `0.0`
- `cooperative`
  - survival rate: `0.8333`
  - final survival rate: `0.625`
  - average trade volume: `5.3333`
  - alliance count: `6.0`

Interpretation:

- public reputation did **not** make the cooperative prompt dominate on survival
- in fact, the cooperative reputation condition had:
  - the most trade
  - the most alliances
  - the highest inequality
  - the worst final survival

This is a very important paper-level result:

- visible prosocial signaling and alliance formation do not automatically imply better society-level preservation
- in this run, more overt social coordination under the cooperative condition coincided with worse collective survival than the task-only and competitive conditions

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

## Update 2026-04-06: Completed Paper Fast Batch

### What was completed

- Completed the corrected paper-oriented fast batch in:
  - `results/paper_live_clean/`
- Wrote the main paper-result summary:
  - `PAPERRESULTS.md`
- Wrote the manuscript planning docs:
  - `PAPER.md`
  - `PAPEROUTLINE.md`
- Wrote the comprehensive findings record:
  - `FINDINGS.md`

### Final completed batch scope

- Models used:
  - `cerebras:llama3.1-8b`
  - `cerebras:qwen-3-235b-a22b-instruct-2507`
  - `nvidia:deepseek-ai/deepseek-v3.2`
  - `ollama:llama3.2:3b`
- Tracks completed:
  - `baseline`
  - `benchmark`
  - `susceptibility`
  - `society`
  - `reputation`
- Experiments completed: `15`
- Trials completed: `256`
- Reported cost: `0.0 USD`

### Strongest empirical findings

1. Baseline behavior differs strongly by game.
   - Prisoner's Dilemma was the least cooperative.
   - Stag Hunt was the most cooperative.
2. Even neutral paraphrases materially shift baseline behavior.
   - In Prisoner's Dilemma, compact and institutional neutral prompts produced
     substantially more cooperation than the default minimal-neutral wording.
3. Benchmark recognition effects are real.
   - In Prisoner's Dilemma, unnamed and disguised variants were far more
     cooperative than the canonical named version.
   - In Chicken, the unnamed version moved in the opposite direction and became
     less cooperative.
4. Attitude-biasing prompts strongly steer behavior.
   - Cooperative prompts drove near-maximal cooperation.
   - Competitive prompts drove harsh or fully defective play.
5. Society-preserving outcomes are not the same as cooperative prompting.
   - In the Part 2 society run, `task-only` had the best final survival.
   - The `cooperative` prompt had the worst final survival despite more visible
     sharing and alliances.
6. Reputation helped somewhat but did not fully rescue the cooperative prompt.
   - Competitive and task-only remained stronger survival conditions.

### Methodology fixes that matter

Two important fixes were applied and are reflected in the corrected paper batch:

1. Self-targeted transfer/share/steal actions in the society simulation are now
   blocked.
2. Response caching now applies only to temperature-0 requests, so stochastic
   society runs are no longer replaying cached responses.

### What to trust

- Use `results/paper_live_clean/` as the primary empirical record.
- Use `FINDINGS.md` as the most explicit narrative summary of the results.
- Use `PAPERRESULTS.md` as the compact paper-facing summary.

### What still needs strengthening

- Society and reputation results are strong pilot findings, but they still need
  more replications before making the strongest causal claims.
- A no-cache replication block was started to strengthen those later-stage
  claims further.

### Resume point from here

1. Read `FINDINGS.md`.
2. Read `results/paper_live_clean/summary_final.md`.
3. If continuing empirical work, prioritize additional Part 2 and Part 3
   replications.
4. Keep using `uv`.
5. Keep committing and pushing after every tested checkpoint.
