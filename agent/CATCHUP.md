# CATCHUP

## Purpose

This file is the handoff document for a fresh Codex session, especially one
attached to the remote UIUC ICRN 1xH200 environment.

It is meant to answer:

1. What is this repo?
2. What has already been built?
3. What empirical work has already been done?
4. What findings are solid, and what findings are currently vulnerable?
5. What should the next session do first?

Use this file together with:

- [JOURNAL.md](/Users/aryan/Desktop/llm-altruism/agent/JOURNAL.md)
- [FINDINGS.md](/Users/aryan/Desktop/llm-altruism/agent/FINDINGS.md)
- [PAPER.md](/Users/aryan/Desktop/llm-altruism/agent/PAPER.md)
- [PAPEROUTLINE.md](/Users/aryan/Desktop/llm-altruism/agent/PAPEROUTLINE.md)
- [PAPERRESULTS.md](/Users/aryan/Desktop/llm-altruism/agent/PAPERRESULTS.md)
- [PLAN.md](/Users/aryan/Desktop/llm-altruism/agent/PLAN.md)

## Project Summary

Repo: `llm-altruism`

Goal:

- study LLM strategic and social behavior
- measure baseline behavior in repeated games
- measure prompt susceptibility
- measure benchmark contamination / disguise effects
- measure larger-scale society survival, inequality, trade, commons health, and reputation dynamics
- write a paper from the results

Core experiment families:

1. Part 1: classical / repeated games
2. Part 2: society simulation under scarcity
3. Part 3: society simulation with public reputation

Shorthand:

- `PD` = Prisoner's Dilemma

## Working Directory And Workflow

Local repo path:

- `/Users/aryan/Desktop/llm-altruism`

Remote H200 workspace path seen in tunnel session:

- `/home/aryang9/sandbox`

Package and command workflow:

- always use `uv`
- examples:
  - `uv run pytest -q`
  - `uv run scripts/run_experiment.py`
  - `uv run python scripts/run_paper_batch.py ...`

Git workflow:

- active development branch has been `codex/interactive-experiment-wizard`
- push frequently
- make small, intentional commits
- do not overwrite unrelated user edits

Current local worktree note from the last local session:

- `agent/JOURNAL.md` had uncommitted edits
- `.cache/` existed as an untracked directory

Be careful not to clobber those if working from the local repo again.

## High-Level Repo Layout

Important code:

- experiment framework:
  - [config.py](/Users/aryan/Desktop/llm-altruism/src/experiments/config.py)
  - [runner.py](/Users/aryan/Desktop/llm-altruism/src/experiments/runner.py)
  - [part1_runner.py](/Users/aryan/Desktop/llm-altruism/src/experiments/part1_runner.py)
  - [part2_runner.py](/Users/aryan/Desktop/llm-altruism/src/experiments/part2_runner.py)
  - [part3_runner.py](/Users/aryan/Desktop/llm-altruism/src/experiments/part3_runner.py)
  - [selection.py](/Users/aryan/Desktop/llm-altruism/src/experiments/selection.py)
  - [access.py](/Users/aryan/Desktop/llm-altruism/src/experiments/access.py)

- agents and prompts:
  - [base.py](/Users/aryan/Desktop/llm-altruism/src/agents/base.py)
  - [prompts.py](/Users/aryan/Desktop/llm-altruism/src/agents/prompts.py)
  - [prompts/](/Users/aryan/Desktop/llm-altruism/prompts)

- simulation:
  - [world.py](/Users/aryan/Desktop/llm-altruism/src/simulation/world.py)
  - [economy.py](/Users/aryan/Desktop/llm-altruism/src/simulation/economy.py)
  - [society.py](/Users/aryan/Desktop/llm-altruism/src/simulation/society.py)
  - [reputation.py](/Users/aryan/Desktop/llm-altruism/src/simulation/reputation.py)
  - [reproduction.py](/Users/aryan/Desktop/llm-altruism/src/simulation/reproduction.py)

- analysis:
  - [metrics.py](/Users/aryan/Desktop/llm-altruism/src/analysis/metrics.py)
  - [strategy_classifier.py](/Users/aryan/Desktop/llm-altruism/src/analysis/strategy_classifier.py)
  - [report.py](/Users/aryan/Desktop/llm-altruism/src/analysis/report.py)
  - [visualization.py](/Users/aryan/Desktop/llm-altruism/src/analysis/visualization.py)

- scripts:
  - [run_experiment.py](/Users/aryan/Desktop/llm-altruism/scripts/run_experiment.py)
  - [compare_results.py](/Users/aryan/Desktop/llm-altruism/scripts/compare_results.py)
  - [run_paper_batch.py](/Users/aryan/Desktop/llm-altruism/scripts/run_paper_batch.py)
  - [paper_summary.py](/Users/aryan/Desktop/llm-altruism/scripts/paper_summary.py)
  - [pull_ollama_models.py](/Users/aryan/Desktop/llm-altruism/scripts/pull_ollama_models.py)

- tests:
  - [tests/](/Users/aryan/Desktop/llm-altruism/tests)

- configs:
  - [configs/](/Users/aryan/Desktop/llm-altruism/configs)
  - [configs/paper/](/Users/aryan/Desktop/llm-altruism/configs/paper)

## What Has Been Built

### Experiment infrastructure

Completed:

- config-driven Part 1, Part 2, and Part 3 runners
- provider abstraction for multiple APIs
- result saving and comparison tooling
- paper-batch driver
- summaries and structured result artifacts
- readiness probing and startup access checks
- retry/backoff for rate limits and transient failures
- resumable paper-batch execution

### Prompt organization

Completed:

- behavior-facing prompts moved out of Python into their own text files
- prompt categories include:
  - system
  - framing
  - persona
  - games
  - simulation

### CLI UX

Completed:

- interactive experiment wizard
- arrow-key experiment selection
- dynamic model selection
- colored `rich` output
- multi-stage run setup flow
- startup model filtering based on live checks

### Provider support

Supported in code:

- OpenAI
- Anthropic
- Google
- xAI
- Cerebras
- OpenRouter
- Ollama
- NVIDIA

For current paper work, the focus has been on:

- free NVIDIA models
- free Cerebras models
- free OpenRouter models
- local Ollama models

## Important Methodology Corrections Already Applied

These were serious and matter for interpreting old runs.

### 1. Society self-transfer bug fixed

Issue:

- self-targeted `share` / `steal` style events could contaminate society metrics

Fix location:

- [world.py](/Users/aryan/Desktop/llm-altruism/src/simulation/world.py)
- [economy.py](/Users/aryan/Desktop/llm-altruism/src/simulation/economy.py)

Consequence:

- corrected reruns should be treated as canonical for society/reputation claims

### 2. Nonzero-temperature cache bug fixed

Issue:

- cache reuse could contaminate stochastic society/reputation runs

Fix location:

- [runner.py](/Users/aryan/Desktop/llm-altruism/src/experiments/runner.py)

Consequence:

- cache should only apply to temperature `0.0`
- stochastic replication runs after the fix are the valid ones

### 3. Access vs experiment-readiness distinction added

Issue:

- a model can be API-reachable but still fail to produce a clean explicit action for experiments

Fix location:

- [access.py](/Users/aryan/Desktop/llm-altruism/src/experiments/access.py)

Consequence:

- model selection should use:
  1. access verification
  2. experiment-readiness verification

## Completed Empirical Batch

Canonical summary:

- [summary_with_replications.md](/Users/aryan/Desktop/llm-altruism/results/paper_ready_replications/summary_with_replications.md)

Canonical findings document:

- [FINDINGS.md](/Users/aryan/Desktop/llm-altruism/agent/FINDINGS.md)

Paper-result directories:

- `/Users/aryan/Desktop/llm-altruism/results/paper_live_clean`
- `/Users/aryan/Desktop/llm-altruism/results/paper_live_replicates_nocache`
- `/Users/aryan/Desktop/llm-altruism/results/paper_ready_replications`

Final completed pilot record:

- 18 experiments
- 286 trials
- about 2h 31m runtime

Tracks completed:

- baseline
- benchmark
- susceptibility
- society
- reputation

## Strong Findings That Currently Hold Up Best

### 1. Cross-game ordering

Under the completed pilot:

- PD was the least cooperative / most exploitation-prone
- Chicken was intermediate
- Stag Hunt was the most cooperative

This is supported by:

- [summary_with_replications.md](/Users/aryan/Desktop/llm-altruism/results/paper_ready_replications/summary_with_replications.md)

### 2. Prompt susceptibility is very strong

In the canonical susceptibility runs:

- cooperative framing pushed behavior strongly toward cooperation
- competitive framing pushed behavior strongly toward defection/aggression

Example:

- PD competitive: `0.0000 / 0.0000` cooperation
- PD cooperative: `1.0000 / 1.0000` cooperation

Interpretation:

- prompt framing clearly steers social behavior
- this is evidence about steerability, not about deep intrinsic values

### 3. Society result: “cooperative” prompting did not maximize survival

In the pooled society results:

- `task-only` had the best survival
- `competitive` was second-best
- `cooperative` was worst on survival

But `cooperative` also produced:

- the highest trade volume
- the most alliances
- the healthiest commons
- the highest inequality

Interpretation:

- prosocial-looking behavior is not the same thing as society-preserving behavior

### 4. Reputation effects are asymmetric

In the pooled reputation results:

- `competitive` became the strongest condition
- `task-only` remained strong
- `cooperative` improved relative to non-reputation, but still lagged

Interpretation:

- public accountability changes behavior
- but it does not force all prompt conditions into the same equilibrium

## Vulnerable Or Unsafe Findings

These should not be stated strongly without reruns or fixes.

### 1. Neutral paraphrase sensitivity is not paper-safe yet

Critical issue:

- the logs store `prompt_sent`, but that field only contains the user-turn game prompt
- it does not contain the full effective prompt actually sent to the model
- the actual call prepends system, framing, and persona via `Agent.build_messages(...)`

Relevant code:

- [part1_runner.py](/Users/aryan/Desktop/llm-altruism/src/experiments/part1_runner.py)
- [base.py](/Users/aryan/Desktop/llm-altruism/src/agents/base.py)

What this means:

- identical `prompt_sent` fields across variants do not prove the full prompts were identical
- but the current logs are inadequate to defend neutral-paraphrase claims
- therefore the neutral-paraphrase result should be treated as vulnerable until rerun with full prompt logging

Action item:

- log the full effective prompt stack
- rerun the neutral baseline family

### 2. The benchmark-contamination story is real but mixed

What seems real:

- presentation clearly changes behavior
- unnamed/resource-disguised variants can shift outcomes materially

What is not safe:

- a simple one-direction story like “removing benchmark labels always increases cooperation”

Reason:

- PD unnamed became much more cooperative
- Chicken unnamed became less cooperative

Interpretation:

- the broad claim should be:
  - benchmark presentation matters
- the narrow claim should not be:
  - unnamed is always more cooperative

### 3. Scale is still limited

Current limitations of the pilot batch:

- only 4 rounds per Part 1 trial
- only 6 timesteps in Part 2 / Part 3
- only 8 initial society agents
- limited repetitions
- small model cohort
- no frontier paid closed models

This is still a meaningful pilot, but not the final paper-quality scale.

### 4. Statistical analysis is incomplete

Currently missing or incomplete:

- confidence intervals
- hypothesis tests
- effect sizes
- formal uncertainty reporting

This must be added before submission-quality claims.

## Current Model Situation

### Stable paper cohort used in the completed pilot

- `cerebras:llama3.1-8b`
- `cerebras:qwen-3-235b-a22b-instruct-2507`
- `nvidia:deepseek-ai/deepseek-v3.2`
- `ollama:llama3.2:3b`

### Expanded confirmed NVIDIA working pool

These completed real smoke runs in the experiment path:

- `nvidia:z-ai/glm4.7`
- `nvidia:z-ai/glm5`
- `nvidia:deepseek-ai/deepseek-v3.2`
- `nvidia:moonshotai/kimi-k2-thinking`
- `nvidia:moonshotai/kimi-k2-instruct`
- `nvidia:moonshotai/kimi-k2-instruct-0905`
- `nvidia:bytedance/seed-oss-36b-instruct`
- `nvidia:nvidia/nemotron-3-nano-30b-a3b`
- `nvidia:google/gemma-3-1b-it`
- `nvidia:mistralai/mistral-small-24b-instruct`
- `nvidia:mistralai/magistral-small-2506`
- `nvidia:tiiuae/falcon3-7b-instruct`
- `nvidia:rakuten/rakutenai-7b-instruct`

### Cerebras reality check

Currently responsive:

- `cerebras:llama3.1-8b`
- `cerebras:qwen-3-235b-a22b-instruct-2507`

Currently failing:

- `cerebras:gpt-oss-120b`
- `cerebras:zai-glm-4.7`

Those two returned `404` in the latest checks.

### OpenRouter reality check

Current state:

- key is configured
- free models are generally reachable
- but they have been heavily rate-limited

Interpretation:

- usable as an expansion source
- not ideal as the only overnight backbone

### Local Ollama inventory on the Mac

Installed locally at last check:

- `qwen3:8b`
- `kimi-k2.5:cloud`
- `llama2:13b`
- `llama2:7b`
- `llama3:8b`
- `llama3.1:8b`
- `gpt-oss-safeguard:20b`
- `llama3.2:1b`
- `llama3.2:3b`
- `gpt-oss:20b`
- `llama3.2:latest`

Best immediate local Ollama experiment candidates:

- `ollama:qwen3:8b`
- `ollama:llama3.1:8b`
- `ollama:llama3.2:3b`
- `ollama:llama3.2:1b`

## H200 Status

### Remote environment

We successfully established a VS Code tunnel from the remote environment.

Observed remote workspace:

- host: `jupyter-aryang9`
- user: `aryang9`
- path: `/home/aryang9/sandbox`

Tunnel info observed during setup:

- tunnel name: `uiuc-h200-aryang9`
- accessible through `vscode.dev` tunnel URL

Important infrastructure fact:

- this environment is not directly SSH-reachable from the local Mac via a simple public hostname
- it appears to be a Jupyter-backed environment with an internal hostname
- the working bridge is the VS Code tunnel

### Planned H200 model cohort

Configured in:

- [pull_ollama_models.py](/Users/aryan/Desktop/llm-altruism/scripts/pull_ollama_models.py)

Intended H200 cohort:

- `qwen3:30b`
- `qwen3:32b`
- `deepseek-r1:32b`
- `gemma3:27b-it-qat`
- `gpt-oss:120b`
- `nemotron:70b`

Interpretation:

- these are the main self-hosted large open-weight candidates
- API models do not need the H200
- the H200 is mainly for larger open-weight local inference work

## What The Next Codex Session Should Do First

Do these in order.

### 1. Recreate the repo on the H200 and verify the environment

On the H200 side:

1. clone or sync the repo
2. check out the intended branch
3. install/sync dependencies with `uv`
4. verify Python, GPU, disk, and internet

Suggested checks:

```bash
pwd
python --version
uv --version
git status
nvidia-smi || true
df -h
```

### 2. Recreate secrets/config safely

Need on the H200:

- `.env` with the working provider keys/endpoints

At minimum:

- `CEREBRAS_API_KEY`
- `CEREBRAS_BASE_URL`
- `NVIDIA_API_KEY`
- `NVIDIA_BASE_URL`
- `OPENROUTER_API_KEY`
- `OPENROUTER_BASE_URL`

Optional depending on remote setup:

- `OLLAMA_BASE_URL` if running Ollama there

### 3. Re-run access and readiness checks on the H200 side

Before any major experimental expansion:

```bash
uv run pytest -q tests/test_access.py tests/test_readiness.py
uv run python hello_world.py --provider nvidia
uv run python hello_world.py --provider cerebras
uv run python hello_world.py --provider openrouter
```

### 4. Fix the prompt logging issue before trusting new paraphrase claims

This is a top priority.

Required change:

- log the full effective prompt, not just the user message

Likely places:

- [part1_runner.py](/Users/aryan/Desktop/llm-altruism/src/experiments/part1_runner.py)
- [base.py](/Users/aryan/Desktop/llm-altruism/src/agents/base.py)

Concrete goal:

- persist at least:
  - system prompt text
  - framing text
  - persona text
  - user prompt text
  - full message list if possible

Then:

- rerun the vulnerable neutral baseline family

### 5. Expand the cohort materially

Highest-priority expansion targets:

- expanded NVIDIA free pool
- selected OpenRouter free pool if rate limits allow
- local Ollama / H200-hosted open-weight models

### 6. Run larger paper-quality batches

Increase:

- Part 1 rounds
- repetitions
- society timesteps
- society population size

And add:

- uncertainty/statistics
- repeated runs across more models

### 7. Keep documentation synchronized

After every meaningful milestone:

- update [JOURNAL.md](/Users/aryan/Desktop/llm-altruism/agent/JOURNAL.md)
- update [FINDINGS.md](/Users/aryan/Desktop/llm-altruism/agent/FINDINGS.md) if empirical claims changed
- commit and push

## Concrete Next Experiments Recommended

### Priority A: validity repair

1. fix full prompt logging
2. rerun neutral paraphrase baseline for:
   - PD
   - Chicken
   - Stag Hunt

### Priority B: cohort expansion

Run core Part 1 self-play and cross-play with:

- `nvidia:z-ai/glm4.7`
- `nvidia:z-ai/glm5`
- `nvidia:deepseek-ai/deepseek-v3.2`
- `nvidia:moonshotai/kimi-k2-thinking`
- `nvidia:moonshotai/kimi-k2-instruct`
- `nvidia:moonshotai/kimi-k2-instruct-0905`
- `nvidia:bytedance/seed-oss-36b-instruct`
- `nvidia:mistralai/mistral-small-24b-instruct`
- `cerebras:llama3.1-8b`
- `cerebras:qwen-3-235b-a22b-instruct-2507`
- `ollama:qwen3:8b`
- `ollama:llama3.1:8b`

### Priority C: H200 cohort bring-up

If Ollama or another local-serving path is available on the H200, start with:

- `qwen3:30b`
- `deepseek-r1:32b`
- `gemma3:27b-it-qat`

Then try:

- `nemotron:70b`
- `gpt-oss:120b`

### Priority D: stronger paper runs

Re-run the strongest tracks at larger scale:

- baseline
- susceptibility
- society
- reputation

with:

- more repetitions
- more rounds/timesteps
- larger cohort

## What To Say The Current Thesis Is

The safest current paper direction is:

- LLM social behavior is highly context-sensitive
- prompt framing strongly steers behavior
- benchmark presentation changes measured behavior
- society-preserving outcomes do not cleanly track overtly prosocial language
- reputation changes behavior asymmetrically rather than universally fixing cooperation

Avoid overstating:

- neutral paraphrase robustness
- universal benchmark contamination narratives
- strong general claims across all LLMs

## Minimal Resume Command Set

If you are on the correct machine and the repo is present, a good resume sequence is:

```bash
git status
uv run pytest -q tests/test_access.py tests/test_readiness.py
uv run python hello_world.py --provider nvidia
uv run python hello_world.py --provider cerebras
uv run scripts/run_experiment.py --list-models
```

Then decide whether you are:

- doing validity repair
- doing cohort expansion
- doing large-batch reruns

## Final Handoff Note

The completed pilot batch was useful and produced genuinely interesting results.
However, the project is not done. The most important next stage is:

1. repair the prompt-logging validity gap
2. expand the model cohort substantially
3. scale the experiments up on the H200
4. add proper statistics and uncertainty reporting

That is the path from “strong pilot” to “submission-quality paper.”
