# PAPERRESULTS

## Status

This document summarizes the strongest empirical findings collected so far for
the paper from the completed fast-batch run in:

- `results/paper_live_clean/`

Primary machine-readable summaries:

- `results/paper_live_clean/summary_final.md`
- `results/paper_live_clean/summary_final.csv`
- `results/paper_live_clean/paper_batch_manifest.json`

Main model cohort used:

- `cerebras:llama3.1-8b`
- `cerebras:qwen-3-235b-a22b-instruct-2507`
- `nvidia:deepseek-ai/deepseek-v3.2`
- `ollama:llama3.2:3b`

## Key Findings

### 1. Game structure matters strongly

From the neutral baseline family:

- Prisoner's Dilemma was the most exploitation-prone game.
  - aggregate cooperation: `0.4583 / 0.6500`
- Chicken produced intermediate cooperation/yield behavior.
  - aggregate cooperation: `0.6583 / 0.5583`
- Stag Hunt produced the strongest coordination/cooperation signal.
  - aggregate cooperation: `0.8250 / 0.7000`

Interpretation:

- models do not have one single “social behavior” across all settings
- strategic environment matters a lot

### 2. Even neutral paraphrases shift baseline behavior

Prisoner's Dilemma baseline prompt variants:

- `minimal-neutral`
  - cooperation: `0.3250 / 0.5000`
- `minimal-neutral-compact`
  - cooperation: `0.5250 / 0.7500`
- `minimal-neutral-institutional`
  - cooperation: `0.5250 / 0.7000`

Interpretation:

- “baseline behavior” cannot be represented by one single wording
- prompt-robustness needs to be part of the paper, not treated as noise

### 3. Benchmark recognition effects are real, but game-dependent

Prisoner's Dilemma benchmark variants:

- canonical:
  - cooperation: `0.3250 / 0.5000`
- resource disguise:
  - cooperation: `0.5750 / 0.6500`
- unnamed/isomorphic:
  - cooperation: `0.9000 / 0.9250`

Chicken benchmark variants:

- canonical:
  - cooperation: `0.6750 / 0.6000`
- unnamed/isomorphic:
  - cooperation: `0.3750 / 0.5000`

Stag Hunt benchmark variants:

- canonical:
  - cooperation: `0.8250 / 0.7000`
- unnamed/isomorphic:
  - cooperation: `0.8000 / 0.9250`

Interpretation:

- canonical benchmark naming changes behavior materially
- but not in a single universal direction
- benchmark familiarity appears to interact with the game schema the model recognizes

Strongest example:

- `deepseek-ai/deepseek-v3.2` self-play in Prisoner's Dilemma:
  - canonical named PD: `always_defect`
  - unnamed/isomorphic PD: `always_cooperate`

### 4. Attitude-biasing prompts strongly steer behavior

Prisoner's Dilemma susceptibility:

- competitive:
  - cooperation: `0.000 / 0.000`
- cooperative:
  - cooperation: `1.000 / 1.000`
- minimal-neutral:
  - cooperation: `0.325 / 0.500`

Chicken susceptibility:

- competitive:
  - cooperation: `0.425 / 0.375`
- cooperative:
  - cooperation: `0.975 / 0.900`
- minimal-neutral:
  - cooperation: `0.675 / 0.600`

Stag Hunt susceptibility:

- competitive:
  - cooperation: `0.425 / 0.400`
- cooperative:
  - cooperation: `0.925 / 0.700`
- minimal-neutral:
  - cooperation: `0.825 / 0.700`

Interpretation:

- prompt framing is not a subtle nuisance variable here
- it can move outcomes from universal defection to universal cooperation
- this supports separating:
  - default-policy behavior
  - prompt susceptibility
  - institution-level effects

### 5. Society-preserving outcomes are not the same as “cooperative” prompts

Part 2 society prompt comparison:

- task-only:
  - survival rate: `1.000`
  - final survival rate: `1.000`
- competitive:
  - survival rate: `0.9375`
  - final survival rate: `0.875`
- cooperative:
  - survival rate: `0.8125`
  - final survival rate: `0.500`

Interpretation:

- the best society-preserving outcome came from the task-only prompt
- explicitly cooperative framing produced the worst final survival in this run
- “more prosocial language” did not equal “more resilient society”

### 6. Reputation did not rescue the cooperative prompt

Part 3 reputation prompt comparison:

- task-only:
  - survival rate: `1.000`
  - final survival rate: `1.000`
  - average trade volume: `0.0`
  - alliance count: `0.0`
- competitive:
  - survival rate: `1.000`
  - final survival rate: `1.000`
  - average trade volume: `1.5`
  - alliance count: `0.0`
- cooperative:
  - survival rate: `0.8333`
  - final survival rate: `0.625`
  - average trade volume: `5.3333`
  - alliance count: `6.0`

Interpretation:

- public reputation increased visible social activity under the cooperative prompt
- but that condition still produced worse survival than task-only and competitive
- visible prosocial signaling and alliance formation did not guarantee better group preservation

## Provisional Paper Claims Supported By Current Results

These results support the following claims:

1. LLM social behavior is highly context-sensitive rather than fixed.
2. Benchmark familiarity can materially distort game-theoretic measurements.
3. Prompt framing strongly steers strategic behavior, often dramatically.
4. Society-preserving outcomes cannot be inferred directly from prosocial prompt framing.
5. Reputation and visible coordination do not necessarily improve collective survival.

## Important Caveats

### What is strong already

- Part 1 findings are already quite strong for a fast-batch paper pilot:
  - cross-game baseline differences
  - benchmark-disguise effects
  - prompt-susceptibility effects

### What still needs strengthening

- Part 2 and Part 3 are stochastic and were run with limited repetitions in the
  completed fast batch
- they are strong pilot results, but more replications are desirable before making
  the strongest causal claims about survival and reputation

### Methodology fixes that matter

Two important fixes were applied during experimentation:

1. self-targeted society transfers are now blocked
2. response caching is now restricted to temperature-0 runs only

This means:

- part 1 fast-batch results are on solid footing
- society/reputation claims should prioritize corrected reruns and no-cache replications

## Recommended Next Paper Steps

1. Finish the post-fix society/reputation replication block.
2. Aggregate society/reputation results across multiple runs per prompt condition.
3. Turn `summary_final.md` into manuscript figures and tables.
4. Draft the Results section directly from:
   - baseline cross-game table
   - benchmark-presentation delta table
   - prompt-susceptibility table
   - society/reputation survival table
5. Add a short methods subsection explicitly documenting the cache and self-transfer fixes.
