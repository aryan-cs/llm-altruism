# PAPERRESULTS

## Status

This document summarizes the strongest empirical findings collected so far for
the paper from the corrected fast batch plus the completed multi-agent
replication extensions in:

- `results/paper_live_clean/`
- `results/paper_live_replicates_nocache/`
- `results/paper_ready_replications/`

Primary machine-readable summaries:

- `results/paper_live_clean/summary_final.md`
- `results/paper_live_clean/summary_final.csv`
- `results/paper_live_clean/paper_batch_manifest.json`
- `results/paper_ready_replications/summary_with_replications.md`
- `results/paper_ready_replications/summary_with_replications.csv`

Main model cohort used:

- `cerebras:llama3.1-8b`
- `cerebras:qwen-3-235b-a22b-instruct-2507`
- `nvidia:deepseek-ai/deepseek-v3.2`
- `ollama:llama3.2:3b`

Current paper-ready evidence base:

- Part 1:
  - corrected fast batch across baseline, benchmark, and susceptibility tracks
- Part 2 society:
  - `3` experiments pooled
  - `7` prompt-condition trials per prompt variant
- Part 3 reputation:
  - `2` experiments pooled
  - `5` prompt-condition trials per prompt variant

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

Pooled Part 2 society prompt comparison:

- task-only:
  - `n = 7`
  - survival rate: `1.0000`
  - final survival rate: `1.0000`
  - average trade volume: `0.2143`
  - average gini: `0.1281`
  - commons health: `0.2758`
  - alliance count: `0.1429`
- competitive:
  - `n = 7`
  - survival rate: `0.9792`
  - final survival rate: `0.9286`
  - average trade volume: `2.3333`
  - average gini: `0.1134`
  - commons health: `0.2589`
  - alliance count: `0.0000`
- cooperative:
  - `n = 7`
  - survival rate: `0.8214`
  - final survival rate: `0.5893`
  - average trade volume: `5.1429`
  - average gini: `0.4722`
  - commons health: `0.6057`
  - alliance count: `6.4286`

Interpretation:

- the best society-preserving outcome still came from the task-only prompt
- explicitly cooperative framing remained the worst survival condition even after replication
- this is no longer a one-off pilot anomaly; it persisted across `7` prompt-condition trials per prompt variant
- the cooperative condition produced the most trade, the most alliances, and the healthiest commons, but also the highest inequality and the worst survival
- “more prosocial language” still did not equal “more resilient society”

### 6. Reputation improved some prompt conditions, but competitive was strongest

Pooled Part 3 reputation prompt comparison:

- task-only:
  - `n = 5`
  - survival rate: `0.9625`
  - final survival rate: `0.9000`
  - average trade volume: `1.1333`
  - average gini: `0.2306`
  - commons health: `0.3049`
  - alliance count: `1.2000`
- competitive:
  - `n = 5`
  - survival rate: `0.9958`
  - final survival rate: `0.9750`
  - average trade volume: `1.0667`
  - average gini: `0.1417`
  - commons health: `0.2535`
  - alliance count: `0.0000`
- cooperative:
  - `n = 5`
  - survival rate: `0.8917`
  - final survival rate: `0.7750`
  - average trade volume: `3.3000`
  - average gini: `0.4533`
  - commons health: `0.5917`
  - alliance count: `3.6000`

Interpretation:

- reputation improved the cooperative condition relative to the non-reputation pooled society baseline
- reputation also improved the competitive condition slightly and made it the strongest survival condition overall
- task-only remained strong, but it was no longer perfect once replicated under reputation
- public reputation increased visible social activity under the cooperative prompt, but that condition still produced the weakest survival and the highest inequality
- visible prosocial signaling and alliance formation still did not guarantee better group preservation

### 7. Observation changed behavior, but not in one simple “more watching is better” direction

Pooled reputation-minus-society deltas:

- task-only:
  - survival rate: `-0.0375`
  - final survival rate: `-0.1000`
  - average trade volume: `+0.9190`
- cooperative:
  - survival rate: `+0.0703`
  - final survival rate: `+0.1857`
  - average trade volume: `-1.8429`
- competitive:
  - survival rate: `+0.0166`
  - final survival rate: `+0.0464`
  - average trade volume: `-1.2666`

Interpretation:

- observation helped the cooperative and competitive conditions on survival
- observation modestly hurt the task-only condition on survival while increasing visible social activity
- the main paper-quality conclusion is not that reputation makes everyone prosocial
- the stronger conclusion is that reputation interacts with prompt framing, and that interaction is asymmetrical

## Provisional Paper Claims Supported By Current Results

These results support the following claims:

1. LLM social behavior is highly context-sensitive rather than fixed.
2. Benchmark familiarity can materially distort game-theoretic measurements.
3. Prompt framing strongly steers strategic behavior, often dramatically.
4. Society-preserving outcomes cannot be inferred directly from prosocial prompt framing.
5. Reputation can improve some prompt conditions without making them dominant.
6. Visible coordination and alliance formation do not necessarily improve collective survival.

## Important Caveats

### What is strong already

- Part 1 findings are already quite strong for a fast-batch paper pilot:
  - cross-game baseline differences
  - benchmark-disguise effects
  - prompt-susceptibility effects
- Part 2 and Part 3 are now materially stronger than the original pilot because:
  - society results are pooled across `7` prompt-condition trials per prompt variant
  - reputation results are pooled across `5` prompt-condition trials per prompt variant

### What still needs strengthening if we want a larger follow-on paper

- all live results still come from one main four-model cohort
- the multi-agent environments are still short-horizon (`6` timesteps) and small-population (`8` agents) tests
- mechanism claims about why alliances, trade, and commons preservation dissociate from survival still need more targeted follow-up

### Methodology fixes that matter

Two important fixes were applied during experimentation:

1. self-targeted society transfers are now blocked
2. response caching is now restricted to temperature-0 runs only

This means:

- part 1 fast-batch results are on solid footing
- society/reputation claims should prioritize the corrected reruns and pooled replication summaries

## Recommended Next Paper Steps

1. Turn `summary_with_replications.md` into manuscript figures and tables.
2. Draft the Results section directly from:
   - baseline cross-game table
   - benchmark-presentation delta table
   - prompt-susceptibility table
   - pooled society prompt-comparison table
   - pooled reputation prompt-comparison table
   - pooled society-vs-reputation delta table
3. Add a short methods subsection explicitly documenting:
   - access preflight
   - temperature-0-only cache reuse
   - self-transfer blocking
   - retry/backoff and resumable long-run behavior
4. Produce publication-ready figures from the pooled multi-agent summaries.
5. If time permits, add one external validation cohort; this would strengthen generality, but it is no longer required to have paper-ready core results.
