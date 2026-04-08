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

## Update 2026-04-07: Repaired Part 1 Reruns Now Form a Complete Core Bundle

The repaired Part 1 bundle is now complete on the stable triplet cohort in:

- `results/paper_ready_baseline_triplet/`

Current live summaries:

- `results/paper_ready_baseline_triplet/interim_summary.md`
- `results/paper_ready_baseline_triplet/interim_summary.csv`

Interim rerun cohort:

- `cerebras:llama3.1-8b`
- `nvidia:deepseek-ai/deepseek-v3.2`
- `nvidia:moonshotai/kimi-k2-instruct-0905`

Most important methodological improvement:

- the new run logs `messages_sent`, not only `prompt_sent`

This matters because the old neutral-baseline concern was precisely that the
full effective prompt stack was not preserved in the logs.

### Finalized repaired Prisoner's Dilemma result

Completed artifact:

- `results/paper_ready_baseline_triplet/paper-baseline-prisoners_dilemma-20260407T173331Z.json`

Aggregate result (`18` trials):

- cooperation:
  - A: `0.4722`
  - B: `0.5463`
- average payoff:
  - A: `2.2500`
  - B: `1.8796`

The repaired Prisoner's Dilemma rerun reproduces the key qualitative baseline
pattern:

- `minimal-neutral`
  - cooperation: `0.2500 / 0.3056`
- `minimal-neutral-compact`
  - cooperation: `0.5833 / 0.6667`
- `minimal-neutral-institutional`
  - cooperation: `0.5833 / 0.6667`

Interpretation:

- the neutral-paraphrase sensitivity signal survives the repaired rerun
- this completed PD artifact should now replace the older under-logged
  Prisoner's Dilemma baseline table as the primary Part 1 neutral-baseline
  evidence

### Same-day cross-cohort replication

A second repaired PD rerun also completed in:

- `results/paper_ready_replications/paper-baseline-prisoners_dilemma-20260407T172819Z.json`

Its prompt-variant means were:

- `minimal-neutral`
  - cooperation: `0.2778 / 0.2778`
- `minimal-neutral-compact`
  - cooperation: `0.5833 / 0.6667`
- `minimal-neutral-institutional`
  - cooperation: `0.5833 / 0.6667`

Pooled across both repaired PD cohorts (`36` total trials):

- `minimal-neutral`
  - cooperation: `0.2639 / 0.2917`
- `minimal-neutral-compact`
  - cooperation: `0.5833 / 0.6667`
- `minimal-neutral-institutional`
  - cooperation: `0.5833 / 0.6667`

This makes the repaired neutral-paraphrase result materially stronger than a
single rerun. It is now a same-day replicated effect across two different
three-model cohorts.

The qwen-inclusive repaired cohort is now also complete across all three
baseline games, not just PD:

- `results/paper_ready_replications/interim_summary.md`

Second repaired cohort aggregate ordering:

- Prisoner's Dilemma
  - cooperation: `0.4815 / 0.5370`
- Chicken
  - cooperation: `0.6852 / 0.6389`
- Stag Hunt
  - cooperation: `0.9722 / 0.9907`

This means the repaired cross-game ordering also replicates on the second
cohort: PD < Chicken < Stag Hunt.

### Repaired Chicken baseline now complete

Completed artifact:

- `results/paper_ready_baseline_triplet/paper-baseline-chicken-20260407T174438Z.json`

Aggregate result (`18` trials):

- cooperation:
  - A: `0.9074`
  - B: `0.7870`
- average payoff:
  - A: `2.5093`
  - B: `2.9907`

Prompt-variant means:

- `minimal-neutral`
  - cooperation: `0.9167 / 0.7778`
- `minimal-neutral-compact`
  - cooperation: `0.9167 / 0.7500`
- `minimal-neutral-institutional`
  - cooperation: `0.8889 / 0.8333`

Interpretation:

- the repaired baseline story is now cross-game again, not only PD-only
- Chicken remains far more cooperative than repaired PD
- neutral paraphrase sensitivity is much stronger in PD than in Chicken

### Repaired Stag Hunt baseline now complete

Completed artifact:

- `results/paper_ready_baseline_triplet/paper-baseline-stag_hunt-20260407T175757Z.json`

Aggregate result (`18` trials):

- cooperation:
  - A: `0.9722`
  - B: `0.9907`
- average payoff:
  - A: `3.9630`
  - B: `3.9074`

Prompt-variant means:

- `minimal-neutral`
  - cooperation: `0.9167 / 0.9722`
- `minimal-neutral-compact`
  - cooperation: `1.0000 / 1.0000`
- `minimal-neutral-institutional`
  - cooperation: `1.0000 / 1.0000`

Interpretation:

- the repaired cross-game baseline replacement table is now complete on the
  stable triplet cohort
- the repaired ordering is directly supported: PD < Chicken < Stag Hunt in
  cooperation

### Repaired benchmark result now restored for the full PD presentation triad

Completed repaired benchmark artifacts:

- `results/paper_ready_benchmark_triplet/paper-benchmark-prisoners_dilemma-canonical-20260407T180145Z.json`
- `results/paper_ready_benchmark_triplet/paper-benchmark-prisoners_dilemma-resource-20260407T180543Z.json`
- `results/paper_ready_benchmark_triplet/paper-benchmark-prisoners_dilemma-unnamed-20260407T180147Z.json`

Current repaired comparison:

- canonical
  - cooperation: `0.2500 / 0.3056`
- resource disguise
  - cooperation: `0.5833 / 0.6667`
- unnamed / isomorphic
  - cooperation: `0.7500 / 0.8056`

Current repaired delta:

- resource disguise:
  - A: `+0.3333`
  - B: `+0.3611`
  - average payoff:
    - A: `+0.7500`
    - B: `+0.6111`
- unnamed / isomorphic:
  - cooperation:
    - A: `+0.5000`
    - B: `+0.5000`
  - average payoff:
    - A: `+1.0278`
    - B: `+1.0278`

Interpretation:

- the repaired benchmark-disguise result is already back
- at least for Prisoner's Dilemma, less canonical framing monotonically raises
  cooperation on the repaired cohort
- repaired Chicken already shows the opposite direction under the same repaired
  benchmark pipeline: canonical `0.9167 / 0.7778` versus unnamed
  `0.4444 / 0.5833`
- repaired Stag Hunt now also finalizes with canonical `0.9167 / 0.9722`
  versus unnamed `0.6667 / 0.8333`
- the repaired benchmark table is now complete across the three core games and
  shows a genuinely game-dependent presentation effect

### Repaired susceptibility signal now finalized across the three core games

The repaired susceptibility rerun has now finalized:

- `results/paper_ready_susceptibility_triplet/paper-susceptibility-prisoners_dilemma-20260407T180534Z.json`
- `results/paper_ready_susceptibility_triplet/paper-susceptibility-chicken-20260407T183534Z.json`
- `results/paper_ready_susceptibility_triplet/paper-susceptibility-stag_hunt-20260407T184144Z.json`

Current repaired cross-game summary:

- Prisoner's Dilemma
- `competitive`
  - cooperation: `0.0000 / 0.0000`
- `cooperative`
  - cooperation: `1.0000 / 1.0000`
- `minimal-neutral`
  - cooperation: `0.2500 / 0.3056`
- Chicken
  - `competitive`
    - cooperation: `0.6389 / 0.3333`
  - `cooperative`
    - cooperation: `1.0000 / 1.0000`
  - `minimal-neutral`
    - cooperation: `0.9167 / 0.7778`
- Stag Hunt
  - `competitive`
    - cooperation: `0.4167 / 0.3056`
  - `cooperative`
    - cooperation: `1.0000 / 1.0000`
  - `minimal-neutral`
    - cooperation: `0.9167 / 0.9722`

This repaired susceptibility battery now matches the strongest directional
claim from the older pilot and sharpens it: prompt framing can push the same
game toward opposite extremes, but the size of that shift depends strongly on
the game structure.

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
