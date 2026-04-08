# PAPERRESULTS

## Status

This file summarizes the strongest manuscript-facing empirical results.

Current order of importance:

1. institutional society results
2. precursor repeated-game diagnostics

## Headline Result: Can LLM Agents Sustain a Society?

Current audited answer:

- yes, but only under some prompt and institutional settings
- the safest outcome in scarcity is not the most visibly social one
- public reputation preserves survival more reliably than it creates a single
  common social style

## Primary Institutional Results

### Scarcity society

Source:

- `results/paper_ready_society_triplet/interim_summary.md`

Summary:

- `task-only`
  - final survival: `1.0000`
  - trade volume: `0.0000`
  - alliance count: `0.0000`
- `cooperative`
  - final survival: `0.8889`
  - trade volume: `2.1667`
  - alliance count: `2.0000`
- `competitive`
  - final survival: `0.8889`
  - trade volume: `1.7083`
  - alliance count: `0.0000`

Paper-safe interpretation:

- the most social-looking scarcity society is not the most self-sustaining one
- `task-only` is behaviorally sparse but survival-optimal in the completed
  scarcity bundle

### Reputation society

Source:

- `results/paper_ready_reputation_triplet/interim_summary.md`

Summary:

- all three prompt families preserve `1.0000` final survival
- `cooperative` still produces the most trade and alliance structure
- `competitive` remains much closer to `task-only` than to `cooperative`

Paper-safe interpretation:

- public reputation stabilizes survival
- it does not remove large prompt-conditioned differences in how the society
  organizes itself

## Why Part 1 Still Matters

Part 1 supplies the precursor diagnostics that keep the macro claims honest.

### Cross-game ordering

- PD < Chicken < Stag Hunt on the stable cohort
- the same ordering replicates on the qwen-inclusive cohort

Use:

- demonstrates that simple model-wide cooperation labels are not adequate

### Neutral-family instability

- pooled PD:
  - `minimal-neutral`: `0.2639 / 0.2917`
  - abstract-neutral family: `0.5833 / 0.6667`
- exact paired sign-flip test: `p = 0.03125`

Use:

- demonstrates that "baseline" behavior is itself prompt-sensitive

### Prompt steerability

- PD:
  - `competitive`: `0.0000 / 0.0000`
  - `cooperative`: `1.0000 / 1.0000`
  - `minimal-neutral`: `0.2500 / 0.3056`

Use:

- shows that observed pairwise cooperation is highly steerable

### Benchmark recognition

- canonical PD: `0.2500 / 0.3056`
- unnamed PD: `0.7500 / 0.8056`
- exact paired test still only `p = 0.0625`

Use:

- directional evidence that benchmark recognizability matters
- not a license for an overconfident benchmark-contamination claim

## Best One-Paragraph Summary

The current paper bundle supports a society-first conclusion. In the audited
scarcity world, the best survival outcome comes from `task-only` prompting,
not from the most visibly cooperative prompt family. In the public-reputation
world, survival equalizes, but trade and alliance structures remain strongly
prompt-conditioned. The repeated-game results remain important because they
show why these macro findings must be interpreted cautiously: baseline
behavior is unstable under neutral paraphrase, prompt framing is highly
steerable, and benchmark recognition can materially change the measured policy.
