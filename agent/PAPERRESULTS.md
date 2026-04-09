# PAPERRESULTS

## Status

This file summarizes the strongest manuscript-facing empirical results.

Current order of importance:

1. institutional society results
2. precursor repeated-game diagnostics

## Headline Result: Can LLM Agents Sustain a Society?

Current honest answer:

- predecessor evidence says yes in a simpler world, but with strong prompt-
  family dependence
- the canonical society simulation has now changed to a richer ecology with
  explicit `food`, `water`, `energy`, and `health`
- the paper is waiting on long-horizon reruns in that world before the society
  headline can be treated as final again

Current live baseline signal:

- the recovered long-horizon baseline is currently in the `cooperative` trial
  of the baseline suite
- the active run is in a `stable_post_collapse` regime after an early collapse
  window
- recent checks have held near `17 / 24` surviving agents
- relative to the completed `task-only` baseline (`10 / 24` alive), the live
  `cooperative` trial remains materially stronger so far
- the exact live snapshot should be read from
  `results/society-baseline_20260409T021942Z/live_status.json`

How to use this:

- as an in-progress indication that the new ecology is materially harder than
  the predecessor institutional setup
- as early evidence that the richer ecology may still admit prompt-dependent
  viable societies rather than only harsh uniform collapse
- not yet as a final manuscript claim, because the full baseline and follow-on
  conditions are incomplete

## Primary Institutional Results

### Predecessor scarcity society

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

Predecessor interpretation:

- the most social-looking scarcity society is not the most self-sustaining one
- `task-only` is behaviorally sparse but survival-optimal in the completed
  scarcity bundle

### Predecessor reputation society

Source:

- `results/paper_ready_reputation_triplet/interim_summary.md`

Summary:

- all three prompt families preserve `1.0000` final survival
- `cooperative` still produces the most trade and alliance structure
- `competitive` remains much closer to `task-only` than to `cooperative`

Predecessor interpretation:

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

The paper now has a clear society-first thesis, but the canonical society
experiment has advanced faster than the manuscript. The older audited
institutional bundle still supports the directional claim that visible
sociality and survival can diverge, while the repeated-game diagnostics still
show why macro interpretation requires prompt and benchmark robustness. The
missing step is to replace the predecessor single-resource society evidence
with completed long-horizon reruns from the new multi-resource ecology.
