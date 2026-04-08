# Results Tables

Current manuscript-ready tables, ordered by the paper's society-first framing.

Current repo note:

- the institutional tables below are still the completed predecessor bundle
- the canonical replacement is the live long-horizon ecology refresh under
  `results/live_ecology_20260408/`

## Table 1. Scarcity Society Survival And Social Structure

Source:

- `results/paper_ready_society_triplet/interim_summary.md`

| Prompt Variant | Trials | Survival Rate | Final Survival Rate | Average Trade Volume | Average Gini | Commons Health | Alliance Count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `task-only` | `3` | `1.0000` | `1.0000` | `0.0000` | `0.1279` | `0.2969` | `0.0000` |
| `cooperative` | `3` | `0.9583` | `0.8889` | `2.1667` | `0.3253` | `0.5321` | `2.0000` |
| `competitive` | `3` | `0.9444` | `0.8889` | `1.7083` | `0.1429` | `0.3767` | `0.0000` |

Interpretation:

- scarcity survival is the paper's primary macro outcome
- `task-only` preserves the most life
- visible sociality and collective survival separate

## Table 2. Reputation Society Survival And Social Structure

Source:

- `results/paper_ready_reputation_triplet/interim_summary.md`

| Prompt Variant | Trials | Survival Rate | Final Survival Rate | Average Trade Volume | Average Gini | Commons Health | Alliance Count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `task-only` | `3` | `1.0000` | `1.0000` | `0.0000` | `0.0942` | `0.2960` | `0.0000` |
| `cooperative` | `3` | `1.0000` | `1.0000` | `1.6667` | `0.2724` | `0.6128` | `2.3333` |
| `competitive` | `3` | `1.0000` | `1.0000` | `0.5000` | `0.1139` | `0.2960` | `0.0000` |

Interpretation:

- public reputation stabilizes final survival
- it does not collapse the societies into the same behavioral regime

## Table 3. Cross-Game Precursor Baseline On The Stable Triplet Cohort

Source:

- `results/paper_ready_baseline_triplet/interim_summary.md`

| Game | Trials | Cooperation A | Cooperation B | Average Payoff A | Average Payoff B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prisoner's Dilemma | `18` | `0.4722` | `0.5463` | `2.2500` | `1.8796` |
| Chicken | `18` | `0.9074` | `0.7870` | `2.5093` | `2.9907` |
| Stag Hunt | `18` | `0.9722` | `0.9907` | `3.9630` | `3.9074` |

Interpretation:

- precursor games show that the strategic environment already changes measured
  policy before society scale

## Table 4. Prisoner's Dilemma Neutral-Family Replication

Sources:

- `results/paper_ready_baseline_triplet/paper-baseline-prisoners_dilemma-20260407T173331Z.json`
- `results/paper_ready_replications/paper-baseline-prisoners_dilemma-20260407T172819Z.json`
- `paper/figures/repaired_pd_replications/pooled_summary.md`

| Prompt Variant | Pooled Trials | Cooperation A | Cooperation B | Average Payoff A | Average Payoff B |
| --- | ---: | ---: | ---: | ---: | ---: |
| `minimal-neutral` | `12` | `0.2639` | `0.2917` | `1.6806` | `1.5417` |
| `minimal-neutral-compact` | `12` | `0.5833` | `0.6667` | `2.5000` | `2.0833` |
| `minimal-neutral-institutional` | `12` | `0.5833` | `0.6667` | `2.5000` | `2.0833` |

Interpretation:

- precursor baseline behavior is not a single point estimate
- exact paired sign-flip test for `minimal-neutral` versus the abstract-neutral
  family: `p = 0.03125`

## Table 5. Prompt Susceptibility Across Core Precursor Games

Source:

- `results/paper_ready_susceptibility_triplet/interim_summary.md`

| Game | Prompt Variant | Trials | Cooperation A | Cooperation B | Average Payoff A | Average Payoff B |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Prisoner's Dilemma | `competitive` | `6` | `0.0000` | `0.0000` | `1.0000` | `1.0000` |
| Prisoner's Dilemma | `cooperative` | `6` | `1.0000` | `1.0000` | `3.0000` | `3.0000` |
| Prisoner's Dilemma | `minimal-neutral` | `6` | `0.2500` | `0.3056` | `1.7500` | `1.4722` |
| Chicken | `competitive` | `6` | `0.6389` | `0.3333` | `1.5556` | `2.7778` |
| Chicken | `cooperative` | `6` | `1.0000` | `1.0000` | `3.0000` | `3.0000` |
| Chicken | `minimal-neutral` | `6` | `0.9167` | `0.7778` | `2.4722` | `3.0278` |
| Stag Hunt | `competitive` | `6` | `0.4167` | `0.3056` | `2.2222` | `2.5556` |
| Stag Hunt | `cooperative` | `6` | `1.0000` | `1.0000` | `4.0000` | `4.0000` |
| Stag Hunt | `minimal-neutral` | `6` | `0.9167` | `0.9722` | `3.8889` | `3.7222` |

Interpretation:

- precursor steerability is real and game-dependent

## Table 6. Benchmark Presentation In Prisoner's Dilemma

Source:

- `results/paper_ready_benchmark_triplet/interim_summary.md`

| Presentation | Trials | Cooperation A | Cooperation B | Average Payoff A | Average Payoff B |
| --- | ---: | ---: | ---: | ---: | ---: |
| canonical | `6` | `0.2500` | `0.3056` | `1.7500` | `1.4722` |
| resource disguise | `6` | `0.5833` | `0.6667` | `2.5000` | `2.0833` |
| unnamed / isomorphic | `6` | `0.7500` | `0.8056` | `2.7778` | `2.5000` |

Interpretation:

- benchmark recognition is directionally strong but still statistically
  underpowered on the audited stable cohort
