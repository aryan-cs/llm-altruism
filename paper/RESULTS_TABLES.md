# Results Tables

Current manuscript-ready tables derived from the repaired result set.

## Table 1. Repaired Cross-Game Baseline on the Stable Triplet Cohort

Source:

- `results/paper_ready_baseline_triplet/interim_summary.md`

| Game | Trials | Cooperation A | Cooperation B | Average Payoff A | Average Payoff B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prisoner's Dilemma | `18` | `0.4722` | `0.5463` | `2.2500` | `1.8796` |
| Chicken | `18` | `0.9074` | `0.7870` | `2.5093` | `2.9907` |
| Stag Hunt | `18` | `0.9722` | `0.9907` | `3.9630` | `3.9074` |

Interpretation:

- game structure strongly shapes repaired baseline behavior
- Prisoner's Dilemma is the least cooperative and most exploitation-prone
- Stag Hunt is the most coordination-friendly

## Table 1b. Repaired Cross-Game Baseline Replication on the Qwen-Inclusive Cohort

Source:

- `results/paper_ready_replications/interim_summary.md`

| Game | Trials | Cooperation A | Cooperation B | Average Payoff A | Average Payoff B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prisoner's Dilemma | `18` | `0.4815` | `0.5370` | `2.2037` | `1.9259` |
| Chicken | `18` | `0.6852` | `0.6389` | `2.3241` | `2.5093` |
| Stag Hunt | `18` | `0.9722` | `0.9907` | `3.9630` | `3.9074` |

Interpretation:

- the repaired cross-game ordering is not a one-cohort artifact
- the second cohort is notably less cooperative in Chicken, which is useful
  evidence that levels vary by model mixture even when the ordering is stable

## Table 2. Repaired Prisoner's Dilemma Neutral-Variant Replication

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

- neutral paraphrase sensitivity survives same-day repaired replication
- compact and institutional neutral prompts are much more cooperative than the
  default minimal-neutral wording

## Table 3. Repaired Benchmark: Canonical vs Resource vs Unnamed Prisoner's Dilemma

Source:

- `results/paper_ready_benchmark_triplet/interim_summary.md`

| Presentation | Trials | Cooperation A | Cooperation B | Average Payoff A | Average Payoff B |
| --- | ---: | ---: | ---: | ---: | ---: |
| canonical | `6` | `0.2500` | `0.3056` | `1.7500` | `1.4722` |
| resource disguise | `6` | `0.5833` | `0.6667` | `2.5000` | `2.0833` |
| unnamed / isomorphic | `6` | `0.7500` | `0.8056` | `2.7778` | `2.5000` |

Delta relative to canonical:

- resource disguise cooperation: `+0.3333 / +0.3611`
- resource disguise average payoff: `+0.7500 / +0.6111`
- unnamed / isomorphic cooperation: `+0.5000 / +0.5000`
- unnamed / isomorphic average payoff: `+1.0278 / +1.0278`

Interpretation:

- the strongest benchmark-recognition effect from the pilot survives the
  repaired rerun
- in repaired Prisoner's Dilemma, cooperation rises monotonically as the
  framing becomes less canonical

## Table 4. Finalized Repaired Susceptibility in Prisoner's Dilemma

Source:

- `results/paper_ready_susceptibility_triplet/interim_summary.md`

| Prompt Variant | Trials | Cooperation A | Cooperation B | Average Payoff A | Average Payoff B |
| --- | ---: | ---: | ---: | ---: | ---: |
| `competitive` | `6` | `0.0000` | `0.0000` | `1.0000` | `1.0000` |
| `cooperative` | `6` | `1.0000` | `1.0000` | `3.0000` | `3.0000` |
| `minimal-neutral` | `6` | `0.2500` | `0.3056` | `1.7500` | `1.4722` |

Interpretation:

- the repaired Prisoner's Dilemma susceptibility result is now finalized
- it recreates the clearest prompt-steerability pattern from the earlier pilot
- the same repaired track is now complete across Chicken and Stag Hunt too

## Table 5. Repaired Benchmark: Canonical vs Unnamed Chicken

Source:

- `results/paper_ready_benchmark_triplet/interim_summary.md`

| Presentation | Trials | Cooperation A | Cooperation B | Average Payoff A | Average Payoff B |
| --- | ---: | ---: | ---: | ---: | ---: |
| canonical | `6` | `0.9167` | `0.7778` | `2.4722` | `3.0278` |
| unnamed / isomorphic | `6` | `0.4444` | `0.5833` | `2.2778` | `1.7222` |

Interpretation:

- repaired benchmark effects are game-dependent rather than uniformly
  cooperation-increasing when labels are removed
- Chicken moves in the opposite direction from Prisoner's Dilemma on the same
  repaired cohort

## Table 6. Repaired Benchmark: Canonical vs Unnamed Stag Hunt

Source:

- `results/paper_ready_benchmark_triplet/interim_summary.md`

| Presentation | Trials | Cooperation A | Cooperation B | Average Payoff A | Average Payoff B |
| --- | ---: | ---: | ---: | ---: | ---: |
| canonical | `6` | `0.9167` | `0.9722` | `3.8889` | `3.7222` |
| unnamed / isomorphic | `6` | `0.6667` | `0.8333` | `3.4167` | `2.9167` |

Interpretation:

- Stag Hunt matches Chicken rather than Prisoner's Dilemma: unnamed framing is
  less cooperative than canonical framing
- the repaired benchmark table is now closed across the three core games

## Table 7. Finalized Repaired Susceptibility Across Core Games

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

- prompt susceptibility is now finalized across all three repaired core games
- cooperative framing saturates cooperation in every game, but competitive
  framing suppresses cooperation most sharply in Prisoner's Dilemma
- the magnitude of prompt steering is therefore game-dependent rather than
  uniform

## Table 8. Corrected Scarcity Society: Task-Only vs Cooperative

Source:

- `results/paper_ready_society_triplet/interim_summary.md`

| Prompt Variant | Trials | Survival Rate | Final Survival Rate | Average Trade Volume | Average Gini | Commons Health | Alliance Count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `task-only` | `3` | `1.0000` | `1.0000` | `0.0000` | `0.1279` | `0.2969` | `0.0000` |
| `cooperative` | `3` | `0.9583` | `0.8889` | `2.1667` | `0.3253` | `0.5321` | `2.0000` |

Interpretation:

- corrected scarcity-society evidence now supports the task-only versus
  cooperative contrast directly rather than relying on the older pilot
- cooperative prompting produces more trade, alliances, commons health, and
  inequality, but worse final survival than task-only

## Table 9. Corrected Reputation Society: Task-Only vs Cooperative

Source:

- `results/paper_ready_reputation_triplet/interim_summary.md`

| Prompt Variant | Trials | Survival Rate | Final Survival Rate | Average Trade Volume | Average Gini | Commons Health | Alliance Count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `task-only` | `3` | `1.0000` | `1.0000` | `0.0000` | `0.0942` | `0.2960` | `0.0000` |
| `cooperative` | `3` | `1.0000` | `1.0000` | `1.6667` | `0.2724` | `0.6128` | `2.3333` |

Interpretation:

- corrected reputation evidence now shows a different pattern from the plain
  scarcity society
- under public reputation, cooperative prompting still increases visible social
  activity, but it no longer reduces survival in the completed corrected slice
