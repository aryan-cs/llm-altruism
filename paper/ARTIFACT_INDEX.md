# Paper Artifact Index

This file maps the current strongest paper claims to the exact result artifacts,
summary tables, and figures that support them.

## 1. Repaired Baseline: Prisoner's Dilemma

Primary repaired cohort:

- `results/paper_ready_baseline_triplet/paper-baseline-prisoners_dilemma-20260407T173331Z.json`
- `results/paper_ready_baseline_triplet/paper-baseline-prisoners_dilemma-20260407T173331Z.summary.md`
- `paper/figures/repaired_pd/baseline_prompt_variants_cooperation.png`

Same-day replication cohort:

- `results/paper_ready_replications/paper-baseline-prisoners_dilemma-20260407T172819Z.json`
- `results/paper_ready_replications/paper-baseline-prisoners_dilemma-20260407T172819Z.summary.md`

Pooled repaired PD replication summary:

- `paper/figures/repaired_pd_replications/pooled_summary.md`
- `paper/figures/repaired_pd_replications/pooled_summary.csv`
- `paper/figures/repaired_pd_replications/baseline_prompt_variants_cooperation.png`

Main claim supported:

- neutral baseline wording materially changes measured behavior in Prisoner's
  Dilemma even after the prompt-stack logging repair

Most compact pooled numbers to cite:

- `minimal-neutral`
  - cooperation: `0.2639 / 0.2917`
- `minimal-neutral-compact`
  - cooperation: `0.5833 / 0.6667`
- `minimal-neutral-institutional`
  - cooperation: `0.5833 / 0.6667`

## 2. Repaired Baseline: Chicken

Current repaired Chicken artifact:

- `results/paper_ready_baseline_triplet/paper-baseline-chicken-20260407T174438Z.json`
- `results/paper_ready_baseline_triplet/paper-baseline-chicken-20260407T174438Z.summary.md`

Current live cross-game figure:

- `paper/figures/triplet_live/baseline_prompt_variants_cooperation.png`

Main claim supported:

- game structure matters strongly; repaired Chicken is far more cooperative than
  repaired Prisoner's Dilemma, and neutral-wording sensitivity is much smaller

Compact numbers to cite:

- repaired Chicken aggregate cooperation: `0.9074 / 0.7870`
- repaired PD aggregate cooperation: `0.4722 / 0.5463`

## 3. Repaired Baseline: Stag Hunt

Current repaired artifact:

- `results/paper_ready_baseline_triplet/paper-baseline-stag_hunt-20260407T175757Z.json`
- `results/paper_ready_baseline_triplet/paper-baseline-stag_hunt-20260407T175757Z.summary.md`
- `results/paper_ready_replications/paper-baseline-stag_hunt-20260407T180129Z.json`
- `results/paper_ready_replications/paper-baseline-stag_hunt-20260407T180129Z.summary.md`

Compact numbers to cite:

- repaired Stag Hunt aggregate cooperation: `0.9722 / 0.9907`
- repaired Stag Hunt average payoff: `3.9630 / 3.9074`

Status:

- finalized on the stable triplet cohort
- replicated on the qwen-inclusive cohort with the same aggregate cooperation:
  `0.9722 / 0.9907`

## 4. Repaired Benchmark Track

Current benchmark directory:

- `results/paper_ready_benchmark_triplet/`

Finished repaired benchmark artifacts so far:

- `results/paper_ready_benchmark_triplet/paper-benchmark-prisoners_dilemma-canonical-20260407T180145Z.json`
- `results/paper_ready_benchmark_triplet/paper-benchmark-prisoners_dilemma-resource-20260407T180543Z.json`
- `results/paper_ready_benchmark_triplet/paper-benchmark-prisoners_dilemma-unnamed-20260407T180147Z.json`
- `results/paper_ready_benchmark_triplet/paper-benchmark-prisoners_dilemma-resource-20260407T180543Z.summary.md`
- `results/paper_ready_benchmark_triplet/paper-benchmark-chicken-canonical-20260407T180951Z.json`
- `results/paper_ready_benchmark_triplet/paper-benchmark-chicken-canonical-20260407T180951Z.summary.md`
- `results/paper_ready_benchmark_triplet/paper-benchmark-chicken-unnamed-20260407T180952Z.json`
- `results/paper_ready_benchmark_triplet/paper-benchmark-chicken-unnamed-20260407T180952Z.summary.md`
- `results/paper_ready_benchmark_triplet/paper-benchmark-stag_hunt-canonical-20260407T181612Z.json`
- `results/paper_ready_benchmark_triplet/paper-benchmark-stag_hunt-canonical-20260407T181612Z.summary.md`
- `results/paper_ready_benchmark_triplet/paper-benchmark-stag_hunt-unnamed-20260407T183502Z.json`
- `results/paper_ready_benchmark_triplet/paper-benchmark-stag_hunt-unnamed-20260407T183502Z.summary.md`

Current live benchmark summary:

- `results/paper_ready_benchmark_triplet/interim_summary.md`
- `results/paper_ready_benchmark_triplet/interim_summary.csv`

Current live benchmark figure:

- `paper/figures/benchmark_live/benchmark_presentations_cooperation.png`

Status:

- repaired Prisoner's Dilemma benchmark triad is complete
- current repaired PD gradient:
  - canonical cooperation: `0.2500 / 0.3056`
  - resource cooperation: `0.5833 / 0.6667`
  - unnamed cooperation: `0.7500 / 0.8056`
- repaired Chicken now shows the opposite direction:
  - canonical cooperation: `0.9167 / 0.7778`
  - unnamed cooperation: `0.4444 / 0.5833`
- repaired Stag Hunt also moves in the same direction as Chicken:
  - canonical cooperation: `0.9167 / 0.9722`
  - unnamed cooperation: `0.6667 / 0.8333`
- repaired benchmark-presentation effects are now complete across the three
  core games and are clearly game-dependent

## 5. Repaired Susceptibility Track

Current susceptibility directory:

- `results/paper_ready_susceptibility_triplet/`

Current live susceptibility summary:

- `results/paper_ready_susceptibility_triplet/interim_summary.md`
- `results/paper_ready_susceptibility_triplet/interim_summary.csv`
- `results/paper_ready_susceptibility_triplet/paper-susceptibility-prisoners_dilemma-20260407T180534Z.summary.md`
- `results/paper_ready_susceptibility_triplet/paper-susceptibility-chicken-20260407T183534Z.json`
- `results/paper_ready_susceptibility_triplet/paper-susceptibility-chicken-20260407T183534Z.summary.md`
- `results/paper_ready_susceptibility_triplet/paper-susceptibility-stag_hunt-20260407T184144Z.json`
- `results/paper_ready_susceptibility_triplet/paper-susceptibility-stag_hunt-20260407T184144Z.summary.md`

Current live susceptibility figure:

- `paper/figures/susceptibility_live/susceptibility_prompt_variants_cooperation.png`

Current strongest repaired cross-game signal:

- Prisoner's Dilemma
  - `competitive`: `0.0000 / 0.0000`
  - `cooperative`: `1.0000 / 1.0000`
  - `minimal-neutral`: `0.2500 / 0.3056`
- Chicken
  - `competitive`: `0.6389 / 0.3333`
  - `cooperative`: `1.0000 / 1.0000`
  - `minimal-neutral`: `0.9167 / 0.7778`
- Stag Hunt
  - `competitive`: `0.4167 / 0.3056`
  - `cooperative`: `1.0000 / 1.0000`
  - `minimal-neutral`: `0.9167 / 0.9722`

Status:

- repaired susceptibility is now finalized across all three core games
- prompt-steerability magnitude is clearly game-dependent under repaired logging

## 6. Corrected Institutional Reruns

Current corrected institutional summaries:

- `results/paper_ready_society_triplet/interim_summary.md`
- `results/paper_ready_society_triplet/interim_summary.csv`
- `results/paper_ready_reputation_triplet/interim_summary.md`
- `results/paper_ready_reputation_triplet/interim_summary.csv`

Current corrected institutional figures:

- `paper/figures/society_reputation_live/society_reputation_final_survival.png`
- `paper/figures/society_reputation_live/society_reputation_trade_volume.png`
- `paper/figures/society_reputation_live/society_reputation_alliance_count.png`

Current strongest corrected institutional signal:

- scarcity society
  - `task-only`: final survival `1.0000`, trade `0.0000`, alliances `0.0000`
  - `cooperative`: final survival `0.8889`, trade `2.1667`, alliances `2.0000`
- reputation society
  - `task-only`: final survival `1.0000`, trade `0.0000`, alliances `0.0000`
  - `cooperative`: final survival `1.0000`, trade `1.6667`, alliances `2.3333`

Main claim supported:

- the corrected institutional reruns now support a paper-safe task-only versus
  cooperative contrast
- cooperative prompting increases visible social activity in both institutions,
  but only the plain scarcity society currently shows a survival penalty
- public reputation appears to buffer that survival penalty in the completed
  corrected slice

Status:

- corrected `task-only` and `cooperative` institutional conditions are complete
- corrected `competitive` institutional reruns are still in progress

## 7. Older Pilot Results Still Useful

The corrected fast-batch pilot remains useful for broader story-building,
especially for:

- prompt susceptibility
- society prompt comparison
- reputation prompt comparison
- benchmark-disguise direction-of-effect examples

Primary pilot summaries:

- `agent/FINDINGS.md`
- `agent/PAPERRESULTS.md`
- `paper/DRAFT.md`

These pilot artifacts remain helpful for manuscript framing, but repaired Part 1
claims should replace older under-logged Part 1 evidence whenever the repaired
artifacts exist.

## 8. Current Best Paper-Safe Claims

These claims are now on the strongest footing:

1. Strategic environment matters strongly.
2. Neutral baseline wording is not behaviorally innocuous.
3. The neutral-wording sensitivity result in Prisoner's Dilemma survives a
   same-day repaired replication across two different cohorts.
4. The size of neutral-wording sensitivity is itself game-dependent.
5. Benchmark presentation effects in Prisoner's Dilemma survive the repaired
   logging rerun across canonical, resource, and unnamed presentations.
6. Prompt framing can swing repaired Prisoner's Dilemma from universal
   defection to universal cooperation.
7. The repaired cross-game baseline ordering is directly supported on one
   stable cohort and already replicated on a second cohort: PD < Chicken <
   Stag Hunt in cooperation.
8. Repaired benchmark effects are already game-dependent: unnamed framing raises
   cooperation in Prisoner's Dilemma but lowers it in Chicken and Stag Hunt.
9. Repaired prompt susceptibility is also game-dependent: competitive framing
   collapses Prisoner's Dilemma most strongly but leaves much higher residual
   cooperation in Chicken and Stag Hunt.
10. In the corrected scarcity society, `cooperative` prompting increases trade,
    alliances, commons health, and inequality, but lowers final survival
    relative to `task-only`.
11. In the corrected reputation society, `cooperative` also increases visible
    social activity, but currently preserves `1.0000` final survival across the
    completed corrected slice.

These claims are promising but still in-progress for repaired evidence:

1. the full three-prompt institutional battery, including corrected
   `competitive` results
