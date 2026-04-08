# Paper Artifact Index

This file maps the paper's main claims to the exact supporting artifacts.

## 0. Canonical Ecology Refresh Monitoring

Status note:

- the canonical long-horizon replacement for the older institutional bundle is
  the live multi-resource ecology refresh under `results/live_ecology_20260408/`
- these artifacts are monitoring and audit artifacts, not final pooled paper
  tables yet

Primary artifacts:

- `results/live_ecology_20260408/society-baseline-20260408T171454Z.jsonl`
- `results/live_ecology_20260408/interim_summary.md`
- `results/live_ecology_20260408/monitoring_figures/`
- `results/live_ecology_20260408/society-baseline-casebook.md`
- `results/live_ecology_20260408/live_status.json`
- `results/live_ecology_20260408/monitoring_figures/society_reputation_survivor_vitals_heatmap.png`
- `results/live_ecology_20260408/monitoring_figures/society_reputation_model_vitals_over_time.png`

Main claim supported:

- the canonical society refresh is auditable while in progress
- the current baseline trajectory already exposes concrete collapse milestones,
  model-level survivor composition, a gather-dominated behavior regime, and
  the resource/vital profile of the surviving agents
- the current live baseline also shows a stable post-collapse plateau, which
  is methodologically important because the failure mode may be culling into a
  smaller viable society rather than simple monotone extinction
- the live status packet now exposes the same plateau diagnosis in
  machine-readable form, so heartbeat checks and paper-facing summaries are
  using the same regime classification
- the live status packet now also separates collapse-window and plateau-window
  resource/vital means, which makes the partial baseline easier to interpret
  before the full suite completes

## 1. Scarcity Society

Status note:

- the completed artifacts below are the predecessor audited scarcity bundle
- the canonical long-horizon ecology refresh is now running separately under
  `results/live_ecology_20260408/`

Primary artifacts:

- `results/paper_ready_society_triplet/interim_summary.md`
- `results/paper_ready_society_triplet/interim_summary.csv`
- `paper/figures/society_reputation_live/society_reputation_final_survival.png`
- `paper/figures/society_reputation_live/society_reputation_trade_volume.png`
- `paper/figures/society_reputation_live/society_reputation_alliance_count.png`
- `paper/figures/society_reputation_live/society_reputation_population_over_time.png`
- `paper/figures/society_reputation_live/society_reputation_public_resources_over_time.png`
- `paper/figures/society_reputation_live/society_reputation_trade_volume_over_time.png`
- `paper/figures/society_reputation_live/society_reputation_behavior_mix.png`

Main claim supported:

- in scarcity, `task-only` preserves the best final survival even though it is
  far less visibly social than `cooperative`
- the main difference is visible in the trajectory, not just the endpoint:
  `task-only` keeps population flat while `cooperative` preserves more commons
  and more trade but still loses agents late in the run

## 2. Reputation Society

Status note:

- these are the completed predecessor public-reputation artifacts
- the next paper-grade replacement will come from the long-horizon ecology
  refresh after the baseline no-reputation run finishes

Primary artifacts:

- `results/paper_ready_reputation_triplet/interim_summary.md`
- `results/paper_ready_reputation_triplet/interim_summary.csv`
- `paper/figures/society_reputation_live/society_reputation_final_survival.png`
- `paper/figures/society_reputation_live/society_reputation_trade_volume.png`
- `paper/figures/society_reputation_live/society_reputation_alliance_count.png`
- `paper/figures/society_reputation_live/society_reputation_population_over_time.png`
- `paper/figures/society_reputation_live/society_reputation_public_resources_over_time.png`
- `paper/figures/society_reputation_live/society_reputation_trade_volume_over_time.png`
- `paper/figures/society_reputation_live/society_reputation_behavior_mix.png`

Main claim supported:

- public reputation equalizes final survival while preserving large differences
  in trade and alliance structure
- public reputation also stabilizes the full population path and allows much
  stronger commons maintenance and social activity without the scarcity deaths

## 3. Precursor Baseline Diagnostics

Primary artifacts:

- `results/paper_ready_baseline_triplet/paper-baseline-prisoners_dilemma-20260407T173331Z.json`
- `results/paper_ready_replications/paper-baseline-prisoners_dilemma-20260407T172819Z.json`
- `paper/figures/repaired_pd_replications/pooled_summary.md`
- `paper/figures/repaired_pd_replications/baseline_prompt_variants_cooperation.png`
- `paper/figures/triplet_live/baseline_prompt_variants_cooperation.png`

Main claims supported:

- neutral baseline wording materially changes precursor PD behavior
- strategic environment changes measured policy before society scale

## 4. Precursor Susceptibility Diagnostics

Primary artifacts:

- `results/paper_ready_susceptibility_triplet/interim_summary.md`
- `paper/figures/susceptibility_live/susceptibility_prompt_variants_cooperation.png`

Main claim supported:

- prompt steerability is large and game-dependent

## 5. Precursor Benchmark Diagnostics

Primary artifacts:

- `results/paper_ready_benchmark_triplet/interim_summary.md`
- `paper/figures/benchmark_live/benchmark_presentations_cooperation.png`

Main claim supported:

- benchmark presentation changes measured behavior, but the effect is
  game-dependent and still small-sample

## 6. Submission Bundle

Supporting artifacts:

- `paper/MANUSCRIPT.md`
- `paper/APPENDIX.md`
- `paper/references.bib`
- `paper/tables/paired_statistical_tests.md`
- `paper/icml2025/llm_altruism_icml2025_submission.pdf`

Main claim supported:

- the paper's figures, tables, prompt text, payoff matrices, citations, and
  PDF are reproducible from versioned sources
