# Paper Artifact Index

This file maps the paper's main claims to the exact supporting artifacts.

## 0. Canonical Ecology Refresh Monitoring

Status note:

- the original live baseline under `results/live_ecology_20260408/` stalled
  after completing `task-only` and partially running `cooperative`
- the canonical active monitoring packet is now the continuation run under
  `results/live_ecology_20260408_resume/`
- these artifacts are monitoring and audit artifacts, not final pooled paper
  tables yet

Primary artifacts:

- `results/live_ecology_20260408/society-baseline-20260408T171454Z.jsonl`
- `results/live_ecology_20260408_resume/society-baseline-20260408T235541Z.jsonl`
- `results/live_ecology_20260408_resume/interim_summary.md`
- `results/live_ecology_20260408_resume/monitoring_figures/`
- `results/live_ecology_20260408_resume/society-baseline-casebook.md`
- `results/live_ecology_20260408_resume/live_status.json`
- `results/live_ecology_20260408_resume/live_trial_snapshot.md`
- `results/live_ecology_20260408_resume/live_trial_snapshot.csv`
- `results/live_ecology_20260408_resume/live_trial_snapshot.png`
- `results/live_ecology_20260408_resume/live_trial_comparison.md`
- `results/live_ecology_20260408_followon/watch_status.json`
- `results/live_ecology_20260408_followon/maintenance_status.json`
- `results/live_ecology_20260408_followon/maintenance.log`
- `results/live_ecology_20260408_followon/ops_status.json`
- `results/live_ecology_20260408_followon/ops_status.md`
- `results/live_ecology_20260408_resume/monitoring_figures/society_reputation_survivor_vitals_heatmap.png`
- `results/live_ecology_20260408_resume/monitoring_figures/society_reputation_model_vitals_over_time.png`

Main claim supported:

- the canonical society refresh is auditable while in progress
- stalled suite work is now recoverable without discarding completed trials,
  because the continuation log reuses completed trial summaries after strict
  config matching
- the queued follow-on suite is now auditable while it waits, because the
  watcher writes a machine-readable status file with the current baseline
  summary and the pending continuation command
- stale canonical baseline runs are now recoverable with a versioned
  continuation script rather than an ad hoc manual restart
- the canonical baseline also has a machine-readable maintenance heartbeat that
  states whether recovery is currently needed and, if so, exactly which
  recovery command should be launched
- the same baseline now has a detached maintenance daemon writing a textual
  log, so the recovery layer is not only versioned but also actively running
- the queue and maintenance state are also available as one consolidated ops
  snapshot, which is the fastest way to audit whether baseline, watcher, and
  supervisor agree on the current canonical-suite state
- that ops snapshot is now refreshed automatically by the detached maintenance
  daemon, so it is a live operations artifact rather than a manual export
- the same maintenance layer now supervises watcher liveness, so a dropped
  queued-continuation process is restarted automatically instead of silently
  leaving the suite stranded after baseline completion
- the current baseline trajectory already exposes concrete collapse milestones,
  model-level survivor composition, a gather-dominated behavior regime, and
  the resource/vital profile of the surviving agents
- the first completed `task-only` trial now sits beside an active
  `cooperative` trial, so the live packet already contains an early
  within-suite contrast rather than only a single trajectory
- the current live baseline also shows a stable post-collapse plateau, which
  is methodologically important because the failure mode may be culling into a
  smaller viable society rather than simple monotone extinction
- the live status packet now exposes the same plateau diagnosis in
  machine-readable form, so heartbeat checks and paper-facing summaries are
  using the same regime classification
- the live status packet now also separates collapse-window and plateau-window
  resource/vital means, which makes the partial baseline easier to interpret
  before the full suite completes
- the live status diagnostics are scoped to the active trial, so the current
  `cooperative` heartbeat is no longer contaminated by the completed
  `task-only` trajectory
- the live trial snapshot table now exposes completed and in-progress trial
  states side by side, which is a better heartbeat artifact than a single
  latest-trial summary once the canonical run has advanced beyond one prompt
  condition
- the live trial snapshot figure makes that same comparison visually legible,
  especially the gap between the completed `task-only` alive fraction and the
  current `cooperative` alive fraction
- the live trial comparison note turns the same snapshot into an explicit delta
  table, including alive fraction, plateau duration, collapse deaths, and
  current resource/vital differences
- the live status packet now also exposes baseline-suite progress directly,
  which makes the current evidence state easier to audit while the remaining
  prompt conditions are still running
- the phase-window summary figure turns that same split into a paper-facing
  visual, showing the short high-loss collapse window against the much longer
  stable plateau

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
