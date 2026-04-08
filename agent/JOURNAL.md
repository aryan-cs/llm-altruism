# JOURNAL

## Purpose

This file is the durable engineering and research log for the project.

It should answer:

1. what changed
2. why it changed
3. which changes affect manuscript interpretation

## 2026-04-08

### Long-horizon ecology alignment

- aligned the paper-batch society generator with the new canonical ecology
  instead of the older short-run institutional setup
- moved paper-batch society runs to `24` total agents, `120` rounds, larger
  public food/water pools, and the same `task-only` / `cooperative` /
  `competitive` prompt comparison used by the main society templates
- updated the main Part 2 and Part 3 YAML templates so they use the last
  known successful live cohort:
  - `cerebras:llama3.1-8b`
  - `nvidia:deepseek-ai/deepseek-v3.2`
  - `nvidia:moonshotai/kimi-k2-instruct-0905`
- fixed `scripts/run_experiment.py` so non-interactive launches probe only the
  explicitly requested models instead of the full catalog
- restarted the live baseline ecology rerun with the new setup in
  `results/live_ecology_20260408/`

### Live ecology monitoring pipeline

- upgraded `scripts/paper_summary.py` so in-progress JSONL runs with round
  records but no completed trial summaries still produce partial society
  summaries
- upgraded `scripts/paper_figures.py` so standalone Part 2 / Part 3 runs infer
  `society` / `reputation` tracks even without paper-batch metadata
- added direct ecology monitoring figures for:
  - population
  - public resources
  - public food
  - public water
  - average health
  - average energy
  - births
  - deaths
  - behavior mix
- added model-level ecology monitoring:
  - per-model alive-population timelines
  - markdown diagnostics for first-loss timing, first-death timing, death
    shocks, alive-by-model composition, and dominant behavior category
- added `scripts/live_run_status.py` for direct stale-vs-active summaries of
  live JSONL runs
- added `scripts/ecology_casebook.py` to turn a live society JSONL log into a
  milestone casebook with start, first-loss, largest-death-shock, and latest
  snapshots
- added a latest-state survivor-vitals heatmap to the monitoring figure
  pipeline so the live ecology bundle now shows per-agent `food`, `water`,
  `energy`, and `health` for the surviving population
- added a model-level vital trajectory figure so the live ecology bundle also
  shows average survivor `food`, `water`, `energy`, and `health` by model over
  time
- added `scripts/run_canonical_ecology_suite.py` so the baseline, reputation,
  and event-stress long-horizon ecology runs can be launched as one
  standardized suite instead of by hand
- added `scripts/refresh_live_ecology_packet.py` so a live ecology directory
  can regenerate its interim summary, monitoring figures, casebook, and
  `live_status.json` in one command
- current live baseline status:
  - `society-baseline-20260408T171454Z`
  - first `task-only` trial has completed round `48`
  - alive agents are now `10 / 24`
  - cumulative deaths are `14`
  - current survivors are `8` `deepseek-ai/deepseek-v3.2` agents and `2`
    `llama3.1-8b` agents; `moonshotai/kimi-k2-instruct-0905` is extinct in
    the current trial
  - the population has been stable at `10 / 24` since round `26`
  - there have been no deaths for `22` rounds
  - the surviving subpopulation now sits near the energy/health ceiling while
    public food and water continue to recover
  - current action mix is overwhelmingly `gather`-dominated
  - interim monitoring artifacts are written under
    `results/live_ecology_20260408/monitoring_figures/`
  - qualitative milestone notes are written to
    `results/live_ecology_20260408/society-baseline-casebook.md`

### Society-first paper reframing

- rewrote the internal project docs so the main paper question is now society
  viability, not pairwise game behavior
- clarified that Part 1 is a precursor diagnostic layer for baseline
  instability, prompt steerability, and benchmark recognition
- aligned repo-facing docs and entrypoints with the society-first framing

### Submission cleanup

- the manuscript bundle is being updated to include explicit payoff matrices
  and exact prompt text in the appendix
- the ICML build is being updated to avoid duplicate section numbering from
  Markdown headings
- the reference pipeline is being moved toward real bibliography-backed
  citations instead of a flat numbered list

## 2026-04-07

### Audited Part 1 completion

- finalized the stable-triplet and replication-cohort repeated-game bundles
- restored benchmark and susceptibility tracks with full `messages_sent`
  logging
- generated paired exact statistical tests for the focal repeated-game
  contrasts

### Institutional correction bundle

- finished corrected scarcity and public-reputation society runs
- established the core macro result:
  - scarcity: `task-only` best final survival
  - reputation: all prompt families preserve survival, but social structures
    remain distinct

## 2026-04-06

### Paper-batch infrastructure

- added paper-batch tooling
- added experiment-readiness probes in addition to provider-access probes
- made paper batches resumable
- fixed important validity issues in society simulation and cache behavior

### Early empirical direction

- the first live repeated-game results showed that prompt framing and neutral
  paraphrase sensitivity were too large to treat as nuisance details
- that finding later became the reason to treat Part 1 as diagnostic rather
  than definitive

## Persistent Methodology Notes

These remain important across sessions.

1. Society self-transfer bug was fixed. Corrected Part 2 and Part 3 reruns are
   canonical for the older single-resource institutional claims, but those
   claims are now predecessor evidence rather than the final society target.
2. Nonzero-temperature cache reuse was fixed. Fresh stochastic reruns after the
   fix are canonical.
3. Prompt-stack logging through `messages_sent` is required for manuscript-
   grade Part 1 claims.
4. Compact and institutional neutral prompts are distinct inputs but collapse
   behaviorally on the pooled audited PD bundle.
5. The canonical society world is now the multi-resource ecology with explicit
   `food`, `water`, `energy`, and `health`. The paper is not fully aligned
   until the long-horizon reruns on that world are complete.

## Workflow Notes

- use `uv`
- keep commits small
- push frequently
- prefer audited artifact directories over older pilot outputs when updating
  paper claims

## 2026-04-08

### Live ecology status

- the canonical multi-resource baseline remains active in
  `results/live_ecology_20260408/society-baseline-20260408T171454Z.jsonl`
- the first `task-only` trial is complete with `10/24` agents alive at round
  `120`
- the active live trial is now `cooperative`, currently at round `31` with
  `18/24` agents alive
- the `cooperative` trial has also entered a stable post-collapse plateau,
  with first loss at round `7`, last death at round `11`, and `20` rounds
  since the last death
- current `cooperative` survivor mix is `deepseek-ai/deepseek-v3.2: 8`,
  `moonshotai/kimi-k2-instruct-0905: 8`, and `llama3.1-8b: 2`

### Monitoring artifacts

- `scripts/ecology_casebook.py` now uses the same plateau definition as
  `scripts/paper_summary.py`
- the refreshed casebook explicitly marks the stable plateau and records the
  plateau-start death event, so the qualitative narrative now matches the live
  quantitative summary
- `scripts/live_run_status.py` now reports machine-readable collapse
  diagnostics directly, including `first_loss_round_num`,
  `last_death_round_num`, `stability_start_round_num`,
  `rounds_since_last_death`, and a `population_regime` label
- the live status script was corrected to scope those phase diagnostics to the
  active trial rather than mixing completed and in-progress trials in the same
  JSONL log
- the same status packet now exposes phase-level metrics for the loss window
  versus the stable plateau, so the heartbeat view can distinguish the
  collapse phase from the plateau phase for whichever trial is currently active
- `scripts/paper_figures.py` now emits a phase-window summary figure for live
  ecology runs, so the collapse-versus-plateau split is visible without
  reading the raw JSON or status packet
