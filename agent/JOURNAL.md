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
