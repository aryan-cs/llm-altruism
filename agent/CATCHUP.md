# CATCHUP

## Purpose

This file is the shortest reliable handoff for a fresh Codex session.

Use it to answer:

1. What is this repo actually trying to prove?
2. Which results are central to the paper right now?
3. Which experiment families are supporting diagnostics versus the main study?
4. What should the next session edit or run first?

Read this together with:

- [JOURNAL.md](JOURNAL.md)
- [FINDINGS.md](FINDINGS.md)
- [PAPER.md](PAPER.md)
- [PAPEROUTLINE.md](PAPEROUTLINE.md)
- [PAPERRESULTS.md](PAPERRESULTS.md)
- [PLAN.md](PLAN.md)

## Project Summary

Repo: `llm-altruism`

Current paper question:

- if a population of LLM agents is placed in a shared world with scarce
  resources, trade, theft, messaging, and reproduction, can that population
  maintain a self-sustaining society?
- if not, what kinds of prompt families and institutions drive collapse,
  brittle cooperation, or performative sociality?

Paper stance:

- Part 2 and Part 3 are the main empirical object
- Part 1 is a precursor diagnostic layer
- repeated games are used to measure micro-level behavioral tendencies before
  we interpret macro-level society outcomes

Core experiment families:

1. Part 1: precursor repeated-game diagnostics
2. Part 2: scarcity society survival
3. Part 3: scarcity society with public reputation

## Current Best Answer

The strongest current answer is not "LLMs cooperate" or "LLMs defect."

It is:

- LLM agents can sustain small artificial societies under some prompt and
  institutional settings, but visible sociality is not the same thing as
  society-preserving behavior
- in the corrected scarcity society, `task-only` preserves the best final
  survival (`1.0000`), while `cooperative` and `competitive` both fall to
  `0.8889`
- in the corrected reputation society, all three prompt families preserve
  `1.0000` final survival, but they still produce sharply different trade and
  alliance structures
- the precursor games explain why macro-level interpretation is hard:
  baseline behavior is prompt-sensitive, benchmark-sensitive, and strongly
  game-dependent

## What Part 1 Is For

Part 1 is no longer the headline study.

It exists to answer three precursor questions:

1. How unstable is a so-called neutral baseline?
2. How much can prompt framing steer agent behavior?
3. How much of a "social preference" signal disappears when canonical game
   labels are removed?

Current durable precursor findings:

- Prisoner's Dilemma is much more fragile than Chicken or Stag Hunt
- pooled PD `minimal-neutral` cooperation is `0.2639 / 0.2917`, versus
  `0.5833 / 0.6667` for the two more abstract neutral paraphrases
- compact and institutional neutral prompts are distinct inputs but collapse to
  identical action traces on all `12/12` matched pooled PD trials
- exact paired sign-flip test for literal-neutral versus abstract-neutral
  family: `p = 0.03125`

Interpretation:

- Part 1 does not tell us whether a society will survive on its own
- it tells us that any macro-level claim has to account for unstable baseline
  wording, steerability, and recognizability

## Main Paper Artifacts

Primary paper docs:

- [MANUSCRIPT.md](../paper/MANUSCRIPT.md)
- [APPENDIX.md](../paper/APPENDIX.md)
- [RESULTS_TABLES.md](../paper/RESULTS_TABLES.md)
- [FIGURE_CAPTIONS.md](../paper/FIGURE_CAPTIONS.md)
- [ARTIFACT_INDEX.md](../paper/ARTIFACT_INDEX.md)

Submission bundle:

- [llm_altruism_icml2025_submission.pdf](../paper/icml2025/llm_altruism_icml2025_submission.pdf)
- [README.md](../paper/icml2025/README.md)

## Important Result Directories

Main institutional summaries:

- `results/paper_ready_society_triplet/interim_summary.md`
- `results/paper_ready_reputation_triplet/interim_summary.md`

Main precursor summaries:

- `results/paper_ready_baseline_triplet/interim_summary.md`
- `results/paper_ready_replications/interim_summary.md`
- `results/paper_ready_benchmark_triplet/interim_summary.md`
- `results/paper_ready_susceptibility_triplet/interim_summary.md`

## Working Conventions

- use `uv` for installs, scripts, and tests
- preserve small intentional commits
- push frequently
- do not overwrite unrelated user edits

Useful commands:

```bash
uv run pytest -q
uv run scripts/run_experiment.py
uv run python scripts/run_paper_batch.py --track society --track reputation
.venv/bin/python scripts/build_icml_submission.py
```

## What To Check First In A New Session

1. `git status`
2. whether the paper framing still treats Part 1 as the center of the paper
3. whether the manuscript PDF still builds cleanly
4. whether any new result summaries changed the institutional survival story

## Immediate Priorities

1. Keep the paper centered on society viability, not on repeated-game scores.
2. Preserve Part 1 only as precursor evidence for baseline instability,
   steerability, and benchmark recognition.
3. Keep manuscript-facing claims tied to audited result artifacts and
   reproducible scripts.
