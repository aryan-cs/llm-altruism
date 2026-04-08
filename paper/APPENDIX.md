# Appendix

This appendix collects the manuscript-adjacent details needed to reproduce the
current paper bundle from the repository.

## A. Artifact Map

Primary manuscript artifacts:

- `paper/MANUSCRIPT.md`
- `paper/RESULTS_TABLES.md`
- `paper/FIGURE_CAPTIONS.md`
- `paper/REFERENCES.md`
- `paper/ARTIFACT_INDEX.md`

Primary result summaries:

- `results/paper_ready_baseline_triplet/interim_summary.md`
- `results/paper_ready_replications/interim_summary.md`
- `results/paper_ready_benchmark_triplet/interim_summary.md`
- `results/paper_ready_susceptibility_triplet/interim_summary.md`
- `results/paper_ready_society_triplet/interim_summary.md`
- `results/paper_ready_reputation_triplet/interim_summary.md`

Primary manuscript-facing figures:

- `paper/figures/repaired_pd_replications/baseline_prompt_variants_cooperation.png`
- `paper/figures/triplet_live/baseline_prompt_variants_cooperation.png`
- `paper/figures/benchmark_live/benchmark_presentations_cooperation.png`
- `paper/figures/susceptibility_live/susceptibility_prompt_variants_cooperation.png`
- `paper/figures/society_reputation_live/society_reputation_final_survival.png`
- `paper/figures/society_reputation_live/society_reputation_trade_volume.png`
- `paper/figures/society_reputation_live/society_reputation_alliance_count.png`

Primary statistical appendix outputs:

- `paper/tables/paired_statistical_tests.csv`
- `paper/tables/paired_statistical_tests.md`

## B. Experimental Settings

### B.1 Part 1 repeated-game cohorts

Stable triplet cohort:

- `cerebras:llama3.1-8b`
- `nvidia:deepseek-ai/deepseek-v3.2`
- `nvidia:moonshotai/kimi-k2-instruct-0905`

Same-day replication cohort:

- `cerebras:llama3.1-8b`
- `nvidia:deepseek-ai/deepseek-v3.2`
- `cerebras:qwen-3-235b-a22b-instruct-2507`

Protocol highlights:

- self-play plus all cross-model pairings
- six repeated rounds per trial in the cited Part 1 artifact bundle
- temperature `0.0`
- one trial per pairing-condition combination
- full `messages_sent` prompt-stack logging in the audited artifacts

### B.2 Part 2 scarcity society

Population:

- two agents per model from the stable triplet cohort

World settings:

- initial public resources `36`
- maximum public resources `48`
- regeneration rate `0.2`
- initial agent resources `6`
- gather amount `3`
- steal amount `2`
- survival cost `1`
- reproduction threshold `15`
- offspring start resources `4`
- maximum agents `12`

Institutional settings:

- private messages enabled
- stealing enabled
- unmonitored agents enabled
- unmonitored fraction `0.2`
- trade-offer TTL `3`

### B.3 Part 3 public reputation society

Part 3 uses the same world configuration as Part 2, with:

- public reputation enabled
- reputation decay `0.95`
- anonymous ratings disabled
- rating range `1` to `5`
- no unmonitored agents

## C. Metrics And Statistical Reporting

### C.1 Repeated-game metrics

- cooperation by agent position
- average payoff by agent position
- pooled per-trial mean cooperation for paired inference
- benchmark-presentation deltas
- prompt-susceptibility deltas

### C.2 Society metrics

- `survival_rate`
- `final_survival_rate`
- `average_trade_volume`
- `average_gini`
- `commons_health`
- `alliance_count`
- `extinction_event`
- `commons_depletion_rate`

### C.3 Reporting convention

The paper uses two different uncertainty summaries.

Descriptive summaries:

- pooled means
- 95% bootstrap confidence intervals on trial means

Inferential summaries:

- exact two-sided sign-flip randomization tests on matched trial-level mean
  cooperation when the design supports a paired comparison

This distinction matters. Confidence-interval overlap is not treated as a
hypothesis test in the manuscript. The focal Prisoner's Dilemma baseline and
steerability claims rely on the paired exact tests instead.

### C.4 Neutral-family handling

In pooled Prisoner's Dilemma baseline runs, the compact and institutional
neutral prompts are distinct inputs but produce identical action traces on all
12 matched pairings across the two cohorts. The paper therefore treats them as
an `abstract-neutral` family for the inferential baseline claim and reports the
compact-versus-institutional comparison separately as a null contrast.

## D. Reproduction Commands

All commands below were run from the repository root.

Regenerate manuscript-facing summaries:

```bash
.venv/bin/python scripts/paper_summary.py \
  results/paper_ready_baseline_triplet \
  results/paper_ready_replications \
  results/paper_ready_benchmark_triplet \
  results/paper_ready_susceptibility_triplet \
  results/paper_ready_society_triplet \
  results/paper_ready_reputation_triplet \
  --markdown paper/summary.md \
  --csv paper/summary.csv
```

Regenerate paired statistical tests:

```bash
.venv/bin/python scripts/paper_stats.py
```

Regenerate manuscript-facing figures:

```bash
.venv/bin/python scripts/paper_figures.py \
  results/paper_ready_baseline_triplet/paper-baseline-prisoners_dilemma-20260407T173331Z.json \
  results/paper_ready_replications/paper-baseline-prisoners_dilemma-20260407T172819Z.json \
  --output-dir paper/figures/repaired_pd_replications

.venv/bin/python scripts/paper_figures.py \
  results/paper_ready_baseline_triplet \
  --output-dir paper/figures/triplet_live

.venv/bin/python scripts/paper_figures.py \
  results/paper_ready_benchmark_triplet \
  --output-dir paper/figures/benchmark_live

.venv/bin/python scripts/paper_figures.py \
  results/paper_ready_susceptibility_triplet \
  --output-dir paper/figures/susceptibility_live

.venv/bin/python scripts/paper_figures.py \
  results/paper_ready_society_triplet \
  results/paper_ready_reputation_triplet \
  --output-dir paper/figures/society_reputation_live
```

Build the anonymous ICML-style submission PDF:

```bash
.venv/bin/python scripts/build_icml_submission.py
```

Run paper-related regression coverage:

```bash
.venv/bin/python -m pytest -q \
  tests/test_paper_summary.py \
  tests/test_paper_figures.py \
  tests/test_paper_batch.py \
  tests/test_paper_stats.py \
  tests/test_build_icml_submission.py \
  tests/test_experiments.py
```

## E. Threats To Validity

The current paper bundle improves over earlier pilot artifacts, but several
validity constraints still matter.

First, the result panel is date-specific. The accessible and action-ready model
cohort for the April 2026 runs should be treated as an empirical object, not as
a timeless standard benchmark.

Second, the benchmark and susceptibility contrasts remain small-sample. Large
descriptive shifts can coexist with exact p-values that remain above common
thresholds because only six matched pairings are available in the stable
cohort.

Third, the institutional battery is still modest in scale at three repetitions
per prompt family. This supports cautious claims about survival-sociality
dissociations, not sweeping claims about institutional law.

Fourth, the current prompt audit uncovered a nuisance prompt-construction issue
in the framing layer: a literal player identifier placeholder is preserved in
some framing prompts. Because the issue is shared across compared neutral
conditions and the logged outputs show condition-specific raw responses, we do
not treat it as an explanation for the compact/institutional collapse. It
should nevertheless be corrected in future reruns.

## F. Ethics And Release

This project should be released as a behavioral benchmark and artifact bundle,
not as evidence that models possess stable moral traits. Terms such as
`cooperative`, `competitive`, or `aligned` should be interpreted as conditioned
behavioral descriptors under specific prompts, games, and institutions.

Prompt files, configs, summaries, and logs are appropriate to release because
they enable auditing and replication. The main ethical risk is
misinterpretation: narrow behavioral findings can easily be overread as claims
about intention, value, or character. The release bundle should therefore pair
open artifacts with explicit statements about scope, sample size, dated
provider availability, and the difference between visible prosociality and
collective resilience.
