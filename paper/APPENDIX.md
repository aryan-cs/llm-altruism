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

Primary paper figures:

- `paper/figures/repaired_pd_replications/baseline_prompt_variants_cooperation.png`
- `paper/figures/triplet_live/baseline_prompt_variants_cooperation.png`
- `paper/figures/benchmark_live/benchmark_presentations_cooperation.png`
- `paper/figures/susceptibility_live/susceptibility_prompt_variants_cooperation.png`
- `paper/figures/society_reputation_live/society_reputation_final_survival.png`
- `paper/figures/society_reputation_live/society_reputation_trade_volume.png`
- `paper/figures/society_reputation_live/society_reputation_alliance_count.png`

## B. Repaired Experimental Settings

### B.1 Part 1 repeated-game cohort

Stable repaired cohort:

- `cerebras:llama3.1-8b`
- `nvidia:deepseek-ai/deepseek-v3.2`
- `nvidia:moonshotai/kimi-k2-instruct-0905`

Same-day replication cohort:

- `cerebras:llama3.1-8b`
- `nvidia:deepseek-ai/deepseek-v3.2`
- `cerebras:qwen-3-235b-a22b-instruct-2507`

Protocol highlights:

- self-play plus all cross-model pairings
- four repeated rounds per repaired fast-batch trial
- temperature `0.0`
- full `messages_sent` prompt-stack logging

### B.2 Part 2 scarcity society

Population:

- two agents per model from the stable repaired triplet cohort

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

## C. Metrics

### C.1 Repeated-game metrics

- cooperation by agent position
- average payoff by agent position
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

Paper summaries report pooled means and 95% confidence intervals where
available. Whole-directory aggregation prefers the latest completed artifact
for each logical experiment name and excludes stale interrupted JSONL retries.

## D. Reproduction Commands

All commands below were run from the repository root.

Regenerate corrected institutional summaries:

```bash
.venv/bin/python scripts/paper_summary.py results/paper_ready_society_triplet \
  --markdown results/paper_ready_society_triplet/interim_summary.md \
  --csv results/paper_ready_society_triplet/interim_summary.csv

.venv/bin/python scripts/paper_summary.py results/paper_ready_reputation_triplet \
  --markdown results/paper_ready_reputation_triplet/interim_summary.md \
  --csv results/paper_ready_reputation_triplet/interim_summary.csv
```

Regenerate manuscript-facing society/reputation figures:

```bash
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
  tests/test_experiments.py
```

## E. Threats To Validity

The most important repaired validity issue was incomplete prompt logging. Older
artifacts that logged only the final task prompt should not be treated as the
paper record when repaired artifacts exist with full message-stack logging.

Provider availability is also unstable over time. The accessible model panel is
therefore a dated empirical object, not a timeless canonical benchmark. This
paper addresses that by documenting the exact repaired cohorts rather than
claiming they are exhaustive.

Finally, the institutional battery remains modest in scale. The corrected Part
2 and Part 3 reruns are now complete across all three prompt families, but only
at three repetitions per condition. That is sufficient for a defensible
behavioral result set, but not for overconfident claims about universal
institutional laws.

## F. Ethics And Release

This project should be released as a behavioral benchmark and artifact bundle,
not as evidence that models possess stable moral traits. Terms such as
"cooperative," "competitive," or "aligned" should always be interpreted as
conditioned behavioral descriptors under specific prompts, games, and
institutions.

Prompt files, configs, and logs are appropriate to release because they enable
auditing and replication. The main ethical risk is misinterpretation:
anthropomorphic readings can make narrow behavioral findings sound like claims
about intention or character. The paper should therefore pair open artifacts
with careful framing about scope, dated provider access, and the distinction
between visible prosociality and resilient collective outcomes.
