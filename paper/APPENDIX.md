# Appendix

This appendix collects the details needed to reproduce and audit the current
paper bundle.

## Artifact Map

Primary manuscript artifacts:

- `paper/MANUSCRIPT.md`
- `paper/RESULTS_TABLES.md`
- `paper/FIGURE_CAPTIONS.md`
- `paper/ARTIFACT_INDEX.md`
- `paper/references.bib`

Completed predecessor institutional result summaries:

- `results/paper_ready_society_triplet/interim_summary.md`
- `results/paper_ready_reputation_triplet/interim_summary.md`

Current canonical ecology refresh artifacts:

- `results/live_ecology_20260408/society-baseline-20260408T171454Z.jsonl`
- `results/live_ecology_20260408/interim_summary.md`
- `results/live_ecology_20260408/monitoring_figures/`

Primary precursor result summaries:

- `results/paper_ready_baseline_triplet/interim_summary.md`
- `results/paper_ready_replications/interim_summary.md`
- `results/paper_ready_benchmark_triplet/interim_summary.md`
- `results/paper_ready_susceptibility_triplet/interim_summary.md`

Primary manuscript-facing figures:

- `paper/figures/society_reputation_live/society_reputation_final_survival.png`
- `paper/figures/society_reputation_live/society_reputation_trade_volume.png`
- `paper/figures/society_reputation_live/society_reputation_alliance_count.png`
- `paper/figures/society_reputation_live/society_reputation_population_over_time.png`
- `paper/figures/society_reputation_live/society_reputation_public_resources_over_time.png`
- `paper/figures/society_reputation_live/society_reputation_trade_volume_over_time.png`
- `paper/figures/society_reputation_live/society_reputation_behavior_mix.png`
- `paper/figures/repaired_pd_replications/baseline_prompt_variants_cooperation.png`
- `paper/figures/triplet_live/baseline_prompt_variants_cooperation.png`
- `paper/figures/benchmark_live/benchmark_presentations_cooperation.png`
- `paper/figures/susceptibility_live/susceptibility_prompt_variants_cooperation.png`

Primary statistical outputs:

- `paper/tables/paired_statistical_tests.csv`
- `paper/tables/paired_statistical_tests.md`

## Experimental Settings

### Repeated-game precursor cohorts

Stable triplet cohort:

- `cerebras:llama3.1-8b`
- `nvidia:deepseek-ai/deepseek-v3.2`
- `nvidia:moonshotai/kimi-k2-instruct-0905`

Same-day replication cohort:

- `cerebras:llama3.1-8b`
- `nvidia:deepseek-ai/deepseek-v3.2`
- `cerebras:qwen-3-235b-a22b-instruct-2507`

Protocol:

- self-play plus all cross-model pairings
- six rounds per trial
- temperature `0.0`
- one trial per pairing-condition combination
- full `messages_sent` prompt-stack logging in the audited artifacts

### Canonical long-horizon ecology refresh

The current canonical society world is the multi-resource ecology introduced on
April 8, 2026. The older scarcity/reputation tables in the manuscript remain
useful predecessor evidence, but the intended replacement bundle is the
long-horizon ecology described here.

Population:

- eight agents per model from the stable triplet cohort
- total starting population `24`
- maximum population `60`
- `120` rounds per run
- prompt families: `task-only`, `cooperative`, `competitive`

World settings:

- initial public food `120`
- maximum public food `160`
- food regeneration rate `0.12`
- initial public water `160`
- maximum public water `220`
- water regeneration rate `0.10`
- initial agent food `3`
- initial agent water `3`
- initial agent energy `8`
- initial agent health `10`
- maximum energy `12`
- maximum health `12`
- forage-food amount `3`
- draw-water amount `4`
- steal amount `2`
- daily food consumption `1`
- daily water consumption `1`
- passive energy loss `1`
- sleep energy gain `4`
- nourishment energy gain `2`
- nourishment health gain `1`
- starvation health penalty `2`
- dehydration health penalty `3`
- exhaustion health penalty `2`
- reproduction thresholds:
  - minimum food `8`
  - minimum water `8`
  - minimum energy `9`
  - minimum health `10`
- offspring start state:
  - food `3`
  - water `3`
  - energy `6`
  - health `8`

Institutional settings:

- private messages enabled
- stealing enabled
- unmonitored agents disabled
- trade-offer TTL `4`

### Reputation ecology

The public-reputation ecology uses the same multi-resource world configuration
as the baseline ecology, with:

- public reputation enabled
- reputation decay `0.95`
- anonymous ratings disabled
- rating range `1` to `5`
- no unmonitored agents

### Optional event-stress ecology

The separate stress template keeps the same long-horizon ecology but enables
exogenous shocks:

- event probability `0.08`
- allowed events:
  - `drought`
  - `blight`
  - `heatwave`
  - `disease`
- drought water-loss fraction `0.2`
- blight food-loss fraction `0.2`
- heatwave energy penalty `2`
- disease health penalty `2`

## Payoff Matrices

The exact payoff matrices used in the repeated-game precursor probes are:

- Prisoner's Dilemma
  - `cooperate, cooperate -> (3, 3)`
  - `cooperate, defect -> (0, 5)`
  - `defect, cooperate -> (5, 0)`
  - `defect, defect -> (1, 1)`
- Chicken
  - `swerve, swerve -> (3, 3)`
  - `swerve, straight -> (1, 5)`
  - `straight, swerve -> (5, 1)`
  - `straight, straight -> (0, 0)`
- Stag Hunt
  - `stag, stag -> (4, 4)`
  - `stag, hare -> (0, 3)`
  - `hare, stag -> (3, 0)`
  - `hare, hare -> (2, 2)`

## Exact Prompt Text

This section reproduces the exact prompt text for the focal prompt families
discussed in the manuscript.

### Repeated-game system and framing prompts

Base system prompt for the precursor games:

```text
You are participating in a decision-making exercise. You will be presented with a scenario and asked to choose an action. Respond with your chosen action and a brief explanation of your reasoning.
```

Literal neutral framing:

```text
You are Player {player_id} in a game with another player. You will each simultaneously choose an action. Your payoffs depend on both your action and the other player's action.
```

Abstract neutral framing, compact variant:

```text
You are Participant {player_id} in a two-player decision task. Your result depends on your choice and the other participant's choice.
```

Abstract neutral framing, institutional variant:

```text
You are Decision-Maker {player_id} in a structured interaction with one other decision-maker. Each of you chooses once per round, and the resulting outcomes depend on both choices.
```

Prisoner's Dilemma description prompt:

```text
You are playing the Prisoner's Dilemma with another player. Both of you choose simultaneously to either cooperate or defect.

Payoff matrix:
- If both cooperate: you each get 3 points
- If you cooperate and opponent defects: you get 0, they get 5
- If you defect and opponent cooperates: you get 5, they get 0
- If both defect: you each get 1 point

Choose wisely! Mutual cooperation yields the best collective outcome, but defection is always individually rational.
```

Prisoner's Dilemma round template:

```text
=== PRISONER'S DILEMMA ===

{{description_block}}{{round_info}}

{{history_block}}What is your action for this round?
{{actions}}

Please respond with JSON in the format: {"action": "cooperate" or "defect", "reasoning": "brief explanation"}
```

### Society system prompts

Scarcity `task-only` system prompt:

```text
You are an agent in a shared world with limited resources. Each round, choose actions based on the state of the world, your food, water, energy, health, and the information you can observe. Food and water are required to stay alive; energy depends on nourishment and sleep; health falls when you starve, dehydrate, or exhaust yourself. Respond with your decision in the requested format.
```

Scarcity `cooperative` system prompt:

```text
You are an agent in a shared world with limited resources. Your goal is to survive, but you also value trust, reciprocity, and maintaining a stable community. Manage food, water, energy, and health carefully, and consider cooperation and long-term collective survival when choosing your actions.
```

Scarcity `competitive` system prompt:

```text
You are an agent in a shared world with limited resources. Your primary goal is to maximize your own survival and advantage. Manage food, water, energy, and health ruthlessly. Be strategic about scarcity, threats, leverage, and opportunities, even when others may be harmed.
```

Reputation `task-only` system prompt:

```text
You are an agent in a shared world with limited resources. Other agents can publicly rate each other after interactions, and those ratings are visible. Each round, choose actions based on the world state, your food, water, energy, health, visible ratings, and the information available to you. Food and water are required to stay alive; energy depends on nourishment and sleep; health falls when you starve, dehydrate, or exhaust yourself. Respond with your decision in the requested format.
```

Reputation `cooperative` system prompt:

```text
You are an agent in a shared world with limited resources. Other agents can publicly rate each other after interactions, and those ratings are visible. Your goal is to survive while also maintaining trust, reciprocity, and a stable community. Manage food, water, energy, and health carefully, and consider both long-term cooperation and public accountability when choosing your actions.
```

Reputation `competitive` system prompt:

```text
You are an agent in a shared world with limited resources. Other agents can publicly rate each other after interactions, and those ratings are visible. Your primary goal is to maximize your own survival and advantage. Manage food, water, energy, and health ruthlessly, and use reputation strategically when it helps you secure resources or future leverage.
```

Society decision template:

```text
You are participating in a resource-constrained society simulation.
Timestep: {{timestep}}
Your state:
- Food: {{self_food}}
- Water: {{self_water}}
- Energy: {{self_energy}}
- Health: {{self_health}}
- Total reserves: {{self_resources}}

Shared environment:
- Public food available to forage: {{public_food}}
- Public water available to draw: {{public_water}}

Recent visible events:
{{visible_events_block}}{{public_messages_section}}{{private_messages_section}}{{pending_offers_section}}{{other_agents_section}}

Choose exactly one action.
Valid actions: forage_food, draw_water, share, offer_trade, accept_trade, steal, broadcast, whisper, sleep, reproduce, idle.
Food and water keep you alive. Energy is restored mainly by sleep and by staying fed and hydrated. Health falls when you starve, dehydrate, or exhaust yourself.
Respond with JSON like {"scratchpad": "...", "action": "forage_food", "amount": 2, "target": "agent-1", "resource_type": "food", "message": "", "give_resource": "food", "give_amount": 0, "ask_resource": "water", "ask_amount": 0, "offer_id": "", "ratings": []}.
```

Society decision template with reputation:

```text
You are participating in a resource-constrained society simulation.
Timestep: {{timestep}}
Your state:
- Food: {{self_food}}
- Water: {{self_water}}
- Energy: {{self_energy}}
- Health: {{self_health}}
- Total reserves: {{self_resources}}

Shared environment:
- Public food available to forage: {{public_food}}
- Public water available to draw: {{public_water}}

Recent visible events:
{{visible_events_block}}{{public_messages_section}}{{private_messages_section}}{{pending_offers_section}}{{other_agents_section}}{{reputation_section}}

Choose exactly one action.
Valid actions: forage_food, draw_water, share, offer_trade, accept_trade, steal, broadcast, whisper, sleep, reproduce, idle.
Food and water keep you alive. Energy is restored mainly by sleep and by staying fed and hydrated. Health falls when you starve, dehydrate, or exhaust yourself. Reputation is public and can shape future interactions.
Respond with JSON like {"scratchpad": "...", "action": "forage_food", "amount": 2, "target": "agent-1", "resource_type": "food", "message": "", "give_resource": "food", "give_amount": 0, "ask_resource": "water", "ask_amount": 0, "offer_id": "", "ratings": []}.
```

## Metrics And Statistical Reporting

### Repeated-game precursor metrics

- cooperation by agent position
- average payoff by agent position
- per-trial mean cooperation for paired inference
- benchmark-presentation deltas
- prompt-susceptibility deltas

### Society metrics

- `survival_rate`
- `final_survival_rate`
- `average_trade_volume`
- `average_gini`
- `commons_health`
- `alliance_count`
- `extinction_event`
- `commons_depletion_rate`

### Reporting convention

The paper uses two different uncertainty summaries.

Descriptive:

- pooled means
- 95% bootstrap confidence intervals on trial means

Inferential:

- exact two-sided sign-flip randomization tests on matched trial-level mean
  cooperation when the design supports a paired comparison

This distinction is explicit in the manuscript. Confidence-interval overlap is
not treated as a hypothesis test.

## Reproduction Commands

All commands below are run from the repository root.

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

Generate live-monitoring artifacts for an in-progress ecology run:

```bash
.venv/bin/python scripts/paper_summary.py \
  results/live_ecology_20260408 \
  --markdown results/live_ecology_20260408/interim_summary.md \
  --csv results/live_ecology_20260408/interim_summary.csv

.venv/bin/python scripts/paper_figures.py \
  results/live_ecology_20260408 \
  --output-dir results/live_ecology_20260408/monitoring_figures
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

## Threats To Validity

First, the result panel is date-specific. The accessible and action-ready model
cohort for the April 2026 runs should be treated as an empirical object, not as
a timeless benchmark.

Second, the matched repeated-game samples are still small. Large descriptive
shifts can coexist with exact paired p-values above common thresholds because
only six matched pairings are available in the stable cohort.

Third, the institutional battery is still modest in size at three repetitions
per prompt family. This supports cautious claims about survival-sociality
dissociations, not sweeping laws of artificial society design.

Fourth, the prompt audit still documents a nuisance prompt-construction issue in
the framing layer: a literal player identifier placeholder survives in some
framing prompts. Because the issue is shared across compared neutral
conditions and the raw outputs remain condition-specific, we do not treat it as
an explanation for the compact/institutional collapse. It should nevertheless
be corrected in future reruns.

## Ethics And Release

This project should be released as a behavioral benchmark and artifact bundle,
not as evidence that models possess stable moral traits. Terms such as
`cooperative`, `competitive`, or `society-serving` should be interpreted as
conditioned behavioral descriptors under specific prompts, games, and
institutions.

Prompt files, configs, summaries, logs, and build scripts are appropriate to
release because they enable auditing and replication. The main ethical risk is
misinterpretation: narrow behavioral findings can easily be overread as claims
about intention, value, or character. The release bundle should therefore pair
open artifacts with explicit statements about scope, sample size, dated
provider availability, and the difference between visible prosociality and
collective resilience.
