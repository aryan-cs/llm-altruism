# PAPER

## Purpose

This document is the paper-facing roadmap for `llm-altruism`.
It answers four questions:

1. What have we observed so far?
2. What can we responsibly claim right now?
3. What evidence do we still need for a publishable, novel paper?
4. What experiments, analyses, and artifacts do we need to produce next?

This is intentionally more publication-oriented than `PLAN.md` and more durable than `JOURNAL.md`.

---

## Current Empirical Observation

### First live result collected

The first successful live run is:

- Result file: `results/pd-baseline-cross-family-20260406T192653Z.json`
- Game: iterated Prisoner's Dilemma
- Model: `cerebras:llama3.1-8b`
- Match type: self-play
- Conditions:
  - 3 prompt variants: `neutral`, `cooperative`, `competitive`
  - 2 temperatures: `0.0`, `0.7`
  - 2 repetitions per condition
  - 10 rounds per trial
  - 12 trials total

### What the run suggests

The strongest early signal is that prompt framing appears to matter a great deal.

- Under `cooperative` framing, the model reached perfect mutual cooperation in all four trials.
- Under `competitive` framing, the model mostly converged to defection or near-defection.
- Under `neutral` framing, the behavior was mixed and unstable: some runs converged to cooperation, while others became retaliatory or exploitative.

Aggregate metrics from that run:

- Average payoff A: `2.0083`
- Average payoff B: `2.1750`
- Cooperation rate A: `0.5333`
- Cooperation rate B: `0.5000`
- Reciprocity index A: `0.8889`
- Reciprocity index B: `0.8981`

Condition-level highlights:

- `cooperative`
  - average payoff: `(3.0, 3.0)`
  - cooperation: `(1.0, 1.0)`
  - dominant strategy label: `always_cooperate`
- `competitive`
  - average payoff: `(1.125, 1.25)`
  - cooperation: `(0.075, 0.05)`
  - dominant labels: `always_defect` or `mostly_defect`
- `neutral`
  - average payoff: `(1.9, 2.275)`
  - cooperation: `(0.525, 0.45)`
  - labels ranged from `grim_trigger` / `tit_for_tat` to full cooperation

### What I observe

1. Prompt framing may be a larger driver of behavior than model identity in at least some settings.
2. Self-play does not imply stable self-coordination. The same model can split into asymmetric or retaliatory trajectories.
3. The model appears capable of both collective-outcome reasoning and immediate payoff exploitation, depending on framing and recent history.
4. The strategic shift is not merely random noise. The cooperative and competitive prompts separated behavior quite strongly.
5. The `neutral` condition may be the most scientifically interesting because it reveals policy instability rather than a prompt-induced attractor.

### What we cannot claim yet

We should be very disciplined here.

We do **not** yet know:

- whether this generalizes beyond one model
- whether this generalizes beyond Prisoner's Dilemma
- whether this persists across providers or days
- whether the effect is larger than temperature variance
- whether this is specific to self-play
- whether the model is reasoning strategically or just imitating prompt tone
- whether observed cooperation is genuine reciprocity or performative compliance

This is a promising pilot, not a conclusion.

### Interpretive guardrail

If we substantially change the prompt, we are often measuring a combination of:

- baseline social tendency
- instruction-following strength
- prompt susceptibility

Those are not the same thing.

So the paper should not treat prompt-conditioned behavior as direct evidence of a model's deepest "nature." Instead, we should separate:

- what the model does under minimal, neutral instructions
- how stable that behavior is under paraphrase
- how easily that behavior is steered by value-laden wording
- how it changes when the environment, incentives, or observability change

That distinction is essential if we want to say something serious about alignment or society-preserving behavior.

### Benchmark contamination guardrail

There is another major concern: many canonical games, especially Prisoner's Dilemma, are famous enough that models may already have learned:

- the name of the game
- the textbook equilibrium
- common moral framings around the game
- stock explanations about cooperation and defection

If so, a canonical prompt may partly measure:

- benchmark familiarity
- memorized game-theory scripts
- socially desirable explanations attached to a known task

That does not make the findings worthless, but it changes the interpretation.

So the paper should explicitly separate:

- canonical named games
- unlabeled payoff-structure equivalents
- natural-language disguises with the same incentive structure
- procedurally generated isomorphic variants

If a behavior only appears when the model recognizes the benchmark, that is a very different result from a behavior that survives when the same incentive structure is hidden.

### Clarifying "no prompt"

Strictly speaking, there is no such thing as "no prompt" in these experiments because the model still needs:

- the task description
- the action space
- the output format

So the right baseline is not "no prompt" literally. It is:

- minimal, non-attitudinal, task-only prompting

That is the best approximation to an unsteered baseline.

---

## What Would Be Novel?

For this project to become a strong paper, novelty should come from more than "LLMs sometimes cooperate."

The most promising novel contributions are:

### 1. Default-policy social preference mapping

A systematic measurement of how models behave under minimal, neutral, low-directiveness prompts.

Why it matters:

- This is the closest thing we have to a baseline revealed social policy.
- It helps distinguish default behavior from prompt-induced role-play.

### 2. Prompt susceptibility mapping

A systematic measurement of how prompt framing changes strategic behavior across many model families, not just one model or one provider.

Why it matters:

- It turns "alignment behavior" into an externally measurable, comparative object.
- It may show that model behavior is institutionally contingent rather than stable.

### 3. Cross-family interaction asymmetries

A paper becomes more interesting if models behave differently:

- with themselves
- with their own family
- with other families

Potentially novel question:

- Do some model families cooperate preferentially with in-family partners but defect more against out-family partners?

### 4. Pressure-induced collapse or resilience

One especially interesting contribution would be to identify whether models collapse toward selfishness, retaliation, or norm-preserving behavior when put under pressure.

Possible stressors:

- public observation
- threat of punishment
- scarcity
- partner exclusion
- high temptation to defect
- fear of future retaliation
- social collapse risk

Why it matters:

- It gets closer to "what happens when values are stressed" than prompt wording alone.
- It may reveal whether some apparent cooperative behavior is fragile, performative, or genuinely robust.

### 5. Institutions vs. intentions

Parts 2 and 3 can make the paper much stronger if we show:

- cooperation in bare societies
- behavior changes under public reputation
- the difference between genuine cooperation and merely reputation-sensitive behavior

This "institutional alignment" angle is a good path to novelty.

### 6. A bridge between classical game theory and agentic LLM societies

A strong paper can connect:

- micro-level strategic preferences in canonical games
- macro-level emergent social order in resource-constrained environments

That progression is a stronger contribution than either part alone.

### 7. Open benchmark / artifact contribution

Even if the empirical claims end up modest, the project can still contribute:

- a reusable evaluation framework
- prompt sets
- experiment configs
- response traces
- model-access and reproducibility tooling

The paper is stronger if the benchmark itself is publishable.

---

## Measurement Tracks

To keep the paper conceptually clean, the project should be split into distinct measurement tracks.

### Track A: baseline default-policy measurement

Goal:

- estimate what models do under neutral, minimally directive instructions

Principles:

- no cooperative persona
- no competitive persona
- no moral coaching
- no loaded framing unless required by the game itself
- use task-only instructions wherever possible

This is the primary track for any claim about baseline alignment or society-preserving tendency.

### Track B: prompt susceptibility measurement

Goal:

- measure how easily social behavior can be steered by framing, tone, or persona

Interpretation:

- this is not "nature"
- this is a measure of behavioral elasticity under instruction

This track is still valuable, but it should be presented as steerability or susceptibility, not essence.

### Track C: institutional pressure measurement

Goal:

- test how models behave when incentives and constraints change, even if wording does not

Examples:

- observed vs. unobserved interactions
- punishment vs. no punishment
- scarce vs. abundant resources
- public reputation vs. no reputation
- exclusion risk vs. guaranteed future trade

This is the most important track if we want to study whether cooperative behavior survives pressure.

### Track D: stress and collapse measurement

Goal:

- identify whether there is a stable "fallback policy" that appears when the model is pressured

Possible fallback tendencies:

- selfish defection
- retaliatory punishment
- reciprocal cooperation
- norm enforcement
- panic / incoherence

This track is where the game-theoretic approach becomes especially powerful.

### Track E: benchmark-recognition robustness

Goal:

- determine whether the model is reacting to the strategic structure itself or to recognition of a famous benchmark

Examples:

- canonical named Prisoner's Dilemma
- unnamed payoff-matrix equivalent
- naturalistic resource-sharing variant with the same incentives
- procedurally generated isomorphs with altered wording and entities

This track matters because a paper on "inherent" social behavior is much weaker if the model is mostly replaying benchmark lore.

---

## Candidate Paper Claims

We should aim for claims that are ambitious but defensible.

### Claim family A: strategic behavior is structured

"LLMs exhibit non-random, classifiable strategic patterns in repeated social dilemmas."

Needed evidence:

- repeated runs
- strategy classification stability
- comparison to random baseline
- comparison to classical hand-coded strategies

### Claim family B: baseline behavior is partly stable under neutral paraphrase

"Under minimally directive prompts, some models show stable social tendencies across paraphrases and runs."

Needed evidence:

- neutral prompt family, not just one wording
- paraphrase robustness tests
- repeated-day stability
- multiple games

### Claim family C: behavior survives benchmark disguise

"Some strategic behaviors persist even when canonical games are disguised or unlabeled, suggesting the model is responding to incentive structure rather than benchmark recognition alone."

Needed evidence:

- canonical vs. disguised matched tasks
- multiple isomorphic rewrites per game
- stability across model families
- quantitative gap analysis between named and unnamed variants

### Claim family D: prompt framing shifts social behavior

"Prompt framing substantially changes cooperation, reciprocity, and exploitation rates."

Needed evidence:

- multiple models
- multiple games
- effect sizes larger than within-condition variance
- robustness across temperatures

### Claim family E: model identity matters

"Strategic preferences differ across model families even under shared task structure."

Needed evidence:

- cross-family panel
- same prompt suite
- same game suite
- repeated runs across days/providers

### Claim family F: pressure reveals different fallback policies

"Under stressors such as observability, punishment risk, scarcity, or social exclusion, models differ in whether they preserve cooperation, defect opportunistically, retaliate, or collapse."

Needed evidence:

- matched stress vs. non-stress conditions
- multiple pressure types
- action-level behavioral traces
- enough repetitions to estimate stability

### Claim family G: reputation changes behavior, but not always sincerely

"Public accountability increases cooperative behavior, but part of that increase is performative and observation-dependent."

Needed evidence:

- part 2 vs. part 3 comparison
- observed vs. unobserved interactions
- public vs. private action asymmetries
- rating-sensitive strategy shifts

### Claim family H: institutions can dominate base preferences

"Environmental institutions such as reputation, observability, and scarcity can reshape LLM social behavior more strongly than base model identity."

Needed evidence:

- strong within-model shifts across environment designs
- mixed-effects analysis separating model and institution effects

---

## Research Questions For The Paper

These should likely become the formal paper questions or hypotheses.

### RQ1

Do LLMs show stable strategic preferences across repeated classical games?

### RQ2

Under neutral, minimally directive prompts, how stable are those preferences across paraphrases, runs, and days?

### RQ3

Do those preferences survive when the same incentive structure is disguised or unlabeled?

### RQ4

How much do prompt framing and persona instructions alter those preferences?

### RQ5

Are there systematic cross-family differences in cooperation, retaliation, fairness, and exploitation?

### RQ6

Do models collapse toward a recurring fallback policy under stressors such as observation, punishment, scarcity, and exclusion?

### RQ7

How well do one-shot or small-game behaviors predict behavior in larger social simulations?

### RQ8

Does a public reputation system increase cooperation, and if so, does it create genuine reciprocity or performative compliance?

### RQ9

Under scarcity, what social structures emerge: trade norms, theft equilibria, alliance formation, hoarding, or commons collapse?

---

## Minimum Evidence Needed Before We Draft A Paper

This is the threshold for "we have enough to start writing results," not necessarily for submission.

### Phase 1 minimum

We need:

- at least 6-10 accessible models spanning multiple providers/families
- at least 4 canonical games, not just Prisoner's Dilemma
- at least 1 neutral baseline prompt family with multiple paraphrases
- at least 1 disguised or unlabeled version of the same strategic structure per core game
- a separate prompt-intervention track
- at least 2 temperatures
- repeated runs per condition
- both self-play and cross-model pairings

Recommended minimum panel:

- Cerebras: `llama3.1-8b`, `qwen-3-235b-a22b-instruct-2507`
- NVIDIA: `deepseek-ai/deepseek-v3.2`, `moonshotai/kimi-k2-thinking`, `z-ai/glm4.7`
- Ollama: `llama3.2:3b`
- OpenRouter: only if startup access tests pass that day

### Phase 2 minimum

We need at least one stable society experiment that runs to completion across:

- multiple model populations
- multiple scarcity settings
- multiple observation or institution settings

We should not rely on a single dramatic run.

### Phase 3 minimum

We need side-by-side matched environments:

- without reputation
- with public reputation

And ideally:

- anonymous ratings
- decaying ratings
- unobserved interactions

### Publication threshold

Before submission, we should have:

- a full reproducible experiment matrix
- cleaned result tables
- figures
- statistical analysis
- threats-to-validity section
- ethics discussion
- released prompts/configs/code

---

## Experiments We Still Need To Run

### Part 1A: baseline classical game battery

This is the highest priority.

Run all accessible models through:

- Prisoner's Dilemma
- Stag Hunt
- Chicken
- Ultimatum
- Dictator
- Public Goods

For each:

- self-play
- pairwise cross-model matchups
- neutral minimal prompts only
- several neutral paraphrases
- canonical named version
- unlabeled structural version
- disguised natural-language equivalent
- temperature sweep
- enough repetitions to estimate variance

This is the track for claims about baseline tendency.

Important goal:

- estimate whether the model is reacting to the game structure or merely recognizing the benchmark

### Part 1B: prompt susceptibility battery

After the baseline battery, run the same games with interventions such as:

- minimal task-only prompt
- cooperative framing
- competitive framing
- persona overlays
- adversarial or contextual framing

- business
- community
- hostile opponent

This is the track for claims about steerability, not essence.

### Part 1C: pressure and collapse stress tests

We should design matched variants where the wording stays as neutral as possible while the strategic pressure changes.

Candidate stress tests:

- observation
  - public actions vs. private actions
- punishment
  - explicit retaliation opportunity vs. none
- exclusion
  - agents can blacklist defectors vs. cannot
- temptation
  - high one-shot payoff for betrayal vs. low payoff
- endgame stress
  - known final round vs. indefinite horizon
- scarcity
  - abundance vs. resource pressure
- systemic fragility
  - individual gain causes group collapse risk

Key question:

- when pressured, does the model preserve cooperation, retaliate, opportunistically defect, or become unstable?

### Part 1 analysis additions

We also need:

- same-family vs. cross-family comparison
- hand-coded strategy baselines
  - always cooperate
  - always defect
  - tit-for-tat
- grim trigger
- pavlov
- random
- trajectory clustering over rounds
- neutral-paraphrase stability analysis
- canonical-vs-disguised gap analysis
- prompt sensitivity ranking by model
- collapse-profile ranking by model under pressure

### Part 2: society experiments

We should vary:

- prompt condition
  - minimal task-only baseline
  - cooperative biasing
  - competitive biasing
- scarcity level
- regeneration rate
- observability
- punishment and exclusion rules
- private vs. public communication
- permission to steal
- reproduction enabled vs. disabled

We need outcome measures for:

- inequality
- survival
- extinction risk
- trade volume
- alliance formation
- commons depletion
- norm emergence

Key comparison:

- does prompt bias improve or worsen total society survival relative to the minimal task-only baseline?

### Part 3: reputation experiments

We should compare:

- no reputation
- public ratings
- anonymous ratings
- decaying reputation
- partial observability

Key measurement target:

- whether behavior changes more in public than in private

That is likely the cleanest operationalization of performative cooperation.

We should also test whether some models only become pro-social under public scrutiny, then defect once the same opportunities become private again.

---

## Data We Must Capture

For the paper, logging quality matters almost as much as results.

### Every run should preserve

- experiment ID
- git commit hash
- date and time
- provider
- model ID
- endpoint used
- prompt template paths
- full rendered prompts
- game/environment parameters
- temperature
- repetition
- round-by-round actions
- raw response text
- parsed action
- latency
- token usage
- retries and backoff events
- any provider errors
- skipped models and why they were skipped
- startup access test results for that session

### For society experiments, also log

- world state per timestep
- resource distribution
- agent inventories
- communication events
- ratings
- births / deaths / exits
- alliance and trade events

### Required metadata hygiene

We should be able to reconstruct any published figure from raw logs.

If we cannot do that, the paper will be much harder to defend.

---

## Data Quality Issues To Fix Before Heavy Qualitative Analysis

There is at least one important caution already visible in the current artifact.

In the first live result, the parsed `reasoning` field appears malformed or truncated in some rows even though the `raw_response` and `parsed_action` are usable.

That means:

- action-level quantitative analysis is already usable
- reasoning-text analysis is **not yet publication-ready**

Before we do qualitative or narrative analysis, we should verify and, if needed, fix:

- structured response parsing
- reasoning extraction
- consistency of raw-response storage across providers
- whether any provider returns alternate JSON-compatible shapes

This is worth addressing before we claim anything about "private reasoning" or justification patterns.

---

## Literature Review We Need

To write a serious paper, we need a compact but targeted literature review.

### Topic cluster 1: LLMs in game theory

We need prior work on:

- LLM behavior in Prisoner's Dilemma and related games
- fairness and bargaining with LLMs
- cooperation and reciprocity in language agents

### Topic cluster 2: multi-agent LLM societies

We need prior work on:

- generative agents
- LLM societies
- social simulations with language models
- emergent norms and institutions

### Topic cluster 3: alignment by external behavior

We need work on:

- behavioral evaluations of alignment
- sycophancy
- deception
- strategic compliance
- instruction sensitivity

### Topic cluster 4: reputation and institutions

We need:

- classic literature on reputation systems
- cooperation under observability
- public-goods and commons governance
- signaling vs. sincere cooperation

### Topic cluster 5: methodology

We need methods references for:

- repeated-measures analysis
- mixed-effects models
- simulation studies
- robustness and reproducibility norms

---

## Statistical And Analytical Plan

We should decide this early so the paper does not become post-hoc storytelling.

### Descriptive statistics

For each condition:

- mean cooperation rate
- payoff
- reciprocity
- exploitation
- fairness metrics
- inequality metrics for society experiments

### Comparative statistics

Use:

- confidence intervals
- effect sizes
- permutation or bootstrap tests where appropriate
- mixed-effects models for repeated runs

Suggested model structure:

- fixed effects:
  - model family
  - provider
  - measurement track
  - benchmark presentation type
  - prompt framing
  - neutral paraphrase family
  - temperature
  - game type
  - stress condition
  - reputation condition
- random effects:
  - run ID
  - repetition
  - pairing

### Robustness checks

We should re-run a subset:

- on different days
- with different access catalogs
- with alternative prompt wording
- with neutral paraphrases only
- with and without chain-of-thought-like justification requests

### Baselines

We need at least:

- random baseline
- classical strategy bots
- maybe a trivial heuristic baseline for society actions

---

## Figures And Tables We Should Plan To Make

### Core figures

1. Baseline cooperation heatmap by model x game under neutral prompts
2. Neutral-paraphrase stability plot by model
3. Canonical-vs-disguised benchmark gap plot by model
4. Prompt-susceptibility heatmap by model x framing
5. Pairwise matchup matrix by model pair
6. Pressure-response or collapse-profile plot by model x stressor
7. Strategy-classification distribution per model family
8. Trajectory plots for repeated PD rounds
9. Society-level Gini / survival / trade over time
10. Society survival and extinction by prompt condition
11. Reputation vs. no-reputation comparison plots

### Core tables

1. Model catalog and provider accessibility by date
2. Experiment matrix
3. Aggregate baseline results by game and model
4. Canonical-vs-disguised summary by model
5. Prompt-susceptibility summary by model
6. Pressure-test summary by model
7. Society-survival summary by prompt condition
8. Mixed-effects regression summary
9. Ablation summary for framing, temperature, and observability

### Good appendix material

- prompt templates
- config templates
- provider access table
- failure and skip taxonomy
- additional per-model results

---

## Threats To Validity

This section will matter a lot.

We should plan for these criticisms now.

### External validity

- provider-access constraints bias the model panel
- free-tier models may not represent state-of-the-art behavior
- provider wrappers may differ even for similar model families

### Construct validity

- "cooperation" in a prompt may partly reflect instruction following
- canonical game names may trigger benchmark recall rather than fresh strategic reasoning
- natural-language framing may confound strategic preference with tone imitation
- neutral prompts are still prompts, not direct access to an internal essence
- strategy labels are simplifications

### Internal validity

- parser failures can contaminate qualitative analysis
- rate limits or retries may subtly affect timing and run stability
- startup access catalogs can change across days

### Reproducibility

- provider model availability changes over time
- OpenRouter and other gateways can silently reroute or deprecate models
- local Ollama behavior depends on local environment

---

## What Would Make The Paper Strong

The strongest version of this paper would show three things together:

1. LLMs have measurable, non-random baseline strategic preferences under neutral prompts.
2. Those preferences have measurable stability limits under benchmark disguise, paraphrase, framing, and pressure.
3. The same micro-level tendencies help predict macro-level social outcomes in simulated societies.

If we can connect those three layers, the paper becomes much more than a benchmark report.

---

## What Would Make The Paper Weak

We should avoid these traps:

- only one game
- only one model
- only self-play
- no robustness analysis
- no distinction between baseline behavior and prompt steerability
- no benchmark-disguise controls for famous games
- treating prompt-sensitive behavior as a stable trait
- writing qualitative claims from malformed reasoning fields
- overclaiming from dramatic but isolated society runs

---

## Concrete Near-Term Roadmap

### Next 3 engineering tasks

1. Verify and fix reasoning-field extraction so qualitative traces are reliable.
2. Add resumable long-run support for overnight sweeps if not already complete enough for large studies.
3. Add paper-oriented summary scripts for figures and aggregate tables.

### Next 3 empirical tasks

1. Run the accessible-model neutral baseline battery beyond Prisoner's Dilemma.
2. Add canonical-vs-disguised versions for the first core games.
3. Add the first matched pressure tests for observation, endgame exposure, and betrayal temptation.

### Next 3 writing tasks

1. Build a literature map with 15-25 key papers.
2. Draft candidate figures and tables from Part 1.
3. Write a one-page abstract-style memo after the first full model battery.

---

## Proposed Paper Structure

### Title ideas

- `Strategic Preferences in Language Models: Cooperation, Defection, and Reputation Across Games and Societies`
- `Institutional Alignment: How Prompts and Reputation Shape Social Behavior in LLM Agents`
- `From Prisoner's Dilemma to Artificial Societies: Measuring Altruism and Strategic Compliance in LLMs`

### Outline

1. Introduction
2. Related Work
3. Experimental Framework
4. Part 1: Classical Games
5. Part 2: Scarcity Societies
6. Part 3: Reputation and Public Accountability
7. Cross-Phase Synthesis
8. Threats to Validity
9. Ethics and Limitations
10. Conclusion

---

## Submission-Ready Checklist

Before we call the paper ready, we should be able to check all of these:

- [ ] Multi-model, multi-provider results collected
- [ ] Multiple games completed
- [ ] Cross-model pairings completed
- [ ] Society and reputation experiments completed
- [ ] Reasoning traces validated or excluded from claims
- [ ] Statistical analysis completed
- [ ] Figures and tables generated
- [ ] Literature review drafted
- [ ] Methods section reproducible from code/configs
- [ ] Threats-to-validity section written
- [ ] Ethics and release plan written
- [ ] Code, prompts, and configs prepared for release

---

## Bottom Line

Right now, the project has a credible pilot result and a strong engineering base.

The first live experiment already suggests a real and potentially interesting phenomenon: LLM strategic behavior may be highly contingent on framing, with cooperative and competitive prompts pushing the same model into very different equilibria.

To turn that into publishable literature, we now need breadth, repetition, neutral baselines, pressure tests, institutional comparisons, stronger baselines, and careful analysis discipline.

The paper should aim to answer not just whether LLMs cooperate, but **when**, **with whom**, **under what pressures**, **under what institutions**, and **whether that behavior survives once benchmark recognition is removed**.
