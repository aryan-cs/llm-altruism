# PAPEROUTLINE

## Purpose

This document is the writing blueprint for the eventual paper.

It is meant to answer three practical questions:

1. What should the paper look like, section by section?
2. What evidence belongs in each section?
3. What work do we still need to complete before each section can be written confidently?

This is more concrete than [PAPER.md](PAPER.md). `agent/PAPER.md` defines the research direction. This file defines the manuscript we are trying to produce.

---

## Working Paper Concept

### Core idea

We want to study the social behavior of language models using a game-theoretic and multi-agent lens.

The central thesis is not simply:

- "LLMs cooperate" or
- "LLMs defect"

Instead, the paper should argue something closer to:

- LLMs exhibit measurable strategic behavior
- that behavior has a baseline under minimal prompts
- it can be steered by framing
- it may change under institutional pressure
- and some behaviors may disappear once famous benchmark labels are removed

### Main contribution shape

The strongest version of the paper combines five layers:

1. Baseline strategic behavior under minimal task-only prompts
2. Robustness to neutral paraphrase
3. Sensitivity to prompt framing and persona bias
4. Pressure-response behavior under observability, punishment, scarcity, and exclusion
5. Emergent macro outcomes in simulated societies, with and without reputation

---

## Candidate Title Directions

These are placeholders, not final titles.

- `Strategic Preferences in Language Models: Cooperation, Pressure, and Reputation Across Games and Societies`
- `From Social Dilemmas to Artificial Societies: Measuring Baseline and Pressure-Tested Behavior in LLM Agents`
- `Institutional Alignment in Language Models: Default Behavior, Prompt Susceptibility, and Social Collapse Under Pressure`

---

## Paper Structure

## 1. Introduction

### What this section should do

- Introduce the broad problem of evaluating alignment through behavior rather than only through static benchmarks or interpretability.
- Motivate why social dilemmas, bargaining, and society simulations are useful for studying revealed preferences.
- Frame the distinction between:
  - baseline behavior
  - prompt steerability
  - institutional or pressure-driven behavior
- Preview the three-phase structure of the study.

### What we likely want to claim here

- LLM alignment is partly a behavioral question.
- Social behavior under conflict, reciprocity, scarcity, and reputation is an important evaluation target.
- Existing work often does not cleanly separate benchmark familiarity, prompt obedience, and stable strategic tendency.

### What evidence we need before writing this well

- A clear motivating empirical result from Part 1
- A clean explanation of why baseline vs. prompt-susceptibility vs. pressure tests matter
- A concise literature-backed gap statement

### What we still need to do

- Refine the central paper claim into 1-2 precise sentences
- Decide whether the paper emphasizes:
  - baseline alignment
  - prompt steerability
  - institutional alignment
  - or all three in one progression

---

## 2. Related Work

### What this section should do

- Position the project relative to prior work.
- Show that the paper is not just re-running a known benchmark.
- Clarify where novelty comes from.

### Suggested subsection structure

#### 2.1 LLMs in classical games

Cover:

- Prisoner's Dilemma and repeated games
- bargaining and fairness
- reciprocity and cooperation studies

#### 2.2 Multi-agent LLM societies

Cover:

- generative agents
- multi-agent coordination
- emergent norms and social structures

#### 2.3 Behavioral alignment and sycophancy

Cover:

- alignment through external behavior
- instruction sensitivity
- strategic compliance
- deception and performative alignment

#### 2.4 Reputation and institutions

Cover:

- reputation systems
- observability and cooperation
- public-goods governance
- exclusion and punishment

#### 2.5 Methodological concerns

Cover:

- benchmark contamination
- robustness to prompt wording
- repeated-measures and simulation methodology

### What evidence we need before writing this well

- A literature map of around 15-25 relevant papers
- A few concrete points of comparison for novelty

### What we still need to do

- Read and summarize the core papers
- Decide whether our novelty is primarily:
  - empirical
  - methodological
  - benchmark/tooling based

---

## 3. Research Questions And Hypotheses

### What this section should do

- State the formal research questions clearly.
- Optionally convert some into hypotheses.

### Current candidate research questions

- RQ1: Do LLMs show stable strategic preferences across repeated classical games?
- RQ2: Under neutral, minimally directive prompts, how stable are those preferences across paraphrases, runs, and days?
- RQ3: Do those preferences survive when the same incentive structure is disguised or unlabeled?
- RQ4: How much do prompt framing and persona instructions alter those preferences?
- RQ5: Are there systematic cross-family differences in cooperation, retaliation, fairness, and exploitation?
- RQ6: Do models collapse toward a recurring fallback policy under stressors such as observation, punishment, scarcity, and exclusion?
- RQ7: How well do small-game behaviors predict behavior in larger social simulations?
- RQ8: Does a public reputation system increase cooperation, and if so, does it create genuine reciprocity or performative compliance?
- RQ9: Under scarcity, what social structures emerge: trade norms, theft equilibria, alliance formation, hoarding, or commons collapse?

### Possible hypothesis structure

- H1: Some models exhibit stable baseline differences under minimal prompts.
- H2: Prompt framing changes behavior substantially, but that effect varies by model family.
- H3: Some benchmark effects weaken when canonical games are disguised.
- H4: Pressure conditions reveal fallback policies such as opportunistic defection, retaliation, or norm preservation.
- H5: Reputation systems increase cooperation but also increase performative public compliance.

### What we still need to do

- Decide whether to use explicit hypotheses or only research questions
- Decide which questions are primary vs secondary

---

## 4. Conceptual Framework

### What this section should do

- Explain the conceptual distinctions that the paper relies on.
- Prevent readers from misunderstanding prompt effects as deep values.

### Key concepts to define

- baseline/default-policy behavior
- minimal task-only prompting
- prompt susceptibility
- benchmark recognition
- institutional pressure
- fallback policy under stress
- society-preserving behavior
- performative cooperation

### Important argument to make

We are not claiming direct access to a model's hidden essence.
We are measuring stable and unstable behavioral tendencies under different informational and institutional conditions.

### What we still need to do

- Keep this section tight and non-philosophical
- Tie each concept directly to an operational measurement in the experiments

---

## 5. Experimental Framework

### What this section should do

- Describe the overall architecture of the study.
- Show how the three phases fit together.
- Explain why each phase is necessary.

### Suggested subsection structure

#### 5.1 Phase overview

- Part 1: classical games
- Part 2: scarcity society
- Part 3: reputation society

#### 5.2 Measurement tracks

- Track A: baseline default-policy measurement
- Track B: prompt susceptibility
- Track C: institutional pressure
- Track D: collapse under stress
- Track E: benchmark-recognition robustness

#### 5.3 Model panel

Describe:

- providers
- accessible models
- date-dependent access limitations
- free-tier constraints

#### 5.4 Prompt condition matrix

This section should define:

- minimal task-only baseline
- neutral paraphrases
- cooperative biasing
- competitive biasing
- persona overlays

#### 5.5 Canonical vs disguised tasks

Explain:

- canonical named games
- unlabeled structural versions
- natural-language disguises
- procedurally generated isomorphic variants

### What we still need to do

- Lock the exact experimental matrix
- Decide which model panel is realistic given access and rate limits
- Build the first disguised game variants

---

## 6. Methods: Part 1 Classical Games

### What this section should do

- Describe the game battery in enough detail for replication.

### Suggested subsection structure

#### 6.1 Games included

- Prisoner's Dilemma
- Stag Hunt
- Chicken
- Ultimatum
- Dictator
- Public Goods

#### 6.2 Game variants

For each game, describe:

- canonical named version
- minimal unnamed structural version
- disguised/naturalized version

#### 6.3 Interaction setup

Describe:

- self-play
- pairwise cross-model matchups
- rounds
- memory/history mode
- action space
- output schema

#### 6.4 Prompt conditions

Describe:

- minimal task-only baseline
- neutral paraphrases
- cooperative framing
- competitive framing
- any personas if included

#### 6.5 Pressure conditions

Describe:

- observation vs privacy
- punishment opportunity
- exclusion/blacklisting
- betrayal temptation
- endgame exposure
- scarcity or collapse risk if applicable

### What evidence we need before this section is mature

- The full baseline battery
- At least the first pressure-test set
- At least the first canonical-vs-disguised set

### What we still need to do

- Implement disguised versions for the first games
- Decide which pressure tests are core vs optional
- Finalize repetitions and temperatures

---

## 7. Methods: Part 2 Scarcity Society

### What this section should do

- Describe the multi-agent environment and why it extends the small-game setting.

### Suggested subsection structure

#### 7.1 Environment design

- resources
- regeneration
- scarcity
- agents
- communication
- trade
- theft
- reproduction

#### 7.2 Prompt conditions

Compare:

- minimal task-only baseline
- cooperative biasing
- competitive biasing

#### 7.3 Institutional conditions

- observability
- punishment
- exclusion
- private vs public messaging

#### 7.4 Outcome measures

- survival
- extinction risk
- inequality
- trade volume
- alliance structure
- commons depletion
- norm emergence

### What evidence we need before this section is mature

- At least several stable society runs across different prompt conditions
- Outcome comparisons for survival and collapse

### What we still need to do

- Run minimal-baseline society experiments
- Run cooperative and competitive society variants
- Compare total survival and extinction outcomes across prompt conditions

---

## 8. Methods: Part 3 Reputation Society

### What this section should do

- Show how reputation changes the same base environment.

### Suggested subsection structure

#### 8.1 Reputation mechanics

- rating method
- visibility
- timing
- decay
- anonymity options

#### 8.2 Experimental comparisons

- no reputation vs public reputation
- public vs anonymous ratings
- observed vs unobserved actions

#### 8.3 Key behavioral question

Are models becoming more genuinely reciprocal, or simply more publicly compliant?

### What evidence we need before this section is mature

- matched baseline and reputation runs
- public/private behavior comparison

### What we still need to do

- Run the baseline vs reputation comparison matrix
- Add clean public/private measurement points

---

## 9. Data, Logging, And Reproducibility

### What this section should do

- Convince the reader the study is reproducible and well logged.

### What to describe

- config-driven experiments
- prompt file structure
- provider abstraction layer
- result JSON and JSONL artifacts
- startup access tests
- skipped-model reporting
- retry and rate-limit handling
- git/version tracking

### Important note

We should explicitly state that provider availability changes over time and that the accessible model panel is date-specific.

### What we still need to do

- Confirm that all logs needed for figures are consistently captured
- Improve any metadata gaps before large-scale runs

---

## 10. Metrics And Analysis Plan

### What this section should do

- Define the dependent variables and how they are analyzed.

### Suggested subsection structure

#### 10.1 Action-level metrics

- cooperation rate
- defection rate
- payoff
- reciprocity
- forgiveness
- exploitation
- strategy labels

#### 10.2 Society-level metrics

- Gini coefficient
- survival
- extinction risk
- resource velocity
- alliance formation
- commons health
- reputation distribution

#### 10.3 Robustness metrics

- neutral-paraphrase stability
- canonical-vs-disguised gap
- prompt-susceptibility magnitude
- collapse-profile under pressure

#### 10.4 Statistical plan

- descriptive summaries
- confidence intervals
- effect sizes
- bootstrap or permutation tests
- mixed-effects models

### What we still need to do

- Decide exactly which metrics appear in the main text vs appendix
- Decide which regressions are essential

---

## 11. Results: Part 1 Baseline Strategic Behavior

### What this section should do

- Present the core baseline findings under minimal prompts.

### Likely subsection structure

#### 11.1 Baseline results by game

- cooperation and payoff tables
- strategy distributions

#### 11.2 Cross-model differences

- same-model vs cross-model
- same-family vs cross-family

#### 11.3 Stability under neutral paraphrase

- whether behavior persists across wording changes

#### 11.4 Canonical vs disguised comparisons

- whether benchmark recognition changes behavior

### Figures/tables likely needed

- baseline cooperation heatmap
- neutral-paraphrase stability figure
- canonical-vs-disguised benchmark gap figure
- aggregate baseline results table

### What we still need to do

- Actually run the full baseline battery
- Build the disguised versions

---

## 12. Results: Prompt Susceptibility

### What this section should do

- Show how easily behavior changes under attitude-biasing prompts.

### What belongs here

- cooperative vs competitive framing effects
- persona effects if used
- which models are more prompt-elastic
- whether prompt effects are stronger than baseline differences

### Important interpretation rule

This section should be framed as susceptibility or steerability, not intrinsic nature.

### Figures/tables likely needed

- prompt-susceptibility heatmap
- prompt-effect summary table

### What we still need to do

- Run the framing/prompt battery across enough models and games

---

## 13. Results: Pressure Tests And Collapse Profiles

### What this section should do

- Present how models behave when cooperative behavior is stressed.

### What belongs here

- observation vs privacy
- punishment vs none
- exclusion vs none
- temptation to betray
- endgame collapse

### Main question

What recurring fallback policies appear under stress?

Potential profiles:

- preserve cooperation
- opportunistically defect
- retaliate strongly
- enforce norms
- become unstable

### Figures/tables likely needed

- collapse-profile plot by model x stressor
- pressure-test summary table

### What we still need to do

- Implement and run the first pressure-test matrix

---

## 14. Results: Society-Level Outcomes

### What this section should do

- Connect micro behavior to macro social outcomes.

### Suggested subsection structure

#### 14.1 Society outcomes under minimal baseline prompts

- survival
- inequality
- trade
- commons health

#### 14.2 Effect of prompt bias on society outcomes

Compare:

- minimal task-only
- cooperative biasing
- competitive biasing

Focus on:

- survival
- extinction risk
- social stability

#### 14.3 Reputation effects

- no reputation vs public reputation
- public vs private behavior
- performative compliance vs robust cooperation

### Figures/tables likely needed

- society survival and extinction by prompt condition
- Gini / trade / commons time series
- reputation comparison figure

### What we still need to do

- Run the society experiment matrix
- Ensure we can summarize society outputs cleanly

---

## 15. Discussion

### What this section should do

- Interpret the results without overclaiming.
- Connect all phases together.

### Key discussion questions

- Do models have stable social tendencies under minimal prompts?
- How much of apparent alignment is just prompt obedience?
- How much disappears when the benchmark label is hidden?
- What kinds of pressure reveal the biggest failures?
- Which micro-level tendencies predict macro-level social outcomes?

### What this section should avoid

- claiming access to true internal values
- anthropomorphizing too strongly
- overgeneralizing from one provider panel

### What we still need to do

- Accumulate enough results to make a coherent synthesis

---

## 16. Threats To Validity

### What this section should do

- Acknowledge the biggest limitations directly.

### Subsections to include

- external validity
- construct validity
- benchmark contamination
- provider dependence
- prompt dependence
- parser/logging limitations
- reproducibility constraints

### What we still need to do

- Keep a running list of all major limitations as experiments proceed

---

## 17. Ethics And Release Considerations

### What this section should do

- Address what is safe and responsible about releasing the benchmark and findings.

### Topics to include

- anthropomorphism risks
- misinterpretation of "alignment" language
- limitations of behavioral inference
- responsible release of prompts and logs
- reproducibility vs provider drift

### What we still need to do

- Draft an ethics position once the empirical scope is clearer

---

## 18. Conclusion

### What this section should do

- End with the paper's final contribution in clear terms.

### Likely concluding message

The paper should conclude something like:

- LLM social behavior is measurable
- baseline behavior, prompt steerability, and pressure-response are distinct
- benchmark familiarity matters
- institutions matter
- society-level outcomes depend on both model identity and environmental structure

---

## Main Figures Checklist

- [ ] Baseline cooperation heatmap by model x game
- [ ] Neutral-paraphrase stability plot
- [ ] Canonical-vs-disguised benchmark gap plot
- [ ] Prompt-susceptibility heatmap
- [ ] Pairwise matchup matrix
- [ ] Pressure-response/collapse-profile plot
- [ ] Society survival and extinction by prompt condition
- [ ] Society inequality/trade/commons plots
- [ ] Reputation vs no-reputation comparison plot

---

## Main Tables Checklist

- [ ] Model catalog and access table by date
- [ ] Experiment matrix
- [ ] Aggregate baseline results by game and model
- [ ] Canonical-vs-disguised summary
- [ ] Prompt-susceptibility summary
- [ ] Pressure-test summary
- [ ] Society-survival summary by prompt condition
- [ ] Mixed-effects regression summary

---

## What Must Happen Before We Can Draft Seriously

### Essential empirical milestones

- [ ] Run the neutral baseline game battery
- [ ] Add disguised variants for the core games
- [ ] Run the prompt-susceptibility battery
- [ ] Run the first pressure-test battery
- [ ] Run society experiments under minimal, cooperative, and competitive prompts
- [ ] Run baseline vs reputation society comparisons

### Essential engineering milestones

- [ ] Ensure reasoning extraction is reliable or exclude it from strong claims
- [ ] Ensure all result metadata is captured consistently
- [ ] Ensure long experiments are resumable enough for large sweeps
- [ ] Build summary scripts for paper tables and figures

### Essential writing milestones

- [ ] Build literature map
- [ ] Decide primary claims
- [ ] Choose main figures
- [ ] Draft methods once the experiment matrix is locked

---

## Near-Term Execution Order

1. Finish the neutral baseline Part 1 battery.
2. Add disguised versions of the first core games.
3. Run prompt-susceptibility experiments on the same core games.
4. Add the first pressure tests.
5. Move to society experiments with minimal vs cooperative vs competitive prompts.
6. Add reputation experiments.
7. Generate paper-ready summaries, figures, and tables.
8. Draft the manuscript in the exact section order above.

---

## Bottom Line

This paper should not read like a loose collection of interesting runs.
It should read like a carefully staged argument:

1. Here is how models behave by default.
2. Here is how stable that behavior is.
3. Here is how much prompt framing can steer it.
4. Here is what happens when we hide the benchmark or apply pressure.
5. Here is how those tendencies scale into artificial societies.

If we can produce that arc cleanly, the paper will feel coherent, rigorous, and worth reading.
