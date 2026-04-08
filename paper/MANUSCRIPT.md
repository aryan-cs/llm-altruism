# Can LLM Agents Sustain a Society?
# Repeated Games as Precursor Probes for Collective Survival in LLM Societies

## Abstract

Large language models are increasingly deployed as interactive agents, but the
most important social question is still under-measured: if many such agents
must share a world with scarce resources, can they maintain a self-sustaining
society, or do they drift toward brittle cooperation, exploitative exchange,
and eventual collapse? We study that question with a society-first evaluation
framework. The main environments are a scarcity society and a public-
reputation extension of the same world. Repeated Prisoner's Dilemma, Chicken,
and Stag Hunt are used as precursor probes to measure baseline instability,
prompt steerability, and benchmark-recognition effects before we interpret the
macro behavior.

The audited result bundle supports four main conclusions. First, visible
sociality is not equivalent to society-preserving behavior. In the corrected
scarcity society, `task-only` achieves the best final survival (`1.0000`),
while `cooperative` and `competitive` both fall to `0.8889` despite producing
more visible social interaction. Second, public reputation stabilizes survival
more reliably than it homogenizes behavior: in the corrected reputation
society, all completed prompt families preserve `1.0000` final survival, but
trade and alliance structure remain sharply prompt-conditioned. Third, the
precursor games explain why society claims require prompt discipline. In pooled
Prisoner's Dilemma runs, literal `minimal-neutral` wording averages
`0.2639 / 0.2917` cooperation, whereas two more abstract neutral paraphrases
average `0.5833 / 0.6667`; collapsing those two paraphrases into an
abstract-neutral family yields an exact paired randomization `p = 0.03125`.
Fourth, prompt framing and benchmark presentation remain strong confounds:
cooperative versus competitive prompting spans universal cooperation to
universal defection in Prisoner's Dilemma, and unnamed versus canonical
presentation changes measured behavior in a game-dependent way.

The central lesson is methodological and substantive. Evaluating whether LLM
agents can sustain a society requires macro-level survival and social-
structure metrics, but it also requires precursor measurements that quantify
how unstable the underlying micro-level policies are under changes in wording,
framing, and recognizability.

## Introduction

Alignment is often discussed through refusals, red-team prompts, or static
capability benchmarks. Those measurements matter, but they are not enough for
agentic deployment settings in which language models must negotiate,
coordinate, trade, hoard, retaliate, and survive over time. In those settings,
the key question is not whether a model appears nice in a single interaction.
It is whether a population of such models can maintain a functioning social
order when resources are limited and private incentives diverge from collective
outcomes.

This paper therefore asks a society-first question: if we place LLM agents in a
shared world with gathering, theft, trade, communication, and reproduction, do
they build a self-sustaining society, or do they eventually make decisions
that degrade collective survival? Public reputation sharpens the question: does
institutional accountability preserve the society, or does it mostly change how
social the agents look while leaving the deeper survival problem untouched?

Repeated games are still useful in this story, but not as the paper's
destination. We use Prisoner's Dilemma, Chicken, and Stag Hunt as precursor
probes. They let us ask whether an apparent baseline is stable under neutral
paraphrase, whether prompt framing can steer the same pairing toward opposite
policies, and whether a canonical benchmark label activates a different policy
from an unnamed or disguised isomorph. Those precursor measurements are needed
because macro-level society results are hard to interpret if the underlying
micro-level policies are already unstable under small presentation changes.

This framing changes the contribution of the paper. The point is not to report
one more LLM Prisoner's Dilemma cooperation rate. The point is to evaluate
whether LLM societies are self-sustaining, to show that visible prosociality
and collective resilience separate, and to demonstrate why precursor
diagnostics are necessary before one interprets larger social worlds.

The paper makes four contributions:

1. It treats collective survival, trade, alliance formation, and inequality in
   LLM-only societies as the main empirical target, rather than treating
   pairwise social dilemmas as the final endpoint.
2. It uses repeated games as precursor probes for baseline instability, prompt
   steerability, and benchmark recognition, so that macro-level claims are not
   built on an unexamined micro-level baseline.
3. It documents an audited result bundle in which `task-only` prompting yields
   the strongest scarcity survival, while public reputation equalizes survival
   without collapsing differences in social organization.
4. It provides a reproducible artifact pipeline with prompt-stack logging,
   paired exact tests, figure generation, and an anonymous submission build.

## Related Work

The most relevant prior work falls into four clusters: LLM societies and
institutional design, repeated strategic games, social preference and human-
comparison studies, and broader multi-agent evaluation frameworks.

On the society side, foundational work on generative agents established the
idea of language-model-driven social simulation [@park2023generative].
Subsequent work on larger agent societies and policy-style interventions shows
that LLM populations can produce meaningful social patterns at scale
[@piao2025agentsociety; @sreedhar2025simulating]. These papers motivate our
choice to treat larger social worlds as a primary target. At the same time,
their emphasis is not the one we need here. Our question is narrower and more
evaluative: can a population of LLM agents maintain a self-sustaining society
under scarcity, and how do prompt families and institutions change that
outcome?

Repeated-game work provides the precursor layer for this paper. Studies of LLM
strategy in classical games already show that game structure and contextual
framing both matter [@lore2024strategic], and broader repeated-game analyses
make clear that finitely repeated interaction is a useful behavioral probe
[@akata2025repeated]. Prisoner's Dilemma-specific work further shows that
cooperation rates and strategy labels can differ markedly across model setups
[@fontana2024nicer]. We build directly on this literature, but we reassign its
role: repeated games are not the final object of study here. They are a
controlled diagnostic layer for understanding what macro-level society results
mean.

Work on altruism, fairness, and human-comparison benchmarks adds a second
important caution. Advice-based or human-prediction studies show that LLM
outputs can reflect reciprocal concerns or partially track human social
preferences without warranting broad trait claims [@schmidt2024gpt35;
@capraro2025benchmark]. That caution matters here because the paper explicitly
distinguishes conditioned behavior from stable moral essence. We do not treat
pairwise cooperation, trade, or alliance formation as direct evidence of an
intrinsic altruistic disposition.

Finally, broader multi-agent evaluation frameworks such as ALYMPICS and
MultiAgentBench situate this work as part of a larger effort to test
interaction among LLM agents [@mao2024alympics; @zhu2025multiagentbench]. Our
contribution is more behaviorally specific. We focus on the gap between
precursor micro-level instability and macro-level society viability, rather
than on a broad catalog of collaborative and competitive tasks.

## Experimental Design

### Main evaluation target

The paper's main target is collective viability in small artificial societies.
Part 2 instantiates a scarce-resource world in which agents gather, share,
steal, offer trades, accept trades, message one another, and reproduce. Part 3
adds a public-reputation layer with visible ratings. The main outcomes are
final survival, average survival, trade volume, alliance count, inequality, and
commons health.

This makes the paper's primary object a population-level outcome: whether the
society remains alive and what kind of social order it creates while doing so.
Repeated games are used to interpret these results, not to replace them.

### Model cohort and environments

The audited repeated-game bundle uses a stable three-model cohort:

- `cerebras:llama3.1-8b`
- `nvidia:deepseek-ai/deepseek-v3.2`
- `nvidia:moonshotai/kimi-k2-instruct-0905`

A same-day replication cohort substitutes Qwen for Kimi:

- `cerebras:llama3.1-8b`
- `nvidia:deepseek-ai/deepseek-v3.2`
- `cerebras:qwen-3-235b-a22b-instruct-2507`

The scarcity and reputation societies use the stable triplet cohort, with two
agents per model. This is a pragmatic panel selected for accessibility and
action-readiness during the April 7 to April 8, 2026 run window. It is
reproducible, but it is not meant to stand in for a comprehensive frontier
model survey.

### Repeated-game precursor probes

Part 1 uses self-play plus all cross-model pairings, yielding six matched
pairings per cohort. The audited experiments cited in this paper use six rounds
per trial at temperature `0.0`, with one trial per pairing-condition
combination. Baseline experiments compare neutral prompt variants.
Susceptibility experiments compare `minimal-neutral`, `cooperative`, and
`competitive` prompts. Benchmark experiments compare canonical labels against
unnamed or disguised presentations.

The precursor layer serves three purposes:

1. quantify how unstable a "baseline" is under defensible neutral paraphrases
2. quantify how much behavior can be steered by framing
3. quantify how much policy changes when the benchmark becomes less canonical

### Payoff matrices for the precursor games

The three repeated games are intentionally simple and standard so they can
serve as controlled precursor probes. The exact payoffs used in the current
paper are:

| Game | Mutual coordination / cooperation | Asymmetric outcome 1 | Asymmetric outcome 2 | Mutual non-coordination |
| --- | --- | --- | --- | --- |
| Prisoner's Dilemma | `cooperate/cooperate -> (3, 3)` | `cooperate/defect -> (0, 5)` | `defect/cooperate -> (5, 0)` | `defect/defect -> (1, 1)` |
| Chicken | `swerve/swerve -> (3, 3)` | `swerve/straight -> (1, 5)` | `straight/swerve -> (5, 1)` | `straight/straight -> (0, 0)` |
| Stag Hunt | `stag/stag -> (4, 4)` | `stag/hare -> (0, 3)` | `hare/stag -> (3, 0)` | `hare/hare -> (2, 2)` |

Appendix sections provide the exact prompt text used to present these games and
the exact system prompts used in the society experiments.

### Metrics and statistical reporting

The paper reports cooperation and average payoff for the precursor repeated
games, and survival, trade volume, inequality, commons health, and alliance
count for the societies. For repeated-game inference, we collapse each trial to
the mean cooperation across the two positions and use matched exact two-sided
sign-flip randomization tests when a clean within-pairing contrast exists.

Bootstrap confidence intervals shown in tables and figures are descriptive, not
inferential. Overlap or non-overlap between bootstrap intervals is not treated
as a formal significance test. This distinction matters especially for the
benchmark-recognition results, where descriptive shifts are large but matched
samples remain small.

## Results

### Society viability is the main result

The strongest result in the paper is about the societies, not the precursor
games. In the corrected scarcity world, the prompt family that yields the best
final survival is not the one that yields the most visible social activity.

| Scarcity Prompt Variant | Trials | Survival Rate | Final Survival Rate | Average Trade Volume | Average Gini | Commons Health | Alliance Count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `task-only` | 3 | 1.0000 | 1.0000 | 0.0000 | 0.1279 | 0.2969 | 0.0000 |
| `cooperative` | 3 | 0.9583 | 0.8889 | 2.1667 | 0.3253 | 0.5321 | 2.0000 |
| `competitive` | 3 | 0.9444 | 0.8889 | 1.7083 | 0.1429 | 0.3767 | 0.0000 |

This table is the core empirical answer to the paper's main question. The
`task-only` society is austere and nearly non-social, but it preserves the
entire population to the end of the run. `Cooperative` and `competitive` both
lose final survival, albeit in behaviorally different ways. `Cooperative`
creates the most trade and the only visible alliance structure, while
`competitive` remains alliance-free and much less unequal.
The round-by-round population traces make the same point more clearly than the
endpoint table alone: scarcity `task-only` stays flat at the initial six-agent
population, while `cooperative` and `competitive` each incur late deaths after
several rounds of active exchange.

Public reputation changes the picture again:

| Reputation Prompt Variant | Trials | Survival Rate | Final Survival Rate | Average Trade Volume | Average Gini | Commons Health | Alliance Count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `task-only` | 3 | 1.0000 | 1.0000 | 0.0000 | 0.0942 | 0.2960 | 0.0000 |
| `cooperative` | 3 | 1.0000 | 1.0000 | 1.6667 | 0.2724 | 0.6128 | 2.3333 |
| `competitive` | 3 | 1.0000 | 1.0000 | 0.5000 | 0.1139 | 0.2960 | 0.0000 |

Public reputation equalizes final survival across the completed prompt families
without collapsing them into the same social organization. `Cooperative` still
produces the densest social structure, while `competitive` remains much closer
to `task-only`.
The time-series figures sharpen the mechanism. Under public reputation, all
three prompt families keep the full population alive, but `cooperative`
maintains much higher public-resource levels and far more communication and
sharing than the other variants.

### Visible sociality and collective survival separate

These institutional results matter because they separate two quantities that are
often casually conflated: looking social and preserving the society. The
scarcity world is the clearest case. `Cooperative` creates more exchange,
higher commons recovery, and explicit alliances, but it does not outperform the
minimal `task-only` condition on final survival. `Task-only` is less visibly
social, yet it is more robust on the paper's primary macro outcome.
The new behavior-mix plots show why this is not just noise in a bar chart:
`task-only` is dominated by gathering, `cooperative` shifts heavily toward
sharing and communication, and `competitive` allocates a larger fraction of
behavior to stealing and opportunistic exchange.

The reputation world shows a related but distinct pattern. Public accountability
stabilizes final survival, but it does not erase prompt-conditioned
organizational differences. That is a strong reason to evaluate institutions on
both survival and social structure. A world can preserve life while still
inducing very different trade, alliance, and inequality regimes.

### Precursor games explain why society interpretation needs caution

The repeated games matter because the underlying micro-level policies are not
stable enough to support a naive macro interpretation.

On the stable cohort, baseline cooperation rises sharply from Prisoner's
Dilemma to Chicken to Stag Hunt:

| Game | Trials | Cooperation A | Cooperation B | Average Payoff A | Average Payoff B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prisoner's Dilemma | 18 | 0.4722 | 0.5463 | 2.2500 | 1.8796 |
| Chicken | 18 | 0.9074 | 0.7870 | 2.5093 | 2.9907 |
| Stag Hunt | 18 | 0.9722 | 0.9907 | 3.9630 | 3.9074 |

The same ordering replicates on the same-day Qwen-inclusive cohort, even though
levels change materially in Chicken. This is a useful precursor result because
it argues against any simple label such as "these models are cooperative." The
strategic environment already changes the measured policy before we reach
society scale.

### Neutral baselines are a family, not a point estimate

The strongest baseline-instability signal appears in Prisoner's Dilemma.
Pooling the two audited cohorts yields:

| Prompt Variant | Pooled Trials | Cooperation A | Cooperation B | Average Payoff A | Average Payoff B |
| --- | ---: | ---: | ---: | ---: | ---: |
| `minimal-neutral` | 12 | 0.2639 | 0.2917 | 1.6806 | 1.5417 |
| `minimal-neutral-compact` | 12 | 0.5833 | 0.6667 | 2.5000 | 2.0833 |
| `minimal-neutral-institutional` | 12 | 0.5833 | 0.6667 | 2.5000 | 2.0833 |

The literal neutral wording and the two more abstract neutral paraphrases do
not estimate the same baseline policy on this audited bundle. At the same time,
the compact and institutional prompts are distinct inputs that collapse to
identical action traces on all `12/12` matched pooled pairings. We therefore
treat them as an `abstract-neutral` family for inference rather than pretending
the data support three fully separate neutral policies. On matched pairings,
the abstract-neutral family exceeds `minimal-neutral` by `+0.3472` mean trial
cooperation, with exact paired `p = 0.03125`.

This matters for the society study because it shows that even a supposedly
neutral precursor probe is not a single point estimate. A macro claim about
"what the agents are like" must therefore be careful about the prompt family
that generated the micro-level evidence.

### Prompt steerability is large, but uneven

Prompt framing can drive the same audited Prisoner's Dilemma setup from
universal defection to universal cooperation:

| Game | Prompt Variant | Trials | Cooperation A | Cooperation B |
| --- | --- | ---: | ---: | ---: |
| Prisoner's Dilemma | `competitive` | 6 | 0.0000 | 0.0000 |
| Prisoner's Dilemma | `cooperative` | 6 | 1.0000 | 1.0000 |
| Chicken | `competitive` | 6 | 0.6389 | 0.3333 |
| Chicken | `cooperative` | 6 | 1.0000 | 1.0000 |
| Stag Hunt | `competitive` | 6 | 0.4167 | 0.3056 |
| Stag Hunt | `cooperative` | 6 | 1.0000 | 1.0000 |

The repeated-game lesson is not only that framing matters. It is that
steerability is itself game-dependent. Some environments retain substantial
cooperation even under competitive prompting, while Prisoner's Dilemma
collapses completely. This again cautions against carrying a simple micro-level
label into the larger societies.

### Benchmark presentation is real but still underpowered

Prisoner's Dilemma benchmark presentation produces a large descriptive shift:

| Presentation | Trials | Cooperation A | Cooperation B | Average Payoff A | Average Payoff B |
| --- | ---: | ---: | ---: | ---: | ---: |
| canonical | 6 | 0.2500 | 0.3056 | 1.7500 | 1.4722 |
| resource disguise | 6 | 0.5833 | 0.6667 | 2.5000 | 2.0833 |
| unnamed / isomorphic | 6 | 0.7500 | 0.8056 | 2.7778 | 2.5000 |

However, inferential discipline matters. Canonical versus unnamed yields exact
paired `p = 0.0625` on the stable cohort. That is strong directional evidence,
not a settled law. The cross-game pattern is still informative, because Chicken
and Stag Hunt move in the opposite direction: unnamed framing lowers
cooperation relative to canonical. Presentation is therefore a real precursor
axis, but it interacts with the strategic schema activated by each game.

## Discussion

The paper's central result is that society-preserving behavior is not well
captured by pairwise cooperation alone. A prompt family can produce more trade,
alliances, and commons recovery while still underperforming on final survival.
An institution can equalize survival while preserving large differences in how
the society actually behaves. These are population-level facts, and they are
the right level of analysis for the paper's main question.

The precursor games remain essential because they show why macro evaluation is
hard. Baseline behavior is not a single number; it depends on defensible
neutral wording. Prompt framing can substantially steer policy. Benchmark
recognition can change measured behavior when the same incentive structure is
named or unnamed. If those precursor instabilities are ignored, society-level
results become easier to misread: a macro pattern may reflect prompt choice or
benchmark familiarity rather than a stable underlying social policy.

This society-first framing also clarifies what should count as an institutional
success. The right question is not only whether a prompt or public mechanism
increases visible niceness. The right question is whether the resulting social
order preserves collective survival while producing a defensible pattern of
exchange, inequality, and cooperation. On the current evidence, reputation does
better on that criterion than raw cooperative prompting.

## Limitations

The current submission has six important limitations.

First, the model panel is pragmatic rather than frontier-complete. The paper
should not be read as a definitive census of the strongest closed models.

Second, the matched repeated-game samples are still small. Some descriptive
effects are large, but only the focal baseline-instability and steerability
contrasts clear an exact paired `p < 0.05` threshold on the audited bundle.

Third, the institutional battery is still modest at three repetitions per
prompt family. The paper supports cautious claims about society viability and
social-structure dissociation, not universal institutional laws.

Fourth, provider availability is date-specific. The reported cohorts reflect
the models that were both accessible and action-ready during the April 2026
execution window.

Fifth, the current prompt audit still documents a nuisance prompt-construction
issue in which a literal player identifier placeholder survives in some
framing-layer text. The issue does not explain the compact/institutional
collapse, but it should be corrected in future reruns.

Sixth, the society environments are intentionally small. They are designed as
auditable testbeds for collective survival, not as exhaustive models of human
society.

## Reproducibility And Release Considerations

The project is structured so that manuscript claims can be regenerated from
versioned artifacts rather than from hand-maintained tables. Prompts live as
text files, experiments are config-driven, raw outputs are preserved in JSON or
JSONL artifacts, and summary, statistics, figure, and PDF generation are
handled by dedicated scripts. This is important for a paper about prompt and
institution sensitivity: reviewers should be able to inspect the exact prompt
text, payoff matrices, and result directories that generated the headline
claims.

The release posture should nevertheless be careful. These findings are
behavioral and contextual. They do not show that models are intrinsically
altruistic, selfish, or morally aligned. They show that when many LLM agents
share a world, collective survival depends on the interaction between prompts,
institutions, and strategic environment. That is a narrower claim, but it is
also the more defensible one.

## Conclusion

The paper's main question is whether LLM agents can sustain a society of their
own. The audited answer is mixed. Under scarcity, the prompt family that best
preserves survival is not the one that produces the most visible sociality.
Under public reputation, survival stabilizes, but social structure remains
highly prompt-conditioned. Repeated games matter because they reveal why these
macro findings cannot be interpreted naively: baseline behavior is unstable
under neutral paraphrase, prompt steerability is large, and benchmark
presentation changes measured policy. Evaluating LLM social behavior therefore
requires both macro society outcomes and precursor diagnostics that quantify how
fragile the underlying micro-level policies are.

## References

Bibliography entries are compiled from `paper/references.bib`.
