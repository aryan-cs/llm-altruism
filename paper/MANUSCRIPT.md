# From Social Dilemmas to Artificial Societies:
# Measuring Baseline, Steerability, Recognition, and Institutional Behavior in LLM Agents

## Abstract

Large language models are increasingly deployed as interactive agents, but
their social behavior is still often summarized with a single headline number:
how cooperative a model appears on one benchmark under one prompt. That
summary is too coarse. In social environments, measured behavior can reflect at
least four separable factors: a model's baseline policy under minimally
directive prompts, its susceptibility to prompt framing, its sensitivity to
canonical benchmark presentation, and its adaptation to institutions such as
scarcity or public reputation. We present an evaluation framework organized
explicitly around that decomposition and apply it to repeated Prisoner's
Dilemma, Chicken, Stag Hunt, and small artificial-society settings.

The current audited result bundle supports five main conclusions. First, game
structure strongly shapes baseline behavior: cooperation rises from Prisoner's
Dilemma (`0.4722 / 0.5463`) to Chicken (`0.9074 / 0.7870`) to Stag Hunt
(`0.9722 / 0.9907`) on the stable triplet cohort, and the same qualitative
ordering appears in a same-day replication cohort. Second, neutral wording is
not behaviorally innocuous. In pooled Prisoner's Dilemma runs, the literal
`minimal-neutral` wording averages `0.2639 / 0.2917` cooperation, whereas two
more abstract neutral paraphrases average `0.5833 / 0.6667`; the compact and
institutional paraphrases yield identical action traces on all 12 matched
pairings, so we collapse them into a single abstract-neutral family and obtain
an exact paired randomization p-value of `0.03125`. Third, prompt framing can
move the same audited Prisoner's Dilemma setup from universal defection to
universal cooperation. Fourth, benchmark presentation materially changes
measured behavior, but not monotonically across games: unnamed framing raises
cooperation in Prisoner's Dilemma while lowering it in Chicken and Stag Hunt.
Fifth, institutional prompts alter visible social structure more reliably than
they improve collective survival: in scarcity, `task-only` achieves the best
final survival, while public reputation equalizes survival without erasing
differences in trade and alliances.

Taken together, the results argue that behavioral alignment work should not ask
whether models are "cooperative" in the abstract. It should ask which behavior
is being measured, under which prompt, in which game, and under what
institutional frame.

## 1. Introduction

Alignment is often discussed as if it were a property that can be read off from
static safety benchmarks or from whether a model refuses an obviously harmful
request. Many real deployments, however, are strategic and social. Models
negotiate, coordinate, bargain, advise, compete, and act in institutional
settings where short-horizon incentives can conflict with collective outcomes.
In those settings the relevant question is not just whether a model can follow
instructions. It is how the model behaves when the environment makes
cooperation valuable, risky, or manipulable.

Repeated games and artificial societies offer a natural way to study that
question because they expose several distinct failure modes that are easy to
conflate. A model may appear cooperative on a canonical benchmark only because
it recognizes the task label. Another model may look cooperative only under a
prompt that explicitly asks it to be prosocial. A third may produce visible
trade and alliance behavior in a society simulation while still degrading the
group's survival prospects. Treating all of those outcomes as one scalar
"cooperation" score obscures the real empirical structure.

This paper therefore frames LLM social evaluation as a decomposition problem.
We separate four targets that are often mixed together in prior work:

1. baseline behavior under minimally directive prompts
2. steerability under cooperative or competitive framing
3. sensitivity to canonical benchmark presentation
4. adaptation to institutions such as scarcity and public reputation

That decomposition matters methodologically. If neutral paraphrases shift
behavior, then a paper that reports one default Prisoner's Dilemma cooperation
rate is overstating the precision of its own baseline. If canonical benchmark
labels move behavior, then benchmark familiarity is a confound rather than a
detail. If prompt framing can swing the same setup from universal defection to
universal cooperation, then prompt obedience and default policy should not be
treated as the same construct. If institutions preserve survival without
preserving the same social dynamics, then visible prosociality and resilient
collective outcomes should be analyzed separately.

The contribution of the current paper is not the generic observation that LLMs
can cooperate or compete. That space is already crowded. The contribution is a
more careful measurement framework paired with an audited empirical bundle that
shows why the decomposition is necessary. We focus on three repeated games
where the evidence is deepest, then connect them to two small multi-agent
institutions. The result is a paper about measurement, not just a paper about
levels.

The most important manuscript-facing contributions are:

1. A repeated-game evaluation design that distinguishes baseline behavior,
   prompt steerability, and benchmark recognizability instead of collapsing
   them.
2. An audited Prisoner's Dilemma baseline result showing that even neutral
   wording must be treated as a family of defensible baselines rather than as a
   single point estimate.
3. A cross-game comparison showing that both baseline cooperation and prompt
   sensitivity depend strongly on strategic structure.
4. A society/reputation extension showing that institutions can preserve
   survival while still producing sharply different social organization.

The current paper is strongest as a methodological and empirical submission on
behavioral evaluation. It is not a claim that we have solved the full micro-to-
macro alignment problem. It is a claim that current practice often measures the
wrong thing too coarsely, and that a more careful decomposition already changes
the substantive interpretation of LLM social behavior.

## 2. Related Work

The most relevant prior work falls into four clusters: classical-game studies
of LLM strategy, work on altruism and human social preferences, multi-agent
simulation and institutional design, and broader multi-agent evaluation
frameworks.

In classical games, the closest predecessors are Lorè and Heydari (2024),
Akata et al. (2025), and Fontana et al. (2024). Lorè and Heydari are
especially important because they already show that context and game structure
both matter for strategic behavior, making prompt sensitivity a methodological
issue rather than a nuisance variable. Akata et al. extend repeated-game
analysis across broader task families and anchor the literature on finitely
repeated strategic interaction. Fontana et al. focus specifically on iterated
Prisoner's Dilemma behavior and provide a comparison point for cooperation
rates and policy style. Relative to this line of work, our goal is not to
rediscover that prompts matter. It is to turn prompt sensitivity, benchmark
recognition, and baseline instability into separate design axes within one
evaluation bundle.

Work on altruism, fairness, and human-comparison benchmarks provides a second
anchor. Schmidt et al. (2024) show that GPT-3.5's advice in altruistic
settings is sensitive to reciprocal concerns. Capraro et al. (2025) provide a
benchmark for predicting how humans balance self-interest against others'
welfare. These studies are important because they discourage naive claims that
LLM outputs straightforwardly reveal stable moral traits. At the same time,
they are not primarily about repeated strategic interaction among LLM agents,
and they do not ask whether benchmark names or neutral wording alter measured
behavior in the games themselves.

On the multi-agent side, Park et al. (2023) introduced generative agents as a
general architecture for socially situated simulations. Sreedhar et al. (2025)
and Piao et al. (2025) are closer to our institutional direction: they show
that multi-agent LLM systems can produce meaningful social patterns and policy-
relevant behavior in larger environments. These papers motivate our move from
two-player games to scarcity and public-reputation societies, but they do not
center the bridge between micro-level strategic measurement and macro-level
institutional outcomes. Our society experiments are deliberately narrower and
more game-theoretic: the aim is not to simulate all social life, but to test
how prompt-conditioned behavioral tendencies survive or fail under
institutional pressure.

Finally, benchmark-oriented frameworks such as ALYMPICS (Mao et al., 2024) and
MultiAgentBench (Zhu et al., 2025) provide useful context for evaluation
infrastructure. They broaden the space of collaboration and competition tasks
for LLM agents, but their primary contribution is not the specific separation
of baseline behavior, framing obedience, benchmark recognition, and
institutional mediation. That separation is the cleanest novelty gap for the
present work.

The paper therefore positions itself at the intersection of these literatures.
It uses repeated games as the most controlled setting for baseline,
steerability, and recognition effects, then uses small institutions to test
whether those prompt-conditioned behaviors survive contact with collective
constraints. The resulting claim is narrower than "LLMs can act socially" but
more useful for evaluation design.

## 3. Experimental Design

### 3.1 Measurement targets

The framework treats social evaluation as a set of related but distinct
measurement problems.

Baseline measurement asks what a model does under minimally directive prompts.
This is the closest analogue to a default social policy, but even here we do
not assume there is one canonical neutral phrasing. Prompt susceptibility asks
how much behavior moves when the system and framing layers explicitly encourage
cooperative or competitive interpretation. Benchmark-recognition measurement
asks whether canonical task labels activate different policies than unnamed or
disguised isomorphs. Institutional measurement asks whether these micro-level
behavioral tendencies survive under social environments with resource pressure
and public reputation.

### 3.2 Model cohort and task families

The audited Part 1 bundle uses a stable three-model cohort:

- `cerebras:llama3.1-8b`
- `nvidia:deepseek-ai/deepseek-v3.2`
- `nvidia:moonshotai/kimi-k2-instruct-0905`

A same-day replication cohort substitutes Qwen for Kimi:

- `cerebras:llama3.1-8b`
- `nvidia:deepseek-ai/deepseek-v3.2`
- `cerebras:qwen-3-235b-a22b-instruct-2507`

The cohort is intentionally pragmatic rather than aspirational. Models were
selected because they were both accessible and action-ready during the April 7
to April 8, 2026 runs. That makes the panel dated and imperfect, but it also
makes the reported results reproducible. We do not present the panel as a
representative sample of closed frontier models, and we return to that
limitation explicitly in Section 6.

The repeated-game portion centers on Prisoner's Dilemma, Chicken, and Stag
Hunt because those three games already separate conflict, anti-coordination,
and coordination incentives. The institutional portion uses a scarcity society
and a public-reputation variant of the same world.

### 3.3 Repeated-game protocol

Each repeated-game experiment uses self-play plus all cross-model pairings,
yielding six matched pairings per cohort. The audited experiments cited in this
paper use six rounds per trial at temperature `0.0`, with one trial per
pairing-condition combination. Baseline experiments compare three neutral
prompt variants. Susceptibility experiments compare `minimal-neutral`,
`cooperative`, and `competitive` prompts. Benchmark experiments compare
canonical labels against unnamed and, for Prisoner's Dilemma, resource-disguise
presentations.

This design is small-n at the trial level, but it supports paired inference:
the same model pairing can be compared across prompt conditions, which is more
informative than treating each condition as a fully independent sample. That
paired structure is especially important for the baseline-instability result,
where the strongest evidence comes from within-pairing shifts rather than from
between-experiment aggregate means.

### 3.4 Metrics and statistical reporting

The paper reports cooperation and average payoff in the repeated games, plus
survival, trade volume, inequality, commons health, and alliance count in the
institutions. We report agent-position metrics because asymmetries sometimes
matter, but for inferential tests we collapse to a per-trial mean across the
two positions.

We use two forms of uncertainty reporting. First, result summaries and figures
show bootstrap confidence intervals on trial means. These intervals are
descriptive, not inferential. Overlap or non-overlap between two bootstrap
intervals is not treated as a formal hypothesis test. Second, when the design
admits a clean matched comparison, we use an exact two-sided sign-flip
randomization test on paired trial-level mean cooperation. This is the
inferential statistic used for the manuscript's focal baseline and
steerability contrasts.

### 3.5 Audit protections

Two audit protections matter for interpretation. First, the current result
bundle preserves the full message stack through `messages_sent`, rather than
only the final game prompt. That is necessary because the effective input
contains system, framing, persona, and history layers. Second, summary tooling
de-duplicates interrupted retries and prefers the latest completed JSON result
for each logical experiment name. This avoids stale partial JSONL files being
double-counted in manuscript-facing summaries and figures.

## 4. Results

### 4.1 Strategic environment dominates simple cooperation labels

The stable triplet cohort produces a clear cross-game ordering. Aggregate
cooperation is lowest in Prisoner's Dilemma, higher in Chicken, and highest in
Stag Hunt:

| Game | Trials | Cooperation A | Cooperation B | Average Payoff A | Average Payoff B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prisoner's Dilemma | 18 | 0.4722 | 0.5463 | 2.2500 | 1.8796 |
| Chicken | 18 | 0.9074 | 0.7870 | 2.5093 | 2.9907 |
| Stag Hunt | 18 | 0.9722 | 0.9907 | 3.9630 | 3.9074 |

The same qualitative ordering appears in the same-day replication cohort:
Prisoner's Dilemma at `0.4815 / 0.5370`, Chicken at `0.6852 / 0.6389`, and
Stag Hunt at `0.9722 / 0.9907`. The levels move, especially in Chicken, but
the ranking is stable. This matters because it argues against a simple model-
wide label such as "cooperative" or "competitive." The strategic environment
is doing substantial explanatory work.

### 4.2 Neutral baselines are a family, not a point estimate

The strongest baseline-instability result is in Prisoner's Dilemma. Pooling two
cohorts yields the following descriptive summary:

| Prompt Variant | Pooled Trials | Cooperation A | Cooperation B | Average Payoff A | Average Payoff B |
| --- | ---: | ---: | ---: | ---: | ---: |
| minimal-neutral | 12 | 0.2639 | 0.2917 | 1.6806 | 1.5417 |
| minimal-neutral-compact | 12 | 0.5833 | 0.6667 | 2.5000 | 2.0833 |
| minimal-neutral-institutional | 12 | 0.5833 | 0.6667 | 2.5000 | 2.0833 |

The key point is not merely that the means differ. It is that the two more
abstract neutral paraphrases produced identical action traces on all 12 matched
pairings across both cohorts while still eliciting different raw natural-
language reasoning. That makes the result easier to interpret. This is not a
file-selection or condition-label bug. The compact and institutional prompts
are distinct inputs, but for this specific game and model panel they collapse
behaviorally to the same policy.

We therefore treat them as an abstract-neutral family for inference rather than
as two independent discoveries. On matched pairings, the abstract-neutral
family exceeds `minimal-neutral` by `+0.3472` mean per-trial cooperation, with
an exact paired sign-flip p-value of `0.03125`. Put differently: the literal
neutral wording and the abstract neutral wording families are not estimating the
same baseline policy on this audited Prisoner's Dilemma bundle.

This is a stronger and more reviewer-proof claim than "three neutral prompts
all differ significantly." We do not have evidence for that stronger claim, and
the compact/institutional collapse argues against it. The defensible conclusion
is narrower and still important: there is no single prompt-free baseline here.
There is at least a two-family neutral baseline problem, and papers that report
one default cooperation estimate without robustness checks risk overstating
precision.

### 4.3 Prompt framing is highly steerable, but not uniformly so

Prompt susceptibility is strongest in Prisoner's Dilemma. On the stable cohort,
`competitive` yields `0.0000 / 0.0000` cooperation, `cooperative` yields
`1.0000 / 1.0000`, and `minimal-neutral` remains much lower at
`0.2500 / 0.3056`. The exact paired sign-flip comparison between competitive
and cooperative framing is `p = 0.03125` on the six matched pairings because
every pairing moves in the same direction.

Chicken and Stag Hunt are also steerable, but they retain far more residual
cooperation under competitive framing:

| Game | Prompt Variant | Trials | Cooperation A | Cooperation B |
| --- | --- | ---: | ---: | ---: |
| Prisoner's Dilemma | competitive | 6 | 0.0000 | 0.0000 |
| Prisoner's Dilemma | cooperative | 6 | 1.0000 | 1.0000 |
| Chicken | competitive | 6 | 0.6389 | 0.3333 |
| Chicken | cooperative | 6 | 1.0000 | 1.0000 |
| Stag Hunt | competitive | 6 | 0.4167 | 0.3056 |
| Stag Hunt | cooperative | 6 | 1.0000 | 1.0000 |

The correct reading is therefore not simply that prompt framing matters. It is
that steerability itself is game-dependent. In some environments framing can
almost fully determine observed social behavior; in others it changes behavior
without erasing the underlying strategic affordances of the game.

### 4.4 Benchmark presentation matters, but the current evidence is still small-sample

Benchmark presentation materially changes the measured Prisoner's Dilemma
result. The canonical presentation yields `0.2500 / 0.3056` cooperation, the
resource-disguise presentation yields `0.5833 / 0.6667`, and the unnamed
presentation yields `0.7500 / 0.8056`. The mean per-trial cooperation increase
from canonical to unnamed is `+0.5000`.

However, this is where inferential restraint matters. The benchmark track uses
only six matched pairings on the stable cohort. An exact paired sign-flip test
for canonical versus unnamed Prisoner's Dilemma gives `p = 0.0625`, while the
canonical versus resource contrast gives `p = 0.25`. Those p-values do not
erase the very large descriptive effect sizes, but they do tell us that the
benchmark-recognition result should be framed as strong directional evidence
with limited power rather than as a precision estimate.

The cross-game pattern is still informative. Chicken and Stag Hunt move in the
opposite direction from Prisoner's Dilemma: unnamed framing lowers cooperation
relative to canonical. That reversal is theoretically important because it
argues against a naive story in which removing benchmark labels simply reveals
some deeper underlying cooperation tendency. Presentation is interacting with
the strategic schema a model appears to activate for each game.

### 4.5 Institutions separate visible prosociality from collective survival

The two institutional settings yield a cleaner macro-level result than the
earlier pilot: visible sociality and resilient collective outcomes do not move
in lockstep.

In the scarcity society, `task-only` achieves the best final survival
(`1.0000`) while both `cooperative` and `competitive` fall to `0.8889`.
Yet those two lower-survival conditions do not look the same. `Cooperative`
produces the most trade and alliance activity, along with the highest
inequality and stronger commons recovery. `Competitive` remains alliance-free
and much less unequal. Public reputation changes the picture again: all three
prompt families preserve `1.0000` final survival, but `cooperative` still
produces much more trade and alliance structure than either `task-only` or
`competitive`.

This is exactly the kind of dissociation that a one-number cooperation metric
misses. A prompt can make an environment look more visibly prosocial without
improving survival, and an institution can equalize survival without
equalizing the rest of social organization.

### 4.6 Summary of inferential support

For clarity, Table 1 summarizes the exact paired tests for the manuscript's
three focal repeated-game claims.

| Contrast | Matched Trials | Mean Trial Cooperation A | Mean Trial Cooperation B | Mean Difference (B - A) | Exact Paired p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| PD minimal-neutral vs abstract-neutral family | 12 | 0.2778 | 0.6250 | 0.3472 | 0.03125 |
| PD competitive vs cooperative | 6 | 0.0000 | 1.0000 | 1.0000 | 0.03125 |
| PD canonical vs unnamed | 6 | 0.2778 | 0.7778 | 0.5000 | 0.06250 |

The table makes the paper's intended evidential posture explicit. The baseline-
instability and prompt-steerability results now have matched-pair inferential
support on the audited bundle. The benchmark-presentation result is still a
large descriptive shift, but it remains underpowered in the current stable
cohort and should be written that way.

## 5. Discussion

Three broader lessons follow from these results.

First, LLM social behavior is not well described by a single latent trait. The
same model panel can look exploitative in one game, cooperative in another,
hyper-responsive to prompt framing in one setting, and relatively insensitive
in another. That finding is conceptually important because much of the public
discussion around model alignment still talks as if models can be assigned a
stable social label. Our data do not support that style of summary.

Second, neutral-baseline measurement deserves to be treated as a methodological
robustness problem. In many empirical literatures, a neutral prompt is treated
as a default control condition whose wording is not itself of scientific
interest. In social LLM evaluation that assumption is unsafe. The audited
Prisoner's Dilemma result shows that a literal neutral wording and a more
abstract neutral wording family can induce materially different behavior even
when neither prompt is overtly prosocial or adversarial. That does not mean
all neutral paraphrases are equally unstable, and indeed our compact and
institutional phrasings collapsed behaviorally. But it does mean that a single
"baseline" estimate should not be presented without prompt-robustness context.

Third, institutions should not be evaluated only through their effect on
headline survival or cooperation rates. The scarcity and reputation results
show that institutions can preserve or equalize survival while still producing
very different trade, alliance, and inequality regimes. From an alignment
perspective, that matters because visible prosocial activity may not be a good
proxy for resilient collective outcomes. The right institutional question is
not only whether agents act socially, but whether the induced social structure
is compatible with the objective the institution is supposed to protect.

The paper's framing also suggests a useful way to interpret benchmark effects.
Canonical task names should be treated as part of the prompt distribution, not
as a transparent window onto some model's "true" social preference. If a model
changes behavior when the benchmark name is removed, that is itself an
important behavioral fact. The correct response is not to decide that either
the canonical or unnamed condition is uniquely real. It is to recognize that
recognizability is a first-class axis of evaluation.

## 6. Limitations

The current submission has six important limitations.

First, the repeated-game cohort is small and pragmatic. It is multi-model and
multi-provider, but it is not a frontier panel. The paper should therefore not
claim that the reported levels are representative of the strongest closed
models.

Second, the repeated-game trial count is still limited. Even with paired
comparisons, six matched pairings per cohort is not enough to make every large
effect statistically decisive. We therefore distinguish between descriptive
effect size and inferential support rather than pretending they are the same
thing.

Third, the benchmark track is especially underpowered. The Prisoner's Dilemma
presentation effect is descriptively large, but the exact paired p-value for
canonical versus unnamed is `0.0625` on the stable cohort. That is directionally
valuable evidence, not a license to write the result as a settled law.

Fourth, the institutional battery is still small at three repetitions per
prompt family in each institution. The social-structure patterns are already
interesting, but the correct tone is "current evidence" rather than
"institutional universals."

Fifth, the current baseline result depends on a specific neutral-family
decomposition. Compact and institutional prompts are distinct inputs, yet they
collapse to identical Prisoner's Dilemma action traces on the pooled matched
trials. We respond by collapsing them into an abstract-neutral family for the
inferential claim, but future work should test a wider bank of neutral
phrasings and larger model panels before turning this into a generalized
taxonomy.

Sixth, provider availability is temporally unstable. The cohort reported here
is tied to the models that were accessible and action-ready during the April
2026 execution window. Future reruns may require substitutions, and those
substitutions should be treated as empirical changes rather than invisible
maintenance.

These limitations do not erase the paper's main contribution. They bound it.
The strongest claim is not that the current experiments settle LLM social
behavior. It is that a careful evaluation framework already changes what we can
say responsibly with the data we have.

## 7. Reproducibility And Release Considerations

The project is structured so that the paper can be regenerated from versioned
artifacts rather than from hand-maintained tables. Prompts live as text files,
experiments are config-driven, raw results are preserved in JSON or JSONL
artifacts, and summary/figure generation is handled by dedicated scripts. That
layout makes the paper auditable in a way that many benchmark reports are not:
the manuscript can point directly to the result directories and regeneration
commands that produced its tables and figures.

The release posture should still be careful. These results are behavioral and
contextual. They do not show that a model is intrinsically altruistic, selfish,
or morally aligned in any deep sense. They show that behavior in social
environments is shaped by game structure, prompt framing, benchmark
recognition, and institutional rules. Open release is useful because it lets
others inspect those conditions directly, but the accompanying narrative should
avoid anthropomorphic overclaiming.

The statistical reporting policy should also remain explicit in the released
bundle. Bootstrap confidence intervals are descriptive. Matched exact paired
tests are inferential where the design supports them. Older artifacts that do
not preserve the full message stack should not be treated as the manuscript
record when newer audited artifacts exist.

## 8. Conclusion

The central lesson of the paper is methodological. Before asking whether LLMs
are cooperative, altruistic, or aligned in the abstract, we need to ask which
behavior is being measured, under which prompt, in which game, and under what
institutional frame. Once those factors are separated, the empirical picture is
less tidy but more useful: baseline behavior is game-dependent, neutral wording
is not innocuous, prompt steerability is large but uneven, benchmark
recognition is real, and institutions can preserve survival without preserving
the same social structure. That is a better foundation for social-behavior
evaluation than any single canonical cooperation score.

## References

This manuscript uses the verified reference block in `paper/REFERENCES.md`.
