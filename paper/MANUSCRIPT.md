# From Social Dilemmas to Artificial Societies:
# Measuring Baseline, Steerability, and Institutional Behavior in LLM Agents

## Abstract

Large language models are increasingly deployed as interactive agents, but
their social behavior is often evaluated in ways that conflate default
strategic tendencies, prompt obedience, benchmark familiarity, and
institution-mediated adaptation. We present a behavioral evaluation framework
that separates these factors across repeated classical games and larger
multi-agent settings. The repaired empirical core now spans repeated
Prisoner's Dilemma, Chicken, and Stag Hunt experiments plus corrected
scarcity and public-reputation society reruns, all with full prompt-stack
logging after a validity repair. We find five robust results. First, game
structure strongly shapes behavior: on a repaired stable cohort, cooperation
rises from Prisoner's Dilemma (`0.4722 / 0.5463`) to Chicken
(`0.9074 / 0.7870`) to Stag Hunt (`0.9722 / 0.9907`), and the same ordering
replicates on a second same-day cohort. Second, even neutral wording is not
behaviorally innocuous: pooling two repaired Prisoner's Dilemma cohorts,
`minimal-neutral` yields `0.2639 / 0.2917` cooperation versus
`0.5833 / 0.6667` for two other neutral paraphrases. Third, prompt framing can
steer the same game from universal defection to universal cooperation.
Fourth, benchmark presentation materially changes measured behavior, but not in
one direction across games: removing canonical framing raises cooperation in
Prisoner's Dilemma while lowering it in Chicken and Stag Hunt. Fifth,
institutional effects are not reducible to surface prosociality: in the
corrected scarcity society, `task-only` preserves `1.0000` final survival
while both `cooperative` and `competitive` fall to `0.8889`, whereas public
reputation equalizes final survival at `1.0000` across all three prompt
families while preserving large differences in trade, alliances, and
inequality. These findings support a behavioral view of alignment in which
baseline policy, steerability, benchmark recognition, and institutional
mediation must be measured separately rather than collapsed into a single
cooperation score.

## 1. Introduction

Large language model alignment is often evaluated through refusal behavior,
static benchmarks, or capability tests. Yet many realistic deployments are
social: models negotiate, coordinate, compete, advise, and participate in
institutional settings where long-horizon incentives matter. In these
environments, alignment is partly a behavioral question. What does a model do
when cooperation is jointly efficient but individually risky? How stable is
that behavior under neutral paraphrase? Does it change when a famous benchmark
label is removed? And can prosocial language be separated from robust
collective outcomes?

This paper argues that LLM social behavior should not be summarized by a single
scalar notion such as "cooperativeness" or "altruism." Instead, behavioral
evaluation should distinguish at least four targets:

1. baseline behavior under minimally directive prompts
2. sensitivity to prompt framing
3. sensitivity to benchmark presentation and recognizability
4. adaptation to institutions such as scarcity and public reputation

That separation matters methodologically as well as substantively. If canonical
benchmark names shift behavior, then some measured "strategy" may reflect
recognition rather than default policy. If cooperative personas move outcomes,
then we may be measuring instruction-following rather than stable social
preferences. If public institutions change behavior, then those changes should
be attributed to environmental structure rather than model essence.

The repaired empirical core in this manuscript centers on repeated classical
games, where the evidence is deepest, and now includes a small corrected
institutional battery under scarcity and public reputation. These repaired
results already support a strong paper: default behavior is game-dependent,
neutral wording is unstable, prompt framing is highly steerable, benchmark
presentation materially changes measured cooperation, and institutional
structure can separate visible social activity from collective survival.

## 2. Related Work

Recent work shows that LLMs can exhibit strategic behavior in classical games,
but the literature often stops short of separating baseline behavior, prompt
steerability, and benchmark familiarity. Lore and Heydari (2024) provide one of
the closest predecessors by showing that both game structure and contextual
framing shape strategic behavior in LLMs. Akata et al. (2025) extend repeated
game analysis to broader families of cooperation and coordination tasks and
show stable differences across game families. Fontana et al. (2024) focus
specifically on iterated Prisoner's Dilemma behavior.

Work on social preferences and human comparison is also relevant. Schmidt et
al. (2024) study altruistic advice in Dictator and Ultimatum Game settings,
while Capraro et al. (2025) benchmark how well LLMs predict human tradeoffs
between self-interest and others' welfare. These papers are useful for
grounding fairness and altruism claims, but they do not center repeated
strategic interaction plus benchmark-disguise conditions.

On the multi-agent side, Park et al. (2023) introduced generative agents as a
simulation architecture for believable social behavior. More recent work such
as AgentSociety (Piao et al., 2025), Simulating Cooperative Prosocial Behavior
with Multi-Agent LLMs (Sreedhar et al., 2025), ALYMPICS (Mao et al., 2024),
and MultiAgentBench (Zhu et al., 2025) provides important context for
multi-agent evaluation and simulation. Our closest novelty gap relative to this
literature is not the generic claim that LLMs can act socially. It is the
explicit decomposition of four measurement targets that are often conflated:
neutral baseline behavior, prompt susceptibility, benchmark recognition, and
institution-mediated behavior.

## 3. Experimental Design

### 3.1 Measurement targets

The project separates three related but distinct questions.

Baseline measurement asks what models do under minimally directive prompts.
Prompt susceptibility asks how much that behavior changes under cooperative or
competitive framing. Benchmark-recognition measurement asks whether canonical
game labels change behavior relative to unnamed or disguised isomorphs.

### 3.2 Repaired Part 1 protocol

The repaired Part 1 experiments use a stable three-model cohort:

- `cerebras:llama3.1-8b`
- `nvidia:deepseek-ai/deepseek-v3.2`
- `nvidia:moonshotai/kimi-k2-instruct-0905`

The repaired replication cohort swaps in Qwen for the Kimi model. Each game is
run over all cross-model pairings plus self-play, yielding six pairings per
game. In the repaired fast batch, each trial lasts four rounds at temperature
`0.0`, and each pairing is evaluated once per condition. For the repaired
baseline track, each game is evaluated under three neutral prompt variants,
yielding eighteen trials per game. For the repaired benchmark track, each
presentation is evaluated over the same six pairings. For the repaired
susceptibility track, each game is evaluated under `minimal-neutral`,
`cooperative`, and `competitive` prompts, again yielding eighteen trials per
game.

### 3.3 Validity repair

The most important methodological repair is that experiment logs now preserve
the full effective message stack through `messages_sent`, rather than logging
only the final task prompt. This matters because the actual model input includes
the system prompt, framing, persona, and history context. Claims about neutral
behavior are only defendable if the full effective prompt is preserved.

The paper bundle also now de-duplicates interrupted retries correctly. Whole
directory summaries and figures prefer the latest completed JSON artifact for
each logical experiment name and exclude stale partial JSONL retries.

## 4. Results

### 4.1 Game structure strongly shapes baseline behavior

The repaired stable cohort produces a clear cross-game ordering. Aggregate
cooperation is lowest in Prisoner's Dilemma, higher in Chicken, and highest in
Stag Hunt:

| Game | Trials | Cooperation A | Cooperation B | Average Payoff A | Average Payoff B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prisoner's Dilemma | 18 | 0.4722 | 0.5463 | 2.2500 | 1.8796 |
| Chicken | 18 | 0.9074 | 0.7870 | 2.5093 | 2.9907 |
| Stag Hunt | 18 | 0.9722 | 0.9907 | 3.9630 | 3.9074 |

The same qualitative ordering replicates on a second same-day cohort:
Prisoner's Dilemma at `0.4815 / 0.5370`, Chicken at `0.6852 / 0.6389`, and
Stag Hunt at `0.9722 / 0.9907`. This makes the core claim stronger than a
single-cohort result: strategic environment matters at least as much as any
simple model-wide "prosociality" label.

### 4.2 Even neutral wording is behaviorally unstable

The strongest repaired neutral-robustness result is in Prisoner's Dilemma. On
the repaired stable cohort, `minimal-neutral` yields `0.2500 / 0.3056`
cooperation, while both `minimal-neutral-compact` and
`minimal-neutral-institutional` yield `0.5833 / 0.6667`. Pooling the stable and
replication cohorts gives:

| Prompt Variant | Pooled Trials | Cooperation A | Cooperation B | Average Payoff A | Average Payoff B |
| --- | ---: | ---: | ---: | ---: | ---: |
| minimal-neutral | 12 | 0.2639 | 0.2917 | 1.6806 | 1.5417 |
| minimal-neutral-compact | 12 | 0.5833 | 0.6667 | 2.5000 | 2.0833 |
| minimal-neutral-institutional | 12 | 0.5833 | 0.6667 | 2.5000 | 2.0833 |

This result supports a strong methodological conclusion: there is no single
"prompt-free" social baseline here. Baseline behavior is a distribution over
defensible neutral phrasings, and papers that report one default cooperation
estimate without robustness checks risk overstating precision.

### 4.3 Prompt framing strongly steers behavior, but by different amounts across games

Prompt susceptibility is now finalized across the three repaired core games.
Prisoner's Dilemma is the most dramatically steerable: `competitive` yields
`0.0000 / 0.0000`, `cooperative` yields `1.0000 / 1.0000`, and
`minimal-neutral` remains much lower at `0.2500 / 0.3056`. Chicken and Stag
Hunt are also steerable, but they retain much higher residual cooperation under
competitive framing:

| Game | Prompt Variant | Trials | Cooperation A | Cooperation B |
| --- | --- | ---: | ---: | ---: |
| Prisoner's Dilemma | competitive | 6 | 0.0000 | 0.0000 |
| Prisoner's Dilemma | cooperative | 6 | 1.0000 | 1.0000 |
| Chicken | competitive | 6 | 0.6389 | 0.3333 |
| Chicken | cooperative | 6 | 1.0000 | 1.0000 |
| Stag Hunt | competitive | 6 | 0.4167 | 0.3056 |
| Stag Hunt | cooperative | 6 | 1.0000 | 1.0000 |

The safe claim is therefore not simply that prompt framing matters. It is that
prompt steerability is itself game-dependent.

### 4.4 Benchmark presentation materially changes measured behavior

The repaired benchmark track is now complete across the three core games.
Prisoner's Dilemma shows the most dramatic presentation effect. The canonical
condition yields `0.2500 / 0.3056` cooperation, the resource-disguise condition
raises that to `0.5833 / 0.6667`, and the unnamed/isomorphic condition raises
it further to `0.7500 / 0.8056`. Relative to canonical, the unnamed condition
increases cooperation by `+0.5000 / +0.5000`.

Chicken and Stag Hunt move in the opposite direction. In Chicken, canonical
framing yields `0.9167 / 0.7778`, while unnamed framing falls to
`0.4444 / 0.5833`. In Stag Hunt, canonical framing yields `0.9167 / 0.9722`,
while unnamed framing falls to `0.6667 / 0.8333`.

This pattern is theoretically important. Benchmark recognition is a real
behavioral confound, but not a monotonic one. Less canonical presentation does
not always increase cooperation. Instead, benchmark framing appears to interact
with the strategic schema a model activates for a particular game.

### 4.5 Multi-agent institutions reveal survival-sociality dissociations

The corrected multi-agent reruns now complete all three prompt conditions for
both the scarcity society and the public-reputation society. In the scarcity
society, `task-only` is perfectly stable across three repetitions (`1.0000`
final survival, zero trade, zero alliances). Both non-task prompts reduce
final survival to `0.8889 [0.8333, 1.0000]`, but in qualitatively different
ways. `Cooperative` produces the most social and unequal regime, with average
trade volume `2.1667`, alliance count `2.0000`, average gini `0.3253`, and
commons health `0.5321`. `Competitive` remains alliance-free and much less
unequal, with average trade volume `1.7083`, alliance count `0.0000`, average
gini `0.1429`, and commons health `0.3767`.

In the reputation society, by contrast, all three prompt families preserve
`1.0000` survival and final survival. Yet their social styles remain sharply
separated. `Task-only` is nearly inactive (`0.0000` trade, `0.0000`
alliances), `competitive` stays austere (`0.5000` trade, `0.0000` alliances,
`0.2960` commons health), and `cooperative` yields the most active regime
(`1.6667` trade, `2.3333` alliances, `0.6128` commons health, `0.2724`
gini). The repaired institutional battery therefore supports a stronger claim
than the earlier pilot: public reputation appears to buffer survival penalties
without erasing prompt-induced differences in social organization.

## 5. Discussion

These repaired results support three broader claims.

First, LLM social behavior is highly context-sensitive. The same models move
substantially across games, prompt framings, and benchmark presentations.
Second, neutral baseline measurement is fragile enough that it should be
treated as a robustness problem, not a single number. Third, benchmark
recognition deserves to be treated as a first-class methodological issue in
LLM social evaluation. Canonical task names can change behavior enough to alter
the substantive interpretation of a result.

Together, these findings argue for a behavioral evaluation agenda that
decomposes baseline policy, steerability, recognizability, and institutional
mediation rather than collapsing them. A model can look cooperative in one
canonical benchmark, yet defect under a different neutral paraphrase, become
far more cooperative once the benchmark label is removed, or sustain visibly
cooperative behavior only under public accountability. Those are not minor
nuisance effects. They change what the experiment is measuring.

## 6. Limitations

This manuscript has four important limitations.

First, the strongest repaired claims still come from Part 1 rather than the
full micro-to-macro program. Second, the repaired baselines use a small but
stable model cohort chosen by provider access and action-readiness constraints.
Third, the repaired runs use fast-batch settings, which improves throughput but
limits the number of rounds and repetitions per condition. Fourth, the
corrected institutional battery is still small, with only three repetitions
per prompt family in each institution.

These limitations do not undercut the main contribution. The Part 1 repaired
bundle already supports a publishable methodological and empirical paper on
baseline instability, prompt steerability, and benchmark recognition in LLM
social dilemmas, and the corrected institutional reruns now add a closed
three-prompt contrast under scarcity and public reputation. They do, however,
bound the precision of institution-level claims until the same patterns are
replicated at larger scale.

## 7. Conclusion

The repaired empirical core shows that LLM social behavior cannot be summarized
by a single cooperation score. It depends jointly on strategic environment,
neutral-prompt robustness, prompt steerability, and benchmark presentation. The
resulting picture is behaviorally richer and methodologically less naive than a
single canonical-game evaluation. For alignment research, that is the central
lesson: before asking whether models are cooperative, altruistic, or aligned in
the abstract, we must first ask which behavior we are measuring, under which
prompt, in which game, and under what institutional frame.

## References

This manuscript uses the verified reference block in
`paper/REFERENCES.md`. The most central sources are Lore and Heydari (2024),
Akata et al. (2025), Fontana et al. (2024), Schmidt et al. (2024), Capraro et
al. (2025), Park et al. (2023), Sreedhar et al. (2025), Piao et al. (2025),
Mao et al. (2024), and Zhu et al. (2025).
