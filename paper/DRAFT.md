# From Social Dilemmas to Artificial Societies:
# Measuring Baseline, Steerability, and Institutional Behavior in LLM Agents

## Abstract

Large language models are increasingly deployed as interactive agents, but
their social behavior is often evaluated in ways that blur together default
strategic tendencies, prompt obedience, benchmark familiarity, and
institution-mediated adaptation. We present a behavioral evaluation framework
that separates these factors across three linked settings: repeated classical
games, scarcity-driven multi-agent societies, and societies with public
reputation. The framework measures four distinct targets: neutral baseline
behavior, prompt susceptibility, benchmark-recognition robustness, and
institution-mediated behavior under scarcity and public accountability. In a
completed pilot batch documented in this project, we find that (i) game
structure strongly shapes behavior, with Prisoner’s Dilemma producing less
cooperation than Chicken or Stag Hunt; (ii) prompt framing can swing outcomes
from near-universal defection to near-universal cooperation; (iii) canonical
benchmark naming materially changes measured behavior; and (iv) visibly
prosocial prompting does not necessarily yield society-preserving outcomes in
larger simulations. We also identify a major methodological threat: neutral
baseline claims are only defensible when the full effective prompt stack is
logged, not only the final user-turn task prompt. We repair that validity issue
in the present codebase, complete a repaired cross-game baseline on a stable
cohort, and obtain same-day repaired replications for the strongest
Prisoner's Dilemma baseline effect. Taken together, the project argues for a
more behaviorally grounded
evaluation of LLM alignment: one that distinguishes baseline policy from
steerability and distinguishes prosocial signaling from resilient collective
outcomes.

## 1. Introduction

Language-model alignment is often discussed as a matter of refusals, safety
filters, or static benchmark performance. But many real deployments of LLMs are
social: models negotiate, advise, coordinate, compete, and participate in
institutional settings where long-horizon incentives matter. In those contexts,
alignment is partly a behavioral question. What does a model do when mutual
cooperation is profitable but individually risky? How stable is that behavior
under neutral paraphrase? Does the same model behave differently once a famous
benchmark label is removed? Does visibly cooperative language actually preserve
the health of a larger society under scarcity, or does it merely create
performative prosociality?

This paper studies those questions through a game-theoretic and multi-agent
behavioral lens. The central claim is not that LLMs are “cooperative” or
“selfish” in the abstract. Instead, we argue that LLM social behavior is
highly context-sensitive and must be decomposed into at least four distinct
measurement targets:

1. neutral baseline/default-policy behavior
2. prompt susceptibility
3. benchmark-recognition and disguise effects
4. institution-mediated behavior under scarcity and public reputation

That separation matters methodologically and substantively. If a canonical
benchmark name changes behavior, we may be measuring benchmark familiarity
rather than stable strategic tendency. If a cooperative prompt increases
cooperation, we may be measuring instruction-following rather than deep social
preference. If a public reputation system changes behavior, we may be measuring
institutional adaptation rather than intrinsic prosociality. And if a prompt
produces more sharing or alliance formation but worse survival in a scarcity
simulation, then overtly prosocial language is not the same thing as
society-preserving behavior.

Our framework therefore proceeds in three parts. Part 1 measures repeated-game
behavior across canonical social dilemmas, bargaining, and related tasks. Part
2 measures macro-level outcomes in a scarcity society where agents gather,
trade, steal, communicate, and reproduce. Part 3 adds public reputation and
asks whether accountability stabilizes cooperation or simply reshapes strategic
adaptation. The result is a bridge from micro-level strategic choice to
macro-level social order.

## 2. Contributions

The intended contribution is not a single benchmark score but a structured
behavioral measurement program.

### 2.1 Behavioral decomposition

We operationalize four measurement targets that are often conflated in prior
LLM social-behavior work:

- neutral baseline behavior
- prompt susceptibility
- benchmark recognition / disguise sensitivity
- institution-mediated behavior

### 2.2 Micro-to-macro bridge

We connect repeated two-player games to larger scarcity societies and then to
reputation-mediated societies, allowing us to test whether micro-level
strategic tendencies predict macro-level resilience.

### 2.3 Society-preserving behavior as a distinct target

We explicitly distinguish visible cooperation from resilient social order. This
is a central conceptual move: a prompt can increase sharing, trade, or alliance
formation without improving long-run survival or commons health.

### 2.4 Reproducible evaluation framework

The project provides:

- config-driven experiment definitions
- provider access and experiment-readiness probing
- resumable paper-batch execution
- structured JSON and JSONL logs
- prompt-stack logging for validity-sensitive reruns
- summary tooling with pooled prompt-condition aggregation and confidence
  intervals

## 3. Related Work

Several recent papers show that LLMs display strategic behavior in classical
games, but the literature still leaves a gap between repeated-game behavior,
prompt steerability, and institution-level social outcomes.

Lorè and Heydari (2024) show that contextual framing and game structure both
shape LLM strategic behavior, with substantial cross-model differences. This is
one of the closest direct predecessors to our Part 1 framing, but it does not
connect small-game behavior to larger social institutions or make benchmark
recognition a central design axis.

Work on repeated games with LLMs, including *Playing repeated games with large
language models* (Nature Human Behaviour, 2025), provides an important
reference point for finitely repeated interaction and strategy comparison.
However, that line of work is not organized around the separation between
neutral baseline behavior, prompt susceptibility, benchmark familiarity, and
institutional adaptation.

Other work focuses on specific social-preference tasks. *GPT-3.5 altruistic
advice is sensitive to reciprocal concerns but not to strategic risk* (2024)
and *A publicly available benchmark for assessing large language models’ ability
to predict how humans balance self-interest and the interest of others* (2025)
are especially relevant for dictator-style and advice-based interpretations.
These studies are useful for grounding fairness and altruism claims, but they
do not study repeated strategic interaction plus scarcity institutions in one
unified framework.

On the multi-agent side, *Generative Agents* (2023), *AgentSociety* (2025),
*Simulating Cooperative Prosocial Behavior with Multi-Agent LLMs* (2025),
*ALYMPICS* (2023/2025), and *MultiAgentBench* (2025) provide essential
architectural and benchmarking context. These papers show that LLM agents can
form social patterns and participate in larger systems, but they leave open a
more behaviorally aligned question: how do neutral baseline tendencies, prompt
steerability, and benchmark contamination interact with larger-scale social
resilience?

Our paper aims to fill that gap by combining: (i) repeated strategic games,
(ii) explicit prompt-susceptibility measurement, (iii) benchmark disguise
conditions, and (iv) scarcity/reputation institutions in one linked empirical
program.

## 4. Experimental Framework

### 4.1 Part 1: repeated classical games

Part 1 measures strategic behavior in repeated two-player settings such as
Prisoner’s Dilemma, Chicken, Stag Hunt, bargaining, and related tasks. The
critical design choice is to distinguish:

- neutral baseline variants
- value-laden prompt variants
- canonical versus disguised or unnamed benchmark presentations

This makes it possible to separate stable game-sensitive behavior from
benchmark familiarity and from prompt-conditioned role play.

### 4.2 Part 2: scarcity society

Part 2 moves beyond pairwise games into a larger world with gathering, resource
depletion, regeneration, stealing, trade, private messaging, and reproduction.
The key dependent variables include survival, extinction, inequality, trade,
alliance formation, and commons health.

### 4.3 Part 3: public reputation

Part 3 adds public ratings and reputation decay to test whether accountability
creates genuine stabilization, mere signaling, or asymmetric changes across
prompt conditions.

## 5. Methodological Guardrails

This paper should foreground methodological care rather than treating it as an
appendix issue.

### 5.1 Benchmark contamination

Canonical game names may activate memorized scripts, textbook equilibria, or
stock moral explanations. We therefore compare canonical, unnamed, and
disguised/isomorphic variants where possible.

### 5.2 Prompt sensitivity

There is no such thing as literally “no prompt” in this setting. The relevant
baseline is minimal, non-attitudinal, task-only prompting. Even then, neutral
paraphrases may shift behavior materially, so baseline behavior must be treated
as a distribution over defensible neutral phrasings rather than a single
canonical prompt.

### 5.3 Full prompt-stack logging

One of the most important validity repairs in this project is that experiment
logs must preserve the full effective message stack. Logging only the final
task prompt is insufficient because the model’s actual input also includes the
system prompt, framing, persona, and memory/history context. The current
codebase now logs `messages_sent` to address this issue for Part 1 reruns.

### 5.4 Institution-sensitive interpretation

More visible prosociality is not equivalent to more robust collective outcomes.
The paper should keep that distinction explicit whenever it interprets Part 2
and Part 3 results.

## 6. Current Empirical Story

The strongest completed pilot results currently support the following thesis.

### 6.1 Game structure matters

Across the completed pilot batch, Prisoner’s Dilemma is the least cooperative,
Chicken is intermediate, and Stag Hunt is the most coordination-friendly. This
already argues against a single scalar “LLM altruism” score.

The repaired triplet baseline now reinforces that point with finalized
cross-game evidence on the same cohort. Repaired Prisoner’s Dilemma averages
`0.4722 / 0.5463` cooperation, while repaired Chicken averages
`0.9074 / 0.7870` and repaired Stag Hunt averages `0.9722 / 0.9907`. The
cross-game gap is already large enough to support the claim that strategic
environment shapes behavior at least as much as any simple “prosociality”
label.

That ordering is no longer confined to one cohort. A second same-day repaired
baseline cohort that swapped in Qwen now also completes all three games and
reproduces the same cross-game ranking, with repaired Prisoner's Dilemma at
`0.4815 / 0.5370`, repaired Chicken at `0.6852 / 0.6389`, and repaired
Stag Hunt again at `0.9722 / 0.9907`. The second cohort is somewhat less
cooperative in Chicken, but it preserves the same qualitative ordering.

### 6.2 Neutral baseline wording is itself behaviorally unstable

The repaired Prisoner’s Dilemma rerun with full prompt-stack logging now gives
methodologically stronger evidence that even neutral paraphrases can shift
measured behavior materially. In the April 7, 2026 repaired triplet rerun,
`minimal-neutral` yielded cooperation rates of `0.2500 / 0.3056`, while both
`minimal-neutral-compact` and `minimal-neutral-institutional` yielded
`0.5833 / 0.6667`. This means the paper can now defend a stronger statement:
there is no single prompt-free social baseline here. Baseline behavior is a
distribution over defensible neutral phrasings, and papers that report a
single “default” cooperation estimate without robustness checks may be
overstating their measurement precision.

That result is no longer just a single repaired rerun. A second same-day
repaired PD cohort that swapped in Qwen reproduced the same qualitative split,
and pooling both repaired cohorts yields `0.2639 / 0.2917` cooperation for
`minimal-neutral` versus `0.5833 / 0.6667` for both the compact and
institutional neutral variants. This considerably strengthens the claim that
neutral-baseline wording sensitivity is a real empirical feature of the setup,
not a logging artifact or one-cohort accident.

At the same time, the repaired Chicken rerun suggests that the size of this
neutral-wording effect is itself game-dependent. All three repaired Chicken
neutral variants remain highly cooperative, with much smaller separation than
in repaired PD. That interaction is theoretically interesting: prompt wording
matters, but not in a vacuum. Its effect is mediated by the strategic tension
of the underlying game.

### 6.3 Prompt framing strongly steers behavior

In the pilot results, cooperative framing pushes behavior toward near-maximal
cooperation while competitive framing drives severe defection. This implies
that alignment-relevant social behavior is strongly steerable rather than fully
stable.

That effect is now finalized under repaired logging in Prisoner's Dilemma. In
the repaired susceptibility rerun, the cooperative condition yields
`1.0000 / 1.0000` cooperation, the competitive condition yields
`0.0000 / 0.0000`, and the minimal-neutral anchor yields only
`0.2500 / 0.3056`, all with six trials per prompt condition. This means the
same repaired setup can be pushed from universal cooperation to universal
defection by prompt framing alone.

That prompt-susceptibility story is now closed across the three repaired core
games, not only in Prisoner's Dilemma. In repaired Chicken, `competitive`
yields `0.6389 / 0.3333`, `cooperative` yields `1.0000 / 1.0000`, and
`minimal-neutral` remains high at `0.9167 / 0.7778`. In repaired Stag Hunt,
`competitive` yields `0.4167 / 0.3056`, `cooperative` again yields
`1.0000 / 1.0000`, and `minimal-neutral` remains high at `0.9167 / 0.9722`.
The safest claim is therefore not just that prompt framing matters, but that
its magnitude is itself game-dependent: Prisoner's Dilemma can be pushed from
universal defection to universal cooperation, while Chicken and Stag Hunt stay
substantially more cooperative even under competitive framing.

### 6.4 Benchmark presentation changes measured behavior

Canonical versus unnamed/disguised variants can materially change behavior, but
not always in the same direction across games. The safe claim is that benchmark
presentation matters, not that removing labels always increases cooperation.

The repaired benchmark rerun already restores the most important version of
this result. In repaired Prisoner's Dilemma, the canonical condition yields
`0.2500 / 0.3056` cooperation, the resource-disguise condition yields
`0.5833 / 0.6667`, and the unnamed/isomorphic condition yields
`0.7500 / 0.8056`. Relative to canonical, the resource framing increases
cooperation by `+0.3333 / +0.3611`, while the unnamed framing increases it by
`+0.5000 / +0.5000`. That repaired ordered pattern makes the
benchmark-recognition story stronger than a simple canonical-versus-unnamed
comparison: progressively less canonical framing produces progressively more
cooperation in Prisoner's Dilemma on the repaired cohort.

At the same time, repaired Chicken already shows that the direction of the
presentation effect is not universal. In Chicken, the canonical condition is at
`0.9167 / 0.7778` cooperation while the repaired unnamed condition is lower at
`0.4444 / 0.5833`. This reinforces the safest paper-level claim: benchmark
presentation matters, but its direction depends on the strategic schema the
framing evokes.

That cross-game benchmark table is now complete on the repaired stable cohort.
In repaired Stag Hunt, the canonical condition yields `0.9167 / 0.9722`
cooperation while the unnamed condition drops to `0.6667 / 0.8333`. Taken
together, the repaired benchmark results now support a cleaner and stronger
claim: less canonical presentation raises cooperation in Prisoner's Dilemma,
but lowers it in Chicken and Stag Hunt. Benchmark recognition is therefore a
real behavioral confound, but not a monotonic one.

### 6.5 Society-preserving behavior is not the same as prosocial wording

The corrected scarcity-society rerun now reproduces this claim on a stronger
footing than the original pilot. Across three corrected repetitions each,
`task-only` yields `1.0000` final survival with `0.0000` trade volume and
`0.0000` alliances. `Cooperative` drops to `0.8889` final survival while
raising trade volume to `2.1667`, alliance count to `2.0000`, inequality to
`0.3253`, and commons health to `0.5321`. `Competitive` also falls to
`0.8889` final survival, but does so with `1.7083` trade volume, zero
alliances, much lower inequality (`0.1429`), and lower commons health
(`0.3767`). The paper-safe claim is now sharper: in the plain scarcity
society, neither prosocial nor adversarial framing improves survival over
task-only, and the main difference is what kind of society gets produced.

### 6.6 Reputation changes behavior asymmetrically

The corrected reputation rerun now closes the full three-prompt slice. All
three prompt families preserve `1.0000` final survival. Yet the behavioral
styles remain distinct: `task-only` stays inactive with no trade and no
alliances, `competitive` stays near that baseline with `0.5000` trade volume
and `0.0000` alliances, while `cooperative` raises trade volume to `1.6667`,
alliance count to `2.3333`, and commons health to `0.6128`. Relative to the
non-reputation scarcity society, public reputation appears to buffer survival
differences without collapsing prompt-induced coordination styles. That is
more interesting than a simple “reputation fixes cooperation” story, because
it points to a specifically institution-mediated buffering effect.

## 7. Remaining Upgrades Before Submission

The main empirical blockers are now closed. What remains is mostly packaging,
venue formatting, and extra robustness if we want a stronger final version.

### 7.1 Repaired Part 1 evidence should remain the canonical record

The repaired baseline, benchmark, and susceptibility tracks should stay as the
default Part 1 evidence in any submission version. Older under-logged artifacts
are now background context, not the paper record.

### 7.2 Venue-formatting is now a packaging task

An anonymous ICML-style submission build now exists in
`paper/icml2025/llm_altruism_icml2025_submission.pdf`, generated from
the Markdown manuscript bundle by `scripts/build_icml_submission.py`. The
remaining structural work is mostly to adapt that build to the exact target
venue if we choose NeurIPS or ICLR instead.

### 7.3 Stronger institutional replications would still help

Part 2 and Part 3 are now closed across all three prompt families, but only at
three repetitions per condition. Additional reruns would tighten the
institutional confidence intervals and make the society-level claims more
publication-hardened.

### 7.4 Statistics and uncertainty should remain first-class

Confidence intervals are already in the pooled summaries. The final venue
version should preserve that uncertainty-aware presentation rather than
collapsing back to point estimates only.

### 7.5 The novelty claim should stay narrow and defensible

The paper should stay framed around the decomposition of behavioral measurement
targets plus the micro-to-macro bridge, not around generic claims that LLMs can
act socially.

## 8. Immediate Next Steps

1. Adapt the current ICML-style build to the final target venue if needed.
2. Keep the repaired Part 1 and institutional artifacts as the canonical paper
   record.
3. Run deeper Part 2 / Part 3 replications if tighter intervals are needed.
4. Normalize bibliography formatting for the chosen venue.

## 9. Positioning Sentence

The cleanest one-sentence positioning is:

> We show that LLM social behavior cannot be summarized by a single
> cooperation score because it depends jointly on neutral-prompt robustness,
> benchmark recognition, prompt steerability, and the institutions that govern
> repeated interaction under scarcity.
