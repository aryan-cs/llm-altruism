# FINDINGS

## Purpose

This file records the findings that are strong enough to guide manuscript
claims right now.

The ordering is intentional:

1. society viability findings first
2. institutional mechanism findings second
3. repeated-game findings as precursor diagnostics

## Central Paper Finding

The paper's main empirical target is whether LLM agents can build and maintain
a self-sustaining society under scarcity.

Current answer:

- the older audited society bundle suggests that higher visible sociality does
  not reliably imply better collective survival
- but that bundle was collected on the predecessor single-resource world
- the canonical society simulation is now the multi-resource ecology with
  explicit `food`, `water`, `energy`, and `health`
- no manuscript-safe central finding should be stated from the new ecology
  until the long-horizon reruns finish

## Main Institutional Findings

Status:

- predecessor evidence exists and is still useful for shaping the paper angle
- the new canonical ecology reruns are the missing blocker for final
  manuscript-safe institutional claims

Predecessor sources:

- `results/paper_ready_society_triplet/interim_summary.md`
- `results/paper_ready_reputation_triplet/interim_summary.md`

### Scarcity society without reputation

Corrected scarcity result:

- `task-only`
  - final survival: `1.0000`
  - trade volume: `0.0000`
  - alliance count: `0.0000`
- `cooperative`
  - final survival: `0.8889`
  - trade volume: `2.1667`
  - alliance count: `2.0000`
- `competitive`
  - final survival: `0.8889`
  - trade volume: `1.7083`
  - alliance count: `0.0000`

Interpretation:

- the most visibly social prompt family is not the most society-preserving one
- `cooperative` creates trade and alliances, but that extra social activity
  does not translate into the best survival outcome
- `task-only` is austere and nearly non-social, but it is the strongest
  survival baseline in scarcity

### Scarcity society with public reputation

Corrected reputation result:

- `task-only`
  - final survival: `1.0000`
  - trade volume: `0.0000`
  - alliance count: `0.0000`
- `cooperative`
  - final survival: `1.0000`
  - trade volume: `1.6667`
  - alliance count: `2.3333`
- `competitive`
  - final survival: `1.0000`
  - trade volume: `0.5000`
  - alliance count: `0.0000`

Interpretation:

- public reputation equalizes survival across the completed prompt families
- it does not collapse the societies into the same behavioral regime
- `cooperative` still produces a much denser social structure than the other
  prompt families

## Main Claims Safe To Make

These are the claims that now feel paper-safe.

1. Society-serving behavior should be measured at the level of collective
   survival and social structure, not only at the level of pairwise
   cooperation.
2. The predecessor scarcity world suggests that visibly cooperative behavior
   and survival-optimal behavior can diverge.
3. Cooperative prompting increases visible trade and alliances more reliably
   than it guarantees better collective survival.
4. Public reputation is better described as an institutional stabilizer than as
   a simple cooperation booster.
5. The final paper-grade society claim now depends on the in-progress
   multi-resource ecology reruns.

## What Part 1 Adds

Part 1 is still important, but as diagnostic support.

Primary precursor sources:

- `results/paper_ready_baseline_triplet/interim_summary.md`
- `results/paper_ready_replications/interim_summary.md`
- `results/paper_ready_benchmark_triplet/interim_summary.md`
- `results/paper_ready_susceptibility_triplet/interim_summary.md`

### Precursor result 1: game structure dominates any simple cooperation label

Stable cohort baseline ordering:

- Prisoner's Dilemma: `0.4722 / 0.5463`
- Chicken: `0.9074 / 0.7870`
- Stag Hunt: `0.9722 / 0.9907`

Replication cohort ordering:

- Prisoner's Dilemma: `0.4815 / 0.5370`
- Chicken: `0.6852 / 0.6389`
- Stag Hunt: `0.9722 / 0.9907`

Interpretation:

- the same models are not globally "cooperative" or "competitive"
- the strategic environment matters before we even reach society scale

### Precursor result 2: neutral baseline wording is not a single point estimate

Pooled PD neutral-family result across two cohorts:

- `minimal-neutral`: `0.2639 / 0.2917`
- `minimal-neutral-compact`: `0.5833 / 0.6667`
- `minimal-neutral-institutional`: `0.5833 / 0.6667`

Inferentially safe wording:

- literal-neutral and abstract-neutral prompts do not estimate the same
  baseline policy on the audited PD bundle
- exact paired sign-flip test for literal-neutral versus abstract-neutral
  family: `p = 0.03125`

### Precursor result 3: prompt framing can overwhelm default policy

PD susceptibility result:

- `competitive`: `0.0000 / 0.0000`
- `cooperative`: `1.0000 / 1.0000`
- `minimal-neutral`: `0.2500 / 0.3056`

Interpretation:

- prompt obedience is powerful enough that repeated-game behavior cannot be
  treated as a stable moral trait

### Precursor result 4: benchmark recognition is real but still underpowered

PD benchmark result:

- canonical: `0.2500 / 0.3056`
- resource disguise: `0.5833 / 0.6667`
- unnamed: `0.7500 / 0.8056`

Key caveat:

- canonical versus unnamed exact paired test remains `p = 0.0625` on six
  matched pairings

Interpretation:

- the benchmark-recognition story is strong directional evidence, not a high-
  precision estimate

## Overall Manuscript Reading

The paper should now read as:

- a study of whether LLM agents can sustain a society
- a study of how institutions shape survival, trade, alliances, and collapse
- a study that uses repeated games to explain why society claims require prompt
  robustness and benchmark robustness

It should not read as:

- a paper mainly about Prisoner's Dilemma cooperation rates
- a generic claim that LLMs are cooperative or altruistic
- a paper where Parts 2 and 3 are an afterthought to Part 1
