# LITERATURE

## Purpose

This document is the current related-work map for `llm-altruism`.

It is meant to answer:

1. Which prior papers are most relevant to our submission story?
2. What does each paper contribute?
3. Where is our cleanest novelty gap?

This is intentionally practical and paper-facing rather than exhaustive.

---

## Core Prior Work

## 1. Strategic behavior in classical games

### Strategic behavior of large language models and the role of game structure versus contextual framing

- Venue: *Scientific Reports* (2024)
- Authors: Nunzio Lorè, Babak Heydari
- URL: `https://www.nature.com/articles/s41598-024-69032-z`

Why it matters:

- This is one of the closest direct predecessors to our Part 1 framing.
- It explicitly separates game structure from contextual framing.
- It supports the claim that prompt/context sensitivity is already a serious
  methodological issue in LLM strategic evaluation.

What it leaves open:

- It does not connect micro-level game behavior to macro-level society
  simulations under scarcity and reputation.
- It does not make benchmark-recognition / disguise effects a central design
  axis in the way our benchmark track does.

### Playing repeated games with large language models

- Venue: *Nature Human Behaviour* (2025); earlier arXiv version from 2023
- URL: `https://www.nature.com/articles/s41562-025-02172-y`
- arXiv: `https://arxiv.org/abs/2305.16867`

Why it matters:

- This is a major reference for finitely repeated-game behavior in LLMs.
- It provides a strong anchor for repeated-game methodology and for comparing
  strategic behavior against humans and hand-coded strategies.

What it leaves open:

- It is not organized around the distinction between:
  - neutral baseline behavior
  - prompt susceptibility
  - institutional / society-level behavior
- It does not appear to center benchmark contamination or prompt-stack
  validity repair as a contribution.

### Nicer Than Humans: How do LLMs Behave in the Prisoner’s Dilemma?

- arXiv (2024)
- URL: `https://arxiv.org/abs/2406.13605`

Why it matters:

- Strong Part 1 comparison point focused specifically on the Iterated
  Prisoner’s Dilemma.
- Useful for anchoring cooperation-rate comparisons and strategy-profile
  comparisons.

What it leaves open:

- It is narrower in game coverage and does not bridge into larger social
  institutions or societies.
- It does not separate canonical benchmark recognition from isomorphic
  disguise conditions.

## 2. Altruism, fairness, and human-social-preference comparisons

### GPT-3.5 altruistic advice is sensitive to reciprocal concerns but not to strategic risk

- Venue: *Scientific Reports* (2024)
- URL: `https://www.nature.com/articles/s41598-024-73306-x`

Why it matters:

- Important for dictator / ultimatum style interpretation.
- Shows that LLM outputs can track some social-preference dimensions while
  diverging from human behavior on others.

What it leaves open:

- It focuses on advice and social-preference settings rather than repeated
  strategic interaction across multiple games.
- It does not address multi-agent society outcomes or reputation.

### A publicly available benchmark for assessing large language models’ ability to predict how humans balance self-interest and the interest of others

- Venue: *Scientific Reports* (2025)
- URL: `https://www.nature.com/articles/s41598-025-01715-7`

Why it matters:

- Strong comparison point for “LLMs versus human social behavior” claims.
- Useful methodological caution: matching qualitative patterns is not the same
  as quantitatively matching human behavior.

What it leaves open:

- It evaluates prediction of human social decisions, not LLM strategic behavior
  as agentic participants under repeated interaction and institutional change.

## 3. Multi-agent cooperation, societies, and institutional design

### Simulating Cooperative Prosocial Behavior with Multi-Agent LLMs: Evidence and Mechanisms for AI Agents to Inform Policy Decisions

- arXiv (2025)
- URL: `https://arxiv.org/abs/2502.12504`

Why it matters:

- Closest direct reference for our Part 2 / Part 3 public-goods and policy
  angle.
- Shows that multi-agent LLM systems can reproduce key public-goods-game
  treatment effects.

What it leaves open:

- It focuses on public goods and policy-style interventions, not the full
  bridge from repeated two-player games to scarcity societies with reputation.
- It does not center benchmark-recognition robustness or neutral-baseline
  prompt validity.

### AgentSociety: Large-Scale Simulation of LLM-Driven Generative Agents Advances Understanding of Human Behaviors and Society

- arXiv (2025)
- URL: `https://arxiv.org/abs/2502.08691`

Why it matters:

- Strong reference for large-scale generative social simulation.
- Useful when positioning our society system as a more targeted behavioral
  alignment testbed rather than a generic social simulator.

What it leaves open:

- Its emphasis is large-scale social simulation and policy scenarios, not
  strategic micro-to-macro mapping across game theory tasks and institutional
  reputation systems.

### Generative Agents: Interactive Simulacra of Human Behavior

- arXiv (2023)
- URL: `https://arxiv.org/abs/2304.03442`

Why it matters:

- Foundational architecture reference for memory, reflection, and simulated
  social agents.
- Useful for framing our simulation architecture choices.

What it leaves open:

- It is not a game-theoretic or strategic-alignment paper.
- It does not measure cooperation, exploitation, or society-preserving
  behavior under scarcity.

## 4. Benchmarks and evaluation frameworks for multi-agent LLMs

### ALYMPICS: LLM Agents Meet Game Theory -- Exploring Strategic Decision-Making with AI Agents

- arXiv (2023), later COLING 2025 version
- URL: `https://arxiv.org/abs/2311.03220`

Why it matters:

- Very close to our general game-theory framing.
- Useful benchmark/tooling comparison for our experimental framework
  contribution.

What it leaves open:

- Our project is more focused on social dilemmas, bargaining, benchmark
  recognition, and institutional reputation rather than a broad game-theory
  platform alone.

### MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents

- arXiv / ACL (2025)
- URL: `https://arxiv.org/abs/2503.01935`
- ACL Anthology: `https://aclanthology.org/2025.acl-long.421/`

Why it matters:

- Strong evaluation-framework comparison for collaboration and competition.
- Useful for situating our work as a more behavioral-economics-oriented
  benchmark rather than a broad multi-agent capability benchmark.

What it leaves open:

- It is not centered on social-dilemma baselines, prompt susceptibility,
  benchmark contamination, or reputation-mediated survival.

---

## Our Cleanest Novelty Gap

The strongest paper positioning is not:

- “LLMs cooperate”
- “LLMs defect”
- or “LLMs can be used as agents”

Those claims are already too crowded.

The cleaner gap is:

### 1. We explicitly separate four measurement targets that are often conflated

- neutral baseline/default-policy behavior
- prompt susceptibility
- benchmark recognition / disguise effects
- institution-mediated behavior in scarcity societies and public reputation

This separation is central to the paper and should be a major novelty claim.

### 2. We bridge micro behavior and macro behavior

Our strongest conceptual contribution is the progression:

- repeated classical games
- then scarcity society
- then scarcity society with public reputation

That lets us ask whether small-game strategic tendencies predict or fail to
predict larger social-order outcomes.

### 3. We focus on society-preserving behavior, not just overt cooperation

One of our most interesting empirical angles is that:

- cooperative framing can increase visibly prosocial actions
- while still harming survival or resilience

That is stronger and more novel than a simple “cooperation goes up” story.

### 4. We treat benchmark contamination and prompt-stack validity as first-class methodological issues

Our paper should emphasize that:

- canonical benchmark naming changes measured behavior
- neutral paraphrase robustness is a real threat to validity
- full effective prompt logging is necessary for defendable claims

This methodological discipline is itself a contribution if we demonstrate it
carefully in the new reruns.

### 5. We contribute a reproducible behavioral-alignment framework

Beyond the empirical findings, the repo can support a benchmark/tooling
contribution:

- config-driven experiment definitions
- access/readiness probing
- resumable paper batches
- structured prompt/response logs
- cross-phase result summaries

---

## Submission Story That Fits The Literature

The most defensible paper story is:

1. LLM social behavior is highly context-sensitive.
2. Neutral baseline behavior must be measured with multiple paraphrases, not a
   single canonical prompt.
3. Benchmark recognition materially changes measured behavior.
4. Prompt framing strongly steers behavior.
5. Macro-level social resilience does not reduce to visibly prosocial language.
6. Reputation changes behavior asymmetrically rather than universally fixing
   cooperation.

That story is well-grounded in prior literature, but the combined experimental
progression and methodological separation still gives us a credible novelty
claim.

---

## Immediate Paper-Writing Use

Use this document to write:

- the Related Work section
- the novelty/gap paragraph in the Introduction
- the “why this benchmark is different” paragraph in the Methods/Framework
  section

Use `agent/FINDINGS.md` and new rerun artifacts for the actual empirical claims.
