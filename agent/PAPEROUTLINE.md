# PAPEROUTLINE

## Purpose

This is the manuscript blueprint.

The outline is now society-first:

- the main question is whether LLM agents can sustain a society
- repeated games appear early, but as precursor measurement rather than the
  paper's destination

## Candidate Title Directions

- `Can LLM Agents Sustain a Society? Repeated Games as Precursor Probes for Collective Survival`
- `Can LLM Agents Build a Self-Sustaining Society? Measuring Survival, Social Structure, and Prompt Sensitivity in LLM Populations`
- `From Precursor Games to Artificial Societies: Evaluating Collective Survival in LLM Agent Worlds`

## Paper Structure

## Introduction

What this section must do:

- open with the society question, not with Prisoner's Dilemma
- motivate why alignment should be tested through collective survival and
  social organization under scarcity
- explain that repeated games are used as controlled precursor probes before
  macro interpretation

Core message:

- the interesting question is not whether a model defects in one benchmark
- it is whether a population of such models can build a society that does not
  collapse

## Related Work

Suggested order:

1. LLM societies and institutional design
2. repeated games and social dilemmas
3. social preference and human-comparison work
4. multi-agent evaluation frameworks

Key framing:

- larger-society papers motivate the target
- repeated-game papers motivate the precursor diagnostic layer

## Experimental Design

### Primary target

- Part 2 scarcity society
- Part 3 public reputation society

### Precursor diagnostics

- Part 1 repeated games for baseline instability, steerability, and benchmark
  recognition

### Must include

- explicit payoff matrices
- exact prompt text or prompt-stack excerpts
- clear distinction between descriptive bootstrap intervals and inferential
  paired tests

## Results

Recommended order:

### 1. Society viability is the main result

- scarcity survival, trade, alliances, inequality
- reputation survival, trade, alliances

### 2. Institutions reshape social structure more reliably than they improve survival

- `cooperative` can produce denser visible sociality without better survival
- reputation can equalize survival without equalizing behavior

### 3. Repeated games explain why society evaluation needs precursor diagnostics

- cross-game ordering
- neutral-family instability
- prompt steerability
- benchmark-recognition evidence

## Discussion

Main discussion points:

1. visible prosociality is not the same as society-preserving behavior
2. society evaluation needs precursor micro-level diagnostics
3. institutions should be judged on survival and social structure, not only on
   apparent niceness

## Limitations

Must stay explicit about:

- small model cohort
- small repeated-game matched samples
- modest institutional repetition count
- date-specific provider accessibility
- nuisance prompt-construction bug still documented in the appendix

## Appendix

Must contain:

- payoff matrices
- exact prompt text for the studied prompt families
- reproduction commands
- artifact map
- threat-to-validity notes

## Writing Guardrails

Do:

- keep society viability in the abstract and first page
- use Part 1 to support interpretation, not to dominate the paper
- keep claims tied to audited artifacts

Do not:

- let the title or abstract read like a Prisoner's Dilemma paper
- present prompt-conditioned behavior as a stable moral essence
- use "cooperative" as a synonym for "society-serving"
