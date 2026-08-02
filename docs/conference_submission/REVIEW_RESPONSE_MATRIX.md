# NeurIPS 2026 revision matrix

This file makes the revision boundary explicit. The manuscript remains the
original three-axis Prosocial Readiness Bench paper, titled exactly **Safety
Beyond Refusal**. A change belongs in the revision only when it strengthens the
original submission in direct response to the reviews below.

## Locked paper invariants

- Preserve the benchmark's three axes: multilingual safety refusal, one-shot
  focal cooperation, and repeated commons restraint.
- Preserve the central motivation: refusal alone is not a sufficient behavioral
  safety profile.
- Report the three measurements separately; do not replace the paper with a
  benchmark-audit narrative and do not claim a unitary latent prosocial trait.
- Keep the title exactly `Safety Beyond Refusal`.
- Never substitute an unavailable model, pool an identity mismatch, fabricate a
  human label, or describe a sensitivity interval as population uncertainty.

## Reviewer concern to revision mapping

| Review concern | Required response | Evidence/status | Manuscript destination |
|---|---|---|---|
| Weak or contradictory “dissociation” claim; refusal–restraint is (r=0.77) and the other intervals are wide | Replace categorical dissociation with partial coupling; report all pairwise estimates, design-appropriate intervals, and influence/rank-reversal diagnostics; use the expanded model panel only where all required axes are observed | The restored base already says “partially coupled,” foregrounds (r=0.77), and rejects independence. Large-N frontier Part 1 collection is active; cross-axis inference still requires matched Part 0/2 evidence | Abstract, Introduction, Evaluation Protocol, Diagnostics, Limitations |
| Insufficient justification for selecting and grouping the three tasks | Explain the shared cost-shifting contrast and the deliberate progression in social scope, interaction topology, temporal horizon, and feedback; explain why joint measurement is useful without claiming one construct | The restored base includes the micro-to-macro design principles and construct map; this needs a final reviewer-facing pass | Introduction and Benchmark Design |
| Older/smaller model cohort; missing GPT, Claude, and Gemini | Evaluate a broad, exact-identity frontier panel under one frozen Part 1 bank, conservative shared rate limiting, and a separate reserved judge; disclose operationally unavailable routes without substitution | Compatibility selected 81 subjects. The active rate-limited campaign covers the stable main cohort plus isolated long-latency routes; both Kimi routes and GLM 5.2 have repeat throttle evidence and remain quarantined | Experimental Setup, Frontier-Model Results, Model Registry, Limitations |
| LLM judge lacks quantitative human validation | Report a genuine blinded human audit with agreement, class-conditional error, and adjudication statistics; until completed, label automated Part 0 results as provisional and do not present corrected rates as validated | Deterministic packets and scoring code exist. Genuine human labels are still an external evidence requirement and will not be fabricated | Part 0 Method, Judge Validation, Limitations, Checklist |
| Games are toy models with limited ecological validity | State exactly which real-world mechanisms are absent; frame the games as controlled behavioral stress tests; connect each mechanism to an explicit extension rather than claiming real-world prediction | The restored base already lists missing communication, reputation, institutions, heterogeneity, shocks, and repeated-run uncertainty; add a concise deployment mapping | Task Contracts, Scope and Responsible Use |
| Requested discussion/experiments for agent systems | Explain how the three probes extend to tool-using or multi-agent systems and distinguish policy-level action measurement from open-ended agent evaluation | Related work already positions AgentBench, SOTOPIA, and GovSim; add a concrete extension protocol after evidence is available | Related Work and Scope |
| Croissant/data accessibility inconsistency | Ship the anonymous supplement with Croissant metadata and verify the documented clean-extraction workflow; obtain a stable anonymous hosted URL and hosted validation before submission | Local ZIP and metadata validation are reproducible. Stable anonymous hosting and official hosted validation remain external submission gates | Artifact Map and Checklist |

## Acceptance gate

The revision is ready only when the frontier evidence is locked and analyzed,
the paper reports uncertainty at the correct independent unit, the human judge
audit is genuine and quantitative, the anonymous artifact is hosted and
validated, and a final context-free review finds no unresolved scientific or
formatting blocker.
