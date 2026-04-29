# Notes: Style And Practice Playbook For A Stellar Datasets And Benchmarks Paper

Last updated: 2026-04-29

This document is not a paper-summary notebook. It is a writing and presentation playbook distilled from strong NeurIPS Datasets and Benchmarks papers. Use it to plan, write, revise, and package our own paper.

The core lesson is simple: a strong Datasets and Benchmarks paper does not merely report an artifact. It teaches reviewers why the artifact should exist, how it is constructed, why its design choices are trustworthy, what it reveals, and how the community can use it immediately.

## Evidence Base

The practices below were distilled from the supplied NeurIPS 2025 Datasets and Benchmarks spotlight papers and accepted spotlight papers available through OpenReview. The source papers are used only as evidence for style and practice patterns, not as content to copy.

Useful source links:

- NeurIPS 2025 Datasets and Benchmarks spotlight tab: https://openreview.net/group?id=NeurIPS.cc/2025/Datasets_and_Benchmarks_Track#tab-accept-spotlight
- OpenReview spotlight API query: https://api2.openreview.net/notes?content.venue=NeurIPS%202025%20Datasets%20and%20Benchmarks%20Track%20spotlight&limit=1000
- Supplied PDFs:
  - https://openreview.net/pdf?id=DHA9uoeMQx
  - https://openreview.net/pdf?id=4vLYwlA3X5
  - https://openreview.net/pdf?id=FS0voKxELr
  - https://openreview.net/pdf?id=YvuufwkFJY
  - https://openreview.net/pdf?id=fm77rDf9JS

## The Standard To Aim For

A stellar Datasets and Benchmarks paper should satisfy four contracts.

| Contract | Reviewer question | What the paper must make obvious |
|---|---|---|
| Problem contract | Why does this artifact need to exist? | Current practice has a named, consequential measurement or infrastructure gap. |
| Artifact contract | What exactly did we build? | The artifact has a precise scope, schema, task definition, split design, and use case. |
| Evidence contract | Why should reviewers trust it? | Construction, quality control, metrics, baselines, and diagnostics are transparent. |
| Release contract | Can the community use it? | Code, data, prompts, metadata, licenses, examples, and reproduction scripts are ready. |

If one contract is vague, reviewers will infer risk. The writing process should be organized around making all four contracts explicit on the first pass, not patched in during rebuttal.

## One-Sentence Thesis

Before writing, fill this sentence:

> Existing work cannot measure or support **[specific capability]** because it lacks **[missing properties]**; we introduce **[artifact]**, which provides **[design features]** and reveals **[main finding]** through **[evaluation protocol]**.

Examples of strong missing properties:

- Sequential interaction.
- Hidden information.
- Multimodal grounding.
- Long-context stress.
- Domain-specific failure modes.
- Fine-grained constraint following.
- Reproducible interface standardization.
- Scientifically meaningful metrics.

Avoid weak missing properties:

- More realism.
- Better diversity.
- Richer data.
- More comprehensive coverage.
- More challenging tasks.

Those words are acceptable only after they are operationalized into measurable axes.

## Choose One Protagonist

The paper can contain a dataset, benchmark, system, environment, model, and analysis, but one object must be the protagonist.

| Archetype | First sentence should sound like | Main risk if unfocused |
|---|---|---|
| Dataset | "We introduce X, a dataset of Y for Z." | Reviewers ask why the data unlocks anything new. |
| Benchmark | "We introduce X, a benchmark for evaluating Y under Z." | Reviewers ask whether the task measures a real capability. |
| Environment | "We introduce X, an interactive environment where agents must Y." | Reviewers ask whether the state/action protocol is realistic and reproducible. |
| System | "We introduce X, a system for Y at Z scale." | Reviewers ask whether it is just engineering. |
| Diagnostic study | "We introduce X to measure whether Y happens under Z." | Reviewers ask whether the phenomenon is robust and useful. |

If multiple artifacts are necessary, state the hierarchy:

> We introduce benchmark X. To support X, we release dataset Y and baseline system Z.

Do not write:

> We introduce a framework, dataset, benchmark, and analysis...

That reads like a list of disconnected deliverables.

## The Introduction Formula

Use seven short moves.

1. Field goal: what people are trying to do.
2. Current evaluation or infrastructure gap: what existing artifacts fail to measure or support.
3. Consequence: why this failure changes conclusions, blocks progress, or creates risk.
4. Artifact: what we built, with scale and scope.
5. Design principles: two to four choices that make the artifact different.
6. Findings: what the artifact reveals that prior work could not reveal.
7. Contributions: exact, concrete bullets.

Contribution bullet template:

- Artifact: "We release X, containing N examples/cases/tasks/environments across K domains, with Y annotations and Z splits."
- Protocol: "We define evaluation protocol P with metrics M and diagnostic categories C."
- Empirical finding: "Across B baselines, X reveals Y; notably, ..."
- Release: "We provide code, data, benchmark harness, prompts, metadata, and reproduction scripts at URL."

Style rules:

- The first page should name the artifact, scale, release, and strongest finding.
- The first page should not make reviewers wait for the task definition.
- The introduction should not begin with generic AI hype.
- The contribution bullets should be testable. A reviewer should be able to check each one against a section, figure, table, or release file.

## Framing The Gap

A weak gap says:

> Existing benchmarks are not realistic.

A strong gap says:

> Existing benchmarks evaluate isolated final answers, so they cannot measure whether early-stage errors propagate through later decisions.

Procedure:

1. List the closest prior datasets, benchmarks, environments, or systems.
2. Identify what each can and cannot measure.
3. Choose measurable axes that map to our thesis.
4. Build a comparison matrix.
5. Write the gap as a missing measurement capability, not as a vague quality judgment.

Good comparison axes:

- Interaction: static, single-turn, multi-turn, tool-using, sequential, adversarial.
- Information: full information, hidden information, partial observation, delayed feedback.
- Scope: domains, languages, modalities, user groups, scenario families.
- Difficulty: easy, compositional, long-tail, ambiguous, out-of-distribution.
- Evaluation: final answer, process trace, error category, calibration, robustness, cost.
- Release: data, code, prompts, schemas, harness, leaderboard, version tags.
- Validity: human validation, expert review, leakage checks, ethical review, license clarity.

Avoid comparison axes that merely flatter our artifact:

- "Ours is more comprehensive."
- "Ours is more realistic."
- "Ours is more diverse."

Replace them with measurable claims:

- "Ours covers K scenario families."
- "Ours evaluates each model over T sequential stages."
- "Ours reports performance by C capability categories."
- "Ours includes held-out splits by source, topic, and time."

## Artifact Design Principles

Strong papers explain design before construction. Do not start with scraping, scripts, or implementation details. First explain the principles.

Use this structure:

1. Design goal.
2. Why the goal matters.
3. Design choice.
4. How it appears in the artifact.
5. How it is evaluated or ablated.

Example pattern:

> Goal: measure sequential dependence.
> Design choice: later stages consume outputs from earlier stages.
> Evaluation: compare full sequential evaluation against independent-stage evaluation.

Every design principle should later reappear in at least one of:

- A task definition.
- A metric.
- A split.
- A figure.
- An ablation.
- A failure analysis.
- A limitation.

If a design principle does not reappear, it is probably prose decoration.

## Artifact Contract Checklist

Write this contract before drafting the Methods section.

For a dataset:

- Sources and provenance.
- Inclusion and exclusion criteria.
- Unit of data.
- Schema and field definitions.
- Annotation or generation process.
- Filtering and deduplication.
- PII, copyright, consent, and safety handling.
- Splits and leakage controls.
- Licenses and allowed use.
- Version, hashes, file sizes, and release layout.

For a benchmark:

- Capability axes.
- Task definitions.
- Inputs and outputs.
- Output format constraints.
- Metrics and aggregation.
- Baselines.
- Test-set access policy.
- Anti-leakage controls.
- Leaderboard and update policy.

For an environment:

- State representation.
- Observation function.
- Action space.
- Tool interface.
- Hidden information.
- Transition rules.
- Termination conditions.
- Scoring or reward.
- Trace logging and replay.

For a system:

- User-facing API.
- Core abstractions.
- Architecture layers.
- Supported workloads.
- Dependency model.
- Scalability path.
- Failure handling.
- Monitoring and debugging.
- Compatibility with existing ecosystems.

For a diagnostic study:

- Measurement target.
- Null or reference condition.
- Data and model populations.
- Controls.
- Sensitivity checks.
- Statistical tests.
- Alternative explanations.
- Failure cases.

## Section Order

Recommended order:

1. Abstract.
2. Introduction.
3. Related work and gap matrix.
4. Artifact design.
5. Construction or implementation.
6. Dataset, benchmark, or system statistics.
7. Evaluation protocol.
8. Results.
9. Diagnostics and ablations.
10. Release, reproducibility, and intended use.
11. Limitations, ethics, and broader impact.
12. Conclusion.

Why this order works:

- Reviewers first understand the need.
- Then they understand the artifact.
- Then they trust the construction.
- Then they see what the artifact reveals.
- Then they can judge whether it is usable and bounded.

Do not put the model or baseline method before the artifact contract unless the paper is primarily a methods paper.

## Abstract Style

Use four moves:

1. Gap: one sentence naming the capability current artifacts miss.
2. Artifact: one sentence naming the release, scale, scope, and task type.
3. Evidence: one sentence with the strongest result or diagnostic finding.
4. Usefulness: one sentence explaining how others can use the artifact.

Checklist:

- Includes artifact name.
- Includes scale or scope.
- Includes task or capability.
- Includes at least one empirical finding.
- Includes release signal if public.
- Avoids generic opening sentences.
- Avoids unsupported "comprehensive" claims.

Bad abstract pattern:

> Recent progress in LLMs has been remarkable. However, evaluation remains challenging. We propose a new benchmark and show it is useful.

Better abstract pattern:

> Existing benchmarks for X evaluate Y in isolation, missing Z. We introduce A, a benchmark with N tasks across K conditions designed to measure Z. Across B baselines, A reveals that models fail primarily on C despite high aggregate performance on D. We release the benchmark, harness, prompts, and analysis scripts to support reproducible evaluation of X.

## Related Work Style

Related work should make the gap undeniable.

Use three layers:

1. Closest artifacts: what a reviewer might claim already solves the problem.
2. Adjacent methods: models or systems that motivate the artifact.
3. Domain context: why the chosen metrics and constraints matter.

End the section with a compact gap statement:

> Prior work covers A and B, but does not evaluate C under D. Our artifact targets this missing evaluation setting.

Best practices:

- Include a comparison table only when the axes are meaningful.
- Do not list every prior paper in prose if the table already carries the comparison.
- Distinguish by evaluation capability, not just topic.
- Admit where prior work is stronger.
- Do not make the new artifact look magically best on every axis.

## Construction Style

Construction sections should read like a reproducible protocol.

Use this order:

1. Source selection.
2. Raw collection.
3. Parsing or normalization.
4. Filtering.
5. Deduplication.
6. Annotation or generation.
7. Validation.
8. Split construction.
9. Final artifact statistics.

For every major filter, report:

- Input count.
- Output count.
- Rule or model used.
- Reason for the filter.
- Known side effect.

For synthetic generation, report:

- Generator model.
- Prompt template.
- Sampling parameters.
- Rejection criteria.
- Acceptance rate.
- Human audit procedure.
- Failure examples.
- Ablation or sanity check.

For human annotation, report:

- Annotator qualifications.
- Instructions.
- Interface or context shown.
- Number of annotators per item.
- Disagreement resolution.
- Inter-annotator agreement or audit rate.
- Compensation or ethical handling where relevant.

## Dataset And Benchmark Statistics

Statistics should support the paper thesis, not merely fill space.

Always include:

- Total size.
- Split sizes.
- Units of data.
- Category distribution.
- Source distribution.
- Difficulty or length distribution.
- Missingness or filtering impact where relevant.

Use distributional statistics when data is skewed:

- Min.
- Max.
- Mean.
- Median.
- Quartiles.
- Mode when meaningful.
- Tail counts.

Tie every statistic to interpretation:

- "This distribution shows that the benchmark tests long-tail cases."
- "This split prevents source leakage."
- "This skew motivates reporting per-category scores."

Avoid:

- A giant statistics table with no narrative.
- Reporting only averages.
- Percentages without counts.
- Counts that do not reconcile with release files.

## Task Definition Style

Every task needs a compact contract.

For each task, specify:

- Input.
- Context.
- Allowed model actions.
- Output format.
- Metric.
- Scoring edge cases.
- Example.
- Failure modes the task is designed to expose.

Show examples before showing results. New tasks are often hard to understand from prose, especially in agentic, multimodal, or domain-specific settings.

Example block format:

```text
Input:
Expected output:
Scoring:
Capability tested:
Why this is hard:
```

For visual tasks, annotate the figure directly. For agent tasks, show a short trajectory. For scientific tasks, include the domain interpretation.

## Metric Style

Metrics should teach readers what behavior matters.

For every metric, answer:

1. What capability does it measure?
2. What real failure does it proxy?
3. What does it miss?
4. Can it be gamed?
5. Is it comparable across model families?
6. Does higher always mean better?

Use metric layers:

- Primary score: one simple headline.
- Category scores: capability or domain breakdown.
- Robustness scores: shift, difficulty, long-tail, adversarial, or degraded-input cases.
- Process scores: trajectory quality, tool use, grounding, or intermediate correctness.
- Cost scores: latency, calls, tokens, compute, or annotation cost when relevant.
- Interpretability outputs: examples, traces, failure categories, or subgroup plots.

For exact match:

- Explain whether semantically equivalent outputs exist.
- Add normalization rules.
- Include human or programmatic equivalence checks when possible.

For LLM-as-judge:

- Include the judge prompt.
- Calibrate against human labels.
- Report agreement.
- Test for bias toward specific model styles.
- Show examples of accepted and rejected judgments.
- Include a deterministic fallback metric where possible.

## Baseline Style

Baselines should make the benchmark credible, not just make our method win.

Include baseline families:

- Trivial baseline.
- Standard existing method.
- Strong open model.
- Strong closed model if appropriate.
- Domain-specific method.
- Agentic or tool-using method if relevant.
- Ablated version of our pipeline.
- Oracle or upper-bound variant where useful.

Report:

- Model versions.
- Prompts.
- Decoding settings.
- Tool access.
- Number of runs.
- Uncertainty.
- Hardware or API cost where relevant.

Use multi-run reporting for stochastic systems. Agentic and LLM evaluations should include standard deviations, confidence intervals, or bootstrap intervals unless deterministic by construction.

## Results Style

The results section should answer:

1. What is the headline performance?
2. What capability drives success or failure?
3. Which design choices matter?
4. What does the artifact reveal that prior benchmarks hide?
5. What should future researchers do differently because of this benchmark?

Recommended order:

1. Main result table.
2. Category breakdown.
3. Robustness or difficulty analysis.
4. Ablations.
5. Qualitative examples.
6. Cost or efficiency analysis if relevant.

Never stop at a leaderboard. A leaderboard answers who wins. A strong paper explains why, where, and under what conditions.

## Diagnostic Analysis

Include at least three diagnostic analyses.

Options:

- Category breakdown: which capabilities fail?
- Difficulty breakdown: when do models degrade?
- Source or domain breakdown: does performance transfer?
- Scaling analysis: does more data, context, or compute help?
- Ablation: which design choices matter?
- Failure taxonomy: what kinds of mistakes happen?
- Human comparison: what is model-specific versus generally hard?
- Robustness test: what happens under perturbation?
- Process analysis: where in the trajectory does failure start?
- Calibration analysis: are confidence and correctness aligned?

Tie diagnostics to the thesis:

- If the thesis is sequentiality, show error propagation.
- If the thesis is hidden information, show visibility ablations.
- If the thesis is long context, show length curves.
- If the thesis is domain realism, show domain-expert or domain-metric validation.
- If the thesis is interface standardization, show how interface choices affect reproducibility.

## Figure Strategy

Plan figures before drafting.

Minimum figure set:

1. Overview figure.
   - Dataset: source-to-release pipeline.
   - Benchmark: capability taxonomy and task flow.
   - Environment: state/action/observation loop.
   - System: layered architecture.
   - Diagnostic study: measurement setup and causal chain.
2. Coverage figure.
   - Domains, modalities, tasks, languages, difficulties, labels, scenarios, or users.
3. Headline result figure or table.
   - Baseline comparison, scaling curve, downstream improvement, or category performance.
4. Diagnostic figure.
   - Error breakdown, ablation, failure taxonomy, subgroup performance, trace, or qualitative case.

Figure rules:

- Every figure has one job.
- The intended takeaway is visible within 10 seconds.
- Axis labels are readable at conference size.
- Legends are short or replaced with direct labels.
- Screenshots are cropped and annotated.
- Small visual details are zoomed.
- Colors are consistent across figures.
- Captions state the conclusion, not just the contents.
- No decorative graphics substitute for evidence.

Caption template:

1. "This figure shows X under setting Y."
2. "The main takeaway is Z."
3. "Error bars/details/caveats are ...", if needed.

## Table Strategy

Strong papers use tables as arguments.

Core tables:

- Prior-work comparison table: shows the gap.
- Artifact statistics table: shows size, coverage, and skew.
- Schema or task contract table: shows fields, inputs, outputs, labels, tools, or metrics.
- Main results table: compares baseline families.
- Diagnostic breakdown table: shows categories, difficulty, domains, or failure types.
- Ablation table: proves design choices matter.
- Release table: files, versions, licenses, URLs, and usage notes.

Table rules:

- Keep column names short.
- Put units in headers.
- Put counts next to percentages.
- Include uncertainty where results vary.
- Group rows by method family or condition.
- Bold sparingly.
- Do not shrink tables until unreadable.
- If a table is too wide, split it by question.

## Writing Style

Use concrete nouns.

Prefer:

- "five-stage workflow"
- "32 UI element types"
- "three held-out source splits"
- "programmatic checks for format constraints"
- "source counts before and after deduplication"

Avoid:

- "more realistic"
- "diverse"
- "comprehensive"
- "high-quality"
- "robust"
- "real-world"

These words are allowed only when immediately backed by a measurement, example, or procedure.

Make claims auditable:

- Every adjective connects to a number, example, rubric, or validation step.
- Every capability maps to a metric or category.
- Every release claim maps to a file, URL, version tag, or script.
- Every limitation names its scope and effect.

Use short section-openers:

- "We design the benchmark around three capabilities."
- "We construct the dataset in four stages."
- "We evaluate models under two information regimes."
- "We ablate each design choice to test whether it matters."

Avoid prose that asks reviewers to trust us:

- "Carefully curated."
- "High quality."
- "Realistic scenarios."
- "Extensive experiments."

Replace with evidence:

- "Two annotators labeled each item; disagreements were resolved by an expert."
- "The benchmark includes 12 scenario families and reports category scores for each."
- "We run five seeds and report standard deviations."

## Limitations Style

Limitations should build trust by defining boundaries.

For each limitation, include:

1. What is limited.
2. Why it matters.
3. Whether it affects the main claim.
4. How the release or future work can address it.

Strong limitation:

> The benchmark focuses on desktop interfaces, so it may not cover mobile-specific interaction patterns. This does not affect our claim about desktop GUI grounding, but it limits generalization to mobile agents.

Weak limitation:

> The benchmark could be larger.

Limitations to consider:

- Source bias.
- Language bias.
- Domain coverage.
- Synthetic-data artifacts.
- Annotation uncertainty.
- LLM-judge bias.
- Leakage risk.
- Release restrictions.
- Safety and misuse.
- Cost of reproduction.
- External validity.

## Ethics And Intended Use

Ethics should be woven into artifact construction.

Include:

- Data provenance.
- Consent or license basis.
- Privacy and PII handling.
- Harmful-content handling.
- Annotation labor handling.
- Intended use.
- Out-of-scope use.
- Misuse risks.
- Release restrictions.
- Takedown or correction process if relevant.

High-stakes rule:

Do not imply deployment validation when the artifact is only a benchmark. For medical, legal, financial, safety, or human-subject domains, state the difference between evaluation utility and real-world readiness.

## Release Practices

The release is part of the paper's argument.

Before submission, the artifact should include:

- README with quick start.
- Data card or dataset card.
- Model card if models are released.
- License for paper, code, and data.
- Citation file.
- Version tag matching the paper.
- Checksums or manifest.
- Download instructions.
- Schema documentation.
- Evaluation script.
- Baseline outputs.
- Environment file or container.
- Prompt files if LLMs are used.
- Example notebook or minimal run.
- Reproduction script for main tables.
- Known issues and limitations.
- Contact or maintenance policy.

Release consistency checks:

- Paper counts match released files.
- Dataset viewer loads if using a hosted dataset.
- Code runs from a clean checkout.
- Prompts in the paper match prompt files.
- License terms are consistent across paper, data, and code.
- Version tags preserve the submitted artifact.
- Tables can be regenerated or at least audited from released outputs.

If a resource cannot be public, state why and provide the best substitute:

- Redacted subset.
- Synthetic sample.
- Access request process.
- Evaluation server.
- Hash manifest.
- Reproducible schema and scripts.

## Appendix Strategy

The appendix should be artifact documentation, not a dumping ground.

Include:

- Full prompts.
- Annotation instructions.
- Rubrics.
- Data schema.
- Full statistics.
- Extra examples.
- Additional baselines.
- Ablations.
- Failure cases.
- Human validation details.
- Reproduction commands.
- Hyperparameters.
- Compute and cost details.
- License and data statements.

Do not hide core task definitions, main metrics, or critical limitations only in the appendix.

## Reviewer Objection Pre-Mortem

Write the likely critical review before submission.

Common objections and fixes:

- Not novel: sharpen the gap matrix and specify what prior artifacts cannot measure.
- Just engineering: explain the scientific consequence of the design choices.
- Too synthetic: add human validation, real-data subset, generator-bias analysis, and ablations.
- Too small: clarify diagnostic purpose, show quality controls, or expand coverage.
- Too broad: make the taxonomy and contribution hierarchy clearer.
- Metrics are weak: calibrate metrics, add category scores, or include human agreement.
- Baselines are weak: add current strong baselines and family-level comparisons.
- Not reproducible: release exact scripts, configs, seeds, prompts, and environment files.
- Ethically risky: document provenance, consent, license, PII handling, restrictions, and intended use.
- Claims overreach: narrow the claims and state boundaries.

## Dynamic Adaptation For Our Paper

For an LLM altruism or social-behavior benchmark, use the general playbook with these topic-specific choices.

Define the construct precisely:

- Cooperation.
- Restraint.
- Altruism.
- Fairness.
- Safety refusal.
- Welfare tradeoff.
- Norm compliance.
- Resource preservation.

Separate behavior from framing:

- Vary wording.
- Vary order.
- Vary social role.
- Vary stakes.
- Vary language.
- Vary domain.
- Include neutral controls.

Make scenario design explicit:

- What does the model observe?
- What action choices are available?
- What information is hidden?
- What is the payoff or consequence structure?
- Is the task one-shot or sequential?
- Is the model asked to explain itself?
- Is the score based on action, explanation, or both?

Report diagnostic breakdowns:

- By model family.
- By instruction tuning.
- By safety tuning.
- By prompt frame.
- By game or scenario type.
- By language.
- By domain.
- By round or stage in sequential settings.

Use controls:

- Randomized ordering.
- Paraphrase sets.
- Blind scoring.
- Programmatic scoring where possible.
- Human validation of ambiguous cases.
- Sensitivity checks for prompt wording.
- Multiple runs for stochastic models.

Release:

- Prompts.
- Scenario schemas.
- Raw model outputs.
- Scoring scripts.
- Analysis notebooks.
- Model registry.
- Run manifest.
- Exclusion rules.
- Known failure modes.

Claim boundaries:

- Do not equate benchmark behavior with real moral character.
- Do not overclaim deployment behavior from artificial scenarios.
- Distinguish expressed preference, selected action, and actual interactive behavior.
- State how prompt framing limits interpretation.

## Drafting Workflow

Use this workflow for our next paper draft.

1. Write the one-sentence thesis.
2. Choose the protagonist artifact.
3. Build the prior-work comparison matrix.
4. Write the artifact contract.
5. Draft Figure 1 before writing the introduction.
6. Define tasks and metrics in contract form.
7. Build the construction pipeline with counts and validation.
8. Choose baseline families and uncertainty reporting.
9. Plan three diagnostic analyses.
10. Draft abstract and introduction.
11. Draft artifact design before construction details.
12. Draft results as answers to the thesis, not as a leaderboard.
13. Draft limitations and ethics before final polishing.
14. Prepare release files.
15. Run the reviewer objection pre-mortem.
16. Verify every claim against a table, figure, script, or release file.

## Final Pre-Submission Checklist

- The first page names the gap, artifact, scale, release, and strongest finding.
- The artifact has one dominant identity.
- Prior work comparison uses measurable axes.
- Every claimed capability appears in a task, metric, example, or ablation.
- Task definitions include inputs, outputs, context, metric, and edge cases.
- The construction pipeline reports counts before and after major filters.
- Data provenance, licenses, consent, PII, and intended use are explicit.
- Splits and leakage controls are documented.
- Metrics are justified and calibrated where needed.
- Baselines include trivial, standard, strong open, strong closed, domain, and ablated variants where appropriate.
- Results include uncertainty for stochastic evaluations.
- At least three diagnostic analyses go beyond aggregate scores.
- Figures are readable at conference size and each has one job.
- Tables include units, counts, and uncertainty where appropriate.
- The appendix contains prompts, schemas, examples, extra statistics, and reproduction details.
- Limitations are specific and do not hide material risks.
- Public code and data links work from a clean browser session.
- README quick start reproduces at least one small result.
- Paper claims match released artifact versions exactly.
- No shortcut, fake, placeholder, or filler data is used anywhere.
