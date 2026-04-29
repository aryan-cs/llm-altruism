# Notes: Writing A Stellar NeurIPS Datasets And Benchmarks Paper

Last updated: 2026-04-29

This document is a working research notebook and writing procedure distilled from NeurIPS 2025 Datasets and Benchmarks Track spotlight papers. It is intentionally practical: the goal is to produce a repeatable framework for writing a strong paper on any dataset, benchmark, evaluation suite, diagnostic study, or research infrastructure artifact.

## Contents

- Source protocol and score caveats.
- Highest-scored accepted spotlight table.
- Live research log for the five supplied papers.
- Cross-paper pattern bank.
- Lessons from the top score tier.
- Universal paper-writing procedure.
- Topic-specific adaptation guidance.
- Final pre-submission checklist.

## Source Protocol

Primary sources:

- NeurIPS 2025 Datasets and Benchmarks Track spotlight tab: https://openreview.net/group?id=NeurIPS.cc/2025/Datasets_and_Benchmarks_Track#tab-accept-spotlight
- Supplied OpenReview PDFs:
  - Data-Juicer 2.0: https://openreview.net/pdf?id=DHA9uoeMQx
  - Sheetpedia: https://openreview.net/pdf?id=4vLYwlA3X5
  - Scaling Computer-Use Grounding: https://openreview.net/pdf?id=FS0voKxELr
  - MedChain: https://openreview.net/pdf?id=YvuufwkFJY
  - AgentRecBench: https://openreview.net/pdf?id=fm77rDf9JS
- OpenReview forum and review metadata where public and accessible.

Score caveat: public OpenReview review ratings are useful signals, but they are not a perfect proxy for final acceptance quality. Scores may reflect reviewer calibration, rebuttal dynamics, confidence, novelty expectations, and track-fit judgments. For this document, score-wise rankings should be treated as a starting point, then interpreted through paper craft and review evidence.

## Highest-Scored Accepted Spotlights

Method: query OpenReview for submissions whose venue metadata is `NeurIPS 2025 Datasets and Benchmarks Track spotlight`, collect public review notes with a `rating` field, and rank by arithmetic mean of those public ratings. This found 56 accepted spotlight papers. The ranking does not include private area-chair discussion, reviewer confidence weighting, hidden review changes, or final program-committee deliberation.

Official sources:

- Accepted spotlight tab: https://openreview.net/group?id=NeurIPS.cc/2025/Datasets_and_Benchmarks_Track#tab-accept-spotlight
- API query: https://api2.openreview.net/notes?content.venue=NeurIPS%202025%20Datasets%20and%20Benchmarks%20Track%20spotlight&limit=1000

| Rank | Mean | Public ratings | OpenReview ID | Paper | Artifact signal |
|---:|---:|---|---|---|---|
| 1 | 5.75 | 5, 6, 6, 6 | `iKwHwCaddB` | [Measuring Fingerprints of Web-filtered Text Datasets and Fingerprint Propagation Through Training](https://openreview.net/forum?id=iKwHwCaddB) | Detects preprocessing fingerprints in web-filtered LLM data and traces propagation into trained model generations. |
| 2 | 5.50 | 5, 5, 6, 6 | `DgH9YCsqWm` | [MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models](https://openreview.net/forum?id=DgH9YCsqWm) | Broad MLLM benchmark over perception and cognition subtasks with manually designed instruction-answer pairs. |
| 2 | 5.50 | 5, 6, 5, 6 | `EPQi0v0OxL` | [MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly](https://openreview.net/forum?id=EPQi0v0OxL) | Long-context VLM benchmark across interleaved image-text tasks and controlled 8K-128K token settings. |
| 2 | 5.50 | 5, 5, 6, 6 | `KI8XRNjPI2` | [Augmenting Biological Fitness Prediction Benchmarks with Landscapes Features from GraphFLA](https://openreview.net/forum?id=KI8XRNjPI2) | Adds fitness-landscape topology features to biological sequence benchmarks. |
| 2 | 5.50 | 6, 6, 4, 6 | `qPMLvJxtPK` | [Gymnasium: A Standard Interface for Reinforcement Learning Environments](https://openreview.net/forum?id=qPMLvJxtPK) | Standardizes the single-agent reinforcement-learning environment API and interoperable tooling. |
| 6 | 5.25 | 4, 5, 6, 6 | `DHA9uoeMQx` | [Data-Juicer 2.0](https://openreview.net/forum?id=DHA9uoeMQx) | Scalable multimodal data-processing system with large operator suite, cloud execution, and multiple interfaces. |
| 6 | 5.25 | 6, 5, 5, 5 | `fAjbYBmonr` | [Why Do Multi-Agent LLM Systems Fail?](https://openreview.net/forum?id=fAjbYBmonr) | Annotated multi-agent traces, failure taxonomy, and LLM-judge pipeline for diagnosing failures. |
| 6 | 5.25 | 5, 6, 5, 5 | `FLiMxTkIeu` | [AGENTIF](https://openreview.net/forum?id=FLiMxTkIeu) | Agentic instruction-following benchmark with long, constraint-heavy realistic instructions. |
| 6 | 5.25 | 5, 5, 6, 5 | `SAT0KPA5UO` | [MLIP Arena](https://openreview.net/forum?id=SAT0KPA5UO) | Benchmark for interatomic potentials using physics-aware, stability, reactivity, and thermodynamic criteria. |
| 6 | 5.25 | 5, 6, 5, 5 | `GgKHKZgg0Y` | [TIDMAD](https://openreview.net/forum?id=GgKHKZgg0Y) | Ultra-long dark-matter experiment time series plus denoising metrics and physics-standard analysis tooling. |

High-score pattern:

The strongest score-wise papers frame a concrete measurement gap, then make the benchmark useful by pairing data with protocol, metrics, baselines, and reusable artifacts. They do not just release data. They define what should be measured, why current evaluation is misleading or incomplete, and what the new artifact reveals that the field could not previously inspect.

Common strengths:

- Direct problem statements and clear "why this benchmark now" positioning.
- Structure that moves from gap to artifact design to evaluation protocol to broad baselines to diagnostic analysis.
- Visuals built for inspection: task taxonomies, dataset-composition tables, leaderboard matrices, ablations, scaling curves, and failure taxonomies.
- Public code, datasets, metadata, supplementary material, and documented evaluation pipelines.
- Controlled splits, leakage mitigation, realistic scenarios, standardized interfaces or input lengths, and domain-relevant metrics.
- Reviewer-facing clarity: artifact URLs, limitations, ablations, and evidence that benchmark choices change conclusions.

## Live Research Log

### Data-Juicer 2.0

OpenReview ID: `DHA9uoeMQx`

Paper: https://openreview.net/pdf?id=DHA9uoeMQx

Forum: https://openreview.net/forum?id=DHA9uoeMQx

Core gap:

Foundation-model data pipelines need multimodal, model-assisted, large-scale processing across text, image, video, and audio. The paper frames three concrete gaps: existing tools are often text-only or modality-specific; general big-data engines do not fit foundation-model per-sample mapper/filter workloads cleanly at huge scale; and practitioners face fragmented APIs across local workflows, Hugging Face, Ray, cloud systems, and production environments.

Core artifact:

Data-Juicer 2.0 is an open-source data-processing system for foundation-model data. Its contribution is a production-grade infrastructure artifact rather than a new dataset:

- 150+ operators extending Data-Juicer 1.0 from text to image, video, audio, multimodal processing, and post-training workflows.
- A layered architecture covering capabilities, interfaces, and runtime adaptation.
- Multiple interfaces, including Python APIs, REST APIs, web tools, command tools, and natural-language or agent interaction.
- Runtime support for engine-agnostic dataset abstraction, adaptive splitting, fault tolerance, operator reordering, batching, auto resource allocation, and insight mining.
- Evaluation across small to very large settings, including Ray, MaxCompute, and Alibaba Cloud deployments up to 12,800 CPU cores.

Structure:

1. Introduction motivates multimodal data-processing gaps, gives the architecture overview, and lists contributions.
2. Design rationale positions the system against prior tools and defines design goals.
3. Processing capabilities catalog operators by modality, function, and implementation.
4. Accessibility section explains APIs, REST, web tooling, natural-language interaction, dependency management, and testing.
5. Runtime section explains dataset abstraction, operator adaptation, fault tolerance, orchestration, and insight mining.
6. Experiments report scalability, resource tradeoffs, and practical usage guidance.
7. Conclusion and limitations name concrete remaining gaps.
8. Appendices document operator lists, interface examples, implementation details, optimization details, ablations, high-resolution figures, and reproducibility information.

What it does well:

- It makes the artifact concrete immediately through an overview figure, a version-delta table, and explicit contribution bullets.
- It writes like a systems paper inside a Datasets and Benchmarks venue: problem, design goals, architecture, implementation, evaluation, deployment evidence.
- It repeatedly connects engineering choices to user pain: installation burden, dependency conflicts, out-of-memory failures, I/O bottlenecks, and backend fragmentation.
- It frames experiments as decision guidance, not just speed numbers. The reader learns when standalone, Ray, Ray-DLC, or MaxCompute is appropriate.
- It states limitations in the main paper, including Ray bottlenecks, GPU backend support, governance and safety, multilingual coverage, and larger-scale optimization.

Visual and table practices:

- Use a single overview figure that maps users, interfaces, operators, runtime, adapters, execution engines, and infrastructure.
- Put a delta table early: previous system versus new system, or existing resources versus the proposed artifact.
- Use taxonomy figures for broad artifacts. Operator distributions by modality, function, and implementation type make breadth inspectable.
- Compare workloads, engines, dataset sizes, and resource utilization instead of reporting one headline benchmark.
- Place complete asset lists in appendices with columns for modality, name, description, function type, and implementation type.

Reproducibility and release practices:

- Public code repository: https://github.com/datajuicer/data-juicer
- The paper uses public datasets for experiments and says no new dataset is released.
- Reproducibility details include dataset scales, recipes, engines, and resource settings.
- The release is treated as an artifact: documentation, demos, tests, Dockerfiles, quick-start commands, release notes, and issue/PR activity.
- The paper reports automated unit and regression testing above 85% coverage.

Reviewer-score signals found by the paper-reading agent:

- Decision: accept spotlight.
- Public review ratings found: `6`, `6`, `5`, `4`; mean `5.25`.
- Main praised signals: mature implementation, important problem, open-source release, strong documentation, broad operator coverage, detailed experiments, and real-world deployment.
- Main concerns: track fit for infrastructure, novelty clarity, figure readability, evidence balance between text-scale and multimodal-scale experiments, and dependence on Alibaba infrastructure for the largest-scale evidence.

Reusable writing practices:

- Define the gap as a workflow failure that real users experience.
- Put the artifact delta in a table early.
- Separate contribution types: artifact, technical design, evaluation, and community or release impact.
- Make breadth inspectable with taxonomies rather than adjectives.
- Include deployment evidence, but pair it with public experiments a reviewer can reproduce.
- Treat appendices as artifact documentation, not leftovers.
- State limitations concretely and technically.

### Sheetpedia

OpenReview ID: `4vLYwlA3X5`

Paper: https://openreview.net/pdf?id=4vLYwlA3X5

Forum: https://openreview.net/forum?id=4vLYwlA3X5

Core gap:

Spreadsheet intelligence is under-resourced relative to table QA and text-to-SQL. The paper makes the domain-specific argument that spreadsheets are not ordinary tables: they combine layout, cell references, formulas, implicit semantics, workbook structure, and operational business use. Existing spreadsheet corpora are too small, narrow, inaccessible, or weak in formula coverage.

Core artifact:

Sheetpedia is a spreadsheet corpus built from Enron, Fuse, and a new ExcelForum crawl. The paper reports `290,509` unique worksheets after parsing, filtering, and MinHash/LSH deduplication. It also introduces two tasks:

- `NL2SR`: map a natural-language request to a semantic spreadsheet range.
- `NL2Formula`: generate an Excel formula from a natural-language request.

The benchmark uses LLM-generated synthetic queries with rejection sampling and human review for the test set. Reported best results are `97.5%` accuracy on NL2SR and `71.7%` on NL2Formula.

Structure:

1. Introduction motivates spreadsheets as formula-rich, semi-structured data.
2. Related work distinguishes spreadsheet datasets from table datasets and formula synthesis.
3. Dataset section explains sources, format standardization, cleaning, formula filtering, language filtering, and deduplication.
4. Statistics section reports workbook, worksheet, language, and formula distributions.
5. Downstream tasks define NL2SR and NL2Formula, generation, baselines, fine-tuning, metrics, and results.
6. Conclusion and limitations summarize value and constraints.
7. Appendices include PII masking, extra tables and figures, prompts, chain-of-thought analysis, and access instructions.

What it does well:

- It gives a clear "why this domain is different" argument before introducing the artifact.
- It separates corpus contribution, benchmark contribution, and modeling demonstration.
- The data pipeline is operational: source counts, parsing tools, filtering rules, deduplication method, and thresholds are visible.
- It ties dataset statistics back to modeling implications, especially formula diversity and skewed spreadsheet sizes.
- It includes prompt templates and implementation details in the appendix, which makes synthetic-data construction inspectable.

Visual and table practices:

- Report min, max, mean, median, quartiles, and mode when distributions are skewed. A mean alone would hide spreadsheet-size pathologies.
- Use formula visualizations so the artifact feels domain-specific rather than like another generic table corpus.
- Show task examples before metrics, especially when introducing a nonstandard task such as NL2SR.
- Compare zero-shot, few-shot, and fine-tuned baselines across tasks in one clean performance table.
- Use a heatmap when the core result is a tradeoff, such as single-task versus mixed-task fine-tuning.
- Put before/after preprocessing statistics in appendices to show how cleaning changes the corpus.

Reproducibility and release practices:

- Dataset: https://huggingface.co/datasets/tianzl66/Sheetpedia_xlsx
- Code: https://github.com/TTtianTT/Sheetpedia/tree/main
- The paper reports implementation details including `pyexcel`, `openpyxl`, `lingua`, MinHash/LSH parameters, generation and scoring models, rejection thresholds, fine-tuning hyperparameters, GPU setup, and metrics.
- Release caveat: the Hugging Face visible row count observed by the paper-reading agent did not obviously match the paper's `290,509` unique worksheets. A stellar paper should make paper-release-version matching unambiguous.

Reviewer-score signals found by the paper-reading agent:

- Decision: accept spotlight.
- Public review ratings found: `5`, `5`, `5`; mean `5.00`.
- One reviewer explicitly raised the score to accept after clarification.
- The forum metadata marks the paper as flagged for ethics review, which is important because dataset source privacy and consent were central risks.

Weaknesses and caveats:

- Ethical and legal risk is material because sources include enterprise archives and public forums.
- PII masking is described, but preserving email domains and last four phone digits can still leave residual re-identification risk.
- The rejection-sampling pipeline would be stronger with acceptance rate, judge calibration, and ablations against no-rejection or human-only filtering.
- The benchmark test set is small relative to the corpus.
- Synthetic query generation may bake in generator and judge model biases.
- Exact-match formula evaluation may penalize semantically equivalent formulas.
- Cross-sheet formulas and some formula types are filtered, reducing coverage of complex spreadsheet workbooks.
- The corpus is predominantly English, limiting multilingual claims.

Reusable writing practices:

- Explain what makes the data type scientifically or operationally distinct before introducing the dataset.
- Report source counts before and after every major preprocessing step.
- Make filtering choices explicit, including data intentionally excluded and why.
- Show domain-specific examples early.
- Treat ethics as part of dataset construction, not a detached checklist.
- Include release-version checks: dataset cardinality, hashes, file manifests, licenses, schema, and scripts should reconcile exactly with paper claims.
- Provide reviewer-proof diagnostics: dedup precision checks, synthetic-data acceptance rates, human validation protocol, known failure modes, and release/version matching.

### Scaling Computer-Use Grounding

OpenReview ID: `FS0voKxELr`

Paper: https://openreview.net/pdf?id=FS0voKxELr

Forum: https://openreview.net/forum?id=FS0voKxELr

Project: https://osworld-grounding.github.io/

Core gap:

GUI grounding is a bottleneck for computer-use agents, but existing benchmarks mostly test simplified referring-expression tasks such as finding text or an icon. They under-measure software commonsense, layout hierarchy, fine cursor placement, drag, slider, table interactions, and refusal of infeasible instructions.

Core artifact:

The paper contributes a benchmark, dataset, and model stack:

- `OSWorld-G`: 564 finely annotated desktop GUI grounding samples across 32 UI element types.
- Capability axes: text matching, element recognition, layout understanding, fine-grained manipulation, and refusal.
- `Jedi`: a 4M-example multimodal computer-use grounding dataset built through UI decomposition and synthesis.
- `Jedi-3B` and `Jedi-7B` grounding models.
- Evidence that stronger grounding improves full agent performance, not only isolated coordinate prediction.

Structure:

1. Introduction motivates GUI grounding as a missing capability for reliable computer-use agents.
2. Approach defines grounding, introduces OSWorld-G, and describes Jedi data construction.
3. Experiments evaluate grounding on ScreenSpot-v2, ScreenSpot-Pro, UI-Vision, and OSWorld-G.
4. Downstream evaluation measures agentic ability on OSWorld and WindowsAgentArena.
5. Analysis studies refined instructions, data scaling, and qualitative cases.
6. Related work, conclusion, limitations, and appendices cover benchmark statistics, annotation details, cost, data pipelines, examples, failure cases, and detailed agent results.

What it does well:

- It names the missing competencies, not merely the missing benchmark: layout hierarchy, fine-grained manipulation, and infeasible-action refusal.
- It uses a benchmark taxonomy that is easy to remember and reuse.
- It connects dataset design to measurable failure modes.
- It validates the artifact at three levels: diagnostic benchmark, broader grounding benchmarks, and downstream agent success.
- Its appendix supports serious artifact review with annotation process, data statistics, cost analysis, generation pipeline details, and failure cases.

Visual and table practices:

- Use a one-screen overview figure that connects benchmark, dataset, training, and outcome.
- Make abstract capabilities tangible through annotated examples of layout understanding and fine-grained manipulation.
- Visualize the synthetic-data generation pipeline, not just its final output.
- Map capability categories to element types and sample counts in a table.
- Break performance down by capability, not only overall score.
- Use refined-instruction experiments to separate pure grounding from background task knowledge.
- Show data-scaling curves when scale is a central claim.
- Include qualitative figures with predicted click locations so error modes are visually inspectable.

Reproducibility and release practices:

- Code: https://github.com/xlang-ai/OSWorld-G
- Dataset: https://huggingface.co/datasets/xlangai/Jedi
- The project page links benchmark JSONs, original and refined instructions, evaluation scripts, data collection code, and Jedi checkpoints.
- OpenReview lists CC BY 4.0 and Croissant metadata.
- Release caveat: a reviewer noted that fine-tuning scripts were not as clear as evaluation and data-generation code.

Reviewer-score signals found by the paper-reading agent:

- Decision: accept spotlight.
- Public review ratings found: `5`, `5`, `5`, `5`, `5`; mean `5.00`.
- Reviewers praised benchmark and dataset value, scale, and relevance to GUI agents.
- Concerns included refusal performance, annotation details, cost, and disentangling planner gains from grounding gains.

Weaknesses and caveats:

- Refusal remains weak despite large-scale refusal data.
- OSWorld-G is high-quality but small, so it is diagnostic rather than exhaustive.
- The benchmark is desktop-focused.
- The data-generation pipeline depends heavily on expensive synthetic annotation from frontier models.
- Downstream agent results do not fully isolate planner versus grounder attribution.
- Better GUI grounding has safety implications because it improves automated interaction with live systems.

Reusable writing practices:

- Start from a measurement gap, not merely a model gap.
- Define capability axes that future work can reuse.
- Pair a benchmark with a training dataset when feasible: evaluation plus a path to improvement is stronger than either alone.
- Show concrete examples for every new task category.
- Report diagnostic breakdowns before aggregate leaderboards.
- Demonstrate downstream utility outside the benchmark itself.
- Include ablations that test the dataset thesis: scale, source mix, refined instructions, and compositional generalization.
- Put annotation protocol, statistics, filtering, cost, and failure cases in the appendix.
- Release original and normalized versions of benchmark data when normalization changes task interpretation.

### MedChain

OpenReview ID: `YvuufwkFJY`

Paper: https://openreview.net/pdf?id=YvuufwkFJY

Forum: https://openreview.net/forum?id=YvuufwkFJY

Core gap:

Medical LLM evaluation is still dominated by static licensing-exam or QA benchmarks, while clinical decision-making is personalized, interactive, and sequential. The key benchmark gap is error propagation: mistakes in specialty referral, history-taking, or examination can contaminate diagnosis and treatment, but isolated QA benchmarks cannot measure that chain.

Core artifact:

MedChain is a benchmark of 12,163 clinical cases from the Chinese medical site iiYi, spanning 19 specialties and 156 subcategories, with 7,338 medical images and reports. Each case is organized into five sequential stages:

1. Specialty referral.
2. History-taking.
3. Examination.
4. Diagnosis.
5. Treatment.

The paper also contributes MedChain-Agent, a multi-agent baseline with general agents, a summarizing agent, a feedback agent, and MedCase-RAG over structured case vectors.

Structure:

1. Introduction establishes why clinical decision-making differs from exam-style medical QA.
2. Related work positions against medical QA, medical agents, and interactive clinical simulations.
3. Benchmark section describes source, quality control, workflow stages, interaction environment, and metrics.
4. MedChain-Agent section describes agent roles, feedback, and retrieval.
5. Experiments compare models, existing benchmarks, ablations, and generalizability.
6. Conclusion, limitations, and appendices include standardization examples, evaluation prompts, agent prompts, benchmark comparisons, and checklist items.

What it does well:

- It has a memorable benchmark thesis: personalization, interactivity, and sequentiality.
- Each abstract property is converted into a benchmark design decision. Sequentiality means downstream stages consume prior model outputs.
- It makes error propagation the core phenomenon rather than merely reporting per-stage accuracy.
- It pairs the dataset with a baseline system designed around the benchmark's failure modes.
- It keeps the artifact central even while introducing an agent method.

Visual and table practices:

- Use one visual that makes the failure mode undeniable. In this case, error propagation across the clinical stages makes the benchmark premise concrete.
- Use a pipeline figure for the benchmark workflow.
- Use a model/system figure only after the benchmark contract is established.
- Put closed-source models, open-source models, single-agent methods, multi-agent methods, RAG, and the proposed method in one table when the comparison class is broad.
- Include ablations for the proposed system components and for the benchmark's conceptual properties. MedChain's personalization, sequentiality, and interactivity ablations are especially reusable.

Reproducibility and release practices:

- Dataset: https://huggingface.co/datasets/ljwztc/MedChain
- Code: https://github.com/ljwztc/MedChain
- Public materials include framework folders, doctor-patient interaction code, prompts, utilities, `main.py`, and requirements.
- The paper gives split ratio, model families, deployment framework, temperature setting, GPU type, and Chinese-language experiment setting.
- The appendix includes prompts and examples, which is essential for agent benchmark reproducibility.
- Release caveat: the Hugging Face dataset card appeared sparse to the paper-reading agent, and public license surfaces differed across paper, data, and code. A stronger release would reconcile paper license, data license, code license, data viewer schema, and exact version.

Reviewer-score signals found by the paper-reading agent:

- Decision: accept spotlight.
- Public review ratings found: `6`, `4`, `5`; mean `5.00`.
- Praised signals: large-scale benchmark, full sequential clinical workflow, multimodal cases, practical relevance, and useful multi-agent baseline.
- Concerns: baseline comparability, novelty relative to prior clinical-agent datasets, insufficient early detail on the interactive environment, and single-source scraped data.

Weaknesses and caveats:

- Data comes from a single Chinese medical website, so generalization across countries, hospitals, languages, and documentation styles is uncertain.
- Specialty distribution is imbalanced.
- The interaction environment uses a patient simulator, which creates consistency but may underrepresent real patient variability.
- The benchmark is clinically inspired and should not be treated as real clinical deployment validation.
- The interactivity claim needs precise operational detail because reviewers will challenge vague realism claims in high-stakes domains.

Reusable writing practices:

- State the benchmark gap as a small set of memorable missing properties.
- Turn each property into an explicit task, interaction rule, metric, or ablation.
- Include one figure that makes the central failure mode visible.
- Report dataset scale, coverage, modalities, splits, and quality control early.
- Use ablations to validate benchmark design, not only model design.
- Compare against several baseline families.
- For high-stakes domains, separate benchmark usefulness from deployment readiness.
- Put prompts, examples, rubrics, and benchmark comparisons in appendices.
- Keep the artifact as the protagonist; methods should support the benchmark story.

### AgentRecBench

OpenReview ID: `fm77rDf9JS`

Paper: https://openreview.net/pdf?id=fm77rDf9JS

Forum: https://openreview.net/forum?id=fm77rDf9JS

Core gap:

Agentic recommender systems can query environments, reason over context, use memory, and adapt to changing user information, but existing recommender benchmarks are mostly static train/test datasets. Static evaluation cannot fairly measure autonomous, interactive LLM-based recommenders.

Core artifact:

AgentRecBench provides an interactive textual recommendation simulator built from Yelp, Goodreads, and Amazon data. It includes standardized query tools over a user-review-item network, data-visibility control, and three scenarios:

- Classic recommendation.
- Evolving-interest recommendation.
- Cold-start recommendation.

The paper also defines a modular agent framework with planning, reasoning, tool use, and memory, and evaluates traditional, deep-learning, and agentic methods.

Structure:

1. Motivation and contributions.
2. Formal definition of agentic recommender systems.
3. Textual environment simulator and data-visibility control.
4. Experimental setup covering datasets, scenarios, baselines, and HR@N.
5. Results for classic, cold-start, and evolving-interest settings.
6. Related work against recommender agents and general agent benchmarks.
7. Conclusion, limitations, and appendix.

What it does well:

- It defines the new system class before benchmarking it.
- It makes the "why static evaluation fails" argument explicit.
- It turns evaluation into an environment contract with users, items, reviews, tools, hidden information, scenarios, and metrics.
- It uses scenario design to mirror real deployment stresses: standard recommendation, temporal drift, and cold start.
- It compares old and new paradigms side by side rather than only comparing agent variants.
- It uses challenge/community validation as adoption evidence.

Visual and table practices:

- Use a compact simulator overview figure.
- Show dynamic data-visibility control visually when hidden information is central.
- Include a schema table for user, item, and review fields.
- Separate method families, datasets, model backbones, and scenarios in result tables.
- Include a case-study figure for a strong system so readers learn design guidance, not just rankings.

Reproducibility and release practices:

- Dataset: https://huggingface.co/datasets/SGJQovo/AgentRecBench
- Challenge page: https://tsinghua-fib-lab.github.io/AgentSocietyChallenge/pages/overview.html
- OpenReview lists public data and supplementary material.
- The paper reports leaderboard and challenge validation, including many teams and submissions.
- Release caveat: the dataset viewer reportedly showed a JSON/schema issue when inspected by the paper-reading agent. For a stellar release, benchmark files should load cleanly in the default public viewer.

Reviewer-score signals found by the paper-reading agent:

- Decision: accept spotlight.
- Public review ratings found: `4`, `5`, `5`; mean `4.67`.
- Decision summary reportedly noted concerns addressed after rebuttal, including added multi-run experiments with standard deviations and newer baselines.

Weaknesses and caveats:

- Reviewers flagged limited initial statistical rigor.
- Baselines were initially viewed as older or insufficiently agentic.
- Details on benchmark construction challenges, module interactions, memory implementation, and dataset-specific failure analysis could be stronger.
- The paper's own limitations include text-only data, single-agent focus, and future need for more traditional, deep, multimodal, and multi-agent baselines.
- Long-term benchmark maintenance details are relatively light compared with the challenge/adoption claims.

Reusable writing practices:

- Define the evaluated system class before the benchmark.
- Treat the environment as the artifact: specify tools, state, hidden information, actions, observations, termination, and scoring.
- Organize scenarios around real deployment failures.
- Include a schema table.
- Compare old and new paradigms in the same result frame.
- Add a case study of a strong system to translate results into design lessons.
- Distinguish related work by evaluation capability rather than topic alone.
- Include public dataset, code or supplement, leaderboard, challenge evidence, and explicit maintenance notes.

## Cross-Paper Pattern Bank

Observed patterns:

- Strong Datasets and Benchmarks papers make the artifact's contract explicit: task, data, metrics, split design, annotation or construction procedure, expected inputs and outputs, and intended use.
- Reviewers reward diagnostic value beyond aggregate leaderboards: taxonomies, error categories, capability axes, ablations, scaling curves, subgroup analysis, and qualitative examples.
- The best visuals are not decorative. They are inspection tools: pipeline diagrams, taxonomy diagrams, dataset composition plots, task schemas, leaderboard matrices, failure taxonomies, and examples with annotations.
- Release credibility matters. Code, data cards, licenses, model lists, versioning, benchmark servers, evaluation scripts, reproducibility checklists, and maintenance plans all strengthen the paper.
- A stellar paper tells the reviewer exactly why the artifact should exist now, why existing resources fail, how the new artifact was built, how quality was controlled, what it reveals, and how others can use it.

## Lessons From The Top Score Tier

### Fingerprints Of Web-Filtered Text Datasets

Reusable lesson: a diagnostic paper should turn a broad worry into a simple, falsifiable measurement. In this case, dataset provenance and filtering are converted into a classification and propagation question: can filtered datasets be distinguished, and do those fingerprints survive model training?

Best practice:

- Make the measurement question restatable in one sentence.
- Show the phenomenon in the source data first.
- Then show whether it survives downstream processing.
- Include sensitivity analysis and human comparison where interpretation could be contested.

### MME

Reusable lesson: a benchmark becomes memorable when its taxonomy is memorable. MME's perception-versus-cognition framing makes a broad multimodal benchmark easy to understand and easy to cite.

Best practice:

- Put the taxonomy near the beginning.
- Use manually constructed or carefully verified examples when leakage is a risk.
- Report both aggregate and category-level scores.
- Make the input/output format simple enough for broad model coverage.

### GraphFLA

Reusable lesson: domain-science Datasets and Benchmarks papers should map every feature or metric to a scientific question. GraphFLA is not just "more features"; it adds fitness-landscape topology features that help explain biological prediction benchmarks.

Best practice:

- Ground metrics in domain literature.
- Show how the artifact integrates with existing scientific workflows.
- Use domain meaning, not generic ML convenience, to justify design choices.

### MMLongBench

Reusable lesson: when the target capability varies continuously, design the benchmark to expose curves. Long-context evaluation is more persuasive when readers see performance over explicit context lengths rather than one fixed setting.

Best practice:

- Control the variable of interest explicitly.
- Report trend lines, not only point estimates.
- Include diverse task types so the capability is not reduced to one proxy.
- Evaluate recent strong models so the results remain useful at publication time.

### Gymnasium

Reusable lesson: infrastructure papers should justify interface decisions as scientific controls. Gymnasium's value is not only convenience; standard APIs reduce experimental ambiguity and reproducibility errors.

Best practice:

- Explain why API details affect scientific conclusions.
- Show compatibility, documentation, and adoption.
- Treat usability and standardization as part of reproducibility.
- Include migration or interoperability guidance.

### Multi-Agent LLM Failure Analysis

Reusable lesson: for failure papers, the taxonomy is the artifact. A strong failure taxonomy makes errors nameable, countable, reproducible, and actionable.

Best practice:

- Define failure categories precisely.
- Validate annotation reliability.
- Show examples for each category.
- Connect diagnosis to interventions or improved systems.

### AGENTIF

Reusable lesson: constraint-following benchmarks should make constraints first-class objects. The benchmark should reveal which instruction types fail, not only whether the final answer looks right.

Best practice:

- Decompose instructions into formatting, semantic, tool, temporal, and interaction constraints.
- Use programmatic checks where possible.
- Validate LLM-assisted checks instead of assuming they are correct.
- Break results down by constraint family.

### MLIP Arena

Reusable lesson: if standard metrics are insufficient, replace them with metrics tied to real failures. MLIP Arena evaluates interatomic potentials on physics-aware, stability, reactivity, and thermodynamic criteria rather than only static error.

Best practice:

- Explain what real-world failure each metric captures.
- Include extreme or difficult conditions if they matter in practice.
- Make the platform accessible to domain and ML researchers.

### TIDMAD

Reusable lesson: scientific datasets are strongest when the metric connects to the scientific quantity of interest. TIDMAD ties time-series denoising to dark-matter analysis rather than generic reconstruction alone.

Best practice:

- Provide the path from raw data to scientific interpretation.
- Use real experimental data or a clearly justified simulation/calibration process.
- Release tooling that lets ML researchers contribute without reinventing domain pipelines.

## Universal Procedure For Writing The Paper

The procedure below is deliberately artifact-agnostic. Use it for a dataset, benchmark, interactive environment, diagnostic study, or system paper by swapping the dominant artifact type while keeping the same contracts.

### Step 1: Choose The Dominant Artifact

Pick one primary identity. A paper can contain multiple contributions, but one must be the protagonist.

| Archetype | Primary artifact | Central reviewer question | Required evidence |
|---|---|---|---|
| Dataset | Corpus, annotations, splits, metadata | Is the data novel, high-quality, useful, and ethically releasable? | Source analysis, collection pipeline, quality checks, statistics, downstream tasks. |
| Benchmark | Tasks, metrics, protocols, leaderboard | Does this measure an important capability better than prior benchmarks? | Taxonomy, task examples, baselines, reliability checks, failure analysis. |
| Environment | Simulator, interaction protocol, tools | Does this expose realistic agent behavior? | State/action/observation contract, scenario variants, agent baselines, traces. |
| System | Library, platform, infrastructure | Does this solve a real bottleneck at useful scale? | Architecture, comparison to prior systems, throughput, latency, cost, scaling, adoption. |
| Diagnostic study | Measurement method and findings | Does this reveal a robust phenomenon? | Controlled setup, sensitivity analysis, statistical tests, cross-model or cross-data validation. |

Decision rule: write the first sentence in the dominant archetype's language.

- Dataset: "We introduce X, a dataset of Y designed to support Z."
- Benchmark: "We introduce X, a benchmark for evaluating Y under Z conditions."
- Environment: "We introduce X, an interactive environment for measuring agents' ability to Y."
- System: "We introduce X, a system for Y that supports Z at scale."
- Diagnostic study: "We study X and introduce Y to measure how Z occurs."

Avoid the unfocused version: "We introduce a framework, dataset, benchmark, and analysis for X." If multiple artifacts are real, rank them: "We introduce benchmark X; to support it, we release dataset Y and baseline system Z."

### Step 2: Write The One-Page Argument

Before drafting, fill these fields in plain language:

1. Current practice: what researchers or practitioners do today.
2. Hidden failure: what current practice cannot reveal or support.
3. Consequence: why that failure matters scientifically, practically, or ethically.
4. Artifact: what you built and exactly what is released.
5. Design principle: the core idea that makes the artifact different.
6. Evidence: the strongest three empirical facts.
7. Adoption path: how another researcher uses the artifact tomorrow.
8. Boundary: what the artifact does not solve.

This becomes the introduction spine. If one field is weak, fix the work before writing more prose.

### Step 3: Operationalize The Gap

Weak gap:

"Existing benchmarks are not realistic."

Strong gap:

"Existing benchmarks omit sequential interaction, hide tool-use decisions, and score only final answers, so they cannot distinguish models that recover from early mistakes from models that fail silently."

Procedure:

1. List the 5 to 10 most relevant prior datasets, benchmarks, environments, or systems.
2. Build a comparison matrix with columns for missing capabilities.
3. Use domain-specific axes rather than generic claims.
4. Mark the new artifact honestly. It does not need to win every column.
5. Convert the matrix into prose: "Prior work covers A and B, but leaves C and D undermeasured."

Good axes are measurable:

- Scale: examples, tokens, users, episodes, modalities, domains, languages, timesteps.
- Fidelity: real versus synthetic, interactive versus static, sequential versus one-shot, hidden-state versus full-information.
- Coverage: task types, domain categories, difficulty bands, user populations, perturbation families.
- Robustness: distribution shift, adversarial cases, long-tail groups, degraded observations.
- Usefulness: downstream fine-tuning gains, diagnostic granularity, leaderboard stability, deployability.
- Reproducibility: licenses, code, schemas, deterministic splits, evaluation harness.

Avoid words such as "comprehensive," "realistic," "challenging," "diverse," or "high-quality" unless you immediately define how they are measured.

### Step 4: Design The Artifact Contract

Write the contract before writing the Methods section.

For a dataset:

- Sources and provenance.
- Inclusion and exclusion criteria.
- Unit of data: document, episode, case, trajectory, sample, environment state, or user-item interaction.
- Schema and field definitions.
- Annotation or generation process.
- Filtering, deduplication, PII handling, and safety processing.
- Splits and leakage controls.
- Licenses, consent, allowed use, and restrictions.
- Version, hashes, size, and release layout.

For a benchmark:

- Capability axes.
- Task definitions.
- Inputs, outputs, and allowed formats.
- Metrics and what each metric means.
- Aggregation rules.
- Baselines and reference implementations.
- Test-set access policy.
- Anti-leakage and anti-overfitting controls.
- Leaderboard and update policy.

For an environment:

- State representation.
- Observation function.
- Action space and tool calls.
- Hidden information.
- Transition rules.
- Termination conditions.
- Reward or scoring.
- Scenario generation.
- Trace logging and replay.

For a system:

- User-facing API.
- Core abstractions.
- Architecture layers.
- Dependency model.
- Scalability path.
- Failure handling.
- Configuration surface.
- Monitoring, debugging, and reproducibility.
- Compatibility with existing ecosystems.

For a diagnostic study:

- Measurement target.
- Null hypothesis or reference condition.
- Data and model populations.
- Controls.
- Sensitivity checks.
- Statistical tests.
- Failure cases and alternative explanations.

### Step 5: Build Quality Control Into The Story

Quality control should appear in the main paper, not only the checklist.

Required evidence for most Datasets and Benchmarks papers:

- Source counts before and after each major filter.
- Deduplication method and threshold.
- Leakage checks against training, validation, and test sources where relevant.
- Annotation instructions and quality checks.
- Inter-annotator agreement or human verification rate when labels are human-generated.
- Synthetic-data validation when LLMs or heuristics generate examples.
- PII, copyright, consent, and harmful-content handling.
- Known exclusions and their implications.

Synthetic data rule: never present synthetic generation as quality by itself. Show generator prompt, rejection criteria, acceptance rate, human audit, failure modes, and at least one ablation or sanity check.

Human annotation rule: report who annotated, what they saw, how disagreements were resolved, and what quality thresholds were used. If annotator identity cannot be disclosed, disclose qualifications and process.

### Step 6: Design Metrics That Teach

Metrics should expose behavior, not just produce a leaderboard.

For every metric, answer:

1. What capability does it measure?
2. What real failure does it proxy?
3. What does it miss?
4. Can it be gamed?
5. Is it comparable across models?
6. Does higher always mean better?

Use multiple metric layers:

- Primary score: one simple headline.
- Category scores: capability or domain breakdown.
- Robustness scores: shift, difficulty, long-tail, or stress cases.
- Efficiency scores: latency, cost, resource use, or sample efficiency when relevant.
- Interpretability outputs: failure categories, examples, traces, or subgroup plots.

For LLM-judge metrics, include calibration:

- Judge prompt.
- Human agreement.
- Inter-judge agreement if multiple judges are used.
- Bias checks across model families.
- Examples of accepted and rejected judgments.
- A fallback deterministic metric where possible.

For exact-match metrics, explain whether semantically equivalent outputs are possible and how they are handled.

### Step 7: Choose Baselines Like A Reviewer

A weak baseline section asks, "Can our method win?"

A strong baseline section asks, "What should a skeptical reader compare this artifact against?"

Baseline families:

- Trivial baseline: random, majority, nearest neighbor, keyword, simple heuristic.
- Existing benchmark standard: the method people already use.
- Strong closed model or proprietary model if appropriate and allowed.
- Strong open model or reproducible model.
- Specialized domain method.
- Ablated version of the proposed artifact.
- Oracle or upper-bound variant where useful.

For benchmark papers, the proposed model should not overshadow the benchmark. The baseline story should show what the benchmark reveals, not only that one system wins.

Always include uncertainty when results could vary: standard deviation, confidence interval, bootstrap interval, or repeated runs. Agentic and LLM evaluations especially need multi-run reporting.

### Step 8: Plan The Main Figures Before Writing

Minimum figure plan:

1. Figure 1: artifact overview.
   - Dataset: source-to-release pipeline.
   - Benchmark: capability taxonomy and task flow.
   - Environment: state/action/observation loop.
   - System: layered architecture.
   - Diagnostic study: measurement setup and causal chain.
2. Figure 2: coverage or composition.
   - Domains, modalities, tasks, languages, difficulty, labels, scenarios, or users.
3. Figure 3: headline result.
   - Baseline comparison, scaling curve, downstream improvement, or category table.
4. Figure 4: diagnostic analysis.
   - Error breakdown, ablation, failure taxonomy, subgroup performance, trace example, or qualitative case.

Figure rules:

- Every figure must have one job.
- The claim should be visible within 10 seconds.
- Axis labels must be readable when printed at conference size.
- Use direct labels where possible instead of dense legends.
- Avoid tiny screenshots unless the annotation is zoomed.
- Do not use decorative graphics in place of evidence.
- Captions should state what is shown and what the reader should conclude.

Caption template:

1. "This figure shows X under setting Y."
2. "The main takeaway is Z."
3. "Error bars/details/caveats are ...", if needed.

### Step 9: Plan The Tables

Strong D&B papers usually need these tables:

- Prior-work comparison table: shows the gap.
- Dataset or benchmark statistics table: shows size, coverage, and skew.
- Schema or task contract table: shows fields, inputs, outputs, labels, tools, or metrics.
- Main results table: compares baseline families.
- Diagnostic breakdown table: shows categories, difficulty, domains, or failure types.
- Ablation table: proves design choices matter.
- Release table: files, versions, licenses, URLs, and usage notes, often in appendix.

Table style:

- Keep column names short.
- Put units in headers.
- Sort rows by meaningful groups, not arbitrary order.
- Bold only the best or most relevant result; do not overdecorate.
- Include standard deviations when appropriate.
- Put exact sample counts next to percentages.
- If a table is too wide, split by question rather than shrinking font until unreadable.

### Step 10: Write The Paper In A Reviewer-Friendly Order

Recommended section skeleton:

1. Abstract.
2. Introduction.
3. Related work and gap matrix.
4. Artifact design.
5. Construction or implementation.
6. Dataset/benchmark/system statistics.
7. Evaluation protocol.
8. Results.
9. Diagnostic analysis and ablations.
10. Release, reproducibility, and intended use.
11. Limitations, ethics, and broader impact.
12. Conclusion.

If page limits force compression, preserve artifact design, evaluation protocol, and limitations. Move extended examples, prompts, proofs, and extra baselines to appendix.

### Step 11: Write The Abstract With Four Moves

Use this structure:

1. Capability gap: one sentence.
2. Artifact: one sentence with size, scope, and release type.
3. Evidence: one sentence with the strongest result or diagnostic finding.
4. Usefulness and release: one sentence explaining how the community can use it.

Avoid abstract mistakes:

- Do not begin with generic field hype.
- Do not spend half the abstract on model performance.
- Do not claim "comprehensive" without specifying dimensions.
- Do not hide the artifact size, modality, domain, or task.

### Step 12: Write The Introduction As A Funnel

Paragraph plan:

1. The field need: what people are trying to do.
2. The measurement or infrastructure failure: why current artifacts are insufficient.
3. The consequence: what wrong conclusion or blocked workflow results.
4. The proposed artifact: what it is, with numbers.
5. Design principles: the 2 to 4 choices that make it different.
6. Findings: what the artifact reveals.
7. Contributions: exact, concrete bullets.

Contribution bullet template:

- Artifact: "We release X, containing N examples/cases/tasks/environments across K domains/modalities, with Y annotations and Z splits."
- Protocol: "We define evaluation protocol P with metrics M and diagnostic categories C."
- Empirical finding: "Across B baselines, X reveals Y; notably, ..."
- Release: "We provide code, data, benchmark harness, prompts, metadata, and reproducibility scripts at URL."

### Step 13: Make Related Work Do Real Work

Related work should not be a bibliography dump. It should justify why the artifact is needed.

Use three layers:

1. Closest artifacts: datasets, benchmarks, systems, or environments that a reviewer might say already solve the problem.
2. Adjacent methods: models or algorithms that use or motivate the artifact.
3. Domain context: what domain experts care about and what metrics are meaningful.

End related work with the gap matrix or a paragraph that says exactly what remains missing.

### Step 14: Write Artifact Design Before Construction Details

Do not start with scraping, annotation, or implementation. First explain design goals.

Design section template:

1. Goal 1: capability coverage.
2. Goal 2: realism or fidelity.
3. Goal 3: reproducibility and accessibility.
4. Goal 4: diagnostic value.
5. How each goal maps to artifact choices.

Then construction details can follow naturally:

- Data sources.
- Pipeline.
- Annotation or generation.
- Filtering and validation.
- Final statistics.

### Step 15: Show Examples Early

Every new task needs an example before its metric. Examples are the fastest way to prevent reviewer confusion.

Good examples include:

- Input.
- Expected output.
- Metadata or context.
- Why the example is hard.
- How it is scored.
- The relevant capability category.

For visual or multimodal tasks, annotate the example directly. For agent tasks, show a short trajectory. For scientific tasks, show the domain interpretation.

### Step 16: Make Results Diagnostic

A leaderboard answers "who wins." A stellar paper answers "what does the artifact reveal?"

Include at least three diagnostic analyses:

- Category breakdown: which skills fail?
- Difficulty breakdown: where do models degrade?
- Source/domain breakdown: does performance transfer?
- Scaling analysis: does more data, context, or compute help?
- Ablation: which design choices matter?
- Failure taxonomy: what kinds of mistakes happen?
- Human comparison: what is model-specific versus generally hard?
- Robustness: what happens under perturbation or distribution shift?

Tie each diagnostic result to the paper thesis. If the thesis is sequentiality, show error propagation. If the thesis is hidden information, show visibility ablations. If the thesis is long context, show length curves.

### Step 17: Treat Limitations As Design Boundaries

Limitations should be specific enough that they build trust.

Strong limitations:

- "The benchmark is desktop-focused and may not cover mobile UI conventions."
- "The corpus is predominantly English, so multilingual claims are limited."
- "The patient simulator standardizes interaction but may underrepresent real patient variability."
- "Large-scale experiments rely on cloud infrastructure that external users may not replicate exactly."

Weak limitations:

- "More work is needed."
- "The dataset could be larger."
- "There may be bias."

For each limitation, include:

1. What is limited.
2. Why it matters.
3. Whether it affects the main claim.
4. How future work or current release design can address it.

### Step 18: Make The Release Review-Proof

Before submission, the public artifact should include:

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
- Reproducibility script for main tables.
- Known issues and limitations.
- Contact or maintenance policy.

If a resource cannot be public, state why and provide the strongest possible substitute: synthetic sample, access process, hash manifest, evaluation server, or redacted subset.

### Step 19: Reviewer Objection Pre-Mortem

Before submission, write the likely critical review yourself.

Common objections and fixes:

- "Not novel": sharpen the gap matrix and explain what existing artifacts cannot measure.
- "Just engineering": explain the scientific control or measurement consequence of design choices.
- "Too synthetic": add human validation, real-data subset, generator bias analysis, and ablations.
- "Too small": clarify diagnostic purpose, show quality, or expand coverage.
- "Too broad": make the taxonomy and contribution hierarchy clearer.
- "Metrics are weak": calibrate metrics, add category scores, or include human agreement.
- "Baselines are weak": add current strong baselines and family-level comparisons.
- "Not reproducible": release scripts, exact configs, seeds, and environment.
- "Ethically risky": document provenance, consent, license, PII handling, restrictions, and intended use.
- "Claims overreach": narrow language and state boundaries.

### Step 20: Style Rules

Use concrete nouns:

- Prefer "five-stage clinical workflow" over "more realistic medical evaluation."
- Prefer "32 UI element types" over "diverse GUI elements."
- Prefer "source counts before and after MinHash/LSH deduplication" over "careful cleaning."

Make claims auditable:

- Every adjective should connect to a number, example, or procedure.
- Every benchmark axis should have a metric or category.
- Every release claim should have a URL, file, script, or manifest.

Use short section-openers:

- "We design OSWorld-G around five grounding capabilities."
- "We construct Sheetpedia in four stages."
- "We evaluate MedChain under three workflow properties."

Avoid:

- Marketing language.
- Overclaiming "human-level," "real-world," or "comprehensive."
- Burying the artifact behind model details.
- Reporting only aggregate scores.
- Tiny unreadable figures.
- Appendix-only reproducibility.
- Placeholder or fake data. If the evidence does not exist, collect it or narrow the claim.

### Step 21: Dynamic Adaptation By Topic

For an LLM altruism or social-behavior benchmark:

- Define the behavior precisely: cooperation, restraint, altruism, fairness, safety refusal, welfare tradeoff, or norm compliance.
- Separate prompt framing from latent model preference.
- Include multiple domains and scenario variants.
- Use controls for wording, order, language, and role framing.
- Report model-family, instruction-tuning, and safety-tuning differences.
- Include human baseline or expert interpretation if possible.
- Release prompts, scoring rubrics, raw outputs, and analysis scripts.

For a clinical or high-stakes benchmark:

- Avoid deployment claims.
- Include domain-expert review.
- Separate educational realism from clinical validity.
- Report source limitations and population coverage.
- Include safety, privacy, and intended-use restrictions.

For an agent environment:

- Specify state, observation, action, tools, memory, hidden information, termination, and scoring.
- Release traces and replay tools.
- Evaluate planner, tool-use, memory, and perception separately when possible.
- Include multi-run variance.

For a scientific dataset:

- Tie metrics to the domain quantity of interest.
- Include domain baselines.
- Explain units, physical constraints, and invalid outputs.
- Provide the path from raw data to scientific conclusion.

For a system or infrastructure artifact:

- Use a layered architecture.
- Compare against prior systems feature-by-feature.
- Include throughput, cost, failure recovery, usability, and scale.
- Show real user or deployment evidence, but do not rely only on internal production claims.

## Final Pre-Submission Checklist

Use this as the last audit before submission.

- The first page names the gap, artifact, scale, release, and strongest finding.
- The artifact has one dominant identity.
- Prior work comparison uses measurable axes.
- Every claimed capability appears in a task, metric, example, or ablation.
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
- Public code/data links work from a clean browser session.
- README quick start reproduces at least one small result.
- Paper claims match released artifact versions exactly.
- No shortcut, fake, placeholder, or filler data is used anywhere.
