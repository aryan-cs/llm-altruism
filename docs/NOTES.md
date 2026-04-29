# Notes: Writing A Stellar NeurIPS Datasets And Benchmarks Paper

Last updated: 2026-04-29

This document is a working research notebook and writing procedure distilled from NeurIPS 2025 Datasets and Benchmarks Track spotlight papers. It is intentionally practical: the goal is to produce a repeatable framework for writing a strong paper on any dataset, benchmark, evaluation suite, diagnostic study, or research infrastructure artifact.

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

## Cross-Paper Pattern Bank

Working hypotheses to validate against the remaining papers:

- Strong Datasets and Benchmarks papers make the artifact's contract explicit: task, data, metrics, split design, annotation or construction procedure, expected inputs and outputs, and intended use.
- Reviewers reward diagnostic value beyond aggregate leaderboards: taxonomies, error categories, capability axes, ablations, scaling curves, subgroup analysis, and qualitative examples.
- The best visuals are not decorative. They are inspection tools: pipeline diagrams, taxonomy diagrams, dataset composition plots, task schemas, leaderboard matrices, failure taxonomies, and examples with annotations.
- Release credibility matters. Code, data cards, licenses, model lists, versioning, benchmark servers, evaluation scripts, reproducibility checklists, and maintenance plans all strengthen the paper.
- A stellar paper tells the reviewer exactly why the artifact should exist now, why existing resources fail, how the new artifact was built, how quality was controlled, what it reveals, and how others can use it.
