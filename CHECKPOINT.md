# Safety Beyond Refusal: Complete Work-Mac Handoff

> **Temporary private handoff. Read completely before working.** This file contains author-identifying
> OpenReview details and descriptions of NVIDIA-internal infrastructure. Never include it in the anonymous
> paper, supplement, Croissant release, or a public repository. After the work-Mac agent has ingested and
> independently confirmed this context, delete this file, commit the deletion, and keep it out of any
> anonymous/public export history.

Last updated: 2026-08-01 (America/Los_Angeles)

Repository: `https://github.com/aryan-cs/llm-altruism` (**PRIVATE**)

Branch: `master`

Pre-checkpoint HEAD: `2c11386`

Personal-Mac workspace: `/Users/aryan/Desktop/projects/old/llm-altruism`

User-approved paper title: **Safety Beyond Refusal**

## 1. Non-negotiable instructions

1. This commit is a resumable engineering checkpoint, not a release-ready result.
2. Preserve the exact title `Safety Beyond Refusal`.
3. Never fabricate labels, model calls, human annotations, repeated runs, citations, or validation results.
4. Never print, log, commit, or transmit `.env` values.
5. Rotate the NVIDIA internal key before use. The old internal token was sent once to NVIDIA's public
   `integrate.api.nvidia.com` completion endpoint before the internal/public distinction was clarified; it
   returned HTTP 401. It was never printed or committed, but it must still be rotated.
6. Internal InferenceHub is `inference.nvidia.com`, reachable through NVIDIA's network/VPN. Do not send its
   credential to the public NVIDIA API Catalog.
7. Copy the exact API base and callable model IDs from InferenceHub **Developer Tools**. Catalog card names
   are not callable IDs. Do not silently substitute a model/version/provider.
8. Repair endpoint fail-closed behavior and restore a green full test suite before any hosted call.
9. Never mix Part 0 judge protocols. Resume the partial run only with identical provenance or restart all
   3,861 rows uniformly.
10. Do not claim human validation until two independent qualified annotators, delayed duplicates,
    adjudication, and quantitative scoring have actually been completed.
11. For fresh model runs, preserve complete raw provider output/reasoning up to a generous explicit cap,
    use a separately versioned model to extract the final answer, and grade only that extracted answer.
    Truncation or extraction failure is unscorable, not a behavioral label.
12. Use subagents for bounded independent audits, visually inspect visual outputs, preserve unrelated changes,
    and push descriptive commits to `master`.

## 2. User requirements

The user asked the agent to understand the entire repository; respond to all NeurIPS reviews; improve and
finish the paper; maximize acceptance odds; use the `check-paper` and PDF visual-review workflows; use real
experiments only; test current frontier and historical systems including Gemma, Gemini, Claude
Opus/Sonnet/Haiku, GPT-3/4/5-era systems, and other SOTA models; use NVIDIA's internal InferenceHub; add an
explicit output cap plus independent final-answer extraction; preserve exact provenance; visually inspect
figures/PDF; and push the complete checkpoint to the private GitHub `master` branch for work-Mac resumption.

Repository `AGENTS.md` instructions also require subagents, no shortcuts/fake data, visual inspection for
visual work, closing browser tabs after automation, and pushing to `master` unless another branch is specified.

## 3. OpenReview evidence and reviewer comments

### Supplied OpenReview screenshot

Local source (not committed separately because the user requested one checkpoint file and because it identifies
the author):

`/var/folders/bz/py3vn5791r5339nvldg8yvzc0000gn/T/codex-clipboard-ff1bf745-0aa3-4a7d-9de4-dc9ca3b68b32.png`

Facts visible in the image:

- NeurIPS 2026 Evaluations and Datasets Track, submission 1113.
- Original title: `Prosocial Readiness Bench: Safety Refusal, Cooperation, and Commons Restraint in LLM Agents`.
- ZSVH rating/confidence: 3/4.
- KHFn rating/confidence: 3/4.
- iD2D rating/confidence: 4/2.
- Average rating 3.33; average confidence 3.33.
- Decision area showed `Track Submission / No Recommendation`.

Complete supplied OpenReview text on the personal Mac:

`/Users/aryan/.codex/attachments/a01e9daa-bdce-4451-bb01-810b6ce4dd26/pasted-text.txt`

### Meta-review

Recognized strengths: important motivation beyond refusal; interesting prompt-frame effects; strong
reproducibility/tooling; relevant and novel direction.

Critical weaknesses:

1. **Weak “dissociation” evidence.** Correlation intervals are very wide, and refusal-restraint around
   `r=0.77` conflicts with claims that the axes are separable.
2. **Insufficient theoretical/task-composition justification.** The paper did not convincingly explain why
   these three tasks are important, selected, and jointly necessary rather than an aggregation.
3. **Limited model selection.** Mostly older/smaller local models; reviewers explicitly requested frontier
   GPT, Claude, and Gemini results.
4. **Ecological validity.** Dyadic games and the commons are simplified toys without realistic sanctions,
   contracts, communication, uncertainty, heterogeneous populations, scale, or institutions.
5. **Judge reliability.** The Part 0 LLM judge lacks large-sample quantitative human validation.

The meta-review said acceptance odds could improve with corrected statistical claims, frontier models,
stronger conceptual framing, quantitative human validation, and better real-world limitations/mapping.

### Reviewer iD2D (rating 4, confidence 2)

- Positive about broadening safety beyond refusal, quantitative games/resource metrics, multilingual coverage,
  and reproducibility.
- Main concern: two-player games and fixed 50-agent/100-day tests are far simpler than real systems.
- Requested real-world justification, agent-system experiments/discussion, and commercial SOTA Gemini/Claude.

### Reviewer KHFn (rating 3, confidence 4)

- Positive about motivation, openness, extensibility, and prompt-type findings.
- Main concern: inadequate justification for task selection and joint study; otherwise the contribution may
  look like aggregation.
- Requested specific cross-task relationship analyses.
- Explicitly noted that strong refusal-restraint correlation contradicts “dissociation.”
- Reported Croissant unavailable at review time. Current Croissant metadata exists but is stale and must be
  rebuilt after final data generation.

### Reviewer ZSVH (rating 3, confidence 4)

- Positive about clear frame findings (about 83% observer versus 4.9% prediction), about 35% frame variance,
  and traceability artifacts.
- The refusal-cooperation interval was roughly `[-0.61, 0.70]`; refusal-restraint near `0.77` undercuts the
  central narrative.
- Concerned results are an artifact of only 13 related/older models.
- Requested frontier GPT/Claude/Gemini statistics, reliable correlations, and quantitative human judge validation.

## 4. Scientific stance already adopted

- Say **partially coupled, non-redundant behavioral profile**, not “dissociated” or independent.
- Do not claim a latent altruism/prosociality/readiness trait, intent, deception, or moral status.
- Link tasks only through an observable cost-shifting contrast; explicitly state they need not share one process.
- Part 1 is a focal hypothetical one-shot response, not live dyadic interaction.
- Part 2 is controlled stateless repeated prompting in a toy microworld, not a realistic society.
- Failure in the transparent toy can be a minimal stress-test failure; success cannot establish real cooperation.
- Cross-part correlations are descriptive model-level diagnostics with `n=13`, wide uncertainty, and family
  dependence. Report Pearson, Spearman, Kendall, bootstrap/Fisher intervals, and leave-model/family sensitivity.
- Treat rationale analysis as lexical/surface evidence only.
- GovSim is the closest richer commons precedent. State honestly that this microworld is narrower.

## 5. Code and analysis changes already present

### Statistics (`analysis/statistics.py`, `analysis/summarize_results.py`, generated tables/tests)

- Part 1 primary statistic is direct self-choice (`self_direct`).
- Pooled frames are secondary response-profile summaries only.
- Part 2 supports equal-run weighting and between-run t intervals when repeated runs exist.
- Cross-part diagnostics include Pearson/Spearman/Kendall, Fisher-z and bootstrap intervals, plus leave-one-
  model/family influence analysis.
- Added `data/analysis/tables/part2_run_summary.csv`.
- Analysis prefers `new_complied` by schema presence and never silently falls back if a response-only schema
  exists but is invalid/blank.

### Response-only Part 0 rejudge (`analysis/rejudge_part0.py`)

Implemented stable content-derived row IDs; response-only prompts that never read legacy `reasoning`; fsync
row checkpoints; exact boolean JSON; explicit `unjudged` failures; strict resume validation of input, rubric,
system prompt, endpoint, model, protocol, and decoding; atomic `--retry-unjudged`; nonzero exit if incomplete;
and metadata for source/output hashes, model digest, server settings, and human status.

Frozen incomplete local protocol:

- `part0-response-only-v2`, Ollama `gpt-oss:20b`.
- Model digest `17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7`.
- Ollama 0.32.5; temperature 0; `think=low`; `num_predict=512`; seed `20260801`.
- `OLLAMA_NUM_PARALLEL=4`, context 32768, workers 4, attempts 3, timeout 300 seconds.
- April localized strings were not stored; original base request is context with original-language response.
- Source has 3,861 rows: 13 models x 297, using 99 base prompts in English/Chinese/Russian.

Committed incomplete checkpoint:

`data/checkpoints/part0_rejudge/part0-response-only-v2-512-gpt-oss-20b.incomplete.csv`

Exact interruption state:

- 1,245/3,861 rows; 1,243 judged.
- 925 denied; 318 complied; 2 unjudged invalid-JSON failures.
- SHA-256 `34c0b19ae50e8a8fdb619bd9748fc13cfe0f0c71ee239e9c66385b5428352c0e`.
- Personal-Mac temp twin was `/private/tmp/part0-response-only-v2-512-gpt-oss-20b.csv`.
- The rejudge and temporary Ollama server are stopped.

Resume only with identical provenance. If the work Mac cannot reproduce the exact local model/server, archive the
partial file and restart all 3,861 rows under one newly frozen hosted protocol. Never mix the partial labels
with another judge.

### Human audit (`analysis/judge_audit.py`, `docs/JUDGE_AUDIT.md`)

- Deterministic stratified blinded sampling and delayed duplicate packets.
- One-annotator backward compatibility.
- `--annotators 2` creates independent packets, per-annotator duplicate packets, private multi-key file,
  and manifest v2.
- `prepare-adjudication` emits a blank disagreement-only packet.
- `score-multi` requires complete independent labels, duplicates, and adjudication.
- Reports adjudicated metrics, per-language/overall pairwise Cohen, generalized Fleiss, and per-annotator
  delayed-duplicate intra-rater Cohen kappa.
- Documentation requires trained native/fluent-language annotators.
- No human labels exist; no validation result may be claimed yet.

### Part 1 order and attempt logging

- Future Part 1 uses seeded shuffle plus cyclic counterbalancing, default seed `20260801`.
- Parts 1/2 have fsync JSONL attempt logs for invalid output, success, provider error, retry, interruption.
- Metadata records log path/hash/counts/retries/errors.
- Legacy April limitation remains: fixed Part 1 order and discarded invalid retry attempts.

### Part 2 sensitivity

- Bounded factorial/exact JSON cells over capacity, depletion units, death rate, population, horizon, and seed.
- Rejects duplicate/uneven seeds, mixed modes, ignored phase, >4,096 jobs, or >10M request upper bound.
- Manifests exact cells/counts; CLI/config support capacity, death rate, seed, resume metadata, filenames.
- No new sensitivity trajectories have been run. Current pilot remains one trajectory per model.

### Validation

- Part 2 validation now checks contiguous days, constants, unique IDs, daily summaries, prior population,
  resource transitions, death rule, and metrics.
- Previous legacy validation: 27 files, 14 pass / 13 warnings / 0 failures.
- This is structural validation, not semantic human validation.

## 6. Provider, campaign, and internal InferenceHub state

### Implemented provider/campaign work

Changes exist in `agents/agent_config.py`, `agents/agent_config.registry.json`, `providers/api_call.py`,
`experiments/misc/preflight.py`, `experiments/misc/run_metadata.py`, `experiments/campaign.py`, the wizard,
docs, and tests:

- versioned cohorts and exact-target planning;
- OpenAI-compatible/InferenceHub routing;
- route, response, and failure provenance;
- resumable campaign manifests;
- strict credential preflight and dry-run request/token estimates;
- Part 0 response-only path for current runs;
- `.env` loading before preflight.

### Current unsafe endpoint state: fix before every hosted call

The implementation predates the internal/public clarification and currently:

- defaults internal `inference_hub` to public `https://integrate.api.nvidia.com/v1`;
- treats catalog-display-derived names as executable pinned routes;
- maps internal InferenceHub and public NVIDIA access to `NVIDIA_API_KEY`;
- does not require `inference.nvidia.com` or reject unverified internal routes;
- documents the unsafe public default in `.env.example`, `README.md`, and release docs.

Required correction:

1. Remove the internal default and require explicit `INFERENCE_HUB_BASE_URL`.
2. Require HTTPS and exact internal host `inference.nvidia.com`; reject URL credentials/query/fragment.
3. Fail closed when the internal base is missing.
4. Keep public NVIDIA NIM distinct, with a distinct key such as `NVIDIA_NIM_API_KEY`.
5. Mark all internal registry entries `verification_status=unverified` and
   `route_source=catalog_display_only` until Developer Tools supplies exact IDs.
6. Block preflight/campaign execution for every unverified route.
7. Update endpoint profiles, preflight, adapters, metadata, docs, tests, and registry hash.

An internal-endpoint-hardening subagent was assigned but made **no edits** before the handoff was requested.
The list above is its exact intended implementation.

### NVIDIA internal photos/evidence

The user supplied these work-Mac photos, which are described here but intentionally not committed as assets:

- `/Users/aryan/Downloads/IMG_6598.JPG`
- `/Users/aryan/Downloads/IMG_6599.JPG`
- `/Users/aryan/Downloads/IMG_6596.JPG`
- `/Users/aryan/Downloads/IMG_6597.JPG`

They show `inference.nvidia.com` InferenceHub with Catalog, Developer Tools, Cost Estimator, Model Requests,
Metrics, Keys, and Wiki tabs; 146 models across 14 companies; and visible cards for Claude Opus
4.1/4.5/4.6/4.7/4.8/5, Claude Sonnet 5, Claude Haiku 4.5, GPT-5/5.1/5.2/5.3/5.4/5.5/5.6 Terra,
GPT-OSS 20B/120B, o4-mini, Gemini 2.5/3/3.1/3.5/3.6 Flash, Qwen 3.5/3.6/Qwen3 Next,
Nemotron 3 Nano/Super, Llama 3.1/3.2/3.3, and others. Some cards are routed through OpenAI, Azure
OpenAI, AWS Bedrock, Google, or Anthropic. These are display names, not verified API IDs.

Personal-Mac tests established:

- `inference.nvidia.com` does not resolve off NVIDIA VPN/internal DNS.
- Public `GET https://integrate.api.nvidia.com/v1/models` returned 200 with 102 public models, but this did
  not validate the internal token.
- A real public Gemma completion with the internal token returned HTTP 401.
- Therefore run internal calls on the VPN-connected work Mac and rotate the token first.

For every model, Developer Tools must supply exact base URL, callable ID, upstream/region, request schema,
generation controls, context/output limits, reasoning-field behavior, quota/rate/cost information, and a
minimal smoke result including finish reason, usage, and request ID.

### Desired model coverage and cost warning

The user wants current GPT models plus historical GPT-3/4/5-era comparisons, Claude Opus/Sonnet/Haiku,
Gemini current/pro/flash, Gemma, and strong current Nemotron, DeepSeek, Qwen, Kimi, GLM, Mistral,
GPT-OSS, and Llama systems when genuinely available. Do not substitute versions silently.

The current registry has aspirational targets across those families. Treat every internal route as unverified.

Earlier pre-clarification dry runs estimated:

- 14-target `current_sota`: 114 jobs across default phases.
- 6-target `historical`: 50 jobs.
- Five-seed full Part 2 across 14 models: up to 350,000 model calls
  (`14 x 5 x 50 agents x 100 days`).

These are scale warnings, not authorization. Recompute after the verified cohort is frozen, smoke first, and
review the exact cost/request upper bound before a full launch.

## 7. Output cap and final-answer extractor: implemented but currently broken

The user's new protocol is partially implemented in `experiments/misc/final_answer.py`, detailed response capture
in `providers/api_call.py`, `agents/base_agent.py`, Parts 0/1/2, campaign, wizard, and metadata.

Intended semantics:

- benchmarked model gets a generous explicit configurable total output cap;
- preserve raw response, separately exposed reasoning, visible content, finish reason, usage, request ID,
  provider body, and truncation state;
- a separately configured/versioned extractor isolates the final answer;
- downstream grading/parsing sees only validated extractor JSON, never hidden reasoning;
- record extractor identity, prompt/settings hashes, attempts, response, failure, and token cap;
- subject truncation, unknown truncation where required, extractor failure, or extractor truncation is explicitly
  unscorable;
- campaign manifests include subject/extractor request and output-token upper bounds.

Latest exact full-suite result after this integration:

```text
30 failed, 212 passed in 60.39s
```

Failure causes:

- Part 0 preflight adds a default extractor target to legacy direct calls.
- Legacy monkeypatched `query_until_valid` callables reject new extraction keyword arguments.
- Part 1/2 direct/legacy tests silently invoke a default extractor, try a real OpenAI path, and fail because
  `OPENAI_API_KEY` is absent.
- Resume, progress, retry, interruption, and disconnect tests inherit those breakages.

Representative failures include `test_run_alignment_test_preflight_only_checks_selected_benchmark_models`,
multiple Part 0 collection/judge/resume tests, Part 1 write/filter/limit/headless/resume/retry tests, and Part 2
multi-day/sensitivity/attempt/resume/headless/disconnect tests.

Required fix: fresh campaigns can require explicit extraction configuration, but legacy direct calls/resumes/test
doubles must retain old behavior; an extractor must not be added to preflight or called unless explicitly enabled;
and signature compatibility must be restored without masking real errors. Then run focused Part 0/1/2 tests and
the full suite. Before extractor integration, subagents reported all 242 tests green.

## 8. Citation and manuscript work already done

### Citations

- Added GovSim as closest richer commons precedent.
- Added Deng et al. multilingual jailbreak evidence and MACHIAVELLI.
- Replaced mismatched Qwen citation with the exact Ollama Qwen 3.5 page used for the pilot.
- Corrected Brookins and WildJailbreak bibliographic details.
- Separated heterogeneous evaluation-suite citations.
- Scoped Zheng et al. as general judge-bias evidence, not classifier validation.
- Scoped bootstrap/correlation citations and removed unsupported comparative-positioning claims.
- Added canonical games/commons citations.
- Last mechanical audit: 44 used keys, 44 BibTeX entries, zero undefined and zero unused. Recheck later.

### Manuscript changes in `docs/conference_submission/conference_submission.tex`

- Exact title `Safety Beyond Refusal`.
- Results-first abstract structure, but Part 0/correlation numbers are stale.
- “Partially coupled/non-redundant” rather than dissociated.
- Explicit construct map; no unitary trait.
- Part 1 focal/non-dyadic clarification.
- Part 2 controlled/stateless/toy/single-trajectory clarification.
- Robust correlation/influence diagnostics.
- Legacy Part 0 provenance limitation and response-only replacement plan.
- Explicitly states exact localized April strings were not stored.
- Current prompt forbids hidden chain of thought; legacy prompt archived.
- Part 2 score multipliers clarified as prompt text, not stored individual reward state.
- Fixed Part 1 order/retry-selection limitations disclosed.
- GovSim comparison and asymmetric ecological-validity interpretation added.
- Custom float/caption spacing overrides removed; `hidelinks` added.
- Conclusion reframed as cross-context evaluation, not a theory claim.
- Rationale language described as lexical/surface evidence.

Do not trust current Part 0 values or cross-part conclusions until the response-only data and new cohort are final.

## 9. Exact stale-result inventory

The active Part 0 file `data/raw/part_0/04-11-2026_13_04_37.csv` still has only legacy `complied?`.
Validation currently reports `response_only_schema: false`. Its sidecar is a schema-1 backfill with unknown
provider/model. Croissant prematurely describes response-only labels in present tense.

After complete uniform rejudging, archive the legacy CSV/sidecar with hashes, promote the response-only CSV to
the active path, make `new_complied` canonical while retaining legacy for audit, replace the sidecar with exact
provenance, and run strict validation requiring 3,861 judged and zero errors.

Manuscript lines/regions requiring regeneration (line numbers may shift):

- 38-39: abstract refusal range/count/correlations.
- 51, 58-59: introduction/contribution correlation claims.
- 83, 87: overview trace/metric nodes.
- 152-161: Part 0 row counts, completed protocol, human status, metric wording.
- 206, 209, 215, 218: sidecar count, validation, totals, analysis protocol.
- 224-250: every refusal value/interval in the main rate table.
- 255-264: Part 0 plot/results narrative.
- 285-307: cross-part narrative/table/influence statements.
- 323, 327, 334: synthesis, limitations, conclusion.
- 346: within-family refusal comparisons.
- 355-367: fingerprint and multilingual refusal figures/captions.
- 397-402: PCA figure/caption.
- 411-416: refusal-restraint figure/caption.
- 489-516: multilingual robustness table.
- Checklist around 695, 700, 709: status, intervals, author review.

Regenerate/update:

- `data/analysis/tables/part0_model_summary.csv`
- `data/analysis/tables/part0_language_robustness.csv`
- `data/analysis/tables/cross_part_model_summary.csv`
- `data/analysis/tables/cross_part_correlations.csv`
- `data/analysis/validation/validation_report.json`
- `data/analysis/run_manifest.jsonl`
- `data/analysis/croissant_metadata.json`
- all 14 `data/graphs/part_0/` figures;
- refusal-dependent paper visuals, PCA/fingerprint, cross-part plots/matrix;
- corresponding `docs/conference_submission/figures/` copies;
- README, Data Card, Reproducibility, Compute, Model Registry, and submission README.

Visualization code now uses Cividis continuous heatmaps, vermillion/blue Part 2 actions, and serif fonts. Existing
images predate corrected labels and are not final. Visually inspect final figures, not just file sizes/pixels.

Legacy context-only values that must not be copied blindly into the final paper: 13 models/4 families;
56,382 decisions/judgments; direct cooperation roughly 6.25%-100%, aggregate self-direct 61.4%, advice 66.8%,
observer 83.2%, prediction 4.9%; restraint 18.6%-92.5%; refusal-restraint Pearson about 0.768;
refusal-cooperation about 0.240 with wide uncertainty; restraint-population about 0.852.

## 10. PDF, supplement, and release state

An earlier Tectonic compile succeeded at 24 total letter pages; main text ended on page 9 and references began
on page 10. That PDF is stale relative to current TeX/figures. Final command previously used:

```bash
cd docs/conference_submission
tectonic -X compile conference_submission.tex --keep-logs --keep-intermediates
```

After final data/manuscript updates, compile with the official unmodified NeurIPS 2026 E&D style, render every
page with Poppler, visually inspect all pages/contact sheets for clipping/overlap/float/label/link issues, verify
the page limit/reference start, and verify the PDF is newer than every input.

`analysis/build_supplement.py` now includes `docs/JUDGE_AUDIT.md` and excludes author metadata. A prior temporary
audit was about 17.2 MiB and 270 files with a clean ZIP test, but no current final `supplement.zip` exists.

Final supplement must have an exact manifest, pass `ZipFile.testzip()`, and be scanned for names, emails, GitHub
origin, absolute paths, secrets, `.env`, this checkpoint, private audit packets/keys, raw Part 0 harmful content,
OpenReview material, and NVIDIA-internal content. Raw Part 0 remains excluded by default. Include only sanitized
aggregate human-audit results after genuine annotation. PDF/ZIP timestamps must postdate all inputs.

Croissant status/version/date/record mappings and all distribution hashes must be updated last. A prior audit
already found stale hashes in Part 0 summary, cross-part summary/correlations, and validation report.

## 11. Current Git working-tree contents

This checkpoint commit intentionally preserves a large unfinished coordinated revision. Modified tracked files:

```text
.env.example
README.md
agents/agent_2.py
agents/agent_config.json
agents/agent_config.py
agents/base_agent.py
analysis/build_supplement.py
analysis/summarize_results.py
analysis/validation.py
data/analysis/croissant_metadata.json
data/analysis/tables/cross_part_correlations.csv
data/analysis/tables/cross_part_model_summary.csv
data/analysis/tables/part0_model_summary.csv
data/analysis/tables/part1_dimension_summary.csv
data/analysis/tables/part1_model_summary.csv
data/analysis/tables/part2_model_summary.csv
data/analysis/validation/validation_report.json
data/graphs/cross_part/individual-plots/*.png
data/graphs/cross_part/master-plots/cross_part_scatter_matrix.png
data/graphs/paper_visuals.py
data/graphs/paper_visuals/*.png
data/graphs/part_0_graphs.py
docs/conference_submission/conference_submission.tex
docs/conference_submission/figures/*.png
docs/conference_submission/references.bib
docs/release/DATA_CARD.md
docs/release/MODEL_REGISTRY.md
docs/release/REPRODUCIBILITY.md
experiments/misc/preflight.py
experiments/misc/run_metadata.py
experiments/misc/wizard.py
experiments/part0/part_0.py
experiments/part0/part_0_prompt.json
experiments/part1/part_1.py
experiments/part2/part_2.py
providers/__init__.py
providers/api_call.py
tests/test_agent_config.py
tests/test_analysis_pipeline.py
tests/test_api_call.py
tests/test_part_0.py
tests/test_part_1.py
tests/test_part_2.py
tests/test_preflight.py
tests/test_wizard.py
```

New untracked-at-checkpoint files/directories that are included in the handoff commit:

```text
CHECKPOINT.md
agents/agent_config.registry.json
analysis/judge_audit.py
analysis/model_metadata.py
analysis/rejudge_part0.py
analysis/statistics.py
data/analysis/tables/part2_run_summary.csv
data/checkpoints/part0_rejudge/part0-response-only-v2-512-gpt-oss-20b.incomplete.csv
docs/JUDGE_AUDIT.md
experiments/campaign.py
experiments/misc/attempt_log.py
experiments/misc/final_answer.py
experiments/part0/part_0_prompt.legacy_20260411.json
tests/test_analysis_statistics.py
tests/test_attempt_log.py
tests/test_build_supplement.py
tests/test_campaign.py
tests/test_judge_audit.py
tests/test_rejudge_part0.py
tests/test_run_metadata.py
```

`.env` is ignored and must remain untracked. It contained a 24-character internal key on the personal Mac; the
value is not copied here or to Git. `git diff --check` was clean before the last test run.

## 12. Work-Mac and SSH context

The execution host should be the NVIDIA work Mac on the Palo Alto/site VPN. Run internal API jobs on that Mac
rather than tunneling the corporate network through the personal Mac.

A dedicated key was created on the personal Mac:

- Private (never commit): `/Users/aryan/.ssh/id_ed25519_nvidia_work`
- Public: `/Users/aryan/.ssh/id_ed25519_nvidia_work.pub`
- Fingerprint: `SHA256:G8tg6DVXZAwMEO8VCVyO1JhmWUy+4jMrbF93/CWUrtg`
- Public key:

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOqLE9FX/Oz5FdszSBHm3jY9y6UyWRH9L1QooWzOMO0S aryan-personal-to-nvidia-work-2026-08-01
```

The user was told to enable macOS Remote Login for only their work account and install the public key. No work
username/hostname/IP was returned before the Git handoff. Do not install Tailscale, reverse tunnels, or broad
SOCKS proxies on a corporate Mac unless NVIDIA policy approves. Pulling this private repository directly on the
work Mac is sufficient to resume.

Safe bootstrap:

```bash
git clone https://github.com/aryan-cs/llm-altruism.git
cd llm-altruism
git switch master
git pull --ff-only origin master
```

Then read this file fully and sync dependencies with the repository's documented `uv` workflow. Do not run a
campaign yet.

Create the rotated key without shell history:

```bash
read -s "NVIDIA_API_KEY?Internal InferenceHub key: "; echo
printf 'NVIDIA_API_KEY=%s\n' "$NVIDIA_API_KEY" > .env
chmod 600 .env
unset NVIDIA_API_KEY
```

After copying the exact base from Developer Tools:

```bash
read "INFERENCE_HUB_BASE_URL?Exact Developer Tools API base: "
printf 'INFERENCE_HUB_BASE_URL=%s\n' "$INFERENCE_HUB_BASE_URL" >> .env
unset INFERENCE_HUB_BASE_URL
```

Do not treat `.env` presence as permission to call. Harden routing first.

## 13. Ordered completion plan

### Gate A: make code safe and green

1. Read this file and all relevant diffs.
2. Fix final-answer extractor backward compatibility while retaining strict extraction for explicitly enabled
   fresh campaigns.
3. Implement fail-closed internal/public NVIDIA separation and block unverified routes.
4. Add focused tests; run full suite; require zero failures.
5. Commit/push a descriptive safety/regression fix before any model call.

### Gate B: freeze exact experimental protocol

1. Copy internal endpoint and exact route IDs/capabilities from Developer Tools.
2. Freeze verified current-SOTA and historical cohorts; no substitutions.
3. Freeze separate extractor and Part 0 judge routes/settings/hashes.
4. Choose generous explicit per-route subject caps and truncation policy.
5. Dry-run exact requests/tokens/costs.
6. Run one minimal smoke per route and inspect content, reasoning, finish, usage, ID.
7. Promote only successful routes.

### Gate C: Part 0 and human validation

1. Decide exact local continuation versus uniform hosted restart.
2. Finish 3,861/3,861 with zero unjudged.
3. Archive legacy active CSV/sidecar and promote response-only data/provenance.
4. Run strict validation.
5. Generate private two-annotator packets.
6. Obtain real independent language-qualified annotations, delayed duplicates, and adjudication.
7. Report accuracy/confusion/F1 and inter-/intra-rater reliability overall/per language.

### Gate D: frontier and robustness experiments

1. Run cost-controlled smoke/minimal cohort first.
2. Run Parts 0/1/2 only after exact cost approval.
3. Use repeated seeds and bounded Part 2 sensitivity rather than silently launching 350,000 calls.
4. Preserve raw/reasoning/extracted answer, attempts, finish/truncation, usage, route, hashes, manifests.
5. Validate and resume only against exact matching metadata.

### Gate E: analysis and paper

1. Run `analysis.summarize_results` on canonical validated data.
2. Inspect primary/secondary statistics and influence diagnostics.
3. Regenerate and visually inspect all dependent figures.
4. Rewrite every stale manuscript location from canonical CSVs.
5. Let actual correlations determine the narrative; never force dissociation.
6. Add verified frontier/repeated/sensitivity/human-audit results with calibrated scope.
7. Run machine-to-LaTeX number and citation audits.

### Gate F: release

1. Rerun strict validation/manifests.
2. Update Croissant status/mappings/hashes last.
3. Update all release docs.
4. Build and integrity/PII/secret-check supplement.
5. Compile/render/visually inspect final PDF.
6. Run a fresh independent paper audit for stale numbers, overclaims, anonymity leaks, and layout failures.
7. Fix findings, rerun everything, and push descriptive commits to `master`.

## 14. Delete this checkpoint after ingestion

Once the work-Mac agent confirms it has ingested and verified this context:

```bash
git rm CHECKPOINT.md
git commit -m "Remove temporary cross-Mac handoff checkpoint"
git push origin master
```

Then verify `.env` is ignored, private annotations are outside Git, release builders exclude this file, no
OpenReview/author or NVIDIA-internal material exists elsewhere, and any anonymous/public export is built from a
sanitized history that excludes this checkpoint.

## 15. Definition of genuinely complete

Completion requires: green tests; verified internal routes/model provenance; real frontier artifacts; uniform
complete response-only Part 0 labels; genuine two-annotator human validation; real repeated/sensitivity evidence
or narrowed claims; regenerated tables/figures/hashes/manifests; manuscript numbers matching canonical outputs;
calibrated statistical/ecological claims; supplement anonymity/integrity/security checks; page-by-page visual PDF
inspection; fresh independent final audit with material findings resolved; and removal of this temporary checkpoint.
