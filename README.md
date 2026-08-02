# Prosocial Cost-Shifting Bench

Prosocial Cost-Shifting Bench is a traceable evaluation suite for measuring
separate response profiles in harmful-request refusal, focal social-dilemma
choices, and repeated shared-resource use. It does not assume that performance
transfers across those domains or combine them into a readiness score.

The project started under the working name `llm-altruism`, but the benchmark does not claim to measure intrinsic altruism or moral character. It measures observable behaviors under explicit task contracts:

- **Part 0: Safety refusal.** Models answer multilingual harmful-request prompts; outputs are scored as refusal or compliance.
- **Part 1: Focal dilemma choices.** Models produce self-directed choices, advice, observer judgments, and predictions in hypothetical one-shot dilemmas.
- **Part 2: Prompt--engine contract audit.** Homogeneous same-model populations emitted `OPTION_A`/`OPTION_B` tokens while the prompt described private and group scores that the engine never implemented. The stored trajectories diagnose that mismatch; they do not measure commons preference.

The April pilot supports descriptive Part 1 action-label summaries and a Part 2
prompt--engine contract diagnosis. It does not support behavioral Part 2 results.
Its legacy Part 0 labels are invalid for model-level refusal claims because the
old labeler could inspect rationale and default failed adjudications to denial.
The paper therefore withdraws all Part 0 rates and refusal-based cross-part
claims pending complete response-only rejudgment and a human criterion audit.

## What Is In This Repository

```text
agents/                       Provider-agnostic agent wrappers and model config
providers/                    API/Ollama call adapters
experiments/part0/            Harmful-request refusal experiment
experiments/part1/            One-shot social dilemma experiment
experiments/part2/            Repeated common-pool resource simulation
experiments/misc/             Shared prompt loading, result writing, metadata, preflight checks
analysis/                     Validation, summary tables, manifests, supplement build, figure syncing
data/raw/part_0/              Curated tracked pilot corpus; new sensitive runs are ignored
data/raw/part_1/              Curated tracked pilot corpus; new sensitive runs are ignored
data/raw/part_2/              Curated tracked pilot corpus; new sensitive runs are ignored
data/analysis/                Validation reports, derived tables, Croissant metadata, manifests
data/graphs/                  Generated figures and diagnostics
docs/conference_submission/   NeurIPS submission source and figures
docs/release/                 Data card, model registry, release terms, public proposal PDF
tests/                        Unit and integration tests
```

Parts 3-5 are roadmap placeholders. Parts 0-2 are the implemented benchmark
scope. Only the Part 1/Part 2 pilot artifacts are structurally validated; Part 0
model-level evidence is withdrawn and confirmatory inputs remain unapproved.
Do not promote additional parts without implementation, validation, and reruns.

## Setup

Install dependencies with `uv` from the repository root:

```bash
uv sync
cp .env.example .env
```

Fill in only the provider credentials you plan to use. Local Ollama runs do not require cloud API keys, but they do require Ollama to be installed and the requested model tag to be available locally.

For revision-pinned offline Hugging Face controls, use
`experiments/misc/local_hf_smoke.py` with
`agents/local_control.registry.json`. The runner hashes the exact cached assets
and performs real generation without downloading or trusting remote code. These
small-model controls are separate from the hosted frontier panel and never
substitute for an unavailable route. See `docs/LOCAL_MODEL_CONTROLS.md`.

Internal NVIDIA InferenceHub and the public NVIDIA API Catalog/NIM are separate
trust domains. Internal calls require `NVIDIA_API_KEY` and an explicit
`INFERENCE_HUB_BASE_URL=https://inference-api.nvidia.com/v1`; the adapter rejects
any other host, scheme, path, port, URL credentials, query, or fragment. The
portal at `https://inference.nvidia.com` is not the API base. Public NIM calls
use the separate `NVIDIA_NIM_API_KEY` and `NVIDIA_NIM_BASE_URL` variables. Never
reuse either credential in the other endpoint profile. Explicit `api_key`
arguments are disabled for both NVIDIA profiles, and generic OpenAI-compatible
profiles cannot target either NVIDIA host.

Run the test suite:

```bash
uv run pytest -q
```

Experiment entry points run a preflight test gate by default. During controlled local development, you can skip that gate with:

```bash
export LLM_ALTRUISM_SKIP_PREFLIGHT=1
```

## Running Experiments

Part 0 evaluates harmful-request refusal for one model/language configuration:

```bash
uv run python -m experiments.part0.part_0 \
  --benchmark ollama:gpt-oss:20b \
  --language english
```

Part 1 runs the focal-choice and role-conditioned social-dilemma prompt matrix:

```bash
uv run python -m experiments.part1.part_1 \
  --provider ollama \
  --model gpt-oss:20b \
  --headless
```

Part 2 executes the legacy repeated-resource protocol for contract auditing. Its
outputs must be interpreted as mismatched prompt--engine traces, not as commons
preference estimates:

```bash
uv run python -m experiments.part2.part_2 \
  --provider ollama \
  --model gpt-oss:20b \
  --society-size 50 \
  --days 100 \
  --resource water \
  --selfish-gain 2 \
  --depletion-units 2 \
  --community-benefit 5 \
  --headless
```

Interrupted runs can be resumed with `--resume`. Result CSVs are written incrementally. Completed runs retain metadata sidecars with provider/model identifiers, command context, timestamps, prompt/config hashes when available, status, and output paths.

## Reproducing The Paper Artifacts

The graph-independent pipeline validates raw traces, builds derived tables, and writes a manifest:

```bash
uv run pytest -q
uv run python -m analysis.build_legacy_part2_provenance --check
uv run python -m analysis.validation --strict
uv run python -m analysis.summarize_results
uv run python -m analysis.build_manifest
uv run python -m analysis.build_croissant_metadata
```

The provenance check binds the untouched April Part 2 CSVs and sidecars to the
archived execution source and replays every population transition under the
recorded divisor-of-five collapse rule. It does not insert the current runtime
default into legacy metadata.

The reconstructed Part 1 sidecars originally derived model identities from
lossy filename slugs. Their exact identities are repaired from the unique
provider/model values present in every CSV row, while preserving the original
reconstruction provenance:

```bash
uv run python -m analysis.backfill_metadata --repair-part1-identities
```

`build_croissant_metadata` deterministically refreshes distribution hashes,
sizes, schemas, and record sets after tables or manifests change. Once an
anonymous reviewer-accessible landing page exists, pass its real URL with
`--dataset-url https://...`; the generator rejects placeholders. The URL may be
a private preview during review but must be public by camera ready.

The current validation report is written to:

```text
data/analysis/validation/validation_report.json
```

The main derived tables are written under:

```text
data/analysis/tables/
```

Important supported table outputs include:

- `part1_model_summary.csv`
- `part1_dimension_summary.csv`
- `part1_frame_effects.csv`
- `part1_prompt_sensitivity.csv`
- `part1_factor_decomposition.csv`
- `part2_model_summary.csv`
- `part2_run_summary.csv`

The generator retains deprecated Part 0 and dependent cross-part tables in the
working tree for forensic replay. Every row is marked invalid/deprecated, and
those tables and their dependent plots are excluded from Croissant metadata and
the anonymous supplement.

The legacy campaign remains available for reproducing pilot workflows:

```bash
uv run python -m experiments.campaign \
  --cohort current_sota \
  --phase smoke --phase part0 --phase part1 --phase part2 \
  --dry-run
```

New paper-facing collection must use the isolated confirmatory campaign in
`experiments.confirmatory_campaign`, not the legacy runner. It requires a
fresh, complete route-evidence bundle, human-approved Part 0 and Part 1 inputs,
same-target full-path smokes, exact native artifact verification, and the fixed
24-trajectory Part 2 panel. See
`docs/CONFIRMATORY_CAMPAIGN.md` and `docs/CONFIRMATORY_PROTOCOL.md` for the
complete commands and gates. A confirmatory dry run validates all inputs and
prints the exact job matrix without writing files or calling a provider.

The 24-system `current_sota` and six-system `historical` cohorts are defined in
`agents/agent_config.registry.json`. Together they cover current GPT-5.6,
Claude, Gemini/Gemma, Nemotron, DeepSeek, Qwen, Kimi, GLM, Mistral, MiniMax,
and GPT-OSS systems plus GPT-3.5, GPT-4.1, GPT-5, Gemini 2.5, Gemma 2, and
GPT-OSS historical comparisons. All 30 evaluated targets and the separate
judge now use exact backend-namespaced IDs present in the authenticated
InferenceHub `/models` census and corresponding portal cards. They remain
`verification_status=unverified` and `route_source=catalog_display_only`, and
therefore cannot be executed by the production adapter. Preserve discovery and
smoke-test evidence and mark a route verified only after review. Developer
Tools display text or a model-card page alone is insufficient. The provider adapter independently
requires a registered verified route before reading credentials and rejects a
missing or different response-model identity. Registry membership is a run
plan, not a claim that a provider route is available or that its results appear
in the paper. All 30 frozen routes must pass smoke tests, complete native
artifacts, and validate; one incomplete route leaves the campaign incomplete
and prevents confirmatory estimator release.

After setting the exact InferenceHub base URL and credential, verify the full
current-plus-historical panel in one fail-closed batch:

```bash
uv run python -m experiments.misc.inference_hub_discovery verify-cohorts \
  --cohort current_sota \
  --cohort historical \
  --cohort judge_only \
  --catalog-output data/private/inference_hub/catalog.json \
  --attempt-ledger data/private/inference_hub/discovery-attempt-ledger.json \
  --max-workers 16 \
  --output data/private/inference_hub/cohort-evidence.json
```

Before changing any registry route, reconcile the authenticated catalog against
the frozen display-label plan with the outcome-blind exact-suffix policy:

```bash
uv run python -m analysis.reconcile_inference_hub_routes \
  --catalog data/private/inference_hub/catalog.json \
  --registry agents/agent_config.registry.json \
  --output data/private/inference_hub/route-reconciliation.json
```

The reconciliation report is private and non-promotional: it retains every
exact-suffix backend candidate, applies a checked-in backend-priority order, and
leaves renamed, versionless, or merely similar routes unresolved. The current
registry resolves all 31 planned routes exactly, but each candidate remains
`smoke_pending`; only a successful identity-checked chat completion can support
a later reviewed registry promotion.

Attempt every selected exact-suffix candidate in one durable, fail-closed batch:

```bash
uv run python -m experiments.misc.inference_hub_discovery verify-candidates \
  --catalog-input data/private/inference_hub/catalog.json \
  --registry-input agents/agent_config.registry.json \
  --reconciliation data/private/inference_hub/route-reconciliation.json \
  --attempt-ledger data/private/inference_hub/candidate-attempt-ledger.json \
  --max-workers 16 \
  --output data/private/inference_hub/candidate-evidence.json
```

The batch reserves each request before dispatch, continues after individual
failures, retains a complete pass/fail inventory, and never mutates the checked-in
registry. It recomputes the entire reconciliation from the supplied catalog and
registry, and refuses catalog drift, reconciliation tampering, duplicate routes,
or an unsafe automatic-promotion policy before the first request.

For the exhaustive authorized-route census, minimally probe every catalog entry
without asserting optional generation controls:

```bash
uv run python -m experiments.misc.inference_hub_discovery probe-catalog \
  --catalog-input data/private/inference_hub/catalog.json \
  --attempt-ledger data/private/inference_hub/catalog-probe-ledger.json \
  --max-workers 16 \
  --output data/private/inference_hub/catalog-probe-evidence.json
```

This performs one bounded, identity-checked chat request per catalog route and
retains both successes and failures. Minimal chat callability is not equivalent
to confirmatory compatibility: selected paper routes must additionally pass the
seeded JSON-schema smoke and the complete experiment-path smoke. A truncated
minimal response or missing usage block is recorded but does not erase proof
that the exact route returned visible chat content; those conditions remain
fatal in the stricter confirmatory smoke.

This captures the virtual key's authorized `GET /models` catalog, inventories
every returned route with an explicit include/exclude decision, and runs a
structured, seeded, identity-checked completion against every exact frozen route. It writes
each pre-dispatch reservation to the discovery ledger, including failed calls,
and marks the evidence complete only after all selected routes pass. The bundle
contains hashes and request IDs, not generated content or credentials. Because the checked-in
routes are currently display-only placeholders, review the authenticated
catalog and replace them with exact callable IDs before expecting this gate to
pass.

Legacy Part 0 exports can be rejudged without exposing stored rationale text to
the judge:

```bash
uv run python -m analysis.rejudge_part0 \
  --input path/to/legacy.csv \
  --output path/to/response_only.csv \
  --summary-json path/to/response_only_summary.json
```

See `docs/JUDGE_AUDIT.md` for the separate blinded human-audit workflow. The
repository never substitutes synthetic annotations for missing human labels.

Regenerate the figures used by the paper and sync them into the LaTeX figure directory:

```bash
uv run python data/graphs/paper_visuals.py
uv run python data/graphs/cross_part_graphs.py
uv run python -m analysis.sync_conference_figures
```

The first two commands render figures under `data/graphs/`. The sync command copies the paper-used PNGs into:

```text
docs/conference_submission/figures/
```

Build the anonymous supplement:

```bash
uv run python -m analysis.build_supplement
```

The output is:

```text
docs/conference_submission/supplement.zip
```

## Figure And Paper Workflow

The paper source lives in `docs/conference_submission/conference_submission.tex`. To rebuild the PDF after changing text, references, or figures:

```bash
cd docs/conference_submission
pdflatex -interaction=nonstopmode conference_submission.tex
bibtex conference_submission
pdflatex -interaction=nonstopmode conference_submission.tex
pdflatex -interaction=nonstopmode conference_submission.tex
```

The LaTeX source intentionally uses local paths under `figures/` so the
conference submission directory is self-contained. If you regenerate plots
under `data/graphs/`, run `uv run python -m analysis.sync_conference_figures`
from the repository root before compiling the paper.

## Validation Philosophy

Validation is separate from plotting. `analysis.validation` checks artifact structure and transition consistency before figures are rendered:

- CSV headers and required fields
- duplicate rows
- valid Part 1 and Part 2 actions
- Part 1 prompt-matrix coverage
- Part 2 day continuity
- incomplete days
- reserve/population transition consistency
- interrupted versus complete run status

This validation does not prove that the automated Part 0 judge is semantically correct. It verifies that the recorded artifacts are internally consistent and suitable for downstream analysis.

## Data And Safety

Part 0 uses harmful-request prompts and model completions for safety evaluation. Do not casually republish raw harmful prompts or completions. The anonymous supplement excludes raw Part 0 prompt-source CSVs, raw Part 0 metadata sidecars, raw Part 0 harmful completions, and every invalid legacy Part 0 or dependent cross-part table and figure. It includes the sanitized aggregate rejudgment-audit checkpoint plus Part 1/Part 2 raw CSVs and metadata.

Part 1 and Part 2 prompts, traces, and metadata are intended for auditability. Treat the current results as a pilot snapshot, not a final leaderboard. The Part 2 runs in the paper are one contract-mismatched trajectory per model; their stored `OPTION_A` token rates and mechanically downstream state traces are audit records, not behavioral estimates.

## Useful Release Documents

- `docs/release/DATA_CARD.md`: datasheet-style overview of the benchmark and release.
- `docs/release/MODEL_REGISTRY.md`: model tags, families, parameter notes, and source links.
- `docs/release/COMPUTE.md`: compute and execution notes.
- `docs/release/LICENSES_AND_TERMS.md`: code, data, and upstream artifact terms.
- `docs/release/REPRODUCIBILITY.md`: expanded reproduction notes for reviewers.

## Submission Gates

Before using new results in a paper or release:

1. `uv run pytest -q` passes.
2. `uv run python -m analysis.validation --strict` passes, or exceptions are documented.
3. Every paper-used CSV has a metadata sidecar and manifest entry.
4. Every central paper claim traces to a table, validation report, figure, or documented source file.
5. Paper figures are regenerated and synced with `analysis.sync_conference_figures`.
6. The supplement builds with `uv run python -m analysis.build_supplement`.
