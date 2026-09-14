# Safety Beyond Refusal

**Safety Beyond Refusal** studies three complementary, directly observable
behaviors: harmful-request refusal, welfare-preserving self-choice in one-shot
social dilemmas, and restraint in a corrected repeated commons simulation. The
benchmark reports the axes separately; it does not infer intrinsic altruism,
moral character, or a single deployment-readiness score.

The definitive provider-safe-v2 design is:

- **Part 0:** 22 exact subject routes, each scheduled on 48 archived English
  harmful-request roots crossed with three requested response languages (144
  responses per route). One fixed judge is disjoint from every subject at the
  target, route, and served-identity levels. Two exact planning-roster routes
  are operationally excluded without substitution. All 3,168 scheduled units
  are terminal under the explicit all-scheduled operational-invalid policy;
  the estimates remain exploratory because the task lacks benign controls and
  the judge has not completed human validation. The inputs are English requests
  with response-language instructions, not translated request banks.
- **Part 1:** 75 exact subject routes, each scheduled on the same balanced
  384-root direct self-choice bank. Six additional frozen registry targets are
  operationally excluded without substitution. A complete exact-source overlay
  resolves the three transport-null units, while all 704 malformed first
  responses remain scheduled nonsuccesses. The prompt bank still lacks
  independent content approval.
- **Part 2:** 23 exact subject routes completed 12 independent common-seed
  trajectories each (276 total) under the corrected 50-agent, 100-day commons
  engine. The three exact source/overlay pairs contain 1,206,808 scheduled
  living agent-days: 1,180,046 valid actions and 26,762 genuine semantic
  `INVALID` actions, of which 969,640 are restraint and 210,406 are overuse.
  The primary descriptive restraint proportion pools those scheduled living
  agent-days within each route. Figures with trajectory Student-$t$ intervals
  instead show the equal-seed-weighted mean of the 12 trajectory-specific
  proportions; attrition can make the two route estimates differ.
  All 276 trajectories are operationally eligible after 57 whole-trajectory
  repairs. Environmental estimates use the 198 zero-invalid trajectories;
  78 trajectories contain at least one semantic invalid, leaving 18 routes
  with an environmental summary and five zero-eligible routes reported as NE.
  `anthropic/claude-opus-4-5` is the sole frozen exclusion and is not
  substituted.

The April pilot remains historical provenance. Its audit led to the corrected
response-only judge, balanced Part 1 bank, repeated-commons engine, provenance
controls, and exclusions used in the current study. Its unsupported rates and
cross-axis claims are not current paper results.

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
scope. The definitive public aggregate is written to
`data/processed/provider-safe-v2-definitive-analysis` only after every source
phase and required operational overlay (Parts 0-2, role calibration, and
sensitivity) passes the fail-closed analyzer. The sealed 12-day
`data/analysis/final_results/final_results.json` is historical, superseded
evidence and is not a current paper input. Part 0 and Part 1 remain exploratory
because their external validation gates are incomplete.

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

Part 2 executes the corrected repeated-resource protocol:

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

After every definitive source phase and bound operational overlay is terminal,
build the text-free aggregate graph, paper assets, Croissant metadata, and
strict supplement:

```bash
uv run pytest -q
uv run python -m analysis.analyze_provider_safe_v2_definitive \
  --part0 data/private/inference_hub/definitive-part0-large-n48-main22-deadline-v6 \
  --part0-terminal-policy all-scheduled-operational-invalid-v1 \
  --part1 data/private/inference_hub/definitive-part1-large-n384-main75-deadline-v5 \
  --part1-operational-repair data/private/inference_hub/definitive-part1-large-n384-main75-deadline-v5-operational-repair-v1 \
  --part2-source-overlay data/private/inference_hub/full-part2-n12-n50-d100-main21-v5 data/private/inference_hub/full-part2-n12-n50-d100-main21-v5-operational-completion-capability-v4 \
  --part2-source-overlay data/private/inference_hub/full-part2-n12-n50-d100-nemotron-3-ultra-recovered-v1 data/private/inference_hub/full-part2-n12-n50-d100-nemotron-3-ultra-operational-repair-v1 \
  --part2-source-overlay data/private/inference_hub/full-part2-n12-n50-d100-deepseek-v4-flash-recovered-v1 data/private/inference_hub/full-part2-n12-n50-d100-deepseek-v4-flash-operational-repair-v1 \
  --part2-declared-exclusion anthropic/claude-opus-4-5 \
  --role-calibration data/private/inference_hub/definitive-part1-role-calibration-v3 \
  --sensitivity data/private/inference_hub/definitive-part2-sensitivity-deadline-fast-v9 \
  --sensitivity-operational-repair data/private/inference_hub/definitive-part2-sensitivity-deadline-fast-v9-operational-repair-v1 \
  --output-dir data/processed/provider-safe-v2-definitive-analysis
uv run python -m analysis.build_provider_safe_v2_paper_assets \
  --input-dir data/processed/provider-safe-v2-definitive-analysis \
  --local-controls data/analysis/local_hf_part1_controls.json \
  --output-dir data/processed/provider-safe-v2-paper-assets
uv run python -m analysis.build_provider_safe_v2_croissant_metadata
uv run python -m analysis.build_provider_safe_v2_croissant_metadata --check
uv run python -m analysis.build_supplement --require-definitive-artifacts
```

The analyzer intentionally fails while any source phase lacks the terminal
evidence required by its explicit policy. It validates the Part 2 three-pair
composition recursively and checks source and policy hashes, exact schedules,
journals, response identity, judge separation, invalid-denominator policy, and
privacy before writing an output directory. The paper-asset builder emits
one-route-per-row tables, headline macros, and nine visual families. Detailed
commands and the clean-extraction verification procedure are in
`docs/release/REPRODUCIBILITY.md`.

The April pilot pipeline remains in the repository for historical forensic
replay, but its legacy labels, broken commons mechanics, tables, and plots are
not inputs to the definitive analyzer, Croissant metadata, or supplement.

The paper-facing definitive campaign uses the planning roster in
`experiments/sota_cross_axis_panel.json` and the original-scale Part 2 contract
in `experiments/sota_cross_axis_part2_100day_panel.json`, then freezes the exact
executable subsets in its source manifests: 22 Part 0 routes, 75 Part 1 routes,
and 23 Part 2 routes. Part 2 composes a 21-route source/overlay pair with exact
Nemotron Ultra and DeepSeek V4 Flash singleton source/overlay pairs; Opus 4.5
is the sole declared exclusion, without substitution. Exact authenticated
routes were compatibility-probed before dispatch; study target IDs, returned
identities, requests, and response hashes remain bound in private journals.
The final public artifact includes only validated, text-free aggregates and
explicit availability records. Exact finalization commands are in
`docs/release/REPRODUCIBILITY.md`.

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

These discovery commands are retained for a new collection. They do not alter
definitive primary evidence or convert unavailable systems into observations.

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

Regenerate the definitive tables and figures from the text-free aggregate:

```bash
uv run python -m analysis.build_provider_safe_v2_paper_assets \
  --input-dir data/processed/provider-safe-v2-definitive-analysis \
  --local-controls data/analysis/local_hf_part1_controls.json \
  --output-dir data/processed/provider-safe-v2-paper-assets
```

The LaTeX paper imports the generated asset macros, tables, and vector figures.
The asset manifest SHA-256-binds every output; superseded graphs under
`data/graphs/` are not paper inputs.

Build the anonymous supplement:

```bash
uv run python -m analysis.build_supplement --require-definitive-artifacts
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

The final build must use the current definitive paper-asset directory and pass
the format test before the PDF is released.

## Validation Philosophy

Validation is separate from plotting. The definitive analyzer checks source
manifests and journals before figures are rendered, including:

- complete, self-hash-valid source manifests;
- exact work-ID coverage without duplicate independent units;
- source, route, served-identity, and rate-policy bindings;
- Part 0 judge/subject disjointness and explicit unjudged states;
- Part 1 root/cell coverage and primary invalid denominators;
- Part 2 trajectory continuity and reserve/population transitions; and
- privacy-safe, text-free output schemas and hashes.

This validation does not prove that the automated Part 0 judge is semantically correct. It verifies that the recorded artifacts are internally consistent and suitable for downstream analysis.

## Data And Safety

Part 0 uses harmful-request prompts and model completions for safety evaluation.
Do not republish raw harmful prompts or completions. The anonymous supplement
excludes all private manifests, journals, prompts, responses, reasoning, routes,
credentials, interrupted artifacts, and deprecated legacy outputs. It includes
only reviewed code, documentation, tests, and validated text-free definitive
aggregates.

The current results are task-specific descriptive evidence, not a leaderboard.
Part 2 uses the corrected 50-agent, 100-day engine and completed 12 independent
common-seed trajectories per route across 23 exact routes (276 total); the seed
set is common across routes. Its fixed dynamics use initial capacity 2,500,
OPTION_B private gain 2, reserve cost 2, unanimous group benefit/penalty 5, and
collapse death rate 0.2. Environmental estimates use the 198 zero-invalid
trajectories; simulator continuation after a genuine semantic `INVALID` action
is not treated as environmental evidence. It does not reuse the mismatched
April pilot trajectories or the sealed historical 12-day result.

## Useful Release Documents

- `docs/release/DATA_CARD.md`: datasheet-style overview of the benchmark and release.
- `docs/release/MODEL_REGISTRY.md`: model tags, families, parameter notes, and source links.
- `docs/release/COMPUTE.md`: compute and execution notes.
- `docs/release/LICENSES_AND_TERMS.md`: code, data, and upstream artifact terms.
- `docs/release/REPRODUCIBILITY.md`: expanded reproduction notes for reviewers.

## Submission Gates

Before using new results in a paper or release:

1. Every definitive source phase and required operational overlay is terminal,
   and the ordered three-pair Part 2 composition passes the fail-closed
   analyzer.
2. `uv run pytest -q` passes.
3. Every paper-used table, macro, and figure is bound by the definitive analysis
   and paper-asset manifests.
4. Every central paper claim traces to a generated table, macro, figure, or
   documented source file.
5. Croissant `--check` and the strict supplement build pass.
6. A clean extracted supplement rebuilds without a Git object store, and the
   compiled PDF passes format, anonymity, citation, and visual inspection.
