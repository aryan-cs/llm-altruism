# Prosocial Readiness Bench

Prosocial Readiness Bench is a behavioral evaluation suite for asking whether large language model agents that refuse harmful requests also cooperate in social dilemmas and preserve shared resources over repeated interaction.

The project started under the working name `llm-altruism`, but the benchmark does not claim to measure intrinsic altruism or moral character. It measures observable behaviors under explicit task contracts:

- **Part 0: Safety refusal.** Models answer multilingual harmful-request prompts; outputs are scored as refusal or compliance.
- **Part 1: Focal dilemma choices.** Models produce self-directed choices, advice, observer judgments, and predictions in hypothetical one-shot dilemmas.
- **Part 2: Commons restraint.** Homogeneous same-model populations repeatedly choose whether to restrain or overuse a shared resource, producing resource and population trajectories in a controlled microworld.

The paper-facing claim is that the profile is non-redundant but partially coupled. Refusal and commons restraint are positively associated in the pilot, while role-conditioned responses and model-level discordances show that refusal alone cannot substitute for the other probes. The codebase makes these claims auditable from raw traces, metadata sidecars, validation reports, summary tables, figures, and a packaged supplement.

## What Is In This Repository

```text
agents/                       Provider-agnostic agent wrappers and model config
providers/                    API/Ollama call adapters
experiments/part0/            Harmful-request refusal experiment
experiments/part1/            One-shot social dilemma experiment
experiments/part2/            Repeated common-pool resource simulation
experiments/misc/             Shared prompt loading, result writing, metadata, preflight checks
analysis/                     Validation, summary tables, manifests, supplement build, figure syncing
data/raw/part_0/              Local Part 0 CSVs and metadata
data/raw/part_1/              Raw Part 1 CSVs and metadata
data/raw/part_2/              Raw Part 2 CSVs and metadata
data/analysis/                Validation reports, derived tables, Croissant metadata, manifests
data/graphs/                  Generated figures and diagnostics
docs/conference_submission/   NeurIPS submission source and figures
docs/release/                 Data card, model registry, release terms, public proposal PDF
tests/                        Unit and integration tests
```

Parts 3-5 are roadmap placeholders. Treat Parts 0-2 as the validated benchmark scope unless additional parts are implemented, validated, and rerun.

## Setup

Install dependencies with `uv` from the repository root:

```bash
uv sync
cp .env.example .env
```

Fill in only the provider credentials you plan to use. Local Ollama runs do not require cloud API keys, but they do require Ollama to be installed and the requested model tag to be available locally.
The configured NVIDIA-hosted Inference Hub uses `NVIDIA_API_KEY`; its base URL
defaults to `https://integrate.api.nvidia.com/v1` and can be overridden with
`INFERENCE_HUB_BASE_URL`.

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

Part 2 runs a repeated commons simulation:

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
uv run python -m analysis.validation --strict
uv run python -m analysis.summarize_results
uv run python -m analysis.build_manifest
```

The current validation report is written to:

```text
data/analysis/validation/validation_report.json
```

The main derived tables are written under:

```text
data/analysis/tables/
```

Important table outputs include:

- `part0_model_summary.csv`
- `part0_language_robustness.csv`
- `part1_model_summary.csv`
- `part1_dimension_summary.csv`
- `part1_frame_effects.csv`
- `part1_prompt_sensitivity.csv`
- `part1_factor_decomposition.csv`
- `part2_model_summary.csv`
- `part2_run_summary.csv`
- `cross_part_model_summary.csv`
- `cross_part_correlations.csv`

To run an exact-version registry cohort through smoke tests and the three
benchmark parts, use the resumable campaign runner. A dry run prints and records
the planned request counts without contacting any provider:

```bash
uv run python -m experiments.campaign \
  --cohort current_sota \
  --phase smoke --phase part0 --phase part1 --phase part2 \
  --dry-run
```

The `current_sota` and `historical` cohorts are defined in
`agents/agent_config.registry.json`. Registry membership is a run plan, not a
claim that a provider route is available or that its results appear in the
paper. A model enters the result set only after successful endpoint smoke tests,
completed native artifacts, and validation.

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

The LaTeX source intentionally uses local paths like `figures/part0_refusal_rate_by_model.png` so the conference submission directory is self-contained. If you regenerate plots under `data/graphs/`, run `uv run python -m analysis.sync_conference_figures` from the repository root before compiling the paper.

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

Part 0 uses harmful-request prompts and model completions for safety evaluation. Do not casually republish raw harmful prompts or completions. The anonymous supplement excludes raw Part 0 prompt-source CSVs, raw Part 0 metadata sidecars, and raw Part 0 harmful completions by default. It includes derived Part 0 aggregate tables and figures plus Part 1/Part 2 raw CSVs and metadata.

Part 1 and Part 2 prompts, traces, and metadata are intended for auditability. Treat the current results as a pilot snapshot, not a final leaderboard. The Part 2 runs in the paper are point estimates from one same-model trajectory per model unless explicitly stated otherwise.

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
