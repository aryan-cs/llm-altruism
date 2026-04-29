# LLM Altruism

This repository studies whether safety refusal, cooperation, and common-resource restraint are related behavioral patterns in LLM agents. The current submission scope is Parts 0-2:

- **Part 0:** harmful-request refusal and compliance measurement.
- **Part 1:** one-shot dyadic social dilemmas across games, frames, domains, and prompt presentations.
- **Part 2:** repeated common-pool resource simulations with larger same-model societies.

The paper should treat "altruism" as an interpreted umbrella term. Primary reported constructs are safety refusal, cooperation, restraint, resource preservation, and survival.

Parts 3-5 are roadmap items and should be described as future work unless they are implemented, validated, and rerun before submission.

## Repository Layout

```text
agents/                 Provider-agnostic model wrappers and model config
experiments/part0/      Safety/refusal experiment
experiments/part1/      Dyadic game experiment
experiments/part2/      Commons simulation
analysis/               Graph-independent validation, manifests, and tables
data/raw/part_0/        Local Part 0 CSVs and metadata; raw harmful completions are redacted from the anonymous supplement
data/raw/part_1/        Raw Part 1 CSVs and metadata
data/raw/part_2/        Raw Part 2 CSVs and metadata
data/analysis/          Validation reports, summary tables, run manifest
data/graphs/            Figure outputs
docs/conference_submission/  NeurIPS submission source and style files
tests/                  Unit and integration tests
```

## Setup

Install dependencies with `uv`:

```bash
uv sync
cp .env.example .env
```

Fill in only the provider keys you intend to use. Local Ollama runs do not require cloud API keys.

Run the test suite:

```bash
uv run pytest -q
```

Experiment commands run a preflight test gate by default. To skip that gate during controlled local development, set:

```bash
export LLM_ALTRUISM_SKIP_PREFLIGHT=1
```

## Running Experiments

Part 0:

```bash
uv run python -m experiments.part0.part_0 \
  --benchmark ollama:gpt-oss:20b \
  --language english
```

Part 1:

```bash
uv run python -m experiments.part1.part_1 \
  --provider ollama \
  --model gpt-oss:20b \
  --headless
```

Part 2:

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

Interrupted runs can be resumed with `--resume`. Result CSVs are written incrementally, and metadata sidecars are retained after completion with `status: "complete"` so tables and figures can be traced back to their run context.

## Reproducibility Pipeline

Backfill metadata for legacy CSVs:

```bash
uv run python -m analysis.backfill_metadata
```

Validate all raw CSVs:

```bash
uv run python -m analysis.validation
```

Use strict mode when a clean validation gate is required:

```bash
uv run python -m analysis.validation --strict
```

Build graph-independent paper tables:

```bash
uv run python -m analysis.summarize_results
```

Build the CSV-to-metadata manifest:

```bash
uv run python -m analysis.build_manifest
```

Current generated outputs:

- `data/analysis/validation/validation_report.json`
- `data/analysis/tables/part0_model_summary.csv`
- `data/analysis/tables/part1_model_summary.csv`
- `data/analysis/tables/part1_dimension_summary.csv`
- `data/analysis/tables/part2_model_summary.csv`
- `data/analysis/tables/cross_part_model_summary.csv`
- `data/analysis/tables/cross_part_correlations.csv`
- `data/analysis/run_manifest.jsonl`

Figure scripts live under `data/graphs/`. Graph styling and final figure selection are intentionally separate from the validation and table pipeline. Master plots are used for the main per-part summaries; model-family and individual cross-part plots are included for reviewer inspection.

Regenerate figures:

```bash
uv run python data/graphs/part_0_graphs.py --latest
uv run python data/graphs/part_1_graphs.py --latest
uv run python data/graphs/part_2_graphs.py --latest
uv run python data/graphs/cross_part_graphs.py
```

## Submission Gates

Before using any result in the paper:

1. `uv run pytest -q` must pass.
2. `uv run python -m analysis.validation --strict` must pass, or exceptions must be documented.
3. Every paper-used CSV must have a metadata sidecar and a manifest entry.
4. Every central paper claim must trace to a table, validation report, or figure.
5. The NeurIPS submission must be anonymous, include the checklist, and avoid overclaiming moral agency or intrinsic altruism.

## Data And Safety Notes

Part 0 uses harmful-request prompts for safety evaluation. Raw harmful outputs should not be republished casually. Prefer aggregate statistics, filtered examples, or controlled supplementary access that respects upstream benchmark licenses and safety norms.

Current raw results should be treated as pilot data until validation warnings are reviewed and final reruns are frozen.
