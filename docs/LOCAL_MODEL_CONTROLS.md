# Offline local-model controls

The local panel is a separate exploratory scale control. It is not a fallback
name for an unavailable hosted model, does not satisfy an InferenceHub route,
and cannot be promoted into a paper result by a smoke test.

`agents/local_control.registry.json` pins four instruction-model commits:

- HuggingFaceTB/SmolLM2-135M-Instruct
- HuggingFaceTB/SmolLM2-360M-Instruct
- Qwen/Qwen2.5-0.5B-Instruct
- HuggingFaceTB/SmolLM2-1.7B-Instruct

The cache location is deliberately external to the registry. The runner
resolves the exact Hugging Face snapshot layout beneath `HF_LOCAL_CACHE_ROOT`,
rejects a missing commit or a symlink that escapes the cache, hashes every
snapshot asset recursively, disables network access and remote code, then performs a real
greedy generation. Model smokes use a bounded model-level worker pool (two
workers by default). They write an atomic private artifact after every completed
attempt and reserve every selected model before dispatch so
a later failure cannot erase earlier evidence. An existing output path is never
overwritten; every new run must use a new evidence filename.

Run the panel with any Python environment that already contains compatible
`torch` and `transformers` packages:

```bash
HF_LOCAL_CACHE_ROOT=/absolute/path/to/huggingface/hub \
python experiments/misc/local_hf_smoke.py \
  --registry agents/local_control.registry.json \
  --output data/private/local_hf/smoke-evidence-$(date +%Y%m%dT%H%M%S).json \
  --device cpu \
  --max-workers 2
```

`status=passed` means the exact pinned assets loaded and returned visible text.
`format_contract_match` separately records whether the tiny model followed the
smoke's exact `READY` instruction; it is diagnostic and does not change asset
callability. The artifact stores the response because the smoke prompt is
benign. This evidence does not authorize analysis on the unapproved Part 0 or
Part 1 banks. Paper-facing local results require the same approved inputs,
complete native artifacts, validation, estimators, and release review as every
other model panel.

For real scale testing before the human-approved banks exist, the separate
large-N runner executes the complete deterministic 384-root Part 1 draft bank
against every pinned control: 1,536 generations for the four-model registry.
Models run concurrently and prompts are batched within each loaded model. Each
model owns an append-only JSONL file, and `--resume` accepts only an exact prefix
of the hash-bound schedule and an unchanged registry/generation contract.

```bash
HF_LOCAL_CACHE_ROOT=/absolute/path/to/huggingface/hub \
python -m experiments.misc.local_hf_part1_panel \
  --output-dir data/private/local_hf/part1-large-n-YYYYMMDD-v1 \
  --max-workers 2 \
  --batch-size 8 \
  --max-new-tokens 32
```

If the process is interrupted, repeat the identical command with `--resume`.
The manifest explicitly records `draft_bank_human_approved=false` and
`confirmatory_or_paper_promotion_permitted=false`; this large-N evidence tests
execution scale and small-model behavior, not the paper's confirmatory claims.
