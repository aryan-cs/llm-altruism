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
snapshot asset, disables network access and remote code, then performs a real
greedy generation. It writes an atomic private artifact after every attempt so
a later failure cannot erase earlier evidence.

Run the panel with any Python environment that already contains compatible
`torch` and `transformers` packages:

```bash
HF_LOCAL_CACHE_ROOT=/absolute/path/to/huggingface/hub \
python experiments/misc/local_hf_smoke.py \
  --registry agents/local_control.registry.json \
  --output data/private/local_hf/smoke-evidence.json \
  --device cpu
```

`status=passed` means the exact pinned assets loaded and returned visible text.
`format_contract_match` separately records whether the tiny model followed the
smoke's exact `READY` instruction; it is diagnostic and does not change asset
callability. The artifact stores the response because the smoke prompt is
benign. This evidence does not authorize analysis on the unapproved Part 0 or
Part 1 banks. Paper-facing local results require the same approved inputs,
complete native artifacts, validation, estimators, and release review as every
other model panel.
