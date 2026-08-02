# Pilot Model Registry

The paper audits 13 related local/Ollama model tags recorded in the April 2026
Part 1 and Part 2 artifacts. Exact tag strings and their short paper labels are
listed in the paper's Model Registry appendix and in each raw CSV/metadata
sidecar. The cohort spans GPT-OSS, Llama 2, Qwen 2.5, and Ollama-hosted Qwen 3.5
variants.

This registry is descriptive provenance, not a stable model-family taxonomy.
The April metadata did not preserve immutable Ollama digests, quantization,
fixed temperature, or an explicit greedy-decoding setting. Mutable tags may
therefore resolve to different bytes in a later environment. Parameter scale is
reported only where encoded in the recorded tag or release name, and vendor- or
architecture-level inference is unsupported because the variants are related
and were not sampled independently.

For the exact recorded identities, consult:

- `docs/conference_submission/conference_submission.tex` (Model Registry appendix),
- `data/analysis/run_manifest.jsonl`, and
- `data/raw/part_1/*.metadata.json` and `data/raw/part_2/*.metadata.json`.

Post-pilot hosted-route rosters and local scale-control registries are outside
this paper's evidence scope and are intentionally absent from the anonymous
supplement.
