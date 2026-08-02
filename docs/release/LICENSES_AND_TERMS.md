# Licenses And Release Terms

This document summarizes license and release handling for the anonymous paper artifact.

## Repository Code And Generated Artifacts

- Repository code, documentation, generated Part 1/Part 2 benchmark prompts, analysis scripts, validation outputs, derived tables, and figures are released under the repository MIT license unless another file states otherwise.
- The repository license is stored at `LICENSE`.
- The paper-facing Croissant metadata is stored at `data/analysis/croissant_metadata.json`.

## External Part 0 Sources

Part 0 uses harmful-request sources only for safety/refusal evaluation. Public source pages list the following license information:

- `JailbreakBench/JBB-Behaviors`: MIT license, public page at `https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors`.
- `walledai/HarmBench`: MIT license, public page at `https://huggingface.co/datasets/walledai/HarmBench`.
These two sources are cited in the paper and configuration files. The generating code loaded their behavior rows directly; the paper therefore does not claim that TrustAIRLab wrapper templates were used. Even though the public source pages are MIT-labeled, the default anonymous supplement does not redistribute the raw Part 0 prompt-source CSVs.

## Model Terms

The pilot records mutable Ollama tags but not immutable digests, upstream model-card snapshots, or the terms shown when each tag was pulled. The repository therefore cannot establish that one current license page is the exact governing version for every April artifact, especially community-published uncensored, derestricted, and abliterate derivatives. No model weights are redistributed. `MODEL_REGISTRY.md` records the exact retained tags and missing provenance; this gap is disclosed as unresolved rather than represented as complete license verification. Confirmatory runs must archive the model card, license/terms URL and hash, immutable digest, and access date before execution.

## Safety Release Policy

Raw Part 0 model outputs can contain harmful requests, jailbreak wrappers, model reasoning, and model responses. The default anonymous supplement therefore excludes:

- `data/raw/part_0/*.csv`,
- `data/raw/part_0/*_meta.json`,
- `data/raw/part_0/prompts/*.csv`.

The released supplement excludes the invalid legacy Part 0 aggregate tables and figures and every dependent cross-part output, as well as the raw harmful prompt text and completions. It includes only a sanitized aggregate checkpoint documenting partial response-only rejudgment instability. Any future hosted raw Part 0 release should receive a separate safety review and should use explicit access conditions rather than the default anonymous supplement policy.

## Supplement Package

Build the anonymous supplement with:

```bash
uv run python -m analysis.build_supplement
```

The script writes `docs/conference_submission/supplement.zip` and adds `SUPPLEMENT_MANIFEST.json` inside the archive. The manifest lists included files and records the policy exclusions above.
