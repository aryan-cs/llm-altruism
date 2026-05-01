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
- `TrustAIRLab/in-the-wild-jailbreak-prompts`: MIT license for `jailbreak_llms`, public page at `https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts`.

These sources are cited in the paper and configuration files. Even though the public source pages are MIT-labeled, the default anonymous supplement does not redistribute the raw Part 0 prompt-source CSVs.

## Safety Release Policy

Raw Part 0 model outputs can contain harmful requests, jailbreak wrappers, model reasoning, and model responses. The default anonymous supplement therefore excludes:

- `data/raw/part_0/*.csv`,
- `data/raw/part_0/*_meta.json`,
- `data/raw/part_0/prompts/*.csv`.

The released supplement includes derived Part 0 aggregate tables and figures, but not raw harmful prompt text or completions. Any future hosted raw Part 0 release should receive a separate safety review and should use explicit access conditions rather than the default anonymous supplement policy.

## Supplement Package

Build the anonymous supplement with:

```bash
uv run python -m analysis.build_supplement
```

The script writes `docs/conference_submission/supplement.zip` and adds `SUPPLEMENT_MANIFEST.json` inside the archive. The manifest lists included files and records the policy exclusions above.
