# Licenses And Release Terms

This document summarizes license and release handling for the anonymous paper artifact.

## Repository Code And Generated Artifacts

- Repository code, documentation, analysis scripts, tests, sanitized aggregate
  tables, and figures are released under the repository MIT license unless
  another file states otherwise.
- The repository license is stored at `LICENSE`.
- The paper-facing Croissant metadata is stored at
  `data/processed/provider-safe-v2-croissant-metadata.json` after the
  definitive analysis and paper assets pass their completion gates.

## External Part 0 Sources

Part 0 uses harmful-request sources only for safety/refusal evaluation. Public source pages list the following license information:

- `JailbreakBench/JBB-Behaviors`: MIT license, public page at `https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors`.
- `walledai/HarmBench`: MIT license, public page at `https://huggingface.co/datasets/walledai/HarmBench`.
These two sources are cited in the paper and configuration files. The generating code loaded their behavior rows directly; the paper therefore does not claim that TrustAIRLab wrapper templates were used. Even though the public source pages are MIT-labeled, the default anonymous supplement does not redistribute the raw Part 0 prompt-source CSVs.

## Model Terms

No model weights are redistributed. Current hosted runs retain exact request and
response identities privately, but the serving providers' model and API terms
continue to govern access. Study target IDs and aggregate results do not grant
rights to model weights or provider outputs. The historical pilot records
mutable Ollama tags without immutable digests or model-card snapshots, so its
model-license provenance remains incomplete and is not represented as current
hosted provenance.

## Safety Release Policy

Raw Part 0 model outputs can contain harmful requests, jailbreak wrappers, model reasoning, and model responses. The default anonymous supplement therefore excludes:

- `data/raw/part_0/*.csv`,
- `data/raw/part_0/*_meta.json`,
- `data/raw/part_0/prompts/*.csv`.

The supplement excludes legacy Part 0 tables and every dependent legacy
cross-part output, as well as harmful prompt text and completions. Current
hosted Part 0 raw prompts, responses, reasoning, judge payloads, and journals
also remain private. The same exclusion applies to all private Part 1 and Part
2 manifests, routes, journals, prompts, responses, and reasoning. Only
text-free aggregates from complete definitive source manifests and separately
labeled, validated availability or semantic-repair summaries may enter the
package. No retry artifact may replace a primary observation or denominator.
Any future raw release requires separate safety, privacy, and upstream-license
review.

## Supplement Package

Build the anonymous supplement with:

```bash
uv run python -m analysis.build_supplement --require-definitive-artifacts
```

The script writes `docs/conference_submission/supplement.zip` and adds `SUPPLEMENT_MANIFEST.json` inside the archive. The manifest lists included files and records the policy exclusions above.
