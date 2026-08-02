"""Stable model grouping metadata for paper-facing analyses.

The raw legacy files predate explicit family/developer/cohort fields.  This
module supplies those fields for the original pilot and lets newer runs state
them in their metadata sidecars.  Unknown models remain visibly unclassified;
we do not guess a developer from a provider or repository owner.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


ORIGINAL_PILOT = "original_pilot"
CURRENT_SOTA = "current_sota"
HISTORICAL = "historical"
UNCLASSIFIED = "unclassified"
VALID_COHORTS = frozenset({ORIGINAL_PILOT, CURRENT_SOTA, HISTORICAL, UNCLASSIFIED})


@dataclass(frozen=True)
class ModelMetadata:
    family_id: str
    developer_id: str
    cohort: str


# All models in the initially released local/open-weight evaluation.  Family
# and developer identify the upstream base model, rather than the Ollama
# packager or the publisher of a derivative checkpoint.
_PILOT_PREFIXES: tuple[tuple[str, str, str], ...] = (
    ("gpt_oss", "gpt_oss", "openai"),
    ("gurubot/gpt_oss", "gpt_oss", "openai"),
    ("llama2", "llama_2", "meta"),
    ("qwen2.5", "qwen_2_5", "alibaba_qwen"),
    ("huihui_ai/qwen2.5", "qwen_2_5", "alibaba_qwen"),
    ("qwen3.5", "qwen_3_5", "alibaba_qwen"),
    ("aratan/qwen3.5", "qwen_3_5", "alibaba_qwen"),
    ("sorc/qwen3.5", "qwen_3_5", "alibaba_qwen"),
)


# Exact inference-hub routes supplied for the expanded evaluation.  Exact
# matching is intentional: a newly released suffix or preview is not silently
# folded into a cohort it was never evaluated as part of.
_MODEL_REGISTRY: dict[str, ModelMetadata] = {
    "gpt_5.6_sol": ModelMetadata("gpt_5", "openai", CURRENT_SOTA),
    "claude_fable_5": ModelMetadata("claude_5", "anthropic", CURRENT_SOTA),
    "claude_opus_5": ModelMetadata("claude_5", "anthropic", CURRENT_SOTA),
    "claude_sonnet_5": ModelMetadata("claude_5", "anthropic", CURRENT_SOTA),
    "claude_haiku_4_5_20251001": ModelMetadata("claude_4_5", "anthropic", CURRENT_SOTA),
    "gemini_3.1_pro_preview": ModelMetadata("gemini_3", "google", CURRENT_SOTA),
    "gemini_3.6_flash": ModelMetadata("gemini_3", "google", CURRENT_SOTA),
    "google/gemma_4_31b_it": ModelMetadata("gemma_4", "google", CURRENT_SOTA),
    "nvidia/nemotron_3_ultra_550b_a55b": ModelMetadata("nemotron_3", "nvidia", CURRENT_SOTA),
    "deepseek_ai/deepseek_v4_pro": ModelMetadata("deepseek_v4", "deepseek", CURRENT_SOTA),
    "qwen/qwen3_next_80b_a3b_thinking": ModelMetadata("qwen_3_next", "alibaba_qwen", CURRENT_SOTA),
    "moonshotai/kimi_k2_thinking": ModelMetadata("kimi_k2", "moonshot_ai", CURRENT_SOTA),
    "z_ai/glm_5.2": ModelMetadata("glm_5_2", "z_ai", CURRENT_SOTA),
    "mistralai/mistral_nemotron": ModelMetadata(
        "mistral_nemotron", "mistral_ai", CURRENT_SOTA
    ),
    "gpt_3.5_turbo_0125": ModelMetadata("gpt_3_5", "openai", HISTORICAL),
    "gpt_4.1_2025_04_14": ModelMetadata("gpt_4_1", "openai", HISTORICAL),
    "gpt_5_2025_08_07": ModelMetadata("gpt_5", "openai", HISTORICAL),
    "gemini_2.5_pro": ModelMetadata("gemini_2_5", "google", HISTORICAL),
    "google/gemma_3_27b_it": ModelMetadata("gemma_3", "google", HISTORICAL),
    "openai/gpt_oss_120b": ModelMetadata("gpt_oss", "openai", HISTORICAL),
}


def _normalized_model(model: str) -> str:
    return model.strip().lower().replace("-", "_")


def _metadata_value(metadata: Mapping[str, object] | None, key: str) -> str:
    if not metadata:
        return ""
    candidates: list[Mapping[str, object]] = [metadata]
    nested = metadata.get("model_metadata")
    if isinstance(nested, Mapping):
        candidates.insert(0, nested)
    parameters = metadata.get("parameters")
    if isinstance(parameters, Mapping):
        parameter_metadata = parameters.get("model_metadata")
        if isinstance(parameter_metadata, Mapping):
            candidates.insert(0, parameter_metadata)
    for candidate in candidates:
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _stable_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _sidecar_registry_target(
    metadata: Mapping[str, object] | None,
    model: str,
) -> Mapping[str, object] | None:
    if not metadata:
        return None
    registry = metadata.get("model_registry")
    if not isinstance(registry, Mapping):
        return None
    targets = registry.get("targets")
    if not isinstance(targets, list):
        return None
    for target in targets:
        if not isinstance(target, Mapping):
            continue
        identifiers = {str(target.get(key, "")).strip() for key in ("model", "route")}
        if model.strip() in identifiers:
            return target
    return None


def resolve_model_metadata(
    provider: str,
    model: str,
    metadata: Mapping[str, object] | None = None,
) -> ModelMetadata:
    """Resolve explicit sidecar fields, with a legacy-pilot compatibility map.

    New runs should record ``family_id``, ``developer_id``, and ``cohort``
    either at sidecar top level or inside ``model_metadata``.  Requiring
    explicit values for unknown models prevents repository names (which often
    denote a fine-tuner) from being mistaken for upstream developers.
    """

    family_id = _metadata_value(metadata, "family_id")
    developer_id = _metadata_value(metadata, "developer_id")
    cohort = _metadata_value(metadata, "cohort")

    sidecar_target = _sidecar_registry_target(metadata, model)
    if sidecar_target:
        upstream_provider = sidecar_target.get("upstream_provider")
        if not developer_id and upstream_provider is not None:
            developer_id = str(upstream_provider).strip()
        memberships = sidecar_target.get("cohorts")
        if not cohort and isinstance(memberships, list) and len(memberships) == 1:
            cohort = str(memberships[0])

    normalized = _normalized_model(model)
    registered = _MODEL_REGISTRY.get(normalized)
    if registered:
        family_id = family_id or registered.family_id
        developer_id = developer_id or registered.developer_id
        cohort = cohort or registered.cohort
    pilot_match = next(
        (
            (family, developer)
            for prefix, family, developer in _PILOT_PREFIXES
            if normalized.startswith(prefix)
        ),
        None,
    )
    if pilot_match:
        family_id = family_id or pilot_match[0]
        developer_id = developer_id or pilot_match[1]
        cohort = cohort or ORIGINAL_PILOT

    family_id = _stable_id(family_id) if family_id else UNCLASSIFIED
    developer_id = _stable_id(developer_id) if developer_id else UNCLASSIFIED
    cohort = _stable_id(cohort) if cohort else UNCLASSIFIED
    if cohort not in VALID_COHORTS:
        allowed = ", ".join(sorted(VALID_COHORTS - {UNCLASSIFIED}))
        raise ValueError(
            f"Invalid cohort {cohort!r} for {provider}/{model}; expected one of {allowed}"
        )
    return ModelMetadata(family_id, developer_id, cohort)
