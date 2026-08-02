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


def _normalized_model(model: str) -> str:
    return model.strip().lower().replace("-", "_")


def _build_exact_model_registry() -> dict[str, ModelMetadata]:
    """Bind every frozen model and backend route without family guessing."""

    groups: tuple[tuple[ModelMetadata, tuple[str, ...]], ...] = (
        (
            ModelMetadata("gpt_5_6", "openai", CURRENT_SOTA),
            (
                "gpt-5.6-sol",
                "azure/openai/gpt-5.6-sol",
                "gpt-5.6-terra",
                "azure/openai/gpt-5.6-terra",
                "gpt-5.6-luna",
                "azure/openai/gpt-5.6-luna",
            ),
        ),
        (
            ModelMetadata("claude_4_5", "anthropic", CURRENT_SOTA),
            (
                "claude-opus-4-5",
                "aws/anthropic/claude-opus-4-5",
                "claude-haiku-4-5-v1",
                "aws/anthropic/claude-haiku-4-5-v1",
                "bedrock-claude-sonnet-4-5-v1",
                "aws/anthropic/bedrock-claude-sonnet-4-5-v1",
            ),
        ),
        (
            ModelMetadata("claude_5", "anthropic", CURRENT_SOTA),
            (
                "claude-opus-5",
                "azure/anthropic/claude-opus-5",
                "claude-sonnet-5",
                "azure/anthropic/claude-sonnet-5",
            ),
        ),
        (
            ModelMetadata("gemini_3", "google", CURRENT_SOTA),
            (
                "gemini-3.1-pro-preview",
                "gcp/google/gemini-3.1-pro-preview",
                "gemini-3.6-flash",
                "gcp/google/gemini-3.6-flash",
                "gemini-3.5-flash",
                "gcp/google/gemini-3.5-flash",
                "gemini-3-flash-preview",
                "gcp/google/gemini-3-flash-preview",
                "gemini-3.1-flash-lite",
                "gcp/google/gemini-3.1-flash-lite",
            ),
        ),
        (
            ModelMetadata("gemma_4", "google", CURRENT_SOTA),
            ("google/gemma-4-31b-it", "nvidia/google/gemma-4-31b-it"),
        ),
        (
            ModelMetadata("nemotron_3", "nvidia", CURRENT_SOTA),
            (
                "nvidia/nemotron-3-ultra",
                "nvidia/nvidia/nemotron-3-ultra",
                "nvidia/nemotron-3-super-v3",
                "nvidia/nvidia/nemotron-3-super-v3",
            ),
        ),
        (
            ModelMetadata("deepseek_v4", "deepseek", CURRENT_SOTA),
            (
                "deepseek-ai/deepseek-v4-pro",
                "nvidia/deepseek-ai/deepseek-v4-pro",
                "deepseek-ai/deepseek-v4-flash",
                "nvidia/deepseek-ai/deepseek-v4-flash",
            ),
        ),
        (
            ModelMetadata("qwen_3_6", "alibaba_qwen", CURRENT_SOTA),
            ("qwen/qwen3.6-35b-a3b", "nvidia/qwen/qwen3.6-35b-a3b"),
        ),
        (
            ModelMetadata("kimi_k2_6", "moonshot_ai", CURRENT_SOTA),
            ("moonshotai/kimi-k2.6", "nvidia/moonshotai/kimi-k2.6"),
        ),
        (
            ModelMetadata("glm_5_2", "z_ai", CURRENT_SOTA),
            ("zai-org/glm-5.2", "nvidia/zai-org/glm-5.2"),
        ),
        (
            ModelMetadata("mixtral_8x22b", "mistral_ai", CURRENT_SOTA),
            (
                "mistralai/mixtral-8x22b-instruct-v01",
                "nvidia/mistralai/mixtral-8x22b-instruct-v01",
            ),
        ),
        (
            ModelMetadata("minimax_m3", "minimax", CURRENT_SOTA),
            ("minimaxai/minimax-m3", "nvidia/minimaxai/minimax-m3"),
        ),
        (
            ModelMetadata("gpt_oss", "openai", CURRENT_SOTA),
            ("openai/gpt-oss-20b", "nvidia/openai/gpt-oss-20b"),
        ),
        (
            ModelMetadata("gpt_3_5", "openai", HISTORICAL),
            (
                "gpt-3.5-turbo",
                "openai/openai/gpt-3.5-turbo",
                "gpt-3.5-turbo-0125",
            ),
        ),
        (
            ModelMetadata("gpt_4_1", "openai", HISTORICAL),
            ("gpt-4.1", "us/azure/openai/gpt-4.1", "gpt-4.1-2025-04-14"),
        ),
        (
            ModelMetadata("gpt_5", "openai", HISTORICAL),
            ("gpt-5", "us/azure/openai/gpt-5", "gpt-5-2025-08-07"),
        ),
        (
            ModelMetadata("gemini_2_5", "google", HISTORICAL),
            ("gemini-2.5-pro", "gcp/google/gemini-2.5-pro"),
        ),
        (
            ModelMetadata("gemma_2", "google", HISTORICAL),
            ("google/gemma-2-9b-it", "nvidia/google/gemma-2-9b-it"),
        ),
        (
            ModelMetadata("gpt_oss", "openai", HISTORICAL),
            ("openai/gpt-oss-120b", "nvidia/openai/gpt-oss-120b"),
        ),
    )
    registry: dict[str, ModelMetadata] = {}
    for metadata, aliases in groups:
        for alias in aliases:
            normalized = _normalized_model(alias)
            previous = registry.setdefault(normalized, metadata)
            if previous != metadata:
                raise RuntimeError(f"Conflicting exact model metadata: {alias}")
    return registry


# Exact matching is intentional: a newly released suffix or preview is not
# silently folded into a cohort it was never evaluated as part of.
_MODEL_REGISTRY = _build_exact_model_registry()


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
