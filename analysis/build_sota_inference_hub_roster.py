"""Build an outcome-blind SOTA text-chat roster from an Inference Hub catalog.

The catalog only states that routes exist; it does not advertise a trustworthy
modality.  Consequently this module uses a frozen, reviewable taxonomy and
fails closed when a route matches no class (or more than one class).  Model
identity is deliberately conservative: routes are grouped only when both the
upstream-provider path component and the final model identifier are identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
BACKEND_PRIORITY = (
    "openai/openai/",
    "gcp/google/",
    "azure/anthropic/",
    "azure/openai/",
    "azure/",
    "aws/",
    "nvcf/",
    "nvidia/",
    "perplexity/",
    "us/azure/",
    "switchyard/",
    "nvidia_dynamo/",
)

# Every subject is named explicitly.  Adding a catalog route never silently
# expands the research panel.
SOTA_MODEL_KEYS = frozenset(
    {
        # Anthropic: identifiers remain distinct (including the Bedrock prefix).
        *(('anthropic', name) for name in (
            'bedrock-claude-opus-4-6', 'bedrock-claude-opus-4-7',
            'bedrock-claude-opus-4-8', 'bedrock-claude-opus-5',
            'bedrock-claude-sonnet-4-5-v1', 'bedrock-claude-sonnet-4-6',
            'bedrock-claude-sonnet-5', 'claude-haiku-4-5',
            'claude-haiku-4-5-v1', 'claude-opus-4-5', 'claude-opus-4-6',
            'claude-opus-4-7', 'claude-opus-4-8', 'claude-opus-5',
            'claude-sonnet-4-5', 'claude-sonnet-4-6', 'claude-sonnet-5',
        )),
        # Google Gemini text-chat routes (image/omni/embedding routes excluded).
        *(('google', name) for name in (
            'gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.5-pro',
            'gemini-3-flash-preview', 'gemini-3.1-flash-lite',
            'gemini-3.1-pro-preview', 'gemini-3.5-flash', 'gemini-3.6-flash',
        )),
        # OpenAI GPT generations 3, 4 and 5, plus current reasoning models.
        *(('openai', name) for name in (
            'gpt-3.5-turbo', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano',
            'gpt-4o', 'gpt-4o-mini', 'gpt-5', 'gpt-5-chat', 'gpt-5-mini',
            'gpt-5-nano', 'gpt-5.1', 'gpt-5.1-chat', 'gpt-5.1-codex',
            'gpt-5.1-codex-max', 'gpt-5.1-codex-mini', 'gpt-5.2',
            'gpt-5.2-chat', 'gpt-5.2-codex', 'gpt-5.3-chat',
            'gpt-5.3-codex', 'gpt-5.4', 'gpt-5.4-mini', 'gpt-5.4-nano',
            'gpt-5.4-pro', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol',
            'gpt-5.6-terra', 'o1', 'o3', 'o3-mini', 'o4-mini',
            'gpt-oss-20b', 'gpt-oss-120b',
        )),
        # Other leading open/current text-chat families in this catalog.
        *(('deepseek-ai', name) for name in ('deepseek-v4-flash', 'deepseek-v4-pro')),
        *(('moonshotai', name) for name in ('kimi-k2.5', 'kimi-k2.6')),
        *(('zai-org', name) for name in ('glm-5.1', 'glm-5.2')),
        *(('minimaxai', name) for name in ('minimax-m2.7', 'minimax-m3')),
        ('google', 'gemma-4-31b-it'),
        ('meta', 'llama-3.3-70b-instruct'),
        ('mistralai', 'mixtral-8x22b-instruct-v01'),
        *(('qwen', name) for name in (
            'qwen-235b', 'qwen3-5-397b-a17b', 'qwen3-next-80b-a3b-instruct',
            'qwen3.5-35b-a3b', 'qwen3.6-27b', 'qwen3.6-35b-a3b',
        )),
        *(('nvidia', name) for name in (
            'nemotron-3-nano-30b-a3b', 'nemotron-3-super-v3',
            'nemotron-3-ultra', 'nemotron-nano-31b-v3',
        )),
        *(('perplexity', name) for name in (
            'sonar', 'sonar-deep-research', 'sonar-pro', 'sonar-reasoning-pro',
        )),
    }
)

REQUIRED_FAMILIES: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    "claude_haiku": (("anthropic", re.compile(r"^(?:bedrock-)?claude-haiku-")),),
    "claude_opus": (("anthropic", re.compile(r"^(?:bedrock-)?claude-opus-")),),
    "claude_sonnet": (("anthropic", re.compile(r"^(?:bedrock-)?claude-sonnet-")),),
    "gemini": (("google", re.compile(r"^gemini-(?!embedding)")),),
    "openai_gpt_generation_3": (("openai", re.compile(r"^gpt-3(?:\.|-)")),),
    "openai_gpt_generation_4": (("openai", re.compile(r"^gpt-4")),),
    "openai_gpt_generation_5": (("openai", re.compile(r"^gpt-5")),),
    "deepseek": (("deepseek-ai", re.compile(r"^deepseek-")),),
    "gemma": (("google", re.compile(r"^gemma-")),),
    "glm": (("zai-org", re.compile(r"^glm-")),),
    "kimi": (("moonshotai", re.compile(r"^kimi-")),),
    "llama": (("meta", re.compile(r"^llama-")),),
    "minimax": (("minimaxai", re.compile(r"^minimax-")),),
    "mistral": (("mistralai", re.compile(r"^(?:mistral|mixtral)-")),),
    "nemotron": (("nvidia", re.compile(r"^(?:nvidia-)?nemotron-")),),
    "perplexity_sonar": (("perplexity", re.compile(r"^sonar")),),
    "qwen": (("qwen", re.compile(r"^qwen")),),
}


class RosterBuildError(ValueError):
    """The fixed inputs cannot support an unambiguous roster."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RosterBuildError(f"{label} is not readable UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise RosterBuildError(f"{label} must be a JSON object")
    return value


def _catalog_routes(catalog: Mapping[str, Any]) -> list[str]:
    if catalog.get("schema_version") != 2 or catalog.get("source_endpoints") != ["/models"]:
        raise RosterBuildError("catalog must be schema 2 from only the /models endpoint")
    rows = catalog.get("routes")
    if not isinstance(rows, list):
        raise RosterBuildError("catalog routes must be a list")
    routes: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("listed_by_models") is not True:
            raise RosterBuildError(f"catalog route row {index} is not /models-authorized")
        route = row.get("route")
        if not isinstance(route, str) or not route.strip() or route != route.strip():
            raise RosterBuildError(f"catalog route row {index} has an invalid route")
        routes.append(route)
    if len(routes) != len(set(routes)):
        raise RosterBuildError("catalog contains duplicate routes")
    if catalog.get("route_count") != len(routes):
        raise RosterBuildError("catalog route_count does not match routes")
    return routes


def _identity(route: str) -> tuple[str, str]:
    if route in {"fake-openai-endpoint", "fusion-fake-llm"}:
        return "infrastructure", route
    parts = route.split("/")
    if len(parts) < 3 or any(not part for part in parts):
        raise RosterBuildError(f"route has no unambiguous provider/model identity: {route}")
    return parts[-2], parts[-1]


def _route_class_matches(route: str) -> list[tuple[str, str, str]]:
    """Return matching ``(modality, role, exclusion reason)`` classes."""
    _, model = _identity(route)
    value = model.lower()
    matches: list[tuple[str, str, str]] = []

    def add_if(condition: bool, modality: str, role: str, reason: str) -> None:
        if condition:
            matches.append((modality, role, reason))

    add_if(value in {"fake-openai-endpoint", "fusion-fake-llm"}, "none", "test_infrastructure", "non_model_test_infrastructure_route")
    add_if("embedding" in value or "embed" in value, "embedding", "retrieval", "non_chat_embedding_route")
    add_if("rerank" in value, "reranking", "retrieval", "non_chat_reranking_route")
    add_if(value.startswith("evals-"), "text", "judge", "judge_or_evaluation_route")
    add_if("guard" in value or "content-safety" in value, "text", "safety", "safety_route")
    add_if("tts" in value, "audio", "generation", "non_chat_audio_route")
    add_if("omni" in value, "audio_and_text", "multimodal", "multimodal_audio_route")
    add_if(value.startswith("sora-") or value.startswith("veo-"), "video", "generation", "non_chat_video_route")
    image_tokens = ("image", "vision", "-vl", "ocr", "page-elements", "table-structure")
    add_if(
        any(token in value for token in image_tokens)
        and "embed" not in value and "rerank" not in value,
        "image_or_vision", "generation_or_understanding",
        "non_text_image_or_vision_route",
    )
    add_if(value.endswith("-batch"), "text", "batch", "noninteractive_batch_route")
    add_if(value.startswith("riva-translate-"), "text", "translation", "specialized_translation_route")

    text_patterns = (
        r"^(?:bedrock-)?claude-(?:haiku|opus|sonnet)-",
        r"^gemini-(?!embedding|.*image|omni)", r"^gpt-(?!image)", r"^o[134](?:-|$)",
        r"^deepseek-", r"^kimi-", r"^glm-", r"^llama-", r"^gemma-", r"^phi-",
        r"^minimax-", r"^(?:mistral|mixtral)-", r"^(?:nvidia-)?nemotron-",
        r"^nemosmith-", r"^cosmos3-", r"^gpt-oss-", r"^qwen", r"^sonar",
        r"^Nemotron-",  # Retained for readability; matching uses IGNORECASE below.
    )
    is_text = any(re.search(pattern, model, re.IGNORECASE) for pattern in text_patterns)
    # Known non-chat markers dominate a text-family prefix (for example a Llama
    # vision model).  The resulting class is still explicit, while a wholly new
    # identifier continues to fail closed below.
    if is_text and not matches:
        role = "coding_chat" if "codex" in value else "search_chat" if value.startswith("sonar") else "general_chat"
        add_if(True, "text", role, "")
    return matches


def classify_route(route: str) -> dict[str, Any]:
    provider, model = _identity(route)
    matches = _route_class_matches(route)
    if len(matches) != 1:
        detail = "no class" if not matches else f"conflicting classes {matches!r}"
        raise RosterBuildError(f"ambiguous route type ({detail}): {route}")
    modality, role, excluded_reason = matches[0]
    return {
        "route": route,
        "upstream_provider": provider,
        "model_identifier": model,
        "modality": modality,
        "role": role,
        "eligible_text_chat": not excluded_reason,
        "taxonomy_exclusion_reason": excluded_reason,
    }


def _backend_rank(route: str) -> tuple[int, str]:
    for index, prefix in enumerate(BACKEND_PRIORITY):
        if route.startswith(prefix):
            return index, route
    return len(BACKEND_PRIORITY), route


def _registry_rows(registry: Mapping[str, Any]) -> list[dict[str, str]]:
    raw = registry.get("targets")
    if not isinstance(raw, list):
        raise RosterBuildError("registry targets must be a list")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, target in enumerate(raw):
        if not isinstance(target, Mapping):
            raise RosterBuildError(f"registry target {index} is invalid")
        target_id, route, model = target.get("id"), target.get("route"), target.get("model")
        if not all(isinstance(item, str) and item for item in (target_id, route, model)):
            raise RosterBuildError(f"registry target {index} lacks id, route, or model")
        if target_id in seen:
            raise RosterBuildError(f"duplicate registry target id: {target_id}")
        seen.add(target_id)
        upstream = target.get("upstream_provider")
        if not isinstance(upstream, str) or not upstream:
            raise RosterBuildError(f"registry target {index} lacks upstream_provider")
        rows.append({
            "target_id": target_id,
            "planned_route": route,
            "model_identifier": model,
            "upstream_provider": upstream,
        })
    return rows


def build_roster(*, catalog: Mapping[str, Any], registry: Mapping[str, Any]) -> dict[str, Any]:
    routes = _catalog_routes(catalog)
    inventory = [classify_route(route) for route in routes]
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in inventory:
        if row["eligible_text_chat"] is True:
            grouped[(row["upstream_provider"], row["model_identifier"])].append(row["route"])

    registry_rows = _registry_rows(registry)
    roster: list[dict[str, Any]] = []
    selected_routes: set[str] = set()
    for key in sorted(SOTA_MODEL_KEYS):
        candidates = sorted(grouped.get(key, []), key=_backend_rank)
        if not candidates:
            continue
        selected_routes.update(candidates)
        provider, model = key
        exact_registry = sorted(
            row["target_id"]
            for row in registry_rows
            if (
                row["upstream_provider"] == provider
                and row["model_identifier"] == model
            ) or row["planned_route"] in candidates
        )
        role = next(row["role"] for row in inventory if row["route"] == candidates[0])
        roster.append({
            "subject_id": f"{provider}/{model}",
            "upstream_provider": provider,
            "model_identifier": model,
            "modality": "text",
            "role": role,
            "preferred_candidate": candidates[0],
            "all_exact_candidate_backends": candidates,
            "registry_target_ids": exact_registry,
            "selection_basis": "explicit_sota_model_key_and_exact_provider_plus_model_identifier",
        })

    excluded: list[dict[str, str]] = []
    for row in inventory:
        if row["route"] in selected_routes:
            continue
        reason = row["taxonomy_exclusion_reason"] or "eligible_text_chat_not_in_explicit_sota_panel"
        excluded.append({
            "route": row["route"],
            "upstream_provider": row["upstream_provider"],
            "model_identifier": row["model_identifier"],
            "modality": row["modality"],
            "role": row["role"],
            "reason": reason,
        })

    coverage: list[dict[str, Any]] = []
    selected_keys = {(row["upstream_provider"], row["model_identifier"]) for row in roster}
    for family, matchers in REQUIRED_FAMILIES.items():
        members = sorted(
            f"{provider}/{model}"
            for provider, model in selected_keys
            if any(provider == expected and pattern.search(model) for expected, pattern in matchers)
        )
        coverage.append({"family": family, "covered": bool(members), "subjects": members})

    registry_reconciliation = []
    route_set = set(routes)
    for row in registry_rows:
        model = row["model_identifier"]
        planned = row["planned_route"]
        provider = row["upstream_provider"]
        candidates = sorted(
            (
                route for route in routes
                if route == planned or _identity(route) == (provider, model)
            ),
            key=_backend_rank,
        )
        registry_reconciliation.append({
            **row,
            "planned_route_listed": planned in route_set,
            "all_exact_candidates": candidates,
            "preferred_candidate": candidates[0] if candidates else None,
            "status": "exact_candidate_smoke_pending" if candidates else "unresolved_no_exact_candidate",
        })

    counts = Counter((row["modality"], row["role"]) for row in inventory)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "inference_hub_sota_text_chat_roster_and_reconciliation",
        "catalog": {
            "captured_at_utc": catalog.get("captured_at_utc"),
            "route_count": len(routes),
            "source_payload_sha256": catalog.get("source_payload_sha256"),
            "input_sha256": _sha256_json(catalog),
        },
        "registry": {
            "registry_version": registry.get("registry_version"),
            "target_count": len(registry_rows),
            "input_sha256": _sha256_json(registry),
        },
        "policy": {
            "identity_key": ["upstream_provider_path_component", "exact_final_model_identifier"],
            "identity_equivalence_inference": False,
            "backend_priority": list(BACKEND_PRIORITY),
            "unknown_or_conflicting_route_classification": "error",
            "subject_selection": "frozen_explicit_sota_model_keys",
            "live_smoke_required": True,
            "automatic_registry_mutation": False,
        },
        "summary": {
            "classified_route_count": len(inventory),
            "subject_count": len(roster),
            "subject_backend_route_count": len(selected_routes),
            "excluded_route_count": len(excluded),
            "route_class_counts": [
                {"modality": modality, "role": role, "count": count}
                for (modality, role), count in sorted(counts.items())
            ],
        },
        "route_inventory": inventory,
        "subject_roster": roster,
        "explicit_panel_keys_absent_from_catalog": sorted(
            f"{provider}/{model}" for provider, model in SOTA_MODEL_KEYS - selected_keys
        ),
        "excluded_routes": excluded,
        "coverage": coverage,
        "covered_families": [row["family"] for row in coverage if row["covered"]],
        "uncovered_families": [row["family"] for row in coverage if not row["covered"]],
        "registry_reconciliation": registry_reconciliation,
    }
    if len(selected_routes) + len(excluded) != len(routes):
        raise AssertionError("internal route accounting failure")
    payload["artifact_sha256"] = _sha256_json(payload)
    return payload


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a deterministic SOTA text-chat roster")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_roster(
        catalog=_read_json(args.catalog, "catalog"),
        registry=_read_json(args.registry, "registry"),
    )
    _atomic_write(args.output, payload)
    print(
        f"Classified {payload['summary']['classified_route_count']} routes: "
        f"{payload['summary']['subject_count']} exact subjects across "
        f"{payload['summary']['subject_backend_route_count']} backends; "
        f"{payload['summary']['excluded_route_count']} routes excluded."
    )
    print(f"Artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
