from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.build_sota_inference_hub_roster import (
    BACKEND_PRIORITY,
    RosterBuildError,
    build_roster,
    classify_route,
    main,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data/private/inference_hub/catalog-live-20260802-v3.json"
REGISTRY_PATH = ROOT / "agents/agent_config.registry.json"


def _live_inputs() -> tuple[dict, dict]:
    return (
        json.loads(CATALOG_PATH.read_text(encoding="utf-8")),
        json.loads(REGISTRY_PATH.read_text(encoding="utf-8")),
    )


def _catalog(routes: list[str]) -> dict:
    return {
        "schema_version": 2,
        "captured_at_utc": "2026-08-02T00:00:00Z",
        "source_endpoints": ["/models"],
        "source_payload_sha256": {"models": "a" * 64},
        "route_count": len(routes),
        "routes": [
            {"route": route, "listed_by_models": True}
            for route in routes
        ],
    }


def _registry() -> dict:
    return {
        "registry_version": "test",
        "targets": [
            {
                "id": "openai.gpt-5",
                "upstream_provider": "openai",
                "model": "gpt-5",
                "route": "azure/openai/gpt-5",
            }
        ],
    }


def test_live_catalog_is_fully_accounted_and_covers_requested_families() -> None:
    catalog, registry = _live_inputs()
    artifact = build_roster(catalog=catalog, registry=registry)
    summary = artifact["summary"]

    assert catalog["route_count"] == 214
    assert summary["classified_route_count"] == 214
    assert len(artifact["route_inventory"]) == 214
    assert summary["subject_backend_route_count"] + summary["excluded_route_count"] == 214
    assert len(artifact["excluded_routes"]) == summary["excluded_route_count"]
    assert all(row["reason"] for row in artifact["excluded_routes"])
    assert artifact["uncovered_families"] == []
    assert artifact["explicit_panel_keys_absent_from_catalog"] == []
    assert {
        "claude_haiku", "claude_opus", "claude_sonnet", "gemini",
        "openai_gpt_generation_3", "openai_gpt_generation_4",
        "openai_gpt_generation_5", "deepseek", "gemma", "glm", "kimi",
        "llama", "minimax", "mistral", "nemotron", "perplexity_sonar", "qwen",
    } == set(artifact["covered_families"])


def test_exact_alias_dedup_retains_every_backend_in_frozen_priority() -> None:
    catalog, registry = _live_inputs()
    artifact = build_roster(catalog=catalog, registry=registry)
    by_id = {row["subject_id"]: row for row in artifact["subject_roster"]}

    terra = by_id["openai/gpt-5.6-terra"]
    assert terra["all_exact_candidate_backends"] == [
        "openai/openai/gpt-5.6-terra",
        "azure/openai/gpt-5.6-terra",
        "switchyard/openai/gpt-5.6-terra",
    ]
    assert terra["preferred_candidate"] == "openai/openai/gpt-5.6-terra"
    assert artifact["policy"]["backend_priority"] == list(BACKEND_PRIORITY)

    # These similar strings are intentionally separate model identities.
    assert "anthropic/bedrock-claude-opus-5" in by_id
    assert "anthropic/claude-opus-5" in by_id
    assert by_id["anthropic/bedrock-claude-opus-5"]["all_exact_candidate_backends"] == [
        "aws/anthropic/bedrock-claude-opus-5"
    ]
    assert by_id["anthropic/claude-opus-5"]["all_exact_candidate_backends"] == [
        "azure/anthropic/claude-opus-5"
    ]


def test_non_subject_modalities_and_roles_are_excluded() -> None:
    catalog, registry = _live_inputs()
    artifact = build_roster(catalog=catalog, registry=registry)
    selected_routes = {
        route
        for subject in artifact["subject_roster"]
        for route in subject["all_exact_candidate_backends"]
    }
    inventory = {row["route"]: row for row in artifact["route_inventory"]}
    forbidden_modalities = {
        "embedding", "reranking", "image_or_vision", "audio", "audio_and_text", "video"
    }
    forbidden_roles = {"safety", "judge", "batch", "translation", "test_infrastructure"}

    assert not any(
        inventory[route]["modality"] in forbidden_modalities
        or inventory[route]["role"] in forbidden_roles
        for route in selected_routes
    )
    excluded_by_route = {row["route"]: row for row in artifact["excluded_routes"]}
    assert excluded_by_route["azure/openai/text-embedding-3-large"]["reason"] == "non_chat_embedding_route"
    assert excluded_by_route["nvidia/qwen/qwen3-reranker-8b"]["reason"] == "non_chat_reranking_route"
    assert excluded_by_route["azure/openai/gpt-image-2"]["reason"] == "non_text_image_or_vision_route"
    assert excluded_by_route["openai/openai/gpt-4o-mini-tts"]["reason"] == "non_chat_audio_route"
    assert excluded_by_route["nvidia/nvidia/llama-3.1-nemoguard-8b-content-safety"]["reason"] == "safety_route"
    assert excluded_by_route["nvidia/nvidia/evals-nemotron-3-30b-a3b"]["reason"] == "judge_or_evaluation_route"


def test_unknown_route_type_fails_closed() -> None:
    with pytest.raises(RosterBuildError, match="ambiguous route type .*no class"):
        classify_route("nvidia/acme/mystery-9000")

    catalog = _catalog(["azure/openai/gpt-5", "nvidia/acme/mystery-9000"])
    with pytest.raises(RosterBuildError, match="mystery-9000"):
        build_roster(catalog=catalog, registry=_registry())


def test_registry_reconciliation_does_not_merge_other_provider_same_name() -> None:
    catalog = _catalog([
        "azure/openai/gpt-5",
        "nvidia/acme/gpt-5",
        "us/azure/openai/gpt-5",
    ])
    artifact = build_roster(catalog=catalog, registry=_registry())
    resolution = artifact["registry_reconciliation"][0]

    assert resolution["all_exact_candidates"] == [
        "azure/openai/gpt-5",
        "us/azure/openai/gpt-5",
    ]
    assert "nvidia/acme/gpt-5" not in resolution["all_exact_candidates"]


def test_cli_writes_byte_deterministic_machine_readable_artifact(tmp_path: Path, capsys) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    argv = ["--catalog", str(CATALOG_PATH), "--registry", str(REGISTRY_PATH)]

    assert main([*argv, "--output", str(first)]) == 0
    assert main([*argv, "--output", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()
    artifact = json.loads(first.read_text(encoding="utf-8"))
    assert len(artifact["artifact_sha256"]) == 64
    assert artifact["artifact_type"] == "inference_hub_sota_text_chat_roster_and_reconciliation"
    assert "Classified 214 routes" in capsys.readouterr().out
