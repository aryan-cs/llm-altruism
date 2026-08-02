from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.reconcile_inference_hub_routes import (
    RouteReconciliationError,
    main,
    reconcile_routes,
)


def _catalog() -> dict:
    routes = [
        "azure/openai/gpt-5.6-terra",
        "openai/openai/gpt-5.6-terra",
        "switchyard/openai/gpt-5.6-terra",
        "gcp/google/gemini-3.1-pro-preview",
        "openai/openai/gpt-3.5-turbo",
    ]
    return {
        "schema_version": 2,
        "captured_at_utc": "2026-08-02T00:00:00Z",
        "source_endpoints": ["/models"],
        "source_payload_sha256": {"models": "a" * 64},
        "route_count": len(routes),
        "routes": [
            {
                "route": route,
                "listed_by_models": True,
                "chat_capability": "unverified_until_structured_smoke",
            }
            for route in routes
        ],
    }


def _registry() -> dict:
    return {
        "registry_version": "test-v1",
        "targets": [
            {
                "id": "openai.current",
                "model": "gpt-5.6-terra",
                "upstream_provider": "openai",
                "route": "gpt-5.6-terra",
            },
            {
                "id": "google.current",
                "model": "gemini-3.1-pro-preview",
                "upstream_provider": "google",
                "route": "gemini-3.1-pro-preview",
            },
            {
                "id": "openai.historical",
                "model": "gpt-3.5-turbo-0125",
                "upstream_provider": "openai",
                "route": "gpt-3.5-turbo-0125",
            },
        ],
        "cohorts": {
            "current_sota": {
                "targets": ["openai.current", "google.current"],
            },
            "historical": {"targets": ["openai.historical"]},
        },
    }


def test_reconciliation_selects_exact_suffix_by_frozen_priority() -> None:
    report = reconcile_routes(catalog=_catalog(), registry=_registry())

    assert report["candidate_selected_count"] == 2
    assert report["unresolved_count"] == 1
    by_id = {row["target_id"]: row for row in report["resolutions"]}
    assert by_id["openai.current"]["selected_candidate"] == (
        "openai/openai/gpt-5.6-terra"
    )
    assert by_id["openai.current"]["all_exact_suffix_candidates"] == [
        "openai/openai/gpt-5.6-terra",
        "azure/openai/gpt-5.6-terra",
        "switchyard/openai/gpt-5.6-terra",
    ]
    assert by_id["google.current"]["selected_candidate"] == (
        "gcp/google/gemini-3.1-pro-preview"
    )
    assert by_id["openai.historical"]["selected_candidate"] is None
    assert report["selection_policy"]["automatic_promotion"] is False
    assert len(report["report_sha256"]) == 64


def test_reconciliation_rejects_non_models_catalog() -> None:
    catalog = _catalog()
    catalog["source_endpoints"] = ["/models", "/model/info"]
    with pytest.raises(RouteReconciliationError, match="authorized /models"):
        reconcile_routes(catalog=catalog, registry=_registry())


def test_cli_writes_private_report(tmp_path: Path, capsys) -> None:
    catalog_path = tmp_path / "catalog.json"
    registry_path = tmp_path / "registry.json"
    output_path = tmp_path / "report.json"
    catalog_path.write_text(json.dumps(_catalog()), encoding="utf-8")
    registry_path.write_text(json.dumps(_registry()), encoding="utf-8")

    assert (
        main(
            [
                "--catalog",
                str(catalog_path),
                "--registry",
                str(registry_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["target_count"] == 3
    assert "Selected 2 exact-suffix candidates" in capsys.readouterr().out
