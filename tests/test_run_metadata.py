import json

import pytest

from experiments.misc import run_metadata
from providers.api_call import ResponseParseError


def test_base_run_metadata_preserves_registry_route_and_generation_controls(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "secret-value-never-recorded")
    monkeypatch.setenv(
        "INFERENCE_HUB_BASE_URL",
        "https://user:password@hub.example.test/v1?token=secret-query",
    )
    monkeypatch.setattr(run_metadata, "git_commit", lambda: "abc123")
    monkeypatch.setattr(run_metadata, "git_dirty", lambda: False)
    monkeypatch.setattr(
        run_metadata.sys,
        "argv",
        ["experiment.py", "--api-key", "cli-secret", "--seed=11"],
    )

    metadata = run_metadata.base_run_metadata(
        experiment="part_2",
        timestamp="20260801_120000",
        csv_path=tmp_path / "result.csv",
        provider="inference_hub",
        model="gemini-3.1-pro-preview",
        generation_config={
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 512,
            "seed": 11,
            "reasoning_effort": "low",
            "access_token": "config-secret",
        },
        parameters={
            "nested": {"password": "parameter-secret"},
            "request_id": "ordinary-request-provenance",
        },
    )

    assert metadata["generation_config"]["seed"] == 11
    assert metadata["generation_config"]["access_token"] == "<REDACTED>"
    assert metadata["parameters"]["nested"]["password"] == "<REDACTED>"
    assert metadata["parameters"]["request_id"] == "ordinary-request-provenance"
    assert metadata["command"] == [
        "experiment.py",
        "--api-key",
        "<REDACTED>",
        "--seed=11",
    ]
    assert metadata["schema_version"] == 2
    assert metadata["model_registry"]["registry_version"] == "2026-08-02.2"
    assert metadata["cohort"] == {
        "id": "current_sota",
        "version": "2026-08-02.1",
    }
    assert metadata["route"]["provider"] == "inference_hub"
    assert metadata["route"]["upstream_provider"] == "google"
    assert metadata["route"]["model"] == "gemini-3.1-pro-preview"
    assert metadata["route"]["route"] == "gemini-3.1-pro-preview"
    assert metadata["route"]["verification_status"] == "unverified"
    assert metadata["route"]["route_source"] == "catalog_display_only"
    assert metadata["route"]["endpoint"]["base_url_env"] == "INFERENCE_HUB_BASE_URL"
    assert metadata["credential_environment"]["NVIDIA_API_KEY"] is True
    assert list(metadata["credential_environment"]) == ["NVIDIA_API_KEY"]
    assert "secret-value-never-recorded" not in json.dumps(metadata)
    assert metadata["environment"]["INFERENCE_HUB_BASE_URL"] == (
        "https://hub.example.test/v1"
    )
    assert "secret-query" not in json.dumps(metadata)
    assert "parameter-secret" not in json.dumps(metadata)


def test_mark_metadata_failed_records_parser_provenance(tmp_path) -> None:
    metadata_path = tmp_path / "run_meta.json"
    run_metadata.write_metadata(
        metadata_path,
        run_metadata.base_run_metadata(
            experiment="part_1",
            timestamp="20260801_120000",
            csv_path=tmp_path / "result.csv",
            provider="inference_hub",
            model="claude-sonnet-5",
        ),
    )

    run_metadata.mark_metadata_failed(
        metadata_path,
        error=ResponseParseError("invalid response token=do-not-store"),
        provider="inference_hub",
        model="claude-sonnet-5",
    )

    metadata = run_metadata.read_metadata(metadata_path)
    assert metadata["status"] == run_metadata.STATUS_FAILED
    assert "do-not-store" not in metadata["failure"]["message"]
    assert metadata["failure"]["provenance"]["category"] == "parser"
    assert metadata["failure"]["provenance"]["upstream_provider"] == "anthropic"
    assert metadata["failure"]["provenance"]["route"] == "claude-sonnet-5"


def test_run_metadata_payload_hash_detects_missing_or_changed_fields(tmp_path) -> None:
    metadata_path = tmp_path / "integrity_meta.json"
    run_metadata.write_metadata(metadata_path, {"status": "running", "value": 1})
    metadata = run_metadata.read_metadata(metadata_path)
    run_metadata.validate_metadata_integrity(metadata, required=True)

    metadata["value"] = 2
    with pytest.raises(ValueError, match="SHA-256"):
        run_metadata.validate_metadata_integrity(metadata, required=True)

    metadata.pop("metadata_sha256")
    with pytest.raises(ValueError, match="SHA-256"):
        run_metadata.validate_metadata_integrity(metadata, required=True)
