from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

from experiments.misc.local_hf_smoke import (
    LocalHFSmokeError,
    fingerprint_snapshot,
    run_smokes,
)


REVISION = "a" * 40


def _registry(path: Path, *, status: str = "smoke_pending") -> Path:
    payload = {
        "schema_version": 1,
        "registry_version": "test-v1",
        "analysis_role": "exploratory_local_scale_controls",
        "frontier_route_substitution_permitted": False,
        "paper_result_promotion_permitted": False,
        "runtime_contract": {
            "backend": "huggingface_transformers_offline",
            "local_files_only": True,
            "remote_code_permitted": False,
            "generation_mode": "greedy",
        },
        "models": [
            {
                "id": "hf.test",
                "model_id": "example/test",
                "revision": REVISION,
                "cache_repository_dir": "models--example--test",
                "parameter_scale": "tiny",
                "verification_status": status,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _snapshot(cache_root: Path) -> Path:
    snapshot = cache_root / "models--example--test" / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    return snapshot


def test_fingerprint_snapshot_hashes_required_assets(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    result = fingerprint_snapshot(cache_root=tmp_path, snapshot=snapshot)
    assert result["asset_count"] == 3
    assert result["total_size_bytes"] == 11
    by_name = {row["name"]: row for row in result["assets"]}
    assert by_name["model.safetensors"]["sha256"] == hashlib.sha256(
        b"weights"
    ).hexdigest()
    assert len(result["snapshot_tree_sha256"]) == 64


def test_fingerprint_snapshot_hashes_nested_and_hidden_assets(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    nested = snapshot / "tokenizer" / "vocab.json"
    nested.parent.mkdir()
    nested.write_text('{"token": 1}', encoding="utf-8")
    (snapshot / ".metadata").write_text("revision", encoding="utf-8")

    result = fingerprint_snapshot(cache_root=tmp_path, snapshot=snapshot)

    names = {row["name"] for row in result["assets"]}
    assert "tokenizer/vocab.json" in names
    assert ".metadata" in names


def test_run_smokes_retains_real_generator_evidence(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.json")
    _snapshot(tmp_path / "cache")
    output = tmp_path / "evidence.json"
    calls = []

    def generator(**kwargs):
        calls.append(kwargs)
        return "READY", {
            "python": "test",
            "torch": "test",
            "transformers": "test",
            "device": kwargs["device"],
        }

    evidence = run_smokes(
        registry_path=registry,
        cache_root=tmp_path / "cache",
        output_path=output,
        generator=generator,
    )
    assert evidence["complete"] is True
    assert evidence["frontier_route_substitution_permitted"] is False
    assert evidence["paper_result_promotion_permitted"] is False
    assert evidence["attempts"][0]["status"] == "passed"
    assert evidence["attempts"][0]["response_text"] == "READY"
    assert evidence["attempts"][0]["format_contract_match"] is True
    assert evidence["generation_contract"]["parallelization"] == (
        "bounded_model_level_thread_pool"
    )
    assert evidence["generation_contract"]["max_workers"] == 1
    assert calls[0]["seed"] == 20260802
    assert json.loads(output.read_text(encoding="utf-8")) == evidence


def test_run_smokes_retains_failure_and_returns_incomplete(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.json")
    _snapshot(tmp_path / "cache")

    def generator(**_kwargs):
        raise RuntimeError("backend failed")

    evidence = run_smokes(
        registry_path=registry,
        cache_root=tmp_path / "cache",
        output_path=tmp_path / "evidence.json",
        generator=generator,
    )
    assert evidence["complete"] is False
    assert evidence["attempts"][0]["status"] == "failed"
    assert evidence["attempts"][0]["error"]["type"] == "RuntimeError"


def test_run_smokes_rejects_unknown_or_duplicate_selection(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.json")
    _snapshot(tmp_path / "cache")
    with pytest.raises(LocalHFSmokeError, match="Unknown local model"):
        run_smokes(
            registry_path=registry,
            cache_root=tmp_path / "cache",
            output_path=tmp_path / "unknown.json",
            selected_ids=["hf.unknown"],
        )
    with pytest.raises(LocalHFSmokeError, match="more than once"):
        run_smokes(
            registry_path=registry,
            cache_root=tmp_path / "cache",
            output_path=tmp_path / "duplicate.json",
            selected_ids=["hf.test", "hf.test"],
        )


def test_registry_rejects_nonpending_model(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.json", status="verified")
    with pytest.raises(LocalHFSmokeError, match="not smoke_pending"):
        run_smokes(
            registry_path=registry,
            cache_root=tmp_path / "cache",
            output_path=tmp_path / "evidence.json",
        )


def test_snapshot_symlink_may_not_escape_cache_root(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path / "cache")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"secret")
    (snapshot / "escape.bin").symlink_to(outside)
    with pytest.raises(LocalHFSmokeError, match="escapes the cache root"):
        fingerprint_snapshot(cache_root=tmp_path / "cache", snapshot=snapshot)


def test_run_smokes_refuses_to_overwrite_prior_evidence(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.json")
    _snapshot(tmp_path / "cache")
    output = tmp_path / "evidence.json"
    output.write_text('{"prior": true}\n', encoding="utf-8")

    with pytest.raises(LocalHFSmokeError, match="already exists"):
        run_smokes(
            registry_path=registry,
            cache_root=tmp_path / "cache",
            output_path=output,
            generator=lambda **_kwargs: ("READY", {}),
        )
    assert json.loads(output.read_text(encoding="utf-8")) == {"prior": True}


def test_run_smokes_executes_models_concurrently_and_keeps_registry_order(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path / "registry.json")
    payload = json.loads(registry.read_text(encoding="utf-8"))
    second = dict(payload["models"][0])
    second.update(
        {
            "id": "hf.second",
            "model_id": "example/second",
            "cache_repository_dir": "models--example--second",
            "revision": "b" * 40,
        }
    )
    payload["models"].append(second)
    registry.write_text(json.dumps(payload), encoding="utf-8")
    _snapshot(tmp_path / "cache")
    second_snapshot = (
        tmp_path
        / "cache"
        / second["cache_repository_dir"]
        / "snapshots"
        / second["revision"]
    )
    second_snapshot.mkdir(parents=True)
    (second_snapshot / "config.json").write_text("{}", encoding="utf-8")
    (second_snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (second_snapshot / "model.safetensors").write_bytes(b"weights")

    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def generator(**_kwargs):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return "READY", {}

    evidence = run_smokes(
        registry_path=registry,
        cache_root=tmp_path / "cache",
        output_path=tmp_path / "parallel.json",
        max_workers=2,
        generator=generator,
    )

    assert maximum_active == 2
    assert [row["model_id"] for row in evidence["attempts"]] == [
        "hf.test",
        "hf.second",
    ]
    assert evidence["generation_contract"]["max_workers"] == 2
    assert evidence["complete"] is True


@pytest.mark.parametrize("max_workers", [0, -1, True])
def test_run_smokes_rejects_invalid_worker_count(
    tmp_path: Path, max_workers: int
) -> None:
    registry = _registry(tmp_path / f"registry-{max_workers}.json")
    with pytest.raises(LocalHFSmokeError, match="positive integer"):
        run_smokes(
            registry_path=registry,
            cache_root=tmp_path / "cache",
            output_path=tmp_path / f"evidence-{max_workers}.json",
            max_workers=max_workers,
        )
