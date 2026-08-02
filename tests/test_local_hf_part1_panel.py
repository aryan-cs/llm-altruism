from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from experiments.misc.local_hf_part1_panel import (
    LocalHFPanelError,
    _real_model_runner,
    _parse_final_action,
    build_draft_trials,
    run_panel,
)


def _registry(path: Path) -> Path:
    models = []
    for index in range(2):
        models.append(
            {
                "id": f"hf.test-{index}",
                "model_id": f"example/test-{index}",
                "revision": str(index + 1) * 40,
                "cache_repository_dir": f"models--example--test-{index}",
                "parameter_scale": "tiny",
                "verification_status": "smoke_pending",
            }
        )
    path.write_text(
        json.dumps(
            {
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
                "models": models,
            }
        ),
        encoding="utf-8",
    )
    return path


_SNAPSHOT_BINDING = {
    "asset_count": 3,
    "total_size_bytes": 11,
    "snapshot_tree_sha256": "a" * 64,
}
_RUNTIME_BINDING = {
    "python": "test-python",
    "torch": "test-torch",
    "transformers": "test-transformers",
    "device": "cpu",
}


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_valid_model_result(*, model, trials, output_path):
    runtime_sha256 = _canonical_sha256(_RUNTIME_BINDING)
    rows = []
    for trial in trials:
        response = "X"
        rows.append(
            {
                "schema_version": 2,
                "model_id": model["id"],
                "upstream_model_id": model["model_id"],
                "revision": model["revision"],
                "trial_id": trial.trial_id,
                "root_id": trial.root_id,
                "game": trial.game,
                "domain": trial.domain,
                "counterbalance_id": trial.counterbalance_id,
                "prompt_text": trial.prompt_text,
                "prompt_sha256": trial.prompt_hash,
                "response_text": response,
                "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                "parsed_action": "X",
                "format_valid": True,
                "snapshot_tree_sha256": _SNAPSHOT_BINDING[
                    "snapshot_tree_sha256"
                ],
                "runtime_sha256": runtime_sha256,
                "finished_at_utc": "2026-08-02T00:00:00Z",
            }
        )
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    output_path.chmod(0o600)
    return {
        "status": "passed",
        "started_at_utc": "2026-08-02T00:00:00Z",
        "finished_at_utc": "2026-08-02T00:00:01Z",
        "completed_trials": len(trials),
        "valid_actions": len(trials),
        "invalid_actions": 0,
        "output_file": output_path.name,
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "resumed_from_trial_count": 0,
        "snapshot": dict(_SNAPSHOT_BINDING),
        "runtime": dict(_RUNTIME_BINDING),
    }


def test_draft_schedule_is_384_roots_balanced_over_twelve_cells() -> None:
    trials = build_draft_trials(base_seed=20260802)
    assert len(trials) == 384
    assert len({row.root_id for row in trials}) == 384
    assert set(Counter((row.game, row.domain) for row in trials).values()) == {32}
    assert len({row.trial_id for row in trials}) == 384


@pytest.mark.parametrize(
    ("text", "expected"),
    [("X", "X"), ("reason\nY", "Y"), ("X\nextra", None), ("choice: X", None)],
)
def test_parse_final_action_requires_exact_final_line(text: str, expected: str | None) -> None:
    assert _parse_final_action(text) == expected


def test_panel_parallelizes_models_and_retains_deterministic_manifest(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.json")
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def runner(*, model, trials, output_path, **_kwargs):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.05)
        result = _write_valid_model_result(
            model=model, trials=trials, output_path=output_path
        )
        with lock:
            active -= 1
        return result

    manifest = run_panel(
        registry_path=registry,
        cache_root=tmp_path,
        output_dir=tmp_path / "panel",
        limit=12,
        max_workers=2,
        model_runner=runner,
    )

    assert maximum_active == 2
    assert manifest["trial_count_per_model"] == 12
    assert manifest["total_planned_generations"] == 24
    assert manifest["complete"] is True
    assert manifest["confirmatory_or_paper_promotion_permitted"] is False
    assert manifest["generation_contract"]["max_workers"] == 2
    assert manifest["models"]["hf.test-0"]["snapshot"] == _SNAPSHOT_BINDING
    assert manifest["models"]["hf.test-0"]["runtime"] == _RUNTIME_BINDING
    assert json.loads((tmp_path / "panel" / "manifest.json").read_text()) == manifest
    if os.name == "posix":
        assert stat.S_IMODE((tmp_path / "panel").stat().st_mode) == 0o700
        assert stat.S_IMODE(
            (tmp_path / "panel" / "manifest.json").stat().st_mode
        ) == 0o600
        assert stat.S_IMODE(
            (tmp_path / "panel" / "hf_test-0.jsonl").stat().st_mode
        ) == 0o600


def test_panel_refuses_to_overwrite_evidence_directory(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.json")
    output = tmp_path / "prior"
    output.mkdir()
    with pytest.raises(LocalHFPanelError, match="already exists"):
        run_panel(
            registry_path=registry,
            cache_root=tmp_path,
            output_dir=output,
            limit=1,
            model_runner=lambda **_kwargs: {},
        )


def test_panel_resume_is_contract_bound_and_retains_failed_attempt(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.json")
    output = tmp_path / "resume-panel"

    def fail(**_kwargs):
        raise RuntimeError("first attempt failed")

    first = run_panel(
        registry_path=registry,
        cache_root=tmp_path,
        output_dir=output,
        selected_ids=["hf.test-0"],
        limit=3,
        model_runner=fail,
    )
    assert first["complete"] is False

    def pass_run(*, model, trials, output_path, **_kwargs):
        return _write_valid_model_result(
            model=model, trials=trials, output_path=output_path
        )

    resumed = run_panel(
        registry_path=registry,
        cache_root=tmp_path,
        output_dir=output,
        selected_ids=["hf.test-0"],
        limit=3,
        resume=True,
        model_runner=pass_run,
    )
    assert resumed["complete"] is True
    assert resumed["resume_count"] == 1
    assert resumed["model_attempt_history"]["hf.test-0"][0]["status"] == "failed"

    with pytest.raises(LocalHFPanelError, match="does not match"):
        run_panel(
            registry_path=registry,
            cache_root=tmp_path,
            output_dir=output,
            selected_ids=["hf.test-0"],
            limit=4,
            resume=True,
            model_runner=pass_run,
        )


def test_resume_rejects_manifest_self_hash_tampering(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.json")
    output = tmp_path / "manifest-tamper"
    run_panel(
        registry_path=registry,
        cache_root=tmp_path,
        output_dir=output,
        selected_ids=["hf.test-0"],
        limit=2,
        model_runner=lambda **kwargs: _write_valid_model_result(
            model=kwargs["model"],
            trials=kwargs["trials"],
            output_path=kwargs["output_path"],
        ),
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resume_count"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_path.chmod(0o600)

    with pytest.raises(LocalHFPanelError, match="self-hash is invalid"):
        run_panel(
            registry_path=registry,
            cache_root=tmp_path,
            output_dir=output,
            selected_ids=["hf.test-0"],
            limit=2,
            resume=True,
            model_runner=lambda **_kwargs: pytest.fail("runner must not execute"),
        )


def test_resume_rejects_retained_row_content_tampering(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.json")
    output = tmp_path / "row-tamper"
    run_panel(
        registry_path=registry,
        cache_root=tmp_path,
        output_dir=output,
        selected_ids=["hf.test-0"],
        limit=2,
        model_runner=lambda **kwargs: _write_valid_model_result(
            model=kwargs["model"],
            trials=kwargs["trials"],
            output_path=kwargs["output_path"],
        ),
    )
    rows_path = output / "hf_test-0.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    rows[0]["response_text"] = "Y"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    rows_path.chmod(0o600)

    with pytest.raises(LocalHFPanelError, match="content hash mismatch"):
        run_panel(
            registry_path=registry,
            cache_root=tmp_path,
            output_dir=output,
            selected_ids=["hf.test-0"],
            limit=2,
            resume=True,
            model_runner=lambda **_kwargs: pytest.fail("runner must not execute"),
        )


def test_real_runner_binds_snapshot_and_runtime_on_completed_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry(tmp_path / "registry.json")
    model = json.loads(registry.read_text())["models"][0]
    cache_root = tmp_path / "cache"
    snapshot = (
        cache_root
        / model["cache_repository_dir"]
        / "snapshots"
        / model["revision"]
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")

    torch = ModuleType("torch")
    torch.__version__ = _RUNTIME_BINDING["torch"]
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    transformers = ModuleType("transformers")
    transformers.__version__ = _RUNTIME_BINDING["transformers"]
    transformers.AutoModelForCausalLM = object
    transformers.AutoTokenizer = object
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        "experiments.misc.local_hf_part1_panel.platform.python_version",
        lambda: _RUNTIME_BINDING["python"],
    )
    monkeypatch.setattr(
        "experiments.misc.local_hf_part1_panel.fingerprint_snapshot",
        lambda **_kwargs: {**_SNAPSHOT_BINDING, "assets": []},
    )

    trials = build_draft_trials(base_seed=20260802, limit=1)
    output_path = tmp_path / "completed.jsonl"
    _write_valid_model_result(model=model, trials=trials, output_path=output_path)
    result = _real_model_runner(
        model=model,
        cache_root=cache_root,
        trials=trials,
        output_path=output_path,
        batch_size=1,
        max_new_tokens=4,
        device="cpu",
    )

    assert result["snapshot"] == _SNAPSHOT_BINDING
    assert result["runtime"] == _RUNTIME_BINDING
    assert result["resumed_from_trial_count"] == 1
