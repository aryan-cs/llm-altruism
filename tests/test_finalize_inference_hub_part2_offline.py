import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

import analysis.finalize_inference_hub_part2_offline as offline
from analysis.build_final_results import _part2_overlay
from analysis.finalize_inference_hub_part2_offline import (
    OfflinePart2FinalizationError,
    finalize_part2_offline,
)
from experiments.misc.inference_hub_part1_panel import (
    _ChainedJournal,
    _atomic_json,
    _seal,
    _sha256_json,
)
from experiments.misc.inference_hub_part2_panel import _SOURCE_PATHS, run_panel


ENDPOINT = "https://inference-api.nvidia.com/v1"
ALPHA = "subject.alpha"
BETA = "subject.beta"
JUDGE = "judge.route"


def _registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "targets": [
            {
                "id": ALPHA, "provider": "inference_hub",
                "upstream_provider": "developer", "model": "alpha-model",
            },
            {
                "id": BETA, "provider": "inference_hub",
                "upstream_provider": "developer", "model": "beta-model",
            },
            {
                "id": JUDGE, "provider": "inference_hub",
                "upstream_provider": "judge", "model": "judge-model",
            },
        ],
    }


def _compat_target(target_id: str, model: str, route: str) -> dict[str, Any]:
    controls = ["seed", "temperature", "top_p", "structured_response"]
    profile = {
        "attempt_id": f"probe-{target_id}", "controls": controls,
        "profile_id": f"profile-{target_id}",
        "request_sha256": hashlib.sha256(target_id.encode()).hexdigest(),
        "resumed_from_ledger": False, "status": "passed",
        "validation_source": "execution_profile_probe",
    }
    return {
        "target_id": target_id, "model": model, "candidate_count": 1,
        "frozen_candidate_order": [route],
        "selected_execution_candidate": route,
        "selected_execution_profile": {**profile, "route": route},
        "selection_basis": "first_execution_compatible_in_reconciliation_frozen_order",
        "status": "execution_candidate_selected",
        "candidates": [{
            "route": route, "candidate_index": 0, "max_tokens": 64,
            "execution_compatible": True, "selected_execution_profile": profile,
        }],
    }


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    panel = {
        "schema_version": 1, "panel_id": "offline-finalizer-test",
        "judge_target_id": JUDGE, "subject_target_ids": [ALPHA, BETA],
        "part2": {
            "society_size": 5, "days": 12, "independent_trajectories": 12,
            "resource_capacity": 50, "private_gain_for_option_b": 2,
            "reserve_cost_for_option_b": 2, "common_environment_seeds": True,
        },
    }
    registry = _registry()
    compatibility = {
        "schema_version": 2,
        "artifact_type": "inference_hub_provider_compatibility",
        "endpoint": ENDPOINT,
        "registry_sha256": _sha256_json(registry),
        "targets": [
            _compat_target(ALPHA, "alpha-model", "region/alpha-model"),
            _compat_target(BETA, "beta-model", "region/beta-model"),
            _compat_target(JUDGE, "judge-model", "region/judge-model"),
        ],
        "target_count": 3, "selected_count": 3, "unresolved_count": 0,
    }
    compatibility["evidence_sha256"] = _sha256_json(compatibility)
    paths = (
        tmp_path / "panel.json", tmp_path / "compatibility.json",
        tmp_path / "registry.json",
    )
    for path, value in zip(paths, (panel, compatibility, registry)):
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return paths


class _Client:
    base_url = ENDPOINT

    def __init__(self, identity_fail_targets: set[str]) -> None:
        self.identity_fail_targets = identity_fail_targets

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        assert path == "/chat/completions"
        route = str(body["model"])
        target_id = BETA if route.endswith("beta-model") else ALPHA
        return {
            "id": "fixture-response",
            "model": (
                "region/wrong-model"
                if target_id in self.identity_fail_targets else route
            ),
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "action": "OPTION_A", "reasoning": "Preserve the reserve."
                    }),
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
            },
        }


def _prepare(
    tmp_path: Path, *, identity_fail_targets: set[str] = {BETA},
) -> Path:
    panel, compatibility, registry = _inputs(tmp_path)
    output = tmp_path / "panel-output"
    run_panel(
        panel_path=panel, compatibility_path=compatibility,
        registry_path=registry, output_dir=output,
        client=_Client(identity_fail_targets), selected_ids=[ALPHA, BETA],
        trajectory_limit=2, trajectory_workers=4, participant_workers=5,
        max_attempts=1, initial_backoff_seconds=0,
    )
    manifest_path = output / "private/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for reference in manifest["sanitized_artifacts"].values():
        Path(reference["path"]).unlink()
    manifest["complete"] = False
    manifest["summary"] = {}
    manifest["sanitized_artifacts"] = {}
    manifest.pop("completed_at_utc", None)
    _seal(manifest)
    _atomic_json(manifest_path, manifest)
    return manifest_path


def _truncate_bound_journal(
    manifest_path: Path, key: str, *, record_count: int,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = Path(manifest["journals"][key]["path"])
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:record_count]) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    manifest["journals"][key] = _ChainedJournal(path).reference()
    _seal(manifest)
    _atomic_json(manifest_path, manifest)


def _clean_source_root(tmp_path: Path) -> Path:
    repository_root = Path(offline.__file__).resolve().parents[1]
    clean = tmp_path / "clean-source"
    for source in _SOURCE_PATHS:
        relative = source.resolve().relative_to(repository_root)
        destination = clean / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return clean


def test_finalizes_retained_complete_and_unavailable_failure_evidence(
    tmp_path: Path,
) -> None:
    manifest_path = _prepare(tmp_path)
    # Preserve one complete unavailable trajectory and only the first day of
    # the second. The latter is target-bound failure evidence, not a basis for
    # behavioral trajectory metrics.
    _truncate_bound_journal(manifest_path, f"{BETA}::1", record_count=10)
    result = finalize_part2_offline(
        manifest_path, unavailable_targets=[BETA],
    )

    assert result["complete"] is False
    assert result["summary"]["planned_trajectories"] == 4
    assert result["summary"]["completed_trajectories"] == 3
    assert result["summary"]["identity_mismatch_count"] == 65
    assert result["summary"]["transport_failure_count"] == 0
    trajectory_path = Path(
        result["sanitized_artifacts"]["trajectory_metrics"]["path"]
    )
    model_path = Path(result["sanitized_artifacts"]["model_metrics"]["path"])
    trajectories = json.loads(trajectory_path.read_text(encoding="utf-8"))
    models = json.loads(model_path.read_text(encoding="utf-8"))
    alpha_rows = [row for row in trajectories["rows"] if row["target_id"] == ALPHA]
    beta_rows = [row for row in trajectories["rows"] if row["target_id"] == BETA]
    assert len(alpha_rows) == 2 and all("aurc" in row for row in alpha_rows)
    assert len(beta_rows) == 2
    assert all(row["evidence_scope"] == "operational_only_no_behavioral_metrics" for row in beta_rows)
    assert all("aurc" not in row and "restraint_rate" not in row for row in beta_rows)
    beta_model = next(row for row in models["rows"] if row["target_id"] == BETA)
    assert beta_model["complete_matched_panel"] is False
    assert "trajectory_level_95_percent_t_intervals" not in beta_model

    rows, identities, binding = _part2_overlay(
        manifest_path, (), (BETA,),
    )
    assert [row["target_id"] for row in rows] == [ALPHA]
    assert set(identities) == {ALPHA}
    assert binding["unavailable_target_ids"] == [BETA]
    public_text = trajectory_path.read_text(encoding="utf-8") + model_path.read_text(encoding="utf-8")
    for forbidden in (
        "prompt_text", "raw_response", "requested_route", "visible_content",
        '"reasoning"', "region/alpha-model", "region/beta-model",
    ):
        assert forbidden not in public_text


def test_rejects_live_run_lock(tmp_path: Path) -> None:
    manifest_path = _prepare(tmp_path)
    lock_path = manifest_path.parent / ".run.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(OfflinePart2FinalizationError, match="run lock"):
            finalize_part2_offline(manifest_path, unavailable_targets=[BETA])


def test_rejects_unavailable_target_without_operational_failure(tmp_path: Path) -> None:
    manifest_path = _prepare(tmp_path, identity_fail_targets=set())
    with pytest.raises(OfflinePart2FinalizationError, match="lacks identity/transport"):
        finalize_part2_offline(manifest_path, unavailable_targets=[BETA])


def test_rejects_incomplete_retained_trajectory(tmp_path: Path) -> None:
    manifest_path = _prepare(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = f"{ALPHA}::1"
    path = Path(manifest["journals"][key]["path"])
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-2]) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    manifest["journals"][key] = _ChainedJournal(path).reference()
    _seal(manifest)
    _atomic_json(manifest_path, manifest)

    with pytest.raises(OfflinePart2FinalizationError, match="incomplete trajectory"):
        finalize_part2_offline(manifest_path, unavailable_targets=[BETA])


def test_rejects_changed_checkpoint_or_frozen_input(tmp_path: Path) -> None:
    checkpoint_manifest = _prepare(tmp_path / "checkpoint")
    manifest = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    manifest["journals"][f"{ALPHA}::0"]["tail_record_sha256"] = "0" * 64
    _seal(manifest)
    _atomic_json(checkpoint_manifest, manifest)
    with pytest.raises(OfflinePart2FinalizationError, match="checkpoint hash"):
        finalize_part2_offline(checkpoint_manifest, unavailable_targets=[BETA])

    input_manifest = _prepare(tmp_path / "input")
    manifest = json.loads(input_manifest.read_text(encoding="utf-8"))
    panel_path = Path(manifest["input_artifacts"]["panel"]["path"])
    panel_path.write_text(panel_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(OfflinePart2FinalizationError, match="file hash"):
        finalize_part2_offline(input_manifest, unavailable_targets=[BETA])


def test_rejects_private_text_leakage_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _prepare(tmp_path)
    original = offline._aggregate_models

    def contaminated(*args: Any, **kwargs: Any):
        rows = original(*args, **kwargs)
        rows[0]["prompt_text"] = "private"
        return rows

    monkeypatch.setattr(offline, "_aggregate_models", contaminated)
    with pytest.raises(OfflinePart2FinalizationError, match="forbidden field"):
        finalize_part2_offline(manifest_path, unavailable_targets=[BETA])
    assert not (manifest_path.parents[1] / "sanitized/trajectory_metrics.json").exists()


def test_refuses_existing_sanitized_artifacts(tmp_path: Path) -> None:
    manifest_path = _prepare(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sanitized_artifacts"] = {"unexpected": {"path": "existing"}}
    _seal(manifest)
    _atomic_json(manifest_path, manifest)
    with pytest.raises(OfflinePart2FinalizationError, match="already exist"):
        finalize_part2_offline(manifest_path, unavailable_targets=[BETA])


def test_explicit_clean_source_root_is_bound_only_in_private_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = _prepare(tmp_path)
    clean = _clean_source_root(tmp_path)
    result = finalize_part2_offline(
        manifest_path, unavailable_targets=[BETA],
        source_verification_root=clean,
    )
    provenance = result["offline_finalization"]["source_verification"]
    assert provenance["mode"] == "explicit_clean_source_root"
    assert provenance["verification_root"] == str(clean.resolve())
    assert len(provenance["verified_files"]) == len(_SOURCE_PATHS)
    sanitized = "".join(
        Path(reference["path"]).read_text(encoding="utf-8")
        for reference in result["sanitized_artifacts"].values()
    )
    assert str(clean.resolve()) not in sanitized
    assert str(tmp_path.resolve()) not in sanitized


@pytest.mark.parametrize("mutation", ["missing", "wrong_bytes"])
def test_rejects_missing_or_wrong_clean_source_bytes(
    tmp_path: Path, mutation: str,
) -> None:
    manifest_path = _prepare(tmp_path)
    clean = _clean_source_root(tmp_path)
    repository_root = Path(offline.__file__).resolve().parents[1]
    relative = _SOURCE_PATHS[0].resolve().relative_to(repository_root)
    candidate = clean / relative
    if mutation == "missing":
        candidate.unlink()
    else:
        candidate.write_text("changed\n", encoding="utf-8")
    with pytest.raises(
        OfflinePart2FinalizationError,
        match="verification file is missing|verification bytes changed",
    ):
        finalize_part2_offline(
            manifest_path, unavailable_targets=[BETA],
            source_verification_root=clean,
        )
