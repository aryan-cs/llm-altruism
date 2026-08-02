import hashlib
import json
import os
import shutil
import stat
import threading
from pathlib import Path
from typing import Any

import pytest

from analysis.analyze_inference_hub_part1_panel import _self_hash
from analysis.analyze_joint_inference_hub_part1_panels import (
    JointHostedPart1AnalysisError,
    analyze_joint_panels,
    cli,
)
from experiments.misc.inference_hub_part1_panel import _sha256_json, run_panel


ENDPOINT = "https://inference-api.nvidia.com/v1"
JUDGE_ID = "judge.route"


def _registry() -> dict[str, Any]:
    targets = []
    for target_id, provider, model, display_route in (
        ("subject.alpha", "alpha", "alpha-model", "display/alpha-model"),
        ("subject.beta", "beta", "beta-model", "display/beta-model"),
        ("subject.gamma", "gamma", "gamma-model", "display/gamma-model"),
        (JUDGE_ID, "judge-provider", "judge-model", "display/judge-model"),
    ):
        targets.append(
            {
                "id": target_id,
                "provider": "inference_hub",
                "upstream_provider": provider,
                "model": model,
                "route": display_route,
            }
        )
    return {"schema_version": 1, "registry_version": "joint-test-v1", "targets": targets}


def _compatibility_target(target_id: str, model: str, route: str) -> dict[str, Any]:
    profile = {
        "attempt_id": f"probe-{target_id}",
        "controls": ["seed"],
        "profile_id": f"profile-{target_id}",
        "request_sha256": hashlib.sha256(target_id.encode()).hexdigest(),
        "resumed_from_ledger": False,
        "status": "passed",
        "validation_source": "execution_profile_probe",
    }
    return {
        "target_id": target_id,
        "model": model,
        "candidate_count": 1,
        "frozen_candidate_order": [route],
        "selected_execution_candidate": route,
        "selected_execution_profile": {**profile, "route": route},
        "selection_basis": "first_execution_compatible_in_reconciliation_frozen_order",
        "status": "execution_candidate_selected",
        "candidates": [
            {
                "route": route,
                "candidate_index": 0,
                "max_tokens": 48,
                "execution_compatible": True,
                "selected_execution_profile": profile,
            }
        ],
    }


def _write_inputs(root: Path) -> tuple[Path, Path]:
    registry = _registry()
    targets = [
        _compatibility_target(target["id"], target["model"], f"region/{target['model']}")
        for target in registry["targets"]
    ]
    compatibility = {
        "schema_version": 2,
        "artifact_type": "inference_hub_provider_compatibility",
        "endpoint": ENDPOINT,
        "registry_sha256": _sha256_json(registry),
        "targets": targets,
        "target_count": len(targets),
        "selected_count": len(targets),
        "unresolved_count": 0,
    }
    compatibility["evidence_sha256"] = _sha256_json(compatibility)
    root.mkdir(parents=True, exist_ok=True)
    registry_path = root / "registry.json"
    compatibility_path = root / "compatibility.json"
    registry_path.write_text(json.dumps(registry) + "\n", encoding="utf-8")
    compatibility_path.write_text(json.dumps(compatibility) + "\n", encoding="utf-8")
    return registry_path, compatibility_path


class _Client:
    base_url = ENDPOINT

    def __init__(self, *, quarantine_target: str | None = None) -> None:
        self.quarantine_target = quarantine_target
        self.lock = threading.Lock()
        self.calls = 0

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        assert path == "/chat/completions"
        with self.lock:
            self.calls += 1
        model = body["model"]
        target = next(
            target_id
            for target_id in ("subject.alpha", "subject.beta", "subject.gamma", JUDGE_ID)
            if model.endswith(target_id.split(".")[-1] + "-model")
        )
        response_model = "unexpected/provider-model" if target == self.quarantine_target else model
        action = "Y" if target == "subject.beta" else "X"
        return {
            "id": f"request-{target}-{body['seed']}",
            "model": response_model,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"analysis\n{action}",
                        "reasoning_content": "private reasoning must never escape",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }


def _run(
    root: Path,
    inputs: tuple[Path, Path],
    selected_ids: list[str],
    *,
    quarantine_target: str | None = None,
    base_seed: int = 20_260_802,
) -> Path:
    output = root
    manifest = run_panel(
        registry_path=inputs[0],
        compatibility_path=inputs[1],
        output_dir=output,
        client=_Client(quarantine_target=quarantine_target),
        selected_ids=selected_ids,
        judge_target_id=JUDGE_ID,
        base_seed=base_seed,
        max_workers=16,
        max_workers_per_subject=4,
        initial_backoff_seconds=0,
    )
    assert manifest["full_primary_root_count"] == 384
    return output / "private" / "manifest.json"


def _coherent_copy(manifest_path: Path, destination: Path) -> Path:
    source_panel = manifest_path.parent.parent
    shutil.copytree(source_panel, destination)
    copied_manifest_path = destination / "private" / "manifest.json"
    manifest = json.loads(copied_manifest_path.read_text(encoding="utf-8"))
    copied_inputs = destination / "bound-inputs"
    copied_inputs.mkdir()
    for name, reference in manifest["input_artifacts"].items():
        copied = copied_inputs / f"{name}.json"
        shutil.copy2(reference["path"], copied)
        reference["path"] = str(copied.resolve())
    private = destination / "private"
    manifest["journals"]["attempt_ledger"]["path"] = str(
        (private / "attempt_ledger.jsonl").resolve()
    )
    for target_id, reference in manifest["journals"]["raw_responses"].items():
        reference["path"] = str(
            (private / "raw_responses" / f"{target_id}.jsonl").resolve()
        )
    manifest["evidence_sha256"] = _self_hash(manifest)
    copied_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(copied_manifest_path, 0o600)
    return copied_manifest_path


@pytest.fixture(scope="module")
def joint_panels(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("joint-hosted-part1")
    inputs = _write_inputs(root / "inputs")
    return {
        "base_one": _run(
            root / "base-one",
            inputs,
            ["subject.alpha", "subject.beta"],
            quarantine_target="subject.beta",
        ),
        "base_two": _run(root / "base-two", inputs, ["subject.gamma"]),
        "replacement": _run(root / "replacement-beta", inputs, ["subject.beta"]),
        "drifted": _run(
            root / "replacement-beta-drifted",
            inputs,
            ["subject.beta"],
            base_seed=20_260_803,
        ),
    }


def test_joint_cli_repairs_quarantined_target_and_runs_one_safe_shared_bootstrap(
    joint_panels: dict[str, Path], tmp_path: Path
) -> None:
    bases = [joint_panels["base_one"], joint_panels["base_two"]]
    result = analyze_joint_panels(bases, [joint_panels["replacement"]])

    assert [row["target_id"] for row in result["subjects"]] == [
        "subject.alpha",
        "subject.beta",
        "subject.gamma",
    ]
    assert result["coverage"] == {
        "base_manifest_count": 2,
        "replacement_manifest_count": 1,
        "unique_planned_target_count": 3,
        "aggregate_eligible_subject_count": 3,
        "unavailable_target_count": 0,
        "replaced_target_count": 1,
        "quarantined_source_target_count": 1,
        "roots_per_subject": 384,
        "retained_eligible_subject_trial_count": 1152,
        "game_domain_strata": 12,
        "roots_per_stratum_per_subject": 32,
        "judge_rows": 0,
    }
    assert result["replaced_targets"][0]["target_id"] == "subject.beta"
    assert result["replaced_targets"][0]["base_status"] == "operationally_quarantined"
    assert result["quarantined_targets"][0]["target_id"] == "subject.beta"
    assert result["overall_equal_subject"]["primary_action_x_rate_95_ci"]["estimate"] == pytest.approx(2 / 3)
    assert result["parameters"]["bootstrap_replicates"] == 5_000
    assert result["parameters"]["panel_interval_combination"] == (
        "single_joint_bootstrap_distribution_not_averaged_input_intervals"
    )
    assert result["confirmatory_or_paper_promotion_permitted"] is False
    assert result["evidence_sha256"] == _self_hash(result)
    serialized = json.dumps(result)
    assert "private reasoning must never escape" not in serialized
    assert '"prompt_text"' not in serialized
    assert '"response_text"' not in serialized
    assert '"raw_response"' not in serialized
    assert '"reasoning_fields"' not in serialized

    output = tmp_path / "derived" / "joint.json"
    assert cli(
        [
            "--base-manifest",
            str(bases[0]),
            "--base-manifest",
            str(bases[1]),
            "--replacement-manifest",
            str(joint_panels["replacement"]),
            "--output",
            str(output),
        ]
    ) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert not list(output.parent.glob(".joint.json.*.tmp"))


def test_joint_rejects_tamper_duplicate_replacement_and_schedule_drift(
    joint_panels: dict[str, Path], tmp_path: Path
) -> None:
    tampered = _coherent_copy(joint_panels["base_two"], tmp_path / "tampered")
    payload = json.loads(tampered.read_text(encoding="utf-8"))
    payload["summary"]["format_valid"] -= 1
    tampered.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    os.chmod(tampered, 0o600)
    with pytest.raises(JointHostedPart1AnalysisError, match="self-hash"):
        analyze_joint_panels([joint_panels["base_one"], tampered], [joint_panels["replacement"]])

    duplicate = _coherent_copy(joint_panels["replacement"], tmp_path / "duplicate")
    with pytest.raises(JointHostedPart1AnalysisError, match="Duplicate replacements"):
        analyze_joint_panels(
            [joint_panels["base_one"], joint_panels["base_two"]],
            [joint_panels["replacement"], duplicate],
        )

    with pytest.raises(JointHostedPart1AnalysisError, match="Schedule/prompt drift"):
        analyze_joint_panels(
            [joint_panels["base_one"], joint_panels["base_two"]],
            [joint_panels["drifted"]],
        )


def test_joint_rejects_outcome_selection_and_registry_identity_ambiguity(
    joint_panels: dict[str, Path], tmp_path: Path
) -> None:
    eligible_gamma_as_replacement = _coherent_copy(
        joint_panels["base_two"], tmp_path / "eligible-gamma-replacement"
    )
    with pytest.raises(JointHostedPart1AnalysisError, match="Outcome-selection guard"):
        analyze_joint_panels(
            [joint_panels["base_one"], joint_panels["base_two"]],
            [eligible_gamma_as_replacement],
        )

    ambiguous = _coherent_copy(joint_panels["replacement"], tmp_path / "ambiguous")
    manifest = json.loads(ambiguous.read_text(encoding="utf-8"))
    registry_path = Path(manifest["input_artifacts"]["registry"]["path"])
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    next(row for row in registry["targets"] if row["id"] == "subject.beta")[
        "upstream_provider"
    ] = "changed-provider"
    registry_path.write_text(json.dumps(registry) + "\n", encoding="utf-8")
    # This is an independently bound-input tamper and must fail before it can
    # create a cross-manifest identity ambiguity.
    with pytest.raises(JointHostedPart1AnalysisError, match="Registry input file hash changed"):
        analyze_joint_panels(
            [joint_panels["base_one"], joint_panels["base_two"]], [ambiguous]
        )
