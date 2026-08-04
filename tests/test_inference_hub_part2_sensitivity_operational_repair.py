from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from typing import Any

import pytest

from analysis.analyze_provider_safe_v2_definitive import (
    SENSITIVITY_OPERATIONAL_REPAIR_STATUS,
    _sensitivity,
    _validate_sensitivity_operational_repair,
)
from experiments.misc import inference_hub_part2_sensitivity_v1 as runner
from experiments.misc.inference_hub_part2_sensitivity_operational_repair import (
    Part2SensitivityOperationalRepairError,
    TRAJECTORY_ARTIFACT_TYPE,
    _load_source,
    _source_key,
    build_overlay_from_complete_subset,
    run_repair,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = (
    ROOT
    / "data/private/inference_hub/definitive-part2-sensitivity-deadline-fast-v9"
)
SUBSET_RUN = (
    ROOT
    / "data/private/inference_hub/definitive-part2-sensitivity-operational-repair-source-v1"
)


class _ValidClient:
    def __init__(self, *, semantic_invalid_first: bool = False, wrong_identity: bool = False):
        self.semantic_invalid_first = semantic_invalid_first
        self.wrong_identity = wrong_identity
        self.calls: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        assert path == "/chat/completions"
        with self.lock:
            self.calls.append(json.loads(json.dumps(body)))
            call_number = len(self.calls)
        invalid = self.semantic_invalid_first and call_number == 1
        return {
            "id": f"repair-{call_number}",
            "model": "wrong/route" if self.wrong_identity else body["model"],
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "unfinished"
                            if invalid
                            else json.dumps(
                                {
                                    "action": "OPTION_A",
                                    "reasoning": "Preserve the shared reserve.",
                                }
                            )
                        ),
                    },
                    "finish_reason": "length" if invalid else "stop",
                }
            ],
        }


@pytest.mark.skipif(
    not (SOURCE_RUN / "private/manifest.json").is_file()
    or not (SUBSET_RUN / "private/manifest.json").is_file(),
    reason="Private production repair evidence is intentionally not distributed.",
)
def test_complete_subset_selects_only_exact_four_source_failures(
    tmp_path: Path,
) -> None:
    source_path = SOURCE_RUN / "private/manifest.json"
    subset_path = SUBSET_RUN / "private/manifest.json"
    source, _, _, _, source_rows, _ = _load_source(source_path)
    source_hash = runner._sha256_file(source_path)
    subset_hash = runner._sha256_file(subset_path)
    overlay = tmp_path / "subset-overlay"
    manifest = build_overlay_from_complete_subset(
        source_manifest_path=source_path,
        subset_manifest_path=subset_path,
        output_dir=overlay,
    )
    assert manifest["complete"] is True
    assert manifest["subset_trajectory_count"] == 12
    assert manifest["selected_subset_trajectory_count"] == 4
    assert manifest["excluded_subset_trajectory_count"] == 8
    assert len(manifest["journals"]) == 4
    assert runner._sha256_file(source_path) == source_hash
    assert runner._sha256_file(subset_path) == subset_hash

    trajectories, effects, audit, _, _ = (
        _validate_sensitivity_operational_repair(
            source_run=SOURCE_RUN,
            source_manifest_path=source_path,
            source_manifest=source,
            repair_value=overlay,
        )
    )
    effective = {_source_key(row): row for row in trajectories["rows"]}
    eligible = {
        key for key, row in source_rows.items() if not row["operationally_eligible"]
    }
    assert len(eligible) == 4
    assert all(
        runner._canonical_bytes(effective[key])
        == runner._canonical_bytes(source_rows[key])
        for key in set(source_rows) - eligible
    )
    assert len(effects["rows"]) == 25
    assert audit["effective_operational_failure_trajectories"] == 0


@pytest.mark.skipif(
    not (SOURCE_RUN / "private/manifest.json").is_file(),
    reason="Private production-shaped v9 evidence is intentionally not distributed.",
)
def test_production_v9_full_trajectory_overlay_is_immutable_and_complete(
    tmp_path: Path,
) -> None:
    source_manifest = SOURCE_RUN / "private/manifest.json"
    source, _, _, _, source_rows, _ = _load_source(source_manifest)
    source_files = {
        Path(ref["path"]): ref["file_sha256"] for ref in source["journals"].values()
    }
    source_files[source_manifest] = runner._sha256_file(source_manifest)
    source_files[Path(source["attempt_ledger"]["path"])] = source[
        "attempt_ledger"
    ]["file_sha256"]
    eligible = {
        key for key, row in source_rows.items() if not row["operationally_eligible"]
    }
    assert len(source_rows) == 160
    assert len(eligible) == 4
    assert sum(source_rows[key]["transport_failure_count"] for key in eligible) == 6

    client = _ValidClient(semantic_invalid_first=True)
    output = tmp_path / "repair"
    result = run_repair(
        source_manifest_path=source_manifest,
        output_dir=output,
        client=client,
        max_rounds=2,
        campaign_workers=4,
        participant_workers=8,
        max_attempts=1,
        initial_backoff_seconds=0,
    )
    assert result["complete"] is True
    assert result["summary"]["source_trajectory_count"] == 160
    assert result["summary"]["eligible_operational_failure_trajectories"] == 4
    assert result["summary"]["successful_full_trajectory_repairs"] == 4
    assert result["summary"]["effective_trajectory_count"] == 160
    assert result["summary"]["effective_operational_failure_trajectories"] == 0
    assert result["summary"]["effective_identity_mismatch_count"] == 0
    assert result["summary"]["effective_transport_failure_count"] == 0
    assert result["summary"]["source_semantic_invalid_trajectories_retried"] == 0
    assert all(runner._sha256_file(path) == digest for path, digest in source_files.items())

    trajectory_ref = result["sanitized_artifacts"]["effective_trajectory_metrics"]
    trajectory_payload = runner._read_json(
        Path(trajectory_ref["path"]), "effective trajectories"
    )
    assert trajectory_payload["artifact_type"] == TRAJECTORY_ARTIFACT_TYPE
    assert len(trajectory_payload["rows"]) == 160
    effective = {
        (row["cell_id"], row["target_id"], row["trajectory_index"]): row
        for row in trajectory_payload["rows"]
    }
    assert set(effective) == set(source_rows)
    assert all(
        runner._canonical_bytes(effective[key])
        == runner._canonical_bytes(source_rows[key])
        for key in set(source_rows) - eligible
    )
    assert all(effective[key]["operationally_eligible"] for key in eligible)
    assert sum(effective[key]["invalid_count"] for key in eligible) == 1
    outcomes = runner._read_json(
        Path(result["sanitized_artifacts"]["repair_outcomes"]["path"]),
        "repair outcomes",
    )
    assert outcomes["source_semantic_invalid_trajectories_retried"] == 0
    assert outcomes["successful_full_trajectory_repair_count"] == 4

    (
        validated_trajectories,
        validated_effects,
        analyzer_audit,
        validated_manifest_path,
        validated_manifest,
    ) = _validate_sensitivity_operational_repair(
        source_run=SOURCE_RUN,
        source_manifest_path=source_manifest,
        source_manifest=source,
        repair_value=output,
    )
    assert validated_trajectories["evidence_sha256"] == trajectory_payload[
        "evidence_sha256"
    ]
    assert len(validated_effects["rows"]) == 25
    assert analyzer_audit["status"] == SENSITIVITY_OPERATIONAL_REPAIR_STATUS
    assert validated_manifest_path == output / "private/manifest.json"
    assert validated_manifest["evidence_sha256"] == result["evidence_sha256"]
    effect_rows, model_rows = _sensitivity(
        SOURCE_RUN,
        source,
        effective_trajectories=validated_trajectories,
        effective_effects=validated_effects,
    )
    assert len(effect_rows) == 25
    assert len(model_rows) == 5
    assert sum(row["trajectory_count"] for row in model_rows) == 160

    no_op_client = _ValidClient()
    resumed = run_repair(
        source_manifest_path=source_manifest,
        output_dir=output,
        client=no_op_client,
        max_rounds=2,
        campaign_workers=4,
        participant_workers=8,
        max_attempts=1,
        initial_backoff_seconds=0,
        resume=True,
    )
    assert resumed["complete"] is True
    assert no_op_client.calls == []


@pytest.mark.skipif(
    not (SOURCE_RUN / "private/manifest.json").is_file(),
    reason="Private production-shaped v9 evidence is intentionally not distributed.",
)
def test_identity_mismatch_never_becomes_effective_replacement(tmp_path: Path) -> None:
    result = run_repair(
        source_manifest_path=SOURCE_RUN / "private/manifest.json",
        output_dir=tmp_path / "repair",
        client=_ValidClient(wrong_identity=True),
        max_rounds=1,
        campaign_workers=4,
        participant_workers=8,
        max_attempts=1,
        initial_backoff_seconds=0,
    )
    assert result["complete"] is False
    assert result["summary"]["successful_full_trajectory_repairs"] == 0
    assert result["summary"]["unresolved_operational_failure_trajectories"] == 4
    assert result["summary"]["effective_operational_failure_trajectories"] == 4


@pytest.mark.skipif(
    not (SOURCE_RUN / "private/manifest.json").is_file(),
    reason="Private production-shaped v9 evidence is intentionally not distributed.",
)
def test_route_and_source_mutation_are_rejected_before_dispatch(tmp_path: Path) -> None:
    copied = tmp_path / "source-copy"
    shutil.copytree(SOURCE_RUN, copied)
    manifest_path = copied / "private/manifest.json"
    manifest = runner._read_json(manifest_path, "copied manifest")
    original_private = SOURCE_RUN / "private"
    copied_private = copied / "private"
    for reference in manifest["journals"].values():
        relative = Path(reference["path"]).resolve().relative_to(original_private.resolve())
        reference["path"] = str((copied_private / relative).resolve())
    attempt_relative = Path(manifest["attempt_ledger"]["path"]).resolve().relative_to(
        original_private.resolve()
    )
    manifest["attempt_ledger"]["path"] = str((copied_private / attempt_relative).resolve())
    for reference in manifest["sanitized_artifacts"].values():
        relative = Path(reference["path"]).resolve().relative_to(SOURCE_RUN.resolve())
        reference["path"] = str((copied / relative).resolve())
    manifest["selected_subject_routes"][0]["route"] = "mutated/exact-route"
    runner._seal(manifest)
    runner._atomic_json(manifest_path, manifest)
    client = _ValidClient()
    with pytest.raises(Part2SensitivityOperationalRepairError, match="binding|reproduce"):
        run_repair(
            source_manifest_path=manifest_path,
            output_dir=tmp_path / "repair",
            client=client,
            max_rounds=1,
            max_attempts=1,
            initial_backoff_seconds=0,
        )
    assert client.calls == []

    manifest["selected_subject_routes"][0]["route"] = source_route = runner._read_json(
        SOURCE_RUN / "private/manifest.json", "source"
    )["selected_subject_routes"][0]["route"]
    assert source_route
    first_ref = next(iter(manifest["journals"].values()))
    first_ref["file_sha256"] = "0" * 64
    runner._seal(manifest)
    runner._atomic_json(manifest_path, manifest)
    with pytest.raises(Exception, match="checkpoint|hash|bytes"):
        _load_source(manifest_path)
