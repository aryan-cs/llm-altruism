from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from analysis.merge_sota_compatibility_with_judge import (
    COMBINED_REGISTRY_ARTIFACT_TYPE,
    JUDGE_TARGET_ID,
    SotaJudgeMergeError,
    main,
    merge_sota_compatibility_with_judge,
)
from experiments.misc.inference_hub_part1_panel import select_routes


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private" / "inference_hub"
SOTA_REGISTRY_PATH = PRIVATE / "sota-probe-registry-v1.json"
SOTA_COMPATIBILITY_PATH = PRIVATE / "sota-compatibility-evidence-v1.json"
PRODUCTION_REGISTRY_PATH = ROOT / "agents" / "agent_config.registry.json"
PRODUCTION_COMPATIBILITY_PATH = PRIVATE / "compatibility-evidence-v2.json"


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sealed_inputs() -> tuple[dict[str, Any], ...]:
    paths = (
        SOTA_REGISTRY_PATH,
        SOTA_COMPATIBILITY_PATH,
        PRODUCTION_REGISTRY_PATH,
        PRODUCTION_COMPATIBILITY_PATH,
    )
    if not all(path.exists() for path in paths):
        pytest.skip("real private SOTA and production compatibility artifacts absent")
    return tuple(_load(path) for path in paths)


def _merge(inputs: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], dict[str, Any]]:
    return merge_sota_compatibility_with_judge(
        sota_registry=inputs[0],
        sota_compatibility=inputs[1],
        production_registry=inputs[2],
        production_compatibility=inputs[3],
    )


def _reseal_registry(registry: dict[str, Any]) -> None:
    registry.pop("artifact_sha256", None)
    registry["artifact_sha256"] = _canonical_hash(registry)


def _reseal_compatibility(
    compatibility: dict[str, Any], registry: dict[str, Any] | None = None
) -> None:
    if registry is not None:
        compatibility["registry_sha256"] = _canonical_hash(registry)
        compatibility["registry_version"] = registry["registry_version"]
    compatibility.pop("evidence_sha256", None)
    compatibility["evidence_sha256"] = _canonical_hash(compatibility)


def test_real_merge_preserves_every_source_row_and_recomputes_all_bindings(
    sealed_inputs: tuple[dict[str, Any], ...],
) -> None:
    sota_registry, sota_compatibility, production_registry, production_compatibility = (
        sealed_inputs
    )
    registry, compatibility = _merge(sealed_inputs)
    judge_registry = next(
        row for row in production_registry["targets"] if row["id"] == JUDGE_TARGET_ID
    )
    judge_evidence = next(
        row
        for row in production_compatibility["targets"]
        if row["target_id"] == JUDGE_TARGET_ID
    )

    assert registry["artifact_type"] == COMBINED_REGISTRY_ARTIFACT_TYPE
    assert registry["targets"][:-1] == sota_registry["targets"]
    assert registry["targets"][-1] == judge_registry
    assert compatibility["targets"][:-1] == sota_compatibility["targets"]
    assert compatibility["targets"][-1] == judge_evidence
    assert _canonical_hash(registry["targets"][:-1]) == _canonical_hash(
        sota_registry["targets"]
    )
    assert _canonical_hash(compatibility["targets"][:-1]) == _canonical_hash(
        sota_compatibility["targets"]
    )
    assert registry["cohorts"]["exploratory_sota"]["targets"] == [
        row["id"] for row in sota_registry["targets"]
    ]
    assert registry["cohorts"]["judge_only"]["targets"] == [JUDGE_TARGET_ID]
    assert registry["artifact_sha256"] == _canonical_hash(
        {key: value for key, value in registry.items() if key != "artifact_sha256"}
    )
    assert compatibility["registry_sha256"] == _canonical_hash(registry)
    assert compatibility["evidence_sha256"] == _canonical_hash(
        {
            key: value
            for key, value in compatibility.items()
            if key != "evidence_sha256"
        }
    )
    assert compatibility["target_count"] == 85
    assert compatibility["selected_count"] == 82
    assert compatibility["unresolved_count"] == 3
    assert compatibility["candidate_count"] == sum(
        row["candidate_count"] for row in compatibility["targets"]
    )
    for artifact in (registry, compatibility):
        assert artifact["confirmatory_promotion_permitted"] is False
        assert artifact["paper_result_promotion_permitted"] is False
        assert artifact["production_registry_mutation_permitted"] is False
        assert artifact["source_bundles"]["exploratory_subjects"][
            "registry_sha256"
        ] == _canonical_hash(sota_registry)
        assert artifact["source_bundles"]["dedicated_judge"][
            "registry_sha256"
        ] == _canonical_hash(production_registry)
        assert artifact["source_bundles"]["exploratory_subjects"][
            "compatibility_evidence_sha256"
        ] == sota_compatibility["evidence_sha256"]
        assert artifact["source_bundles"]["dedicated_judge"][
            "compatibility_evidence_sha256"
        ] == production_compatibility["evidence_sha256"]


def test_real_output_is_consumable_by_hosted_runner_route_selection(
    sealed_inputs: tuple[dict[str, Any], ...],
) -> None:
    registry, compatibility = _merge(sealed_inputs)

    subjects, judge = select_routes(
        registry=registry,
        compatibility=compatibility,
        selected_ids=None,
        judge_target_id=JUDGE_TARGET_ID,
    )

    assert len(subjects) == 81
    assert len({row["target_id"] for row in subjects}) == 81
    assert judge["target_id"] == JUDGE_TARGET_ID
    assert judge["route"] == "nvidia/nvidia/evals-nemotron-3-30b-a3b"
    assert judge["route"] not in {row["route"] for row in subjects}
    assert (judge["upstream_provider"], judge["model"]) not in {
        (row["upstream_provider"], row["model"]) for row in subjects
    }


@pytest.mark.parametrize(
    ("artifact_index", "field", "message"),
    [
        (0, "artifact_sha256", "artifact_sha256 is invalid"),
        (1, "evidence_sha256", "evidence_sha256 is invalid"),
        (3, "evidence_sha256", "evidence_sha256 is invalid"),
    ],
)
def test_tampered_input_self_hashes_fail_closed(
    sealed_inputs: tuple[dict[str, Any], ...],
    artifact_index: int,
    field: str,
    message: str,
) -> None:
    altered = list(copy.deepcopy(sealed_inputs))
    altered[artifact_index][field] = "0" * 64

    with pytest.raises(SotaJudgeMergeError, match=message):
        _merge(tuple(altered))


def test_resealed_wrong_registry_binding_endpoint_and_catalog_fail_closed(
    sealed_inputs: tuple[dict[str, Any], ...],
) -> None:
    wrong_binding = list(copy.deepcopy(sealed_inputs))
    wrong_binding[1]["registry_sha256"] = "0" * 64
    _reseal_compatibility(wrong_binding[1])
    with pytest.raises(SotaJudgeMergeError, match="canonical supplied registry"):
        _merge(tuple(wrong_binding))

    wrong_endpoint = list(copy.deepcopy(sealed_inputs))
    wrong_endpoint[3]["endpoint"] = "https://different.invalid/v1"
    _reseal_compatibility(wrong_endpoint[3])
    with pytest.raises(SotaJudgeMergeError, match="endpoints do not match"):
        _merge(tuple(wrong_endpoint))

    wrong_catalog = list(copy.deepcopy(sealed_inputs))
    wrong_catalog[3]["catalog_source_payload_sha256"]["models"] = "f" * 64
    _reseal_compatibility(wrong_catalog[3])
    with pytest.raises(SotaJudgeMergeError, match="payload hashes do not match"):
        _merge(tuple(wrong_catalog))


def test_resealed_unselected_wrong_or_nonexclusive_judge_fails_closed(
    sealed_inputs: tuple[dict[str, Any], ...],
) -> None:
    unselected = list(copy.deepcopy(sealed_inputs))
    judge = next(
        row
        for row in unselected[3]["targets"]
        if row["target_id"] == JUDGE_TARGET_ID
    )
    judge["status"] = "unresolved"
    _reseal_compatibility(unselected[3])
    with pytest.raises(SotaJudgeMergeError, match="selected target evidence"):
        _merge(tuple(unselected))

    extra_judge = list(copy.deepcopy(sealed_inputs))
    extra_id = extra_judge[2]["cohorts"]["current_sota"]["targets"].pop()
    extra_judge[2]["cohorts"]["judge_only"]["targets"].append(extra_id)
    next(
        row for row in extra_judge[3]["targets"] if row["target_id"] == extra_id
    )["cohort"] = "judge_only"
    _reseal_compatibility(extra_judge[3], extra_judge[2])
    with pytest.raises(SotaJudgeMergeError, match="exactly the judge"):
        _merge(tuple(extra_judge))

    with pytest.raises(SotaJudgeMergeError, match="must be exactly"):
        merge_sota_compatibility_with_judge(
            sota_registry=sealed_inputs[0],
            sota_compatibility=sealed_inputs[1],
            production_registry=sealed_inputs[2],
            production_compatibility=sealed_inputs[3],
            judge_target_id="judge.some-other-model",
        )


@pytest.mark.parametrize("overlap_kind", ["route", "provider_model"])
def test_resealed_cross_bundle_overlap_fails_closed(
    sealed_inputs: tuple[dict[str, Any], ...], overlap_kind: str
) -> None:
    altered = list(copy.deepcopy(sealed_inputs))
    subject = altered[0]["targets"][0]
    judge = next(
        row for row in altered[2]["targets"] if row["id"] == JUDGE_TARGET_ID
    )
    if overlap_kind == "route":
        subject["route"] = judge["route"]
        message = "Judge route overlaps"
    else:
        subject["upstream_provider"] = judge["upstream_provider"]
        subject["model"] = judge["model"]
        altered[1]["targets"][0]["model"] = judge["model"]
        message = "provider/model identity overlaps"
    _reseal_registry(altered[0])
    _reseal_compatibility(altered[1], altered[0])

    with pytest.raises(SotaJudgeMergeError, match=message):
        _merge(tuple(altered))


def test_cli_is_byte_deterministic_atomic_private_and_does_not_modify_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    sealed_inputs: tuple[dict[str, Any], ...],
) -> None:
    input_paths = (
        SOTA_REGISTRY_PATH,
        SOTA_COMPATIBILITY_PATH,
        PRODUCTION_REGISTRY_PATH,
        PRODUCTION_COMPATIBILITY_PATH,
    )
    before = {path: path.read_bytes() for path in input_paths}
    first_dir = tmp_path / "first-private"
    second_dir = tmp_path / "second-private"
    first_registry = first_dir / "registry.json"
    first_compatibility = first_dir / "compatibility.json"
    second_registry = second_dir / "registry.json"
    second_compatibility = second_dir / "compatibility.json"

    def invoke(registry_output: Path, compatibility_output: Path) -> int:
        return main(
            [
                "--sota-registry",
                str(SOTA_REGISTRY_PATH),
                "--sota-compatibility",
                str(SOTA_COMPATIBILITY_PATH),
                "--production-registry",
                str(PRODUCTION_REGISTRY_PATH),
                "--production-compatibility",
                str(PRODUCTION_COMPATIBILITY_PATH),
                "--registry-output",
                str(registry_output),
                "--compatibility-output",
                str(compatibility_output),
            ]
        )

    assert invoke(first_registry, first_compatibility) == 0
    assert invoke(second_registry, second_compatibility) == 0
    assert first_registry.read_bytes() == second_registry.read_bytes()
    assert first_compatibility.read_bytes() == second_compatibility.read_bytes()
    assert {path: path.read_bytes() for path in input_paths} == before
    assert not list(tmp_path.rglob(".*.tmp"))
    if os.name == "posix":
        for directory in (first_dir, second_dir):
            assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        for path in (
            first_registry,
            first_compatibility,
            second_registry,
            second_compatibility,
        ):
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
    output = capsys.readouterr().out
    assert "84 exploratory subjects (81 selected)" in output
    assert "promotion are prohibited" in output
