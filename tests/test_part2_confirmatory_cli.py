import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys

from analysis import part2_confirmatory


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "analysis.part2_confirmatory",
            *(str(argument) for argument in arguments),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_json(path: Path, value: object, *, private: bool = False) -> bytes:
    encoded = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    if private:
        path.chmod(0o600)
    return encoded


def _verify_seal(value: dict[str, object], hash_field: str) -> None:
    payload = dict(value)
    recorded = payload.pop(hash_field)
    canonical = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    assert recorded == hashlib.sha256(canonical).hexdigest()


def _variance_input(tmp_path: Path) -> dict[str, object]:
    seeds = [str(1_026_080_100 + seed) for seed in range(8)]
    return part2_confirmatory._sealed_artifact({
        "schema_version": 1,
        "artifact_type": "part2_native_identity_masked_variance_input",
        "pilot_campaign_path": str((tmp_path / "pilot-manifest.json").resolve()),
        "pilot_campaign_manifest_sha256": "a" * 64,
        "pilot_campaign_payload_sha256": "b" * 64,
        "native_artifact_set_sha256": "c" * 64,
        "identity_masking_scheme": "automated_plan_bound_location_insensitive_v1",
        "frozen_blinded_group_ids": ["private-alpha", "private-beta"],
        "pilot_by_blinded_group": {
            "private-alpha": {seed: 0.2 for seed in seeds},
            "private-beta": {seed: 0.8 for seed in seeds},
        },
    })


def test_select_variance_cli_is_private_blind_hash_bound_and_fresh_only(
    tmp_path: Path, monkeypatch,
) -> None:
    input_path = tmp_path / "private-pilot.json"
    variance_input = _variance_input(tmp_path)
    raw_input = _write_json(input_path, variance_input, private=True)
    output_path = tmp_path / "selection.json"
    monkeypatch.setattr(
        part2_confirmatory,
        "_derive_native_blinded_variance_input",
        lambda path, expected_sha256: variance_input,
    )

    completed = part2_confirmatory.main([
        "select-variance", "--input", str(input_path),
        "--pilot-campaign", variance_input["pilot_campaign_path"],
        "--pilot-campaign-sha256", "a" * 64,
        "--output", str(output_path),
    ])

    assert completed == 0
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    _verify_seal(artifact, "artifact_sha256")
    assert artifact["private_input_sha256"] == hashlib.sha256(raw_input).hexdigest()
    assert artifact["pilot_campaign_manifest_sha256"] == "a" * 64
    assert artifact["selection"]["selected_common_run_count"] == 20
    serialized = output_path.read_text(encoding="utf-8")
    assert "private-alpha" not in serialized
    assert "private-beta" not in serialized
    assert "0.2" not in serialized
    assert "0.8" not in serialized
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o600

    original = output_path.read_bytes()
    repeated = part2_confirmatory.main([
        "select-variance", "--input", str(input_path),
        "--pilot-campaign", variance_input["pilot_campaign_path"],
        "--pilot-campaign-sha256", "a" * 64,
        "--output", str(output_path),
    ])
    assert repeated == 2
    assert output_path.read_bytes() == original


def test_select_variance_cli_rejects_permissions_duplicates_and_schema_drift(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    variance_input = _variance_input(tmp_path)
    monkeypatch.setattr(
        part2_confirmatory,
        "_derive_native_blinded_variance_input",
        lambda path, expected_sha256: variance_input,
    )
    lineage_args = [
        "--pilot-campaign", variance_input["pilot_campaign_path"],
        "--pilot-campaign-sha256", "a" * 64,
    ]
    public_input = tmp_path / "public.json"
    _write_json(public_input, variance_input)
    public_input.chmod(0o644)
    rejected = part2_confirmatory.main([
        "select-variance", "--input", str(public_input), *map(str, lineage_args),
        "--output", str(tmp_path / "should-not-exist.json"),
    ])
    assert rejected == 2
    assert "private permissions" in capsys.readouterr().err

    extra = _variance_input(tmp_path)
    extra["unexpected"] = True
    extra_path = tmp_path / "extra.json"
    _write_json(extra_path, extra, private=True)
    rejected = part2_confirmatory.main([
        "select-variance", "--input", str(extra_path), *map(str, lineage_args),
        "--output", str(tmp_path / "extra-output.json"),
    ])
    assert rejected == 2
    assert "does not exactly match" in capsys.readouterr().err
    assert not (tmp_path / "extra-output.json").exists()

    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(
        '{"schema_version":1,"schema_version":1,'
        '"frozen_blinded_group_ids":[],"pilot_by_blinded_group":{}}\n',
        encoding="utf-8",
    )
    duplicate_path.chmod(0o600)
    rejected = part2_confirmatory.main([
        "select-variance", "--input", str(duplicate_path), *map(str, lineage_args),
        "--output", str(tmp_path / "duplicate-output.json"),
    ])
    assert rejected == 2
    assert "duplicate JSON object key" in capsys.readouterr().err

    bad_lineage = _variance_input(tmp_path)
    bad_lineage["pilot_campaign_manifest_sha256"] = "A" * 64
    bad_lineage_path = tmp_path / "bad-lineage.json"
    _write_json(bad_lineage_path, bad_lineage, private=True)
    rejected = part2_confirmatory.main([
        "select-variance", "--input", str(bad_lineage_path), *map(str, lineage_args),
        "--output", str(tmp_path / "bad-lineage-output.json"),
    ])
    assert rejected == 2
    assert "does not exactly match" in capsys.readouterr().err


def _design_command(output_path: Path) -> list[object]:
    arguments: list[object] = ["sensitivity-design"]
    for sentinel in range(6):
        arguments.extend(("--sentinel-id", f"sentinel-{sentinel}"))
    for seed in range(901, 907):
        arguments.extend(("--environment-seed", seed))
    arguments.extend(("--output", output_path))
    return arguments


def _completed_sensitivity_input(manifest: dict[str, object]) -> dict[str, object]:
    observations: list[dict[str, object]] = []
    for sentinel in manifest["frozen_sentinel_ids"]:
        for cell in manifest["cells"]:
            coded = cell["coded_levels"]
            for seed_index, seed in enumerate(manifest["common_environment_seeds"]):
                observations.append(
                    {
                        "sentinel_id": sentinel,
                        "cell_id": cell["cell_id"],
                        "capacity_per_initial_agent": cell[
                            "capacity_per_initial_agent"
                        ],
                        "depletion_units": cell["depletion_units"],
                        "collapse_death_rate": cell["collapse_death_rate"],
                        "society_size": cell["society_size"],
                        "horizon_days": cell["horizon_days"],
                        "resource_capacity": cell["resource_capacity"],
                        "environment_seed": seed,
                        "normalized_aurc": (
                            0.5
                            + (0.02 + 0.001 * seed_index)
                            * coded["capacity_per_initial_agent"]
                            + 0.01 * coded["depletion_units"]
                        ),
                    }
                )
    return {
        "schema_version": 1,
        "design_manifest_sha256": manifest["manifest_sha256"],
        "observations": observations,
    }


def test_sensitivity_cli_builds_and_enforces_hash_pinned_complete_design(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "design.json"
    built = _run_cli(*_design_command(manifest_path))
    assert built.returncode == 0, built.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_seal(manifest, "manifest_sha256")
    assert manifest["cell_count"] == 16
    assert manifest["expected_observation_count"] == 576
    assert len(manifest["cells"]) == 16
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600

    observation_path = tmp_path / "completed-observations.json"
    raw_observations = _write_json(
        observation_path,
        _completed_sensitivity_input(manifest),
    )
    output_path = tmp_path / "sensitivity-analysis.json"
    analyzed = _run_cli(
        "analyze-sensitivity",
        "--input",
        observation_path,
        "--design-manifest",
        manifest_path,
        "--output",
        output_path,
    )
    assert analyzed.returncode == 0, analyzed.stderr
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    _verify_seal(artifact, "artifact_sha256")
    assert artifact["design_manifest_sha256"] == manifest["manifest_sha256"]
    assert artifact["observation_input_sha256"] == hashlib.sha256(
        raw_observations
    ).hexdigest()
    assert artifact["observation_count"] == 576
    assert len(artifact["main_effect_results"]) == 30
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o600

    tampered_manifest = dict(manifest)
    tampered_manifest["cell_count"] = 15
    tampered_path = tmp_path / "tampered-design.json"
    _write_json(tampered_path, tampered_manifest)
    rejected = _run_cli(
        "analyze-sensitivity",
        "--input",
        observation_path,
        "--design-manifest",
        tampered_path,
        "--output",
        tmp_path / "tampered-analysis.json",
    )
    assert rejected.returncode == 2
    assert "verification failed" in rejected.stderr
    assert "Traceback" not in rejected.stderr
    assert not (tmp_path / "tampered-analysis.json").exists()


def test_sensitivity_cli_rejects_incomplete_or_unpinned_observations(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "design.json"
    assert _run_cli(*_design_command(manifest_path)).returncode == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    incomplete = _completed_sensitivity_input(manifest)
    incomplete["observations"] = incomplete["observations"][:-1]
    input_path = tmp_path / "incomplete.json"
    _write_json(input_path, incomplete)
    output_path = tmp_path / "incomplete-output.json"
    rejected = _run_cli(
        "analyze-sensitivity",
        "--input",
        input_path,
        "--design-manifest",
        manifest_path,
        "--output",
        output_path,
    )
    assert rejected.returncode == 2
    assert "exactly 576" in rejected.stderr
    assert "Traceback" not in rejected.stderr
    assert not output_path.exists()

    unpinned = _completed_sensitivity_input(manifest)
    unpinned["design_manifest_sha256"] = "0" * 64
    unpinned_path = tmp_path / "unpinned.json"
    _write_json(unpinned_path, unpinned)
    rejected = _run_cli(
        "analyze-sensitivity",
        "--input",
        unpinned_path,
        "--design-manifest",
        manifest_path,
        "--output",
        tmp_path / "unpinned-output.json",
    )
    assert rejected.returncode == 2
    assert "does not pin" in rejected.stderr
    assert not (tmp_path / "unpinned-output.json").exists()

    duplicated = _completed_sensitivity_input(manifest)
    duplicated["observations"][-1] = dict(duplicated["observations"][0])
    duplicated_path = tmp_path / "duplicated.json"
    _write_json(duplicated_path, duplicated)
    rejected = _run_cli(
        "analyze-sensitivity",
        "--input",
        duplicated_path,
        "--design-manifest",
        manifest_path,
        "--output",
        tmp_path / "duplicated-output.json",
    )
    assert rejected.returncode == 2
    assert "duplicate sensitivity cell/seed" in rejected.stderr
    assert "Traceback" not in rejected.stderr

    drifted = _completed_sensitivity_input(manifest)
    drifted["observations"][0]["post_outcome_label"] = "not-prespecified"
    drifted_path = tmp_path / "drifted.json"
    _write_json(drifted_path, drifted)
    rejected = _run_cli(
        "analyze-sensitivity",
        "--input",
        drifted_path,
        "--design-manifest",
        manifest_path,
        "--output",
        tmp_path / "drifted-output.json",
    )
    assert rejected.returncode == 2
    assert "invalid schema" in rejected.stderr
    assert not (tmp_path / "drifted-output.json").exists()


def _baseline_input() -> dict[str, object]:
    return {
        "schema_version": 1,
        "structural_cell": {
            "provider": "no-call",
            "model": "baseline",
            "society_size": 50,
            "horizon_days": 100,
            "resource": "water",
            "resource_capacity": 2500,
            "selfish_gain": 2,
            "depletion_units": 2,
            "community_benefit": 5,
            "collapse_death_rate": 0.2,
        },
    }


def test_baselines_cli_emits_exact_hash_bound_no_call_suite(tmp_path: Path) -> None:
    input_path = tmp_path / "cell.json"
    raw_input = _write_json(input_path, _baseline_input())
    output_path = tmp_path / "baselines.json"
    completed = _run_cli(
        "baselines",
        "--input",
        input_path,
        "--environment-seed",
        501,
        "--environment-seed",
        502,
        "--output",
        output_path,
    )
    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    _verify_seal(artifact, "artifact_sha256")
    assert artifact["structural_cell_input_sha256"] == hashlib.sha256(
        raw_input
    ).hexdigest()
    assert artifact["mechanical_survival_threshold"] == {
        "total_agent_days": 5000,
        "max_safe_overuse_actions": 1249,
        "max_safe_overuse_rate": 0.2498,
        "min_restraint_actions": 3751,
        "min_restraint_rate": 0.7502,
        "resource_floor_units": 2,
    }
    assert len(artifact["baselines"]) == 9
    assert sum(
        row["policy"].startswith("bernoulli") for row in artifact["baselines"]
    ) == 6
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o600

    duplicate_seed = _run_cli(
        "baselines",
        "--input",
        input_path,
        "--environment-seed",
        501,
        "--environment-seed",
        501,
        "--output",
        tmp_path / "duplicate-seed.json",
    )
    assert duplicate_seed.returncode == 2
    assert "must be unique" in duplicate_seed.stderr
    assert "Traceback" not in duplicate_seed.stderr

    drifted = _baseline_input()
    drifted["structural_cell"]["unplanned_parameter"] = 1
    drifted_path = tmp_path / "drifted-cell.json"
    _write_json(drifted_path, drifted)
    rejected = _run_cli(
        "baselines",
        "--input",
        drifted_path,
        "--environment-seed",
        501,
        "--output",
        tmp_path / "drifted-baseline.json",
    )
    assert rejected.returncode == 2
    assert "invalid schema" in rejected.stderr
    assert "Traceback" not in rejected.stderr


def test_sensitivity_design_cli_rejects_nonexact_or_duplicate_freeze(
    tmp_path: Path,
) -> None:
    too_few = _run_cli(
        "sensitivity-design",
        "--sentinel-id",
        "only-one",
        "--environment-seed",
        1,
        "--output",
        tmp_path / "too-few.json",
    )
    assert too_few.returncode == 2
    assert "exactly 6" in too_few.stderr
    assert "Traceback" not in too_few.stderr

    arguments: list[object] = ["sensitivity-design"]
    for sentinel in range(6):
        arguments.extend(("--sentinel-id", f"sentinel-{sentinel}"))
    for seed in (1, 2, 3, 4, 5, 5):
        arguments.extend(("--environment-seed", seed))
    arguments.extend(("--output", tmp_path / "duplicate-design.json"))
    duplicate = _run_cli(*arguments)
    assert duplicate.returncode == 2
    assert "must be unique" in duplicate.stderr
    assert not (tmp_path / "duplicate-design.json").exists()
