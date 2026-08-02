from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments import campaign


def _args(*values: str):
    args = campaign.build_parser().parse_args(list(values))
    args.campaign_id = args.campaign_id or "test-campaign"
    return args


def test_default_scientific_phases_cover_every_current_sota_target() -> None:
    plan = campaign.build_plan(
        _args("--phase", "part0", "--phase", "part1", "--phase", "part2")
    )

    assert plan["cohort"]["id"] == "current_sota"
    assert len(plan["cohort"]["target_ids"]) == 14
    assert len(plan["jobs"]) == 29  # one cohort-wide part 0 plus 14 each for parts 1/2

    part_0 = next(job for job in plan["jobs"] if job["experiment"] == "part_0")
    assert part_0["expected"]["prompt_count"] == len(
        campaign.load_part_0_raw_prompts()
    )
    assert len(part_0["expected"]["languages"]) == 10
    assert part_0["counts"]["decisions"] == 14 * 500 * 10
    assert part_0["counts"]["target_model_requests_baseline_estimate"] == 14 * 500 * 10
    assert part_0["command"][0] == sys.executable
    assert part_0["command"].count("--benchmark") == 14
    assert "--judge-after" in part_0["command"]

    part_1_jobs = [job for job in plan["jobs"] if job["experiment"] == "part_1"]
    assert len(part_1_jobs) == 14
    assert {job["expected"]["row_count"] for job in part_1_jobs} == {384}
    assert [job["expected"]["ordering"]["counterbalance_index"] for job in part_1_jobs] == list(
        range(14)
    )
    assert {job["expected"]["ordering"]["seed"] for job in part_1_jobs} == {
        campaign.DEFAULT_PART_1_ORDER_SEED
    }
    assert all("--order-seed" in job["command"] for job in part_1_jobs)
    assert all("--counterbalance-index" in job["command"] for job in part_1_jobs)

    part_2_jobs = [job for job in plan["jobs"] if job["experiment"] == "part_2"]
    assert len(part_2_jobs) == 14
    assert {job["counts"]["decisions_upper_bound"] for job in part_2_jobs} == {5000}
    assert {
        job["counts"]["target_model_requests_baseline_estimate"]
        for job in part_2_jobs
    } == {5000}
    assert all("--seed" not in job["command"] for job in plan["jobs"])


def test_historical_cohort_and_replicated_part2_subset_are_exact() -> None:
    plan = campaign.build_plan(
        _args(
            "--cohort",
            "historical",
            "--phase",
            "part2",
            "--part2-target",
            "openai.gpt-4.1-2025-04-14",
            "--part2-replicates",
            "3",
            "--part2-society-size",
            "7",
            "--part2-days",
            "9",
        )
    )

    assert len(plan["cohort"]["target_ids"]) == 6
    assert len(plan["jobs"]) == 3
    assert [job["expected"]["replicate"] for job in plan["jobs"]] == [1, 2, 3]
    assert {job["target_ids"][0] for job in plan["jobs"]} == {
        "openai.gpt-4.1-2025-04-14"
    }
    assert {job["counts"]["decisions_upper_bound"] for job in plan["jobs"]} == {63}


def test_factorial_part2_sensitivity_plan_records_exact_balanced_cells() -> None:
    plan = campaign.build_plan(
        _args(
            "--cohort",
            "historical",
            "--phase",
            "part2",
            "--part2-target",
            "openai.gpt-4.1-2025-04-14",
            "--part2-grid-capacity",
            "100",
            "--part2-grid-capacity",
            "200",
            "--part2-grid-depletion-units",
            "1",
            "--part2-grid-depletion-units",
            "3",
            "--part2-grid-death-rate",
            "0.1",
            "--part2-grid-population",
            "5",
            "--part2-grid-horizon",
            "7",
            "--part2-grid-seed",
            "11",
            "--part2-grid-seed",
            "22",
            "--part2-grid-seed",
            "33",
        )
    )

    design = plan["part2_design"]
    assert design == {
        "mode": "factorial",
        "cell_fields": list(campaign.PART2_CELL_FIELDS),
        "trajectory_cells": design["trajectory_cells"],
        "structural_cell_count": 4,
        "replicates_per_structural_cell": 3,
        "trajectory_cell_count": 12,
        "target_count": 1,
        "job_count": 12,
        "target_model_requests_upper_bound": 12 * 5 * 7,
    }
    assert len(plan["jobs"]) == 12
    assert {job["expected"]["generation_seed"] for job in plan["jobs"]} == {
        11,
        22,
        33,
    }
    assert {job["expected"]["resource_capacity"] for job in plan["jobs"]} == {
        100,
        200,
    }
    assert all("--resource-capacity" in job["command"] for job in plan["jobs"])
    assert all("--collapse-death-rate" in job["command"] for job in plan["jobs"])
    assert all("--seed" in job["command"] for job in plan["jobs"])
    assert sum(
        job["counts"]["target_model_requests_baseline_estimate"]
        for job in plan["jobs"]
    ) == design["target_model_requests_upper_bound"]


def test_specified_part2_cells_require_equal_seed_replication() -> None:
    one = json.dumps(
        {
            "resource_capacity": 100,
            "depletion_units": 2,
            "collapse_death_rate": 0.2,
            "society_size": 5,
            "days": 7,
            "seed": 11,
        }
    )
    two = json.dumps(
        {
            "resource_capacity": 200,
            "depletion_units": 2,
            "collapse_death_rate": 0.2,
            "society_size": 5,
            "days": 7,
            "seed": 11,
        }
    )
    three = json.dumps(
        {
            "resource_capacity": 200,
            "depletion_units": 2,
            "collapse_death_rate": 0.2,
            "society_size": 5,
            "days": 7,
            "seed": 22,
        }
    )

    with pytest.raises(campaign.CampaignError, match="same number of replicate seeds"):
        campaign.build_plan(
            _args(
                "--phase",
                "part2",
                "--part2-cell",
                one,
                "--part2-cell",
                two,
                "--part2-cell",
                three,
            )
        )


def test_factorial_part2_plan_is_bounded_before_execution(monkeypatch) -> None:
    monkeypatch.setattr(campaign, "MAX_PART2_SENSITIVITY_CELLS", 1)
    with pytest.raises(campaign.CampaignError, match="bounded job limit"):
        campaign.build_plan(
            _args(
                "--phase",
                "part2",
                "--part2-grid-capacity",
                "100",
                "--part2-grid-seed",
                "1",
                "--part2-grid-seed",
                "2",
            )
        )


def test_part2_subset_rejects_target_outside_exact_cohort() -> None:
    with pytest.raises(campaign.CampaignError, match="exact members"):
        campaign.build_plan(
            _args(
                "--cohort",
                "historical",
                "--phase",
                "part2",
                "--part2-target",
                "openai.gpt-5.6-sol",
            )
        )


def _part1_job(job_id: str = "part1-one") -> dict[str, object]:
    return {
        "id": job_id,
        "phase": "part1",
        "experiment": "part_1",
        "target_ids": ["target.one"],
        "command": [sys.executable, "-m", "experiments.part1.part_1"],
        "command_display": f"{sys.executable} -m experiments.part1.part_1",
        "expected": {
            "provider": "inference_hub",
            "model": "model-one",
            "games": ["game"],
            "frames": ["frame"],
            "domains": ["domain"],
            "presentations": ["presentation"],
            "limit": 1,
            "row_count": 1,
        },
        "counts": {"decisions": 1, "target_model_requests_baseline_estimate": 1},
        "status": "pending",
        "artifact": None,
        "attempts": [],
    }


def _manifest(*jobs: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "campaign_id": "test-campaign",
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "updated_at_utc": "2026-01-01T00:00:00+00:00",
        "status": "planned",
        "cohort": {
            "id": "current_sota",
            "version": "test",
            "registry_version": "test",
            "registry_hash": "test",
            "target_ids": ["target.one"],
        },
        "phases": ["part1"],
        "timeout_seconds": 30,
        "jobs": list(jobs),
    }


def _write_part1_artifact(
    repo: Path,
    job: dict[str, object],
    *,
    name: str = "run",
    status: str = "complete",
) -> Path:
    result_dir = repo / "data" / "raw" / "part_1"
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / f"{name}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prompt_id", "action"])
        writer.writerow(["one", "COOPERATE"])
    expected = job["expected"]
    metadata = {
        "experiment": "part_1",
        "status": status,
        "csv_path": str(csv_path.relative_to(repo)),
        "provider": expected["provider"],
        "model": expected["model"],
        "games": expected["games"],
        "frames": expected["frames"],
        "domains": expected["domains"],
        "presentations": expected["presentations"],
        "limit": expected["limit"],
        "total_prompts": expected["row_count"],
        "completed_rows": 1,
    }
    metadata_path = result_dir / f"{name}_meta.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata_path


def _configure_temp_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(campaign, "REPO_ROOT", repo)
    monkeypatch.setattr(campaign, "CAMPAIGN_ROOT", repo / "data" / "campaigns")
    return repo


def test_strict_preflight_failure_records_status_before_any_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = _configure_temp_repo(monkeypatch, tmp_path)
    campaign_dir = repo / "data" / "campaigns" / "test-campaign"
    campaign_dir.mkdir(parents=True)
    manifest = _manifest(_part1_job())
    campaign._atomic_write_json(campaign_dir / "manifest.json", manifest)
    subprocess_called = False

    def fail_preflight(*args, **kwargs):
        raise EnvironmentError("missing credential")

    def process_runner(*args, **kwargs):
        nonlocal subprocess_called
        subprocess_called = True
        return campaign.ProcessResult(returncode=0)

    with pytest.raises(EnvironmentError, match="missing credential"):
        campaign.execute_manifest(
            campaign_dir,
            manifest,
            fail_fast=False,
            process_runner=process_runner,
            preflight_runner=fail_preflight,
        )

    assert not subprocess_called
    saved = json.loads((campaign_dir / "manifest.json").read_text())
    assert saved["status"] == "preflight_failed"
    assert saved["preflight_failure"]["type"] == "OSError"


def test_success_requires_a_new_verified_native_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = _configure_temp_repo(monkeypatch, tmp_path)
    campaign_dir = repo / "data" / "campaigns" / "test-campaign"
    campaign_dir.mkdir(parents=True)
    job = _part1_job()
    manifest = _manifest(job)
    campaign._atomic_write_json(campaign_dir / "manifest.json", manifest)
    observed: dict[str, object] = {}
    monkeypatch.setenv("LLM_ALTRUISM_SKIP_PREFLIGHT", "1")

    def preflight(name, targets, **kwargs):
        observed["preflight_targets"] = list(targets)
        observed["strict"] = kwargs["strict_provider_checks"]
        observed["skip_visible_during_preflight"] = "LLM_ALTRUISM_SKIP_PREFLIGHT" in campaign.os.environ

    def process_runner(command, cwd, env, log_path, timeout):
        observed["command"] = command
        observed["cwd"] = cwd
        observed["skip_child_preflight"] = env["LLM_ALTRUISM_SKIP_PREFLIGHT"]
        observed["timeout"] = timeout
        _write_part1_artifact(repo, job)
        return campaign.ProcessResult(returncode=0)

    result = campaign.execute_manifest(
        campaign_dir,
        manifest,
        fail_fast=False,
        process_runner=process_runner,
        preflight_runner=preflight,
    )

    assert result == 0
    assert observed["strict"] is True
    assert observed["skip_visible_during_preflight"] is False
    assert campaign.os.environ["LLM_ALTRUISM_SKIP_PREFLIGHT"] == "1"
    assert observed["preflight_targets"] == [("inference_hub", "model-one")]
    assert observed["command"] == job["command"]
    assert observed["cwd"] == repo
    assert observed["skip_child_preflight"] == "1"
    saved = json.loads((campaign_dir / "manifest.json").read_text())
    assert saved["status"] == "complete"
    assert saved["jobs"][0]["status"] == "complete"
    assert saved["jobs"][0]["artifact"]["rows"] == 1


def test_verified_complete_job_is_the_only_kind_skipped_on_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = _configure_temp_repo(monkeypatch, tmp_path)
    campaign_dir = repo / "data" / "campaigns" / "test-campaign"
    campaign_dir.mkdir(parents=True)
    job = _part1_job()
    metadata_path = _write_part1_artifact(repo, job)
    job["status"] = "complete"
    job["artifact"] = {"metadata_path": str(metadata_path.relative_to(repo))}
    manifest = _manifest(job)

    def process_runner(*args):
        raise AssertionError("a verified complete job must not be rerun")

    assert campaign.execute_manifest(
        campaign_dir,
        manifest,
        fail_fast=False,
        process_runner=process_runner,
        preflight_runner=lambda *args, **kwargs: None,
    ) == 0
    assert job["status"] == "complete"
    assert job["artifact"]["rows"] == 1


def test_resume_does_not_trust_complete_status_when_artifact_is_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = _configure_temp_repo(monkeypatch, tmp_path)
    campaign_dir = repo / "data" / "campaigns" / "test-campaign"
    campaign_dir.mkdir(parents=True)
    job = _part1_job()
    invalid_path = _write_part1_artifact(repo, job, name="invalid", status="failed")
    job["status"] = "complete"
    job["artifact"] = {"metadata_path": str(invalid_path.relative_to(repo))}
    manifest = _manifest(job)
    calls = 0

    def process_runner(*args):
        nonlocal calls
        calls += 1
        _write_part1_artifact(repo, job, name="replacement")
        return campaign.ProcessResult(returncode=0)

    assert campaign.execute_manifest(
        campaign_dir,
        manifest,
        fail_fast=False,
        process_runner=process_runner,
        preflight_runner=lambda *args, **kwargs: None,
    ) == 0
    assert calls == 1
    assert job["artifact"]["metadata_path"].endswith("replacement_meta.json")
    assert "not marked complete" in job["resume_verification_error"]


@pytest.mark.parametrize("fail_fast, expected_calls", [(False, 2), (True, 1)])
def test_failures_are_recorded_and_continue_policy_is_respected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fail_fast: bool,
    expected_calls: int,
) -> None:
    repo = _configure_temp_repo(monkeypatch, tmp_path)
    campaign_dir = repo / "data" / "campaigns" / "test-campaign"
    campaign_dir.mkdir(parents=True)
    manifest = _manifest(_part1_job("first"), _part1_job("second"))
    calls = 0

    def process_runner(*args):
        nonlocal calls
        calls += 1
        return campaign.ProcessResult(returncode=17, error="provider unavailable")

    result = campaign.execute_manifest(
        campaign_dir,
        manifest,
        fail_fast=fail_fast,
        process_runner=process_runner,
        preflight_runner=lambda *args, **kwargs: None,
    )

    assert result == 1
    assert calls == expected_calls
    assert manifest["jobs"][0]["status"] == "failed"
    assert manifest["jobs"][0]["attempts"][0]["returncode"] == 17
    if not fail_fast:
        assert manifest["jobs"][1]["status"] == "failed"


def test_run_subprocess_uses_no_shell_and_enforces_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}

    class Process:
        pid = 1234

        def wait(self, timeout=None):
            observed["wait_timeout"] = timeout
            raise subprocess.TimeoutExpired(["python"], timeout)

        def poll(self):
            return None

    def popen(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return Process()

    terminated: list[object] = []
    monkeypatch.setattr(campaign.subprocess, "Popen", popen)
    monkeypatch.setattr(campaign, "_terminate_process_group", terminated.append)
    log_path = tmp_path / "job.log"

    result = campaign.run_subprocess(
        [sys.executable, "-m", "experiments.part1.part_1"],
        tmp_path,
        {"PATH": ""},
        log_path,
        42,
    )

    assert result.timed_out is True
    assert result.returncode is None
    assert observed["wait_timeout"] == 42
    assert observed["kwargs"]["shell"] is False
    assert observed["kwargs"]["stdin"] is subprocess.DEVNULL
    assert terminated and terminated[0].pid == 1234


def test_artifact_csv_must_stay_under_native_result_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = _configure_temp_repo(monkeypatch, tmp_path)
    job = _part1_job()
    metadata_path = _write_part1_artifact(repo, job)
    metadata = json.loads(metadata_path.read_text())
    metadata["csv_path"] = str(tmp_path / "outside.csv")
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(campaign.CampaignError, match="escapes"):
        campaign.verify_artifact(job, metadata_path)


def test_resume_rejects_a_modified_pinned_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure_temp_repo(monkeypatch, tmp_path)
    plan = campaign.build_plan(
        _args(
            "--phase",
            "part2",
            "--part2-target",
            "openai.gpt-5.6-sol",
        )
    )
    campaign_dir = campaign._campaign_dir("test-campaign")
    campaign_dir.mkdir(parents=True)
    plan["jobs"][0]["command"].append("--unexpected")
    campaign._atomic_write_json(campaign_dir / "manifest.json", plan)

    with pytest.raises(campaign.CampaignError, match="plan hash mismatch"):
        campaign._resume_manifest("test-campaign")


def test_dry_run_creates_no_campaign_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _configure_temp_repo(monkeypatch, tmp_path)

    assert campaign.main(
        [
            "--phase",
            "part2",
            "--part2-target",
            "openai.gpt-5.6-sol",
            "--campaign-id",
            "dry-plan",
            "--dry-run",
        ]
    ) == 0
    assert "gpt-5.6-sol" in capsys.readouterr().out
    assert not campaign.CAMPAIGN_ROOT.exists()
