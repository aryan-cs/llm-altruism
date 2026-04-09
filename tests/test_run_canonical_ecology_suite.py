"""Tests for canonical ecology suite orchestration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def _load_suite_module():
    module_path = ROOT / "scripts" / "run_canonical_ecology_suite.py"
    spec = importlib.util.spec_from_file_location("run_canonical_ecology_suite_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_continue_module():
    module_path = ROOT / "scripts" / "continue_canonical_ecology_suite.py"
    spec = importlib.util.spec_from_file_location("continue_canonical_ecology_suite_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_recover_module():
    module_path = ROOT / "scripts" / "recover_canonical_ecology_baseline.py"
    spec = importlib.util.spec_from_file_location("recover_canonical_ecology_baseline_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_maintain_module():
    module_path = ROOT / "scripts" / "maintain_canonical_ecology_suite.py"
    spec = importlib.util.spec_from_file_location("maintain_canonical_ecology_suite_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_ops_module():
    module_path = ROOT / "scripts" / "refresh_canonical_ecology_ops_status.py"
    spec = importlib.util.spec_from_file_location("refresh_canonical_ecology_ops_status_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_suite_runs_cover_baseline_reputation_and_event_stress():
    module = _load_suite_module()

    runs = module.suite_runs("results/canonical")

    assert [run.name for run in runs] == ["baseline", "reputation", "event-stress"]
    assert runs[0].config_path == "configs/part2/society_baseline.yaml"
    assert runs[1].config_path == "configs/part3/society_reputation.yaml"
    assert runs[2].config_path == "configs/part2/society_event_stress.yaml"


def test_selected_runs_can_resume_from_reputation():
    module = _load_suite_module()

    runs = module.selected_runs(module.suite_runs("results/canonical"), "reputation")

    assert [run.name for run in runs] == ["reputation", "event-stress"]


def test_build_command_includes_models_results_dir_and_dry_run():
    module = _load_suite_module()
    run = module.SuiteRun(
        name="baseline",
        config_path="configs/part2/society_baseline.yaml",
        results_dir="results/canonical/baseline",
    )

    command = module.build_command(
        run,
        models=["cerebras:llama3.1-8b", "nvidia:deepseek-ai/deepseek-v3.2"],
        dry_run=True,
    )

    assert command[:5] == [
        module.project_python(),
        "scripts/run_experiment.py",
        "--config",
        "configs/part2/society_baseline.yaml",
        "--results-dir",
    ]
    assert "results/canonical/baseline" in command
    assert command.count("--model") == 2
    assert "--dry-run" in command


def test_build_command_can_forward_resume_log():
    module = _load_suite_module()
    run = module.SuiteRun(
        name="baseline",
        config_path="configs/part2/society_baseline.yaml",
        results_dir="results/canonical/baseline",
    )

    command = module.build_command(
        run,
        models=["cerebras:llama3.1-8b"],
        dry_run=False,
        resume_log="results/live_ecology_20260408/society-baseline-20260408T171454Z.jsonl",
    )

    assert "--resume-log" in command
    assert "results/live_ecology_20260408/society-baseline-20260408T171454Z.jsonl" in command


def test_continue_module_detects_completed_baseline():
    module = _load_continue_module()

    assert module.baseline_is_complete({"total_expected_trials": 3, "completed_trials": 3}) is True
    assert module.baseline_is_complete({"total_expected_trials": 3, "completed_trials": 2}) is False


def test_continue_module_builds_followon_command():
    module = _load_continue_module()

    command = module.build_followon_command(
        results_root="results/followon",
        from_run="reputation",
        models=["cerebras:llama3.1-8b", "nvidia:deepseek-ai/deepseek-v3.2"],
        dry_run=True,
    )

    assert command[:5] == [
        module.project_python(),
        "scripts/run_canonical_ecology_suite.py",
        "--from-run",
        "reputation",
        "--results-root",
    ]
    assert "results/followon" in command
    assert command.count("--model") == 2
    assert "--dry-run" in command


def test_continue_module_builds_refresh_command():
    module = _load_continue_module()

    command = module.build_refresh_command("results/live_ecology_20260408_resume")

    assert command == [
        module.project_python(),
        "scripts/refresh_live_ecology_packet.py",
        "results/live_ecology_20260408_resume",
    ]


def test_continue_module_writes_watcher_status(tmp_path: Path):
    module = _load_continue_module()
    summary = {
        "completed_trials": 1,
        "total_expected_trials": 3,
        "prompt_variant": "cooperative",
        "latest_round_num": 12,
    }
    command = [
        module.project_python(),
        "scripts/run_canonical_ecology_suite.py",
        "--from-run",
        "reputation",
    ]

    output_path = module.write_watcher_status(
        results_root=tmp_path,
        baseline_results="results/live_ecology_20260408_resume",
        summary=summary,
        followon_command=command,
        watcher_state="waiting",
    )

    assert output_path == tmp_path / "watch_status.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["watcher_state"] == "waiting"
    assert payload["baseline_results"] == "results/live_ecology_20260408_resume"
    assert payload["followon_command"] == command
    assert payload["baseline_summary"]["prompt_variant"] == "cooperative"


def test_continue_module_waits_for_first_log(monkeypatch, tmp_path: Path):
    module = _load_continue_module()
    followon_command = [module.project_python(), "scripts/run_canonical_ecology_suite.py"]
    calls: list[str] = []
    summaries = iter(
        [
            FileNotFoundError("No JSONL experiment logs found."),
            {"total_expected_trials": 1, "completed_trials": 1, "latest_trial_id": 0},
        ]
    )

    def fake_summarize(*args, **kwargs):
        value = next(summaries)
        if isinstance(value, Exception):
            raise value
        return value

    def fake_write_watcher_status(**kwargs):
        calls.append(kwargs["watcher_state"])
        return tmp_path / "watch_status.json"

    monkeypatch.setattr(module, "summarize_baseline", fake_summarize)
    monkeypatch.setattr(module, "write_watcher_status", fake_write_watcher_status)
    monkeypatch.setattr(module.time, "sleep", lambda *_args, **_kwargs: None)

    summary = module.wait_for_baseline_completion(
        "results/new_baseline",
        results_root=tmp_path,
        followon_command=followon_command,
        poll_seconds=0.0,
        stale_minutes=15.0,
        refresh_results_packet=False,
    )

    assert calls == ["waiting_for_log", "baseline_complete"]
    assert summary["completed_trials"] == 1


def test_recover_module_detects_needing_recovery():
    module = _load_recover_module()

    assert module.needs_recovery(
        {"state": "stale", "completed_trials": 1, "total_expected_trials": 3}
    ) is True
    assert module.needs_recovery(
        {"state": "active", "completed_trials": 1, "total_expected_trials": 3}
    ) is False
    assert module.needs_recovery(
        {"state": "stale", "completed_trials": 3, "total_expected_trials": 3}
    ) is False


def test_recover_module_builds_resume_and_watcher_commands():
    module = _load_recover_module()

    resume_command = module.build_resume_command(
        config_path="configs/part2/society_baseline.yaml",
        results_dir="results/recovered_run",
        resume_log="results/live_ecology_20260408_resume/society-baseline-20260408T235541Z.jsonl",
        models=["cerebras:llama3.1-8b"],
        dry_run=True,
    )
    watcher_command = module.build_followon_watcher_command(
        baseline_results="results/recovered_run",
        followon_root="results/followon",
        models=["cerebras:llama3.1-8b"],
        dry_run=True,
    )

    assert "--resume-log" in resume_command
    assert "results/recovered_run" in resume_command
    assert "--dry-run" in resume_command
    assert watcher_command[:5] == [
        module.project_python(),
        "scripts/continue_canonical_ecology_suite.py",
        "results/recovered_run",
        "--results-root",
        "results/followon",
    ]
    assert "--refresh-packet" in watcher_command
    assert "--dry-run" in watcher_command


def test_recover_module_builds_maintenance_command():
    module = _load_recover_module()

    command = module.build_maintenance_command(
        baseline_results="results/recovered_run",
        followon_root="results/followon",
        results_parent="results",
        config_path="configs/part2/society_baseline.yaml",
        models=["cerebras:llama3.1-8b"],
        stale_minutes=15.0,
        dry_run=True,
    )

    assert command[:5] == [
        module.project_python(),
        "scripts/maintain_canonical_ecology_suite.py",
        "results/recovered_run",
        "--followon-root",
        "results/followon",
    ]
    assert "--loop" in command
    assert "--poll-seconds" in command
    assert "--dry-run" in command


def test_recover_module_makes_rollover_results_dir():
    module = _load_recover_module()
    path = module.make_rollover_results_dir(
        "results",
        name="society-baseline",
        now=module.datetime(2026, 4, 9, 0, 10, 0, tzinfo=module.UTC),
    )

    assert path == Path("results/society-baseline_20260409T001000Z")


def test_maintain_module_builds_recovery_command():
    module = _load_maintain_module()

    command = module.build_recovery_command(
        baseline_results="results/live_ecology_20260408_resume",
        followon_root="results/followon",
        results_parent="results",
        config_path="configs/part2/society_baseline.yaml",
        models=["cerebras:llama3.1-8b"],
        stale_minutes=15.0,
        dry_run=True,
    )

    assert command[:5] == [
        module.project_python(),
        "scripts/recover_canonical_ecology_baseline.py",
        "results/live_ecology_20260408_resume",
        "--followon-root",
        "results/followon",
    ]
    assert "--dry-run" in command
    assert "--model" in command


def test_maintain_module_builds_ops_refresh_command():
    module = _load_maintain_module()

    command = module.build_ops_refresh_command(
        baseline_results="results/live_ecology_20260408_resume",
        followon_root="results/live_ecology_20260408_followon",
    )

    assert command == [
        module.project_python(),
        "scripts/refresh_canonical_ecology_ops_status.py",
        "results/live_ecology_20260408_resume",
        "--followon-root",
        "results/live_ecology_20260408_followon",
    ]


def test_maintain_module_uses_default_models_for_watcher():
    module = _load_maintain_module()
    recover_module = _load_recover_module()

    class Args:
        model: list[str] = []

    models = module.configured_models(Args(), recover_module)

    assert models == list(recover_module.DEFAULT_MODEL_SELECTORS)


def test_maintain_module_builds_watcher_command():
    module = _load_maintain_module()
    recover_module = _load_recover_module()

    command = module.build_watcher_command(
        baseline_results="results/live_ecology_20260408_resume",
        followon_root="results/live_ecology_20260408_followon",
        models=["cerebras:llama3.1-8b"],
        dry_run=True,
        recover_module=recover_module,
    )

    assert command[:5] == [
        module.project_python(),
        "scripts/continue_canonical_ecology_suite.py",
        "results/live_ecology_20260408_resume",
        "--results-root",
        "results/live_ecology_20260408_followon",
    ]
    assert "--refresh-packet" in command
    assert "--dry-run" in command


def test_maintain_module_detects_followon_started(tmp_path: Path):
    module = _load_maintain_module()
    reputation_dir = tmp_path / "reputation"
    reputation_dir.mkdir(parents=True)

    assert module.followon_has_started(tmp_path) is False

    (reputation_dir / "run.jsonl").write_text("{}", encoding="utf-8")

    assert module.followon_has_started(tmp_path) is True


def test_maintain_module_writes_status_file(tmp_path: Path):
    module = _load_maintain_module()
    summary = {"state": "active", "completed_trials": 1, "total_expected_trials": 3}
    command = [module.project_python(), "scripts/recover_canonical_ecology_baseline.py", "results/live"]
    path = tmp_path / "maintenance_status.json"

    module.write_status(
        path=path,
        baseline_summary=summary,
        recovery_needed=False,
        recovery_command=command,
        recovery_returncode=None,
        watcher_status={"needed": True, "running": True, "pid": 1234},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["recovery_needed"] is False
    assert payload["recovery_command"] == command
    assert payload["baseline_summary"]["state"] == "active"
    assert payload["watcher"]["pid"] == 1234


def test_maintain_module_parses_loop_arguments(monkeypatch):
    module = _load_maintain_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "maintain_canonical_ecology_suite.py",
            "results/live",
            "--loop",
            "--poll-seconds",
            "90",
        ],
    )

    args = module.parse_args()

    assert args.baseline_results == "results/live"
    assert args.loop is True
    assert args.poll_seconds == 90.0


def test_ops_module_builds_payload_and_markdown():
    module = _load_ops_module()
    payload = module.build_payload(
        baseline_summary={"state": "active", "prompt_variant": "cooperative", "latest_round_num": 12, "alive_count": 18, "total_agents": 24, "completed_trials": 1, "total_expected_trials": 3, "provider_retry_count": 9, "estimated_minutes_remaining_in_trial": 12.5, "naive_minutes_remaining_in_baseline_suite": 25.0},
        watch_status={"watcher_state": "waiting", "baseline_results": "results/live"},
        maintenance_status={"recovery_needed": False, "recovery_returncode": None, "watcher": {"needed": True, "running": True, "pid": 4321, "followon_started": False}},
    )

    markdown = module.build_markdown(payload)

    assert payload["baseline"]["prompt_variant"] == "cooperative"
    assert "# Canonical Ecology Ops Status" in markdown
    assert "`cooperative`" in markdown
    assert "`18/24`" in markdown
    assert "`4321`" in markdown
    assert "`12.5`" in markdown


def test_ops_module_default_output_paths():
    module = _load_ops_module()

    json_path, markdown_path = module.default_output_paths("results/followon")

    assert json_path == Path("results/followon/ops_status.json")
    assert markdown_path == Path("results/followon/ops_status.md")
