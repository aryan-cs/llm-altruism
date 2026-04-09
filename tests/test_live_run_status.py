"""Tests for live JSONL run status reporting."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_live_run_status_module():
    module_path = ROOT / "scripts" / "live_run_status.py"
    spec = importlib.util.spec_from_file_location("live_run_status_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summarize_jsonl_log_extracts_society_state(tmp_path: Path):
    module = _load_live_run_status_module()
    log_path = tmp_path / "society-live.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "experiment_start",
                        "timestamp": "2026-04-08T17:14:54+00:00",
                        "experiment_id": "society-baseline-20260408T171454Z",
                        "config": {
                            "experiment": {
                                "name": "society-baseline",
                                "part": 2,
                                "repetitions": 1,
                                "prompt_variants": [{"name": "task-only"}],
                                "parameters": {"temperature": [0.3]},
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "provider_retry",
                        "timestamp": "2026-04-08T17:20:00+00:00",
                        "data": {"model": "deepseek-ai/deepseek-v3.2"},
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "timestamp": "2026-04-08T17:30:00+00:00",
                        "trial_id": 0,
                        "round_num": 30,
                        "data": {
                            "timestep": 30,
                            "alive_count": 10,
                            "total_agents": 24,
                            "public_food": 56,
                            "public_water": 117,
                            "average_health": 11.4,
                            "average_energy": 11.8,
                            "trade_volume": 0,
                            "agent_vitals": {
                                "agent-0": {"model": "llama3.1-8b", "alive": True},
                                "agent-1": {"model": "llama3.1-8b", "alive": True},
                                "agent-2": {"model": "deepseek-ai/deepseek-v3.2", "alive": True},
                                "agent-3": {"model": "moonshotai/kimi-k2-instruct-0905", "alive": False},
                            },
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = module.summarize_jsonl_log(
        log_path,
        stale_minutes=15.0,
        now=datetime(2026, 4, 8, 17, 35, 0, tzinfo=UTC),
    )

    assert summary is not None
    assert summary["track"] == "society"
    assert summary["prompt_variant"] == "task-only"
    assert summary["latest_round_num"] == 30
    assert summary["alive_count"] == 10
    assert summary["public_food"] == 56
    assert summary["alive_models"] == {
        "deepseek-ai/deepseek-v3.2": 1,
        "llama3.1-8b": 2,
    }
    assert summary["provider_retry_count"] == 1
    assert summary["last_retry_model"] == "deepseek-ai/deepseek-v3.2"
    assert summary["state"] == "active"
    assert summary["total_expected_trials"] == 1
    assert summary["completed_trials"] == 0
    assert summary["active_trials"] == 1
    assert summary["remaining_trials"] == 1
    assert summary["completion_fraction"] == 0.0
    assert summary["first_loss_round_num"] == 30
    assert summary["first_death_round_num"] is None
    assert summary["last_death_round_num"] is None
    assert summary["population_regime"] == "losses_observed"
    assert summary["collapse_start_round_num"] == 30
    assert summary["collapse_end_round_num"] is None
    assert summary["plateau_duration_rounds"] is None


def test_summarize_jsonl_log_estimates_trial_eta(tmp_path: Path):
    module = _load_live_run_status_module()
    log_path = tmp_path / "eta.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "experiment_start",
                        "timestamp": "2026-04-08T17:00:00+00:00",
                        "experiment_id": "society-eta",
                        "config": {
                            "experiment": {
                                "name": "society-eta",
                                "part": 2,
                                "rounds": 5,
                                "repetitions": 1,
                                "prompt_variants": [{"name": "task-only"}, {"name": "cooperative"}],
                                "parameters": {"temperature": [0.0]},
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "trial_summary",
                        "timestamp": "2026-04-08T17:00:30+00:00",
                        "trial_id": 0,
                        "summary": {"final_survival_rate": 0.5},
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "timestamp": "2026-04-08T17:01:00+00:00",
                        "trial_id": 1,
                        "round_num": 1,
                        "data": {"alive_count": 4, "total_agents": 4, "agent_vitals": {}},
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "timestamp": "2026-04-08T17:03:00+00:00",
                        "trial_id": 1,
                        "round_num": 3,
                        "data": {"alive_count": 4, "total_agents": 4, "agent_vitals": {}},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = module.summarize_jsonl_log(
        log_path,
        stale_minutes=15.0,
        now=datetime(2026, 4, 8, 17, 3, 30, tzinfo=UTC),
    )

    assert summary is not None
    assert summary["expected_rounds_per_trial"] == 5
    assert summary["remaining_rounds_in_trial"] == 2
    assert summary["observed_rounds_per_minute"] == pytest.approx(1.0)
    assert summary["estimated_minutes_remaining_in_trial"] == pytest.approx(2.0)
    assert summary["naive_minutes_remaining_in_baseline_suite"] == pytest.approx(2.0)
    assert summary["estimated_trial_completion_timestamp"] == "2026-04-08T17:05:00+00:00"


def test_summarize_jsonl_log_marks_stale_runs(tmp_path: Path):
    module = _load_live_run_status_module()
    log_path = tmp_path / "stale.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "experiment_start",
                        "timestamp": "2026-04-08T17:00:00+00:00",
                        "experiment_id": "exp-1",
                        "config": {"experiment": {"name": "exp-1", "part": 1}},
                    }
                ),
                json.dumps(
                    {
                        "type": "trial_summary",
                        "timestamp": "2026-04-08T17:05:00+00:00",
                        "trial_id": 0,
                        "summary": {"cooperation_rate_a": 0.5},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = module.summarize_jsonl_log(
        log_path,
        stale_minutes=10.0,
        now=datetime(2026, 4, 8, 17, 30, 0, tzinfo=UTC),
    )

    assert summary is not None
    assert summary["state"] == "stale"
    assert summary["trial_summary_count"] == 1
    assert summary["latest_event_type"] == "trial_summary"


def test_summarize_jsonl_log_uses_trial_summary_when_rounds_are_missing(tmp_path: Path):
    module = _load_live_run_status_module()
    log_path = tmp_path / "summary-only.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "experiment_start",
                        "timestamp": "2026-04-08T17:00:00+00:00",
                        "experiment_id": "exp-summary-only",
                        "config": {
                            "experiment": {
                                "name": "exp-summary-only",
                                "part": 2,
                                "repetitions": 1,
                                "prompt_variants": [{"name": "task-only"}],
                                "parameters": {"temperature": [0.3]},
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "trial_summary",
                        "timestamp": "2026-04-08T17:05:00+00:00",
                        "trial_id": 0,
                        "summary": {
                            "round_count": 120.0,
                            "final_survival_rate": 0.4166666667,
                            "final_alive_count": 10.0,
                            "final_total_agents": 24.0,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = module.summarize_jsonl_log(
        log_path,
        stale_minutes=15.0,
        now=datetime(2026, 4, 8, 17, 10, 0, tzinfo=UTC),
    )

    assert summary is not None
    assert summary["completed_trials"] == 1
    assert summary["active_trials"] == 0
    row = summary["trial_status_rows"][0]
    assert row["completed"] is True
    assert row["alive_count"] == 10.0
    assert row["total_agents"] == 24.0
    assert row["alive_fraction"] == pytest.approx(0.4166666667)
    assert row["latest_round_num"] == 120.0


def test_summarize_jsonl_log_detects_stable_post_collapse_plateau(tmp_path: Path):
    module = _load_live_run_status_module()
    log_path = tmp_path / "plateau.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "experiment_start",
                        "timestamp": "2026-04-08T17:00:00+00:00",
                        "experiment_id": "exp-plateau",
                        "config": {"experiment": {"name": "exp-plateau", "part": 2}},
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "timestamp": "2026-04-08T17:01:00+00:00",
                        "trial_id": 0,
                        "round_num": 1,
                        "data": {
                            "alive_count": 4,
                            "total_agents": 4,
                            "public_food": 10,
                            "public_water": 20,
                            "average_health": 8.0,
                            "average_energy": 7.0,
                            "trade_volume": 0,
                            "newly_dead": [],
                            "agent_vitals": {},
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "timestamp": "2026-04-08T17:02:00+00:00",
                        "trial_id": 0,
                        "round_num": 2,
                        "data": {
                            "alive_count": 2,
                            "total_agents": 4,
                            "public_food": 8,
                            "public_water": 18,
                            "average_health": 7.0,
                            "average_energy": 6.0,
                            "trade_volume": 0,
                            "newly_dead": ["c", "d"],
                            "agent_vitals": {},
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "timestamp": "2026-04-08T17:03:00+00:00",
                        "trial_id": 0,
                        "round_num": 3,
                        "data": {
                            "alive_count": 2,
                            "total_agents": 4,
                            "newly_dead": [],
                            "agent_vitals": {},
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "timestamp": "2026-04-08T17:04:00+00:00",
                        "trial_id": 0,
                        "round_num": 4,
                        "data": {
                            "alive_count": 2,
                            "total_agents": 4,
                            "newly_dead": [],
                            "agent_vitals": {},
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "timestamp": "2026-04-08T17:05:00+00:00",
                        "trial_id": 0,
                        "round_num": 5,
                        "data": {
                            "alive_count": 2,
                            "total_agents": 4,
                            "newly_dead": [],
                            "agent_vitals": {},
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "timestamp": "2026-04-08T17:06:00+00:00",
                        "trial_id": 0,
                        "round_num": 6,
                        "data": {
                            "alive_count": 2,
                            "total_agents": 4,
                            "newly_dead": [],
                            "agent_vitals": {},
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "timestamp": "2026-04-08T17:07:00+00:00",
                        "trial_id": 0,
                        "round_num": 7,
                        "data": {
                            "alive_count": 2,
                            "total_agents": 4,
                            "newly_dead": [],
                            "agent_vitals": {},
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = module.summarize_jsonl_log(
        log_path,
        stale_minutes=15.0,
        now=datetime(2026, 4, 8, 17, 7, 30, tzinfo=UTC),
    )

    assert summary is not None
    assert summary["first_loss_round_num"] == 2
    assert summary["first_death_round_num"] == 2
    assert summary["last_death_round_num"] == 2
    assert summary["stability_start_round_num"] == 2
    assert summary["rounds_since_last_death"] == 5
    assert summary["stabilized_post_collapse"] is True
    assert summary["population_regime"] == "stable_post_collapse"
    assert summary["collapse_start_round_num"] == 2
    assert summary["collapse_end_round_num"] == 2
    assert summary["collapse_duration_rounds"] == 1
    assert summary["collapse_death_count"] == 2
    assert summary["plateau_start_round_num"] == 2
    assert summary["plateau_end_round_num"] == 7
    assert summary["plateau_duration_rounds"] == 6


def test_summarize_jsonl_log_scopes_phase_diagnostics_to_latest_trial(tmp_path: Path):
    module = _load_live_run_status_module()
    log_path = tmp_path / "multi-trial.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "experiment_start",
                        "timestamp": "2026-04-08T17:00:00+00:00",
                        "experiment_id": "exp-multi",
                        "config": {
                            "experiment": {
                                "name": "exp-multi",
                                "part": 2,
                                "repetitions": 1,
                                "prompt_variants": [{"name": "task-only"}, {"name": "cooperative"}],
                                "parameters": {"temperature": [0.0]},
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "timestamp": "2026-04-08T17:01:00+00:00",
                        "trial_id": 0,
                        "round_num": 1,
                        "data": {
                            "alive_count": 4,
                            "total_agents": 4,
                            "public_food": 10,
                            "public_water": 20,
                            "average_health": 8.0,
                            "average_energy": 7.0,
                            "trade_volume": 0,
                            "newly_dead": [],
                            "agent_vitals": {},
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "timestamp": "2026-04-08T17:02:00+00:00",
                        "trial_id": 0,
                        "round_num": 2,
                        "data": {
                            "alive_count": 2,
                            "total_agents": 4,
                            "public_food": 8,
                            "public_water": 18,
                            "average_health": 7.0,
                            "average_energy": 6.0,
                            "trade_volume": 0,
                            "newly_dead": ["c", "d"],
                            "agent_vitals": {},
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "trial_summary",
                        "timestamp": "2026-04-08T17:03:00+00:00",
                        "trial_id": 0,
                        "summary": {"final_survival_rate": 0.5},
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "timestamp": "2026-04-08T17:04:00+00:00",
                        "trial_id": 1,
                        "round_num": 1,
                        "data": {
                            "alive_count": 4,
                            "total_agents": 4,
                            "public_food": 12,
                            "public_water": 22,
                            "average_health": 9.0,
                            "average_energy": 8.0,
                            "trade_volume": 1,
                            "newly_dead": [],
                            "agent_vitals": {},
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "timestamp": "2026-04-08T17:05:00+00:00",
                        "trial_id": 1,
                        "round_num": 2,
                        "data": {
                            "alive_count": 4,
                            "total_agents": 4,
                            "public_food": 13,
                            "public_water": 23,
                            "average_health": 9.5,
                            "average_energy": 8.5,
                            "trade_volume": 2,
                            "newly_dead": [],
                            "agent_vitals": {},
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = module.summarize_jsonl_log(
        log_path,
        stale_minutes=15.0,
        now=datetime(2026, 4, 8, 17, 6, 0, tzinfo=UTC),
    )

    assert summary is not None
    assert summary["latest_trial_id"] == 1
    assert summary["prompt_variant"] == "cooperative"
    assert summary["trial_summary_count"] == 1
    assert summary["total_expected_trials"] == 2
    assert summary["completed_trials"] == 1
    assert summary["active_trials"] == 1
    assert summary["remaining_trials"] == 1
    assert summary["completion_fraction"] == 0.5
    assert summary["first_loss_round_num"] is None
    assert summary["last_death_round_num"] is None
    assert summary["population_regime"] == "no_losses_yet"
    assert summary["collapse_start_round_num"] is None
    assert summary["plateau_duration_rounds"] is None
    assert len(summary["trial_status_rows"]) == 2
    assert summary["trial_status_rows"][0]["trial_id"] == 0
    assert summary["trial_status_rows"][0]["completed"] is True
    assert summary["trial_status_rows"][0]["prompt_variant"] == "task-only"
    assert summary["trial_status_rows"][0]["public_food"] == 8
    assert summary["trial_status_rows"][1]["trial_id"] == 1
    assert summary["trial_status_rows"][1]["completed"] is False
    assert summary["trial_status_rows"][1]["prompt_variant"] == "cooperative"
    assert summary["trial_status_rows"][1]["population_regime"] == "no_losses_yet"
    assert summary["trial_status_rows"][1]["public_food"] == 13
    assert summary["trial_status_rows"][1]["trade_volume"] == 2


def test_expand_inputs_reads_jsonl_files_from_directories(tmp_path: Path):
    module = _load_live_run_status_module()
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")

    paths = module.expand_inputs([str(tmp_path)])

    assert paths == [first, second]
