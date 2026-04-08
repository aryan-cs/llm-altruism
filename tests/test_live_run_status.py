"""Tests for live JSONL run status reporting."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

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


def test_expand_inputs_reads_jsonl_files_from_directories(tmp_path: Path):
    module = _load_live_run_status_module()
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")

    paths = module.expand_inputs([str(tmp_path)])

    assert paths == [first, second]
