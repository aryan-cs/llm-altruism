"""Tests for live ecology packet refresh orchestration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def _load_refresh_module():
    module_path = ROOT / "scripts" / "refresh_live_ecology_packet.py"
    spec = importlib.util.spec_from_file_location("refresh_live_ecology_packet_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_log_path_picks_newest_jsonl(tmp_path: Path):
    module = _load_refresh_module()
    older = tmp_path / "older.jsonl"
    newer = tmp_path / "newer.jsonl"
    older.write_text("", encoding="utf-8")
    newer.write_text("", encoding="utf-8")
    older.touch()
    newer.touch()

    resolved = module.resolve_log_path(tmp_path)

    assert resolved == newer


def test_packet_paths_use_logical_casebook_name(tmp_path: Path):
    module = _load_refresh_module()
    log_path = tmp_path / "society-baseline-20260408T171454Z.jsonl"

    outputs = module.packet_paths(tmp_path, log_path)

    assert outputs["summary_markdown"] == tmp_path / "interim_summary.md"
    assert outputs["summary_csv"] == tmp_path / "interim_summary.csv"
    assert outputs["figures_dir"] == tmp_path / "monitoring_figures"
    assert outputs["casebook_markdown"] == tmp_path / "society-baseline-casebook.md"
    assert outputs["status_json"] == tmp_path / "live_status.json"
    assert outputs["trial_snapshot_markdown"] == tmp_path / "live_trial_snapshot.md"
    assert outputs["trial_snapshot_csv"] == tmp_path / "live_trial_snapshot.csv"
    assert outputs["trial_snapshot_figure"] == tmp_path / "live_trial_snapshot.png"


def test_write_trial_snapshot_emits_markdown_and_csv(tmp_path: Path):
    module = _load_refresh_module()
    outputs = {
        "trial_snapshot_markdown": tmp_path / "live_trial_snapshot.md",
        "trial_snapshot_csv": tmp_path / "live_trial_snapshot.csv",
    }
    live_status = {
        "trial_status_rows": [
            {
                "trial_id": 0,
                "prompt_variant": "task-only",
                "completed": True,
                "latest_round_num": 120,
                "alive_count": 10,
                "total_agents": 24,
                "population_regime": "stable_post_collapse",
                "last_death_round_num": 26,
                "plateau_duration_rounds": 95,
                "final_survival_rate": 0.4167,
                "alive_models": {"deepseek-ai/deepseek-v3.2": 8, "llama3.1-8b": 2},
            },
            {
                "trial_id": 1,
                "prompt_variant": "cooperative",
                "completed": False,
                "latest_round_num": 32,
                "alive_count": 18,
                "total_agents": 24,
                "population_regime": "stable_post_collapse",
                "last_death_round_num": 11,
                "plateau_duration_rounds": 22,
                "final_survival_rate": None,
                "alive_models": {
                    "deepseek-ai/deepseek-v3.2": 8,
                    "moonshotai/kimi-k2-instruct-0905": 8,
                    "llama3.1-8b": 2,
                },
            },
        ]
    }

    module.write_trial_snapshot(outputs, live_status)

    markdown = outputs["trial_snapshot_markdown"].read_text(encoding="utf-8")
    csv_text = outputs["trial_snapshot_csv"].read_text(encoding="utf-8")
    assert "# Live Trial Snapshot" in markdown
    assert "| 0 | task-only | completed | 120 | 10/24 | stable_post_collapse | 26 | 95 | 0.4167 |" in markdown
    assert "| 1 | cooperative | active | 32 | 18/24 | stable_post_collapse | 11 | 22 |  |" in markdown
    assert "trial_id,prompt_variant,repetition,completed,latest_round_num" in csv_text
    assert "0,task-only,,True,120,10,24" in csv_text


def test_write_trial_snapshot_figure_emits_png(tmp_path: Path):
    module = _load_refresh_module()
    outputs = {
        "trial_snapshot_figure": tmp_path / "live_trial_snapshot.png",
    }
    live_status = {
        "trial_status_rows": [
            {
                "trial_id": 0,
                "prompt_variant": "task-only",
                "completed": True,
                "alive_count": 10,
                "total_agents": 24,
                "alive_fraction": 10 / 24,
                "plateau_duration_rounds": 95,
                "last_death_round_num": 26,
            },
            {
                "trial_id": 1,
                "prompt_variant": "cooperative",
                "completed": False,
                "alive_count": 18,
                "total_agents": 24,
                "alive_fraction": 18 / 24,
                "plateau_duration_rounds": 22,
                "last_death_round_num": 11,
            },
        ]
    }

    module.write_trial_snapshot_figure(outputs, live_status)

    assert outputs["trial_snapshot_figure"].exists()
    assert outputs["trial_snapshot_figure"].stat().st_size > 0
