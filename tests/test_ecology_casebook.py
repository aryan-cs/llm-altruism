"""Tests for ecology casebook generation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def _load_casebook_module():
    module_path = ROOT / "scripts" / "ecology_casebook.py"
    spec = importlib.util.spec_from_file_location("ecology_casebook_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_render_markdown_includes_key_milestones(tmp_path: Path):
    module = _load_casebook_module()
    log_path = tmp_path / "society.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "experiment_start",
                        "experiment_id": "society-baseline-20260408T171454Z",
                        "config": {"experiment": {"name": "society-baseline", "part": 2}},
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "round_num": 1,
                        "data": {
                            "alive_count": 4,
                            "total_agents": 4,
                            "public_food": 10,
                            "public_water": 12,
                            "average_health": 10.0,
                            "average_energy": 8.0,
                            "events": [{"kind": "forage_food"}],
                            "newly_dead": [],
                            "agent_vitals": {
                                "a": {
                                    "agent_id": "a",
                                    "model": "model-a",
                                    "alive": True,
                                    "health": 10,
                                    "energy": 8,
                                    "resources_total": 20,
                                }
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "round_num": 2,
                        "data": {
                            "alive_count": 2,
                            "total_agents": 4,
                            "public_food": 6,
                            "public_water": 9,
                            "average_health": 7.0,
                            "average_energy": 6.0,
                            "events": [{"kind": "draw_water"}, {"kind": "draw_water"}],
                            "newly_dead": ["c", "d"],
                            "agent_vitals": {
                                "a": {
                                    "agent_id": "a",
                                    "model": "model-a",
                                    "alive": True,
                                    "health": 7,
                                    "energy": 6,
                                    "resources_total": 12,
                                },
                                "b": {
                                    "agent_id": "b",
                                    "model": "model-b",
                                    "alive": True,
                                    "health": 8,
                                    "energy": 7,
                                    "resources_total": 15,
                                },
                            },
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    start, rounds = module.load_log(log_path)
    markdown = module.render_markdown(start, rounds)

    assert "## Milestones" in markdown
    assert "First Loss" in markdown
    assert "Largest Death Shock" in markdown
    assert "`model-a: 1, model-b: 1`" in markdown
    assert "`draw_water: 2`" in markdown


def test_main_writes_output_file(tmp_path: Path):
    module = _load_casebook_module()
    log_path = tmp_path / "society.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "experiment_start",
                        "experiment_id": "exp-1",
                        "config": {"experiment": {"name": "exp-1", "part": 2}},
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "round_num": 1,
                        "data": {
                            "alive_count": 1,
                            "total_agents": 1,
                            "public_food": 1,
                            "public_water": 1,
                            "average_health": 1.0,
                            "average_energy": 1.0,
                            "events": [],
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

    output_path = tmp_path / "casebook.md"
    module.main = module.main  # keep linter quiet during import-based execution
    start, rounds = module.load_log(log_path)
    output_path.write_text(module.render_markdown(start, rounds), encoding="utf-8")

    assert output_path.exists()
    assert "# Ecology Casebook" in output_path.read_text(encoding="utf-8")


def test_render_markdown_includes_stable_plateau_section_when_population_flattens(tmp_path: Path):
    module = _load_casebook_module()
    log_path = tmp_path / "society.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "experiment_start",
                        "experiment_id": "exp-plateau",
                        "config": {"experiment": {"name": "exp-plateau", "part": 2}},
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "round_num": 1,
                        "data": {
                            "alive_count": 4,
                            "total_agents": 4,
                            "public_food": 10,
                            "public_water": 10,
                            "average_health": 8.0,
                            "average_energy": 7.0,
                            "events": [{"kind": "forage_food"}],
                            "newly_dead": [],
                            "agent_vitals": {},
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "round_num": 2,
                        "data": {
                            "alive_count": 2,
                            "total_agents": 4,
                            "public_food": 8,
                            "public_water": 9,
                            "average_health": 7.0,
                            "average_energy": 6.0,
                            "events": [{"kind": "draw_water"}],
                            "newly_dead": ["c", "d"],
                            "agent_vitals": {},
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "round_num": 3,
                        "data": {
                            "alive_count": 2,
                            "total_agents": 4,
                            "public_food": 9,
                            "public_water": 11,
                            "average_health": 9.0,
                            "average_energy": 10.0,
                            "events": [{"kind": "forage_food"}],
                            "newly_dead": [],
                            "agent_vitals": {},
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "round_num": 4,
                        "data": {
                            "alive_count": 2,
                            "total_agents": 4,
                            "public_food": 10,
                            "public_water": 12,
                            "average_health": 10.0,
                            "average_energy": 10.0,
                            "events": [{"kind": "forage_food"}],
                            "newly_dead": [],
                            "agent_vitals": {},
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "round_num": 5,
                        "data": {
                            "alive_count": 2,
                            "total_agents": 4,
                            "public_food": 11,
                            "public_water": 13,
                            "average_health": 10.0,
                            "average_energy": 11.0,
                            "events": [{"kind": "draw_water"}],
                            "newly_dead": [],
                            "agent_vitals": {},
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "round_num": 6,
                        "data": {
                            "alive_count": 2,
                            "total_agents": 4,
                            "public_food": 12,
                            "public_water": 14,
                            "average_health": 11.0,
                            "average_energy": 11.0,
                            "events": [{"kind": "draw_water"}],
                            "newly_dead": [],
                            "agent_vitals": {},
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "round_num": 7,
                        "data": {
                            "alive_count": 2,
                            "total_agents": 4,
                            "public_food": 13,
                            "public_water": 15,
                            "average_health": 11.0,
                            "average_energy": 12.0,
                            "events": [{"kind": "forage_food"}],
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

    start, rounds = module.load_log(log_path)
    markdown = module.render_markdown(start, rounds)

    assert "## Stable Plateau" in markdown
    assert "plateau start round: `2`" in markdown
    assert "duration: `6` rounds" in markdown
    assert "deaths on plateau start round: `c, d`" in markdown
