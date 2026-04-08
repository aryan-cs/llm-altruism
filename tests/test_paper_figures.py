"""Tests for paper figure generation helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_paper_figures_module():
    """Import the paper figures CLI module for helper testing."""
    module_path = ROOT / "scripts" / "paper_figures.py"
    spec = importlib.util.spec_from_file_location("paper_figures_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_side_average_metric_frame_averages_agent_metrics():
    """Plotting helpers should collapse A/B metrics into a single mean column."""
    module = _load_paper_figures_module()
    frame = pd.DataFrame(
        [
            {
                "track": "baseline",
                "game": "prisoners_dilemma",
                "prompt_variant": "minimal-neutral",
                "cooperation_rate_a": 0.25,
                "cooperation_rate_b": 0.75,
                "cooperation_rate_a_ci95_low": 0.0,
                "cooperation_rate_b_ci95_low": 0.5,
                "cooperation_rate_a_ci95_high": 0.5,
                "cooperation_rate_b_ci95_high": 1.0,
            }
        ]
    )

    averaged = module.side_average_metric_frame(frame, "cooperation_rate")

    assert averaged.loc[0, "cooperation_rate"] == 0.5
    assert averaged.loc[0, "cooperation_rate_ci95_low"] == 0.25
    assert averaged.loc[0, "cooperation_rate_ci95_high"] == 0.75


def test_save_prompt_variant_track_figure_writes_png(tmp_path: Path):
    """The prompt-variant figure helper should emit a non-empty PNG."""
    module = _load_paper_figures_module()
    frame = pd.DataFrame(
        [
            {
                "track": "baseline",
                "game": "prisoners_dilemma",
                "prompt_variant": "minimal-neutral",
                "cooperation_rate_a": 0.25,
                "cooperation_rate_b": 0.30,
                "cooperation_rate_a_ci95_low": 0.10,
                "cooperation_rate_b_ci95_low": 0.15,
                "cooperation_rate_a_ci95_high": 0.40,
                "cooperation_rate_b_ci95_high": 0.45,
            },
            {
                "track": "baseline",
                "game": "prisoners_dilemma",
                "prompt_variant": "minimal-neutral-compact",
                "cooperation_rate_a": 0.60,
                "cooperation_rate_b": 0.65,
                "cooperation_rate_a_ci95_low": 0.40,
                "cooperation_rate_b_ci95_low": 0.45,
                "cooperation_rate_a_ci95_high": 0.80,
                "cooperation_rate_b_ci95_high": 0.85,
            },
        ]
    )
    output_path = tmp_path / "baseline.png"

    written = module.save_prompt_variant_track_figure(
        frame,
        track="baseline",
        metric_root="cooperation_rate",
        output_path=output_path,
        title="Neutral baseline prompt variants",
        y_label="Mean cooperation",
        limit=(0, 1),
    )

    assert written == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_society_metric_figure_writes_png(tmp_path: Path):
    """The society/reputation metric helper should emit a non-empty PNG."""
    module = _load_paper_figures_module()
    frame = pd.DataFrame(
        [
            {
                "track": "society",
                "prompt_variant": "task-only",
                "average_trade_volume": 0.0,
                "average_trade_volume_ci95_low": 0.0,
                "average_trade_volume_ci95_high": 0.0,
            },
            {
                "track": "society",
                "prompt_variant": "cooperative",
                "average_trade_volume": 2.5,
                "average_trade_volume_ci95_low": 2.0,
                "average_trade_volume_ci95_high": 3.0,
            },
            {
                "track": "reputation",
                "prompt_variant": "task-only",
                "average_trade_volume": 0.0,
                "average_trade_volume_ci95_low": 0.0,
                "average_trade_volume_ci95_high": 0.0,
            },
            {
                "track": "reputation",
                "prompt_variant": "cooperative",
                "average_trade_volume": 3.0,
                "average_trade_volume_ci95_low": 2.5,
                "average_trade_volume_ci95_high": 3.5,
            },
        ]
    )
    output_path = tmp_path / "society_trade.png"

    written = module.save_society_metric_figure(
        frame,
        metric_root="average_trade_volume",
        output_path=output_path,
        title="Scarcity and reputation trade volume by prompt condition",
        y_label="Average trade volume",
    )

    assert written == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_flatten_society_round_rows_extracts_round_metrics_and_events():
    """Society timeline helpers should flatten raw rounds and logged event kinds."""
    module = _load_paper_figures_module()
    summaries = [
        {
            "experiment_id": "paper-society-prompts-20260408T005441Z",
            "config": {"experiment": {"name": "paper-society-prompts"}},
            "run_metadata": {"track": "society"},
            "trials": [
                {
                    "trial_id": 0,
                    "prompt_variant": "task-only",
                    "repetition": 0,
                    "rounds": [
                        {
                            "timestep": 1,
                            "alive_count": 6,
                            "total_agents": 6,
                            "public_food": 10,
                            "public_water": 14,
                            "public_resources": 24,
                            "trade_volume": 0,
                            "average_health": 9.5,
                            "average_energy": 8.0,
                            "spawned_agents": [],
                            "newly_dead": ["agent-5"],
                            "events": [
                                {"kind": "gather", "actor": "agent-0", "target": None, "amount": 1},
                                {"kind": "share", "actor": "agent-1", "target": "agent-2", "amount": 2},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    round_frame, event_frame = module.flatten_society_round_rows(summaries)

    assert len(round_frame) == 1
    assert round_frame.iloc[0]["track"] == "society"
    assert round_frame.iloc[0]["birth_count"] == 0
    assert round_frame.iloc[0]["death_count"] == 1
    assert round_frame.iloc[0]["event_count"] == 2
    assert round_frame.iloc[0]["public_food"] == 10
    assert round_frame.iloc[0]["public_water"] == 14
    assert round_frame.iloc[0]["average_health"] == 9.5
    assert round_frame.iloc[0]["average_energy"] == 8.0
    assert len(event_frame) == 2
    assert set(event_frame["event_kind"].tolist()) == {"gather", "share"}


def test_flatten_society_round_rows_infers_track_from_part_when_run_metadata_missing():
    """Standalone society runs should still feed the timeline figures."""
    module = _load_paper_figures_module()
    summaries = [
        {
            "experiment_id": "society-baseline-20260408T171454Z",
            "config": {"experiment": {"name": "society-baseline", "part": 2}},
            "trials": [
                {
                    "trial_id": 0,
                    "prompt_variant": "task-only",
                    "repetition": 0,
                    "rounds": [
                        {
                            "timestep": 1,
                            "alive_count": 24,
                            "total_agents": 24,
                            "public_food": 67,
                            "public_water": 166,
                            "public_resources": 233,
                            "trade_volume": 0,
                            "average_health": 11.0,
                            "average_energy": 9.0,
                            "spawned_agents": [],
                            "newly_dead": [],
                            "events": [],
                        }
                    ],
                }
            ],
        }
    ]

    round_frame, event_frame = module.flatten_society_round_rows(summaries)

    assert len(round_frame) == 1
    assert round_frame.iloc[0]["track"] == "society"
    assert round_frame.iloc[0]["alive_count"] == 24
    assert event_frame.empty


def test_flatten_society_model_rows_extracts_alive_counts_by_model():
    """Model-level timeline helpers should recover alive-population splits from vitals."""
    module = _load_paper_figures_module()
    summaries = [
        {
            "experiment_id": "society-baseline-20260408T171454Z",
            "config": {"experiment": {"name": "society-baseline", "part": 2}},
            "trials": [
                {
                    "trial_id": 0,
                    "prompt_variant": "task-only",
                    "repetition": 0,
                    "rounds": [
                        {
                            "timestep": 1,
                            "agent_vitals": {
                                "agent-0": {"model": "model-a", "alive": True},
                                "agent-1": {"model": "model-a", "alive": False},
                                "agent-2": {"model": "model-b", "alive": True},
                            },
                        }
                    ],
                }
            ],
        }
    ]

    model_frame = module.flatten_society_model_rows(summaries)

    assert len(model_frame) == 2
    assert set(model_frame["model"].tolist()) == {"model-a", "model-b"}
    assert sorted(model_frame["alive_count_model"].tolist()) == [1, 1]


def test_flatten_society_agent_vital_rows_extracts_latest_survivor_vitals():
    """Agent-level vital extraction should preserve per-agent food, water, energy, and health."""
    module = _load_paper_figures_module()
    summaries = [
        {
            "experiment_id": "society-baseline-20260408T171454Z",
            "config": {"experiment": {"name": "society-baseline", "part": 2}},
            "trials": [
                {
                    "trial_id": 0,
                    "prompt_variant": "task-only",
                    "repetition": 0,
                    "rounds": [
                        {
                            "timestep": 1,
                            "agent_vitals": {
                                "agent-0": {
                                    "agent_id": "agent-0",
                                    "model": "model-a",
                                    "alive": True,
                                    "food": 3,
                                    "water": 2,
                                    "energy": 8,
                                    "health": 9,
                                    "resources_total": 22,
                                },
                                "agent-1": {
                                    "agent_id": "agent-1",
                                    "model": "model-b",
                                    "alive": False,
                                    "food": 0,
                                    "water": 0,
                                    "energy": 0,
                                    "health": 0,
                                    "resources_total": 0,
                                },
                            },
                        }
                    ],
                }
            ],
        }
    ]

    vital_frame = module.flatten_society_agent_vital_rows(summaries)

    assert len(vital_frame) == 1
    assert vital_frame.iloc[0]["agent_id"] == "agent-0"
    assert vital_frame.iloc[0]["food"] == 3
    assert vital_frame.iloc[0]["water"] == 2
    assert vital_frame.iloc[0]["energy"] == 8
    assert vital_frame.iloc[0]["health"] == 9


def test_save_society_timeline_figure_writes_png(tmp_path: Path):
    """Timeline figure helper should emit a non-empty PNG from per-round rows."""
    module = _load_paper_figures_module()
    round_frame = pd.DataFrame(
        [
            {"track": "society", "prompt_variant": "task-only", "timestep": 1, "alive_count": 6},
            {"track": "society", "prompt_variant": "task-only", "timestep": 2, "alive_count": 6},
            {"track": "society", "prompt_variant": "cooperative", "timestep": 1, "alive_count": 6},
            {"track": "society", "prompt_variant": "cooperative", "timestep": 2, "alive_count": 5},
            {"track": "reputation", "prompt_variant": "task-only", "timestep": 1, "alive_count": 6},
            {"track": "reputation", "prompt_variant": "task-only", "timestep": 2, "alive_count": 6},
            {"track": "reputation", "prompt_variant": "cooperative", "timestep": 1, "alive_count": 6},
            {"track": "reputation", "prompt_variant": "cooperative", "timestep": 2, "alive_count": 6},
        ]
    )
    output_path = tmp_path / "population.png"

    written = module.save_society_timeline_figure(
        round_frame,
        metric_root="alive_count",
        output_path=output_path,
        title="Population over time",
        y_label="Alive agents",
    )

    assert written == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_society_model_population_figure_writes_png(tmp_path: Path):
    """Model-level population trajectories should render to a non-empty PNG."""
    module = _load_paper_figures_module()
    model_frame = pd.DataFrame(
        [
            {
                "track": "society",
                "prompt_variant": "task-only",
                "model": "model-a",
                "timestep": 1,
                "alive_count_model": 4,
            },
            {
                "track": "society",
                "prompt_variant": "task-only",
                "model": "model-a",
                "timestep": 2,
                "alive_count_model": 3,
            },
            {
                "track": "society",
                "prompt_variant": "task-only",
                "model": "model-b",
                "timestep": 1,
                "alive_count_model": 2,
            },
            {
                "track": "society",
                "prompt_variant": "task-only",
                "model": "model-b",
                "timestep": 2,
                "alive_count_model": 1,
            },
        ]
    )
    output_path = tmp_path / "population_by_model.png"

    written = module.save_society_model_population_figure(
        model_frame,
        output_path=output_path,
        title="Population trajectories by model",
    )

    assert written == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_society_vitals_heatmap_writes_png(tmp_path: Path):
    """Latest survivor vitals should render to a non-empty PNG."""
    module = _load_paper_figures_module()
    agent_frame = pd.DataFrame(
        [
            {
                "track": "society",
                "prompt_variant": "task-only",
                "trial_id": 0,
                "timestep": 5,
                "agent_id": "agent-0",
                "model": "model-a",
                "food": 5,
                "water": 3,
                "energy": 10,
                "health": 11,
            },
            {
                "track": "society",
                "prompt_variant": "task-only",
                "trial_id": 0,
                "timestep": 5,
                "agent_id": "agent-1",
                "model": "model-b",
                "food": 2,
                "water": 4,
                "energy": 9,
                "health": 10,
            },
        ]
    )
    output_path = tmp_path / "survivor_vitals.png"

    written = module.save_society_vitals_heatmap(
        agent_frame,
        output_path=output_path,
        title="Latest survivor vitals",
    )

    assert written == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_summarize_model_vital_frame_aggregates_agent_vitals_by_model():
    """Model-level vital summaries should average per-agent values by timestep."""
    module = _load_paper_figures_module()
    agent_frame = pd.DataFrame(
        [
            {
                "track": "society",
                "prompt_variant": "task-only",
                "model": "model-a",
                "timestep": 1,
                "food": 4,
            },
            {
                "track": "society",
                "prompt_variant": "task-only",
                "model": "model-a",
                "timestep": 1,
                "food": 6,
            },
            {
                "track": "society",
                "prompt_variant": "task-only",
                "model": "model-b",
                "timestep": 1,
                "food": 3,
            },
        ]
    )

    summary = module.summarize_model_vital_frame(agent_frame, metric_root="food")

    model_a = summary[summary["model"] == "model-a"].iloc[0]
    assert model_a["food"] == 5.0
    assert "food_ci95_low" in model_a.index
    assert "food_ci95_high" in model_a.index


def test_save_society_model_vitals_timeline_figure_writes_png(tmp_path: Path):
    """Model-level vital trajectories should render to a non-empty PNG."""
    module = _load_paper_figures_module()
    agent_frame = pd.DataFrame(
        [
            {
                "track": "society",
                "prompt_variant": "task-only",
                "trial_id": 0,
                "timestep": 1,
                "agent_id": "agent-0",
                "model": "model-a",
                "food": 4,
                "water": 3,
                "energy": 8,
                "health": 9,
            },
            {
                "track": "society",
                "prompt_variant": "task-only",
                "trial_id": 0,
                "timestep": 2,
                "agent_id": "agent-0",
                "model": "model-a",
                "food": 5,
                "water": 4,
                "energy": 9,
                "health": 10,
            },
            {
                "track": "society",
                "prompt_variant": "task-only",
                "trial_id": 0,
                "timestep": 1,
                "agent_id": "agent-1",
                "model": "model-b",
                "food": 2,
                "water": 5,
                "energy": 7,
                "health": 8,
            },
            {
                "track": "society",
                "prompt_variant": "task-only",
                "trial_id": 0,
                "timestep": 2,
                "agent_id": "agent-1",
                "model": "model-b",
                "food": 3,
                "water": 5,
                "energy": 8,
                "health": 9,
            },
        ]
    )
    output_path = tmp_path / "model_vitals_over_time.png"

    written = module.save_society_model_vitals_timeline_figure(
        agent_frame,
        output_path=output_path,
        title="Model-level vital trajectories",
    )

    assert written == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_society_event_mix_figure_writes_png(tmp_path: Path):
    """Behavior-mix helper should emit a non-empty PNG from logged event kinds."""
    module = _load_paper_figures_module()
    event_frame = pd.DataFrame(
        [
            {"track": "society", "prompt_variant": "task-only", "event_kind": "gather"},
            {"track": "society", "prompt_variant": "task-only", "event_kind": "gather"},
            {"track": "society", "prompt_variant": "cooperative", "event_kind": "share"},
            {"track": "society", "prompt_variant": "cooperative", "event_kind": "broadcast"},
            {"track": "reputation", "prompt_variant": "task-only", "event_kind": "gather"},
            {"track": "reputation", "prompt_variant": "competitive", "event_kind": "steal"},
        ]
    )
    output_path = tmp_path / "behavior_mix.png"

    written = module.save_society_event_mix_figure(
        event_frame,
        output_path=output_path,
        title="Behavior mix",
    )

    assert written == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_event_category_handles_new_ecology_actions():
    """Behavior-mix categories should reflect the richer ecology action set."""
    module = _load_paper_figures_module()

    assert module.event_category("forage_food") == "gather"
    assert module.event_category("draw_water") == "gather"
    assert module.event_category("offer_trade") == "share"
    assert module.event_category("sleep") == "rest"
    assert module.event_category("reproduce") == "reproduce"


def test_build_frames_collapses_stale_retries_by_experiment_name(tmp_path: Path):
    """Figure inputs should inherit summary-level retry de-duplication by experiment name."""
    module = _load_paper_figures_module()
    experiment_name = "paper-benchmark-stag_hunt-unnamed"
    completed_id = f"{experiment_name}-20260407T183502Z"
    partial_retry_id = f"{experiment_name}-20260407T184144Z"

    (tmp_path / f"{completed_id}.json").write_text(
        json.dumps(
            {
                "experiment_id": completed_id,
                "config": {
                    "experiment": {
                        "name": experiment_name,
                        "part": 1,
                        "game": "stag_hunt",
                    },
                    "run_metadata": {
                        "track": "benchmark",
                        "presentation": "unnamed",
                    },
                },
                "trials": [
                    {
                        "trial_id": 0,
                        "prompt_variant": "minimal-neutral",
                        "repetition": 0,
                        "summary": {
                            "cooperation_rate_a": 0.5,
                            "cooperation_rate_b": 0.5,
                        },
                    }
                ],
                "aggregate_summary": {
                    "cooperation_rate_a": 0.5,
                    "cooperation_rate_b": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )

    (tmp_path / f"{partial_retry_id}.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "experiment_start",
                        "experiment_id": partial_retry_id,
                        "config": {
                            "experiment": {
                                "name": experiment_name,
                                "part": 1,
                                "game": "stag_hunt",
                                "repetitions": 1,
                                "prompt_variants": [{"name": "minimal-neutral"}],
                                "parameters": {"temperature": [0.0]},
                            },
                            "run_metadata": {
                                "track": "benchmark",
                                "presentation": "unnamed",
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "trial_summary",
                        "trial_id": 0,
                        "summary": {
                            "cooperation_rate_a": 1.0,
                            "cooperation_rate_b": 1.0,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    experiment_frame, prompt_variant_frame, pooled_prompt_frame = module.build_frames([str(tmp_path)])

    assert len(experiment_frame) == 1
    assert experiment_frame.iloc[0]["experiment_id"] == completed_id
    assert len(prompt_variant_frame) == 1
    assert prompt_variant_frame.iloc[0]["experiment_id"] == completed_id
    assert len(pooled_prompt_frame) == 1
