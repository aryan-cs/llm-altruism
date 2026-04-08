"""Tests for paper summary aggregation helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_paper_summary_module():
    """Import the paper summary CLI module for helper testing."""
    module_path = ROOT / "scripts" / "paper_summary.py"
    spec = importlib.util.spec_from_file_location("paper_summary_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pooled_prompt_variant_frame_aggregates_raw_trial_rows():
    """Pooled prompt summaries should combine repeated trials across experiments."""
    module = _load_paper_summary_module()
    frame = pd.DataFrame(
        [
            {
                "experiment_id": "exp-1",
                "track": "society",
                "presentation": "prompt_comparison",
                "game": None,
                "prompt_variant": "task-only",
                "trial_id": 0,
                "survival_rate": 1.0,
                "final_survival_rate": 1.0,
            },
            {
                "experiment_id": "exp-2",
                "track": "society",
                "presentation": "prompt_comparison",
                "game": None,
                "prompt_variant": "task-only",
                "trial_id": 1,
                "survival_rate": 0.75,
                "final_survival_rate": 0.5,
            },
            {
                "experiment_id": "exp-2",
                "track": "society",
                "presentation": "prompt_comparison",
                "game": None,
                "prompt_variant": "competitive",
                "trial_id": 2,
                "survival_rate": 0.9,
                "final_survival_rate": 0.75,
            },
        ]
    )

    pooled = module.pooled_prompt_variant_frame(frame)

    task_only = pooled[pooled["prompt_variant"] == "task-only"].iloc[0]
    assert task_only["experiment_count"] == 2
    assert task_only["trial_count"] == 2
    assert task_only["survival_rate"] == 0.875
    assert task_only["final_survival_rate"] == 0.75
    assert "survival_rate_ci95_low" in task_only.index
    assert "survival_rate_ci95_high" in task_only.index


def test_bootstrap_mean_ci_collapses_to_point_for_single_observation():
    """A single observation should yield a degenerate confidence interval."""
    module = _load_paper_summary_module()

    mean, ci_low, ci_high = module.bootstrap_mean_ci([0.875])

    assert mean == 0.875
    assert ci_low == 0.875
    assert ci_high == 0.875


def test_render_markdown_includes_pooled_prompt_section():
    """Markdown output should include the pooled summary section when available."""
    module = _load_paper_summary_module()
    experiment_frame = pd.DataFrame([{"experiment_id": "exp-1", "track": "baseline"}])
    prompt_variant_frame = pd.DataFrame()
    pooled_prompt_frame = pd.DataFrame(
        [
            {
                "track": "society",
                "presentation": "prompt_comparison",
                "game": None,
                "prompt_variant": "task-only",
                "experiment_count": 2,
                "trial_count": 4,
                "survival_rate": 1.0,
                "final_survival_rate": 0.875,
                "survival_rate_ci95_low": 0.75,
                "survival_rate_ci95_high": 1.0,
                "final_survival_rate_ci95_low": 0.5,
                "final_survival_rate_ci95_high": 1.0,
            }
        ]
    )

    markdown = module.render_markdown(experiment_frame, prompt_variant_frame, pooled_prompt_frame)

    assert "## Pooled Prompt Variant Summary" in markdown
    assert "task-only" in markdown
    assert "1.0000 [0.7500, 1.0000]" in markdown


def test_ecology_diagnostic_rows_summarize_latest_society_state():
    """Ecology diagnostics should expose collapse timing and live model composition."""
    module = _load_paper_summary_module()
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
                            "alive_count": 4,
                            "total_agents": 4,
                            "public_food": 20,
                            "public_water": 30,
                            "average_health": 12.0,
                            "average_energy": 10.0,
                            "events": [{"kind": "forage_food"}],
                            "spawned_agents": [],
                            "newly_dead": [],
                            "agent_vitals": {
                                "agent-0": {"model": "model-a", "alive": True},
                                "agent-1": {"model": "model-a", "alive": True},
                                "agent-2": {"model": "model-b", "alive": True},
                                "agent-3": {"model": "model-c", "alive": True},
                            },
                        },
                        {
                            "timestep": 2,
                            "alive_count": 2,
                            "total_agents": 4,
                            "public_food": 16,
                            "public_water": 24,
                            "average_health": 9.5,
                            "average_energy": 8.0,
                            "events": [{"kind": "draw_water"}, {"kind": "draw_water"}],
                            "spawned_agents": [],
                            "newly_dead": ["agent-3", "agent-1"],
                            "agent_vitals": {
                                "agent-0": {"model": "model-a", "alive": True},
                                "agent-1": {"model": "model-a", "alive": False},
                                "agent-2": {"model": "model-b", "alive": True},
                                "agent-3": {"model": "model-c", "alive": False},
                            },
                        },
                    ],
                }
            ],
        }
    ]

    rows = module.ecology_diagnostic_rows(summaries)

    assert len(rows) == 1
    assert rows[0]["alive"] == "2/4"
    assert rows[0]["first_loss_timestep"] == 2
    assert rows[0]["first_death_timestep"] == 2
    assert rows[0]["cumulative_deaths"] == 2
    assert rows[0]["alive_models"] == "model-a: 1, model-b: 1"
    assert rows[0]["dominant_behavior"] == "gather (100%)"


def test_render_markdown_includes_ecology_diagnostics_section():
    """Markdown output should surface ecology diagnostics when present."""
    module = _load_paper_summary_module()
    experiment_frame = pd.DataFrame([{"experiment_id": "exp-1", "track": "society"}])
    prompt_variant_frame = pd.DataFrame()
    pooled_prompt_frame = pd.DataFrame()
    ecology_frame = pd.DataFrame(
        [
            {
                "experiment_id": "society-baseline-20260408T171454Z",
                "track": "society",
                "prompt_variant": "task-only",
                "trial_id": 0,
                "latest_timestep": 29,
                "alive": "10/24",
                "alive_models": "deepseek: 8, llama: 2",
                "dominant_behavior": "gather (95%)",
            }
        ]
    )

    markdown = module.render_markdown(
        experiment_frame,
        prompt_variant_frame,
        pooled_prompt_frame,
        ecology_frame,
    )

    assert "## Ecology Diagnostics" in markdown
    assert "10/24" in markdown
    assert "gather (95%)" in markdown


def test_load_partial_jsonl_summary_recovers_trial_metadata(tmp_path: Path):
    """Active JSONL logs should be convertible into partial experiment summaries."""
    module = _load_paper_summary_module()
    log_path = tmp_path / "partial.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "experiment_start",
                        "experiment_id": "paper-baseline-prisoners_dilemma-20260407T000000Z",
                        "config": {
                            "experiment": {
                                "name": "paper-baseline-prisoners_dilemma",
                                "part": 1,
                                "game": "prisoners_dilemma",
                                "repetitions": 1,
                                "prompt_variants": [
                                    {"name": "minimal-neutral"},
                                    {"name": "minimal-neutral-compact"},
                                ],
                                "parameters": {"temperature": [0.0]},
                            },
                            "run_metadata": {"track": "baseline", "presentation": "canonical"},
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "trial_summary",
                        "trial_id": 0,
                        "summary": {
                            "average_payoff_a": 1.0,
                            "average_payoff_b": 2.0,
                            "cooperation_rate_a": 0.0,
                            "cooperation_rate_b": 0.5,
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "trial_summary",
                        "trial_id": 1,
                        "summary": {
                            "average_payoff_a": 3.0,
                            "average_payoff_b": 3.0,
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

    summary = module.load_partial_jsonl_summary(log_path)

    assert summary is not None
    assert summary["experiment_id"] == "paper-baseline-prisoners_dilemma-20260407T000000Z"
    assert summary["trials"][0]["prompt_variant"] == "minimal-neutral"
    assert summary["trials"][1]["prompt_variant"] == "minimal-neutral-compact"
    assert summary["aggregate_summary"]["average_payoff_a"] == 2.0


def test_load_partial_jsonl_summary_recovers_in_progress_society_rounds(tmp_path: Path):
    """Live society JSONL logs should yield partial trials from round records alone."""
    module = _load_paper_summary_module()
    log_path = tmp_path / "society-live.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "experiment_start",
                        "experiment_id": "society-baseline-20260408T171454Z",
                        "config": {
                            "experiment": {
                                "name": "society-baseline",
                                "part": 2,
                                "rounds": 120,
                                "repetitions": 1,
                                "prompt_variants": [
                                    {"name": "task-only"},
                                    {"name": "cooperative"},
                                    {"name": "competitive"},
                                ],
                                "parameters": {"temperature": [0.3]},
                                "world": {
                                    "max_public_food": 160,
                                    "max_public_water": 220,
                                },
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "round",
                        "trial_id": 0,
                        "round_num": 1,
                        "data": {
                            "timestep": 1,
                            "alive_count": 24,
                            "total_agents": 24,
                            "public_food": 120,
                            "public_water": 160,
                            "public_resources": 280,
                            "trade_volume": 0,
                            "average_health": 11.0,
                            "average_energy": 9.0,
                            "agent_resources": {"agent-0": 26, "agent-1": 26},
                            "agent_vitals": {
                                "agent-0": {"food": 4, "water": 2},
                                "agent-1": {"food": 4, "water": 2},
                            },
                            "events": [],
                            "spawned_agents": [],
                            "newly_dead": [],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = module.load_partial_jsonl_summary(log_path)
    flattened = module.flatten_summary(summary)
    trial_rows = module.flatten_trial_rows(summary)

    assert summary is not None
    assert summary["trials"][0]["prompt_variant"] == "task-only"
    assert len(summary["trials"][0]["rounds"]) == 1
    assert summary["aggregate_summary"]["final_survival_rate"] == 1.0
    assert flattened["track"] == "society"
    assert flattened["presentation"] == "prompt_comparison"
    assert trial_rows[0]["track"] == "society"
    assert trial_rows[0]["final_survival_rate"] == 1.0


def test_collect_unique_summaries_prefers_final_json_over_partial_jsonl(tmp_path: Path):
    """Directory summaries should not double count a finalized experiment."""
    module = _load_paper_summary_module()
    experiment_id = "paper-baseline-prisoners_dilemma-20260407T000000Z"

    final_path = tmp_path / f"{experiment_id}.json"
    final_path.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "config": {
                    "experiment": {
                        "name": "paper-baseline-prisoners_dilemma",
                        "part": 1,
                        "game": "prisoners_dilemma",
                    },
                    "run_metadata": {
                        "track": "baseline",
                        "presentation": "canonical",
                    },
                },
                "trials": [
                    {
                        "trial_id": 0,
                        "prompt_variant": "minimal-neutral",
                        "repetition": 0,
                        "summary": {
                            "cooperation_rate_a": 0.0,
                            "cooperation_rate_b": 0.5,
                        },
                    },
                    {
                        "trial_id": 1,
                        "prompt_variant": "minimal-neutral-compact",
                        "repetition": 0,
                        "summary": {
                            "cooperation_rate_a": 1.0,
                            "cooperation_rate_b": 1.0,
                        },
                    },
                ],
                "aggregate_summary": {
                    "cooperation_rate_a": 0.5,
                    "cooperation_rate_b": 0.75,
                },
            }
        ),
        encoding="utf-8",
    )

    partial_path = tmp_path / f"{experiment_id}.jsonl"
    partial_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "experiment_start",
                        "experiment_id": experiment_id,
                        "config": {
                            "experiment": {
                                "name": "paper-baseline-prisoners_dilemma",
                                "part": 1,
                                "game": "prisoners_dilemma",
                                "repetitions": 1,
                                "prompt_variants": [
                                    {"name": "minimal-neutral"},
                                    {"name": "minimal-neutral-compact"},
                                ],
                                "parameters": {"temperature": [0.0]},
                            },
                            "run_metadata": {
                                "track": "baseline",
                                "presentation": "canonical",
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "trial_summary",
                        "trial_id": 0,
                        "summary": {
                            "cooperation_rate_a": 0.0,
                            "cooperation_rate_b": 0.5,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summaries = module.collect_unique_summaries([tmp_path])

    assert len(summaries) == 1
    assert summaries[0]["experiment_id"] == experiment_id
    assert len(summaries[0]["trials"]) == 2


def test_collect_unique_summaries_prefers_completed_json_over_newer_partial_retry(
    tmp_path: Path,
):
    """A newer partial retry should not displace the latest completed result for the same name."""
    module = _load_paper_summary_module()
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

    summaries = module.collect_unique_summaries([tmp_path])

    assert len(summaries) == 1
    assert summaries[0]["experiment_id"] == completed_id
    assert summaries[0]["aggregate_summary"]["cooperation_rate_a"] == 0.5


def test_collect_unique_summaries_prefers_latest_completed_json_for_same_name(tmp_path: Path):
    """Multiple completed reruns should collapse to the latest experiment id for one name."""
    module = _load_paper_summary_module()
    experiment_name = "paper-susceptibility-stag_hunt"
    older_id = f"{experiment_name}-20260407T183735Z"
    newer_id = f"{experiment_name}-20260407T184144Z"

    for experiment_id, cooperation_rate in [(older_id, 0.5), (newer_id, 0.75)]:
        (tmp_path / f"{experiment_id}.json").write_text(
            json.dumps(
                {
                    "experiment_id": experiment_id,
                    "config": {
                        "experiment": {
                            "name": experiment_name,
                            "part": 1,
                            "game": "stag_hunt",
                        },
                        "run_metadata": {
                            "track": "susceptibility",
                            "presentation": "canonical",
                        },
                    },
                    "trials": [
                        {
                            "trial_id": 0,
                            "prompt_variant": "minimal-neutral",
                            "repetition": 0,
                            "summary": {
                                "cooperation_rate_a": cooperation_rate,
                                "cooperation_rate_b": cooperation_rate,
                            },
                        }
                    ],
                    "aggregate_summary": {
                        "cooperation_rate_a": cooperation_rate,
                        "cooperation_rate_b": cooperation_rate,
                    },
                }
            ),
            encoding="utf-8",
        )

    summaries = module.collect_unique_summaries([tmp_path])

    assert len(summaries) == 1
    assert summaries[0]["experiment_id"] == newer_id
    assert summaries[0]["aggregate_summary"]["cooperation_rate_a"] == 0.75
