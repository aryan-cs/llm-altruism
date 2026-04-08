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
