"""Tests for paper summary aggregation helpers."""

from __future__ import annotations

import importlib.util
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
            }
        ]
    )

    markdown = module.render_markdown(experiment_frame, prompt_variant_frame, pooled_prompt_frame)

    assert "## Pooled Prompt Variant Summary" in markdown
    assert "task-only" in markdown
