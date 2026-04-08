"""Tests for exact paired statistical summaries."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _load_paper_stats_module():
    """Import the paired-stats CLI module for direct helper testing."""
    module_path = ROOT / "scripts" / "paper_stats.py"
    spec = importlib.util.spec_from_file_location("paper_stats_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_sign_flip_p_value_is_one_for_zero_differences():
    """A zero-effect paired comparison should return p=1."""
    module = _load_paper_stats_module()

    p_value = module.exact_sign_flip_p_value([0.0, 0.0, 0.0])

    assert p_value == 1.0


def test_exact_sign_flip_p_value_matches_small_exact_case():
    """The exact test should agree with a hand-checkable three-pair example."""
    module = _load_paper_stats_module()

    p_value = module.exact_sign_flip_p_value([1.0, 1.0, 1.0])

    assert p_value == 0.25


def test_summarize_paired_values_reports_effect_and_identity_metadata():
    """Paired summaries should preserve means, p-values, and identical-trace counts."""
    module = _load_paper_stats_module()

    row = module.summarize_paired_values(
        analysis_id="demo",
        track="baseline",
        game="prisoners_dilemma",
        condition_a="minimal-neutral",
        condition_b="abstract-neutral-family",
        values_a=[0.0, 0.5, 0.5],
        values_b=[0.5, 0.5, 1.0],
        identical_action_traces=2,
        notes="demo note",
    )

    assert row["matched_trials"] == 3
    assert row["mean_condition_a"] == 1.0 / 3.0
    assert row["mean_condition_b"] == 2.0 / 3.0
    assert row["mean_diff_b_minus_a"] == 1.0 / 3.0
    assert row["exact_p_two_sided"] == 0.5
    assert row["identical_action_traces"] == 2
    assert row["notes"] == "demo note"
