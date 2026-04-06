"""Analysis utilities for metrics, reporting, and visualization."""

from .metrics import (
    COOPERATIVE_ACTIONS_BY_GAME,
    DEFECTIVE_ACTIONS_BY_GAME,
    categorical_distribution,
    commons_health,
    cooperation_rate,
    defection_rate,
    exploitation_index,
    forgiveness_index,
    gini_coefficient,
    reciprocity_index,
    resource_velocity,
    safe_mean,
    summarize_numeric_trial,
    summarize_pairwise_trial,
    summarize_society,
)
from .report import (
    collect_experiment_summaries,
    comparison_table,
    load_result_artifact,
    render_text_report,
    summarize_jsonl_log,
)
from .strategy_classifier import classify_strategy
from .visualization import plot_cooperation_rates, plot_payoff_curve, plot_resource_distribution

__all__ = [
    "COOPERATIVE_ACTIONS_BY_GAME",
    "DEFECTIVE_ACTIONS_BY_GAME",
    "categorical_distribution",
    "classify_strategy",
    "collect_experiment_summaries",
    "commons_health",
    "comparison_table",
    "cooperation_rate",
    "defection_rate",
    "exploitation_index",
    "forgiveness_index",
    "gini_coefficient",
    "load_result_artifact",
    "plot_cooperation_rates",
    "plot_payoff_curve",
    "plot_resource_distribution",
    "reciprocity_index",
    "render_text_report",
    "resource_velocity",
    "safe_mean",
    "summarize_jsonl_log",
    "summarize_numeric_trial",
    "summarize_pairwise_trial",
    "summarize_society",
]
