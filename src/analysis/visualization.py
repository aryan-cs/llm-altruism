"""Visualization helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _ensure_parent(path: str | Path | None) -> None:
    if path is None:
        return
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def plot_cooperation_rates(summary_rows: list[dict[str, Any]], output_path: str | None = None):
    """Plot cooperation rates across conditions."""
    if not summary_rows:
        return None

    frame = pd.DataFrame(summary_rows)
    if "cooperation_rate_a" not in frame.columns:
        return None

    melted = frame.melt(
        id_vars=[col for col in ["pairing", "prompt_variant", "temperature"] if col in frame.columns],
        value_vars=[col for col in ["cooperation_rate_a", "cooperation_rate_b"] if col in frame.columns],
        var_name="agent",
        value_name="cooperation_rate",
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    x_col = "pairing" if "pairing" in melted.columns else "prompt_variant"
    sns.barplot(data=melted, x=x_col, y="cooperation_rate", hue="agent", ax=ax)
    ax.set_ylabel("Cooperation Rate")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()

    if output_path:
        _ensure_parent(output_path)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig


def plot_payoff_curve(round_rows: list[dict[str, Any]], output_path: str | None = None):
    """Plot cumulative payoffs over the course of a trial."""
    if not round_rows:
        return None

    frame = pd.DataFrame(round_rows)
    if "round" not in frame.columns:
        return None

    frame = frame.sort_values("round").copy()
    if "payoff_a" in frame.columns:
        frame["cumulative_payoff_a"] = frame["payoff_a"].cumsum()
    if "payoff_b" in frame.columns:
        frame["cumulative_payoff_b"] = frame["payoff_b"].cumsum()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    if "cumulative_payoff_a" in frame.columns:
        sns.lineplot(data=frame, x="round", y="cumulative_payoff_a", label="Agent A", ax=ax)
    if "cumulative_payoff_b" in frame.columns:
        sns.lineplot(data=frame, x="round", y="cumulative_payoff_b", label="Agent B", ax=ax)
    ax.set_ylabel("Cumulative Payoff")
    ax.set_xlabel("Round")
    fig.tight_layout()

    if output_path:
        _ensure_parent(output_path)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig


def plot_resource_distribution(
    snapshots: list[dict[str, Any]],
    output_path: str | None = None,
):
    """Plot the public commons level over time for a society simulation."""
    if not snapshots:
        return None

    frame = pd.DataFrame(snapshots)
    if "timestep" not in frame.columns or "public_resources" not in frame.columns:
        return None

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=frame, x="timestep", y="public_resources", marker="o", ax=ax)
    ax.set_ylabel("Public Resources")
    ax.set_xlabel("Timestep")
    fig.tight_layout()

    if output_path:
        _ensure_parent(output_path)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig
