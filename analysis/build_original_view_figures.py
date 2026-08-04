"""Rebuild the original paper's bar, line, and raster views on current routes.

The two bar charts use only the sealed public aggregate graph.  The two
longitudinal charts and agent-day raster replay the hash-bound private Part 2 journals without any
network access, validate every retained action and transition against the
original simulator, and then plot matched-trajectory means.  This keeps the
visual grammar of the earlier ``Safety Beyond Refusal`` submission while
respecting the current experiment's 12-trajectory independence unit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from analysis.build_provider_safe_v2_paper_assets import (
    BLUE,
    GREEN,
    GRID,
    INK,
    MUTED,
    ORANGE,
    PROVIDER_DISPLAY_NAMES,
    RED,
    _load_and_validate,
    _provider_grouped_rows,
)
from analysis.finalize_inference_hub_part2_offline import (
    _offline_replay,
    _validate_frozen_inputs,
    _validate_journal_identity,
)
from experiments.misc.inference_hub_part1_panel import (
    _ChainedJournal,
    _read_json,
    _safe_file_stem,
    _self_hash,
    _validate_checkpoint_reference,
)
from experiments.misc.inference_hub_part2_panel import _matched_attrition, _result_index
from experiments.part2.part_2 import _collapse_deaths


DEFAULT_ANALYSIS_DIR = Path("data/processed/provider-safe-v2-definitive-analysis")
DEFAULT_PART2_MANIFEST = Path(
    "data/private/inference_hub/definitive-part2-n12-main19-v3/private/manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("docs/conference_submission/figures")

# The historical figures distinguished model families with turquoise, blue,
# purple, and magenta.  The expanded panel keeps those hues and the paper's
# explicit Okabe--Ito task colors; line style distinguishes routes within a
# provider when their trajectories coincide.
PROVIDER_COLORS = {
    "openai": BLUE,
    "anthropic": ORANGE,
    "google": GREEN,
    "nvidia": RED,
    "qwen": "#7544B8",
    "meta": "#4A84D2",
    "deepseek-ai": "#1F9E83",
    "zai-org": "#A344B5",
    "perplexity": "#6B7280",
}
LINE_STYLES = ("-", "--", ":", "-.")
MARKERS = ("o", "s", "^", "D")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.titlesize": 10.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


class OriginalViewFigureError(RuntimeError):
    """The bound evidence cannot support an original-view figure."""


def _provider_color(row: Mapping[str, Any]) -> str:
    return PROVIDER_COLORS.get(str(row["upstream_provider"]), MUTED)


def _short_label(model: str) -> str:
    replacements = {
        "claude-opus-4-6": "Opus 4.6",
        "claude-sonnet-4-6": "Sonnet 4.6",
        "claude-haiku-4-5": "Haiku 4.5",
        "claude-sonnet-4-5": "Sonnet 4.5",
        "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
        "gemini-3.5-flash": "Gemini 3.5 Flash",
        "nemotron-3-super-v3": "Nemotron 3 Super",
        "nemotron-3-ultra": "Nemotron 3 Ultra",
        "llama-3.3-70b-instruct": "Llama 3.3 70B",
        "deepseek-v4-flash": "DeepSeek V4 Flash",
    }
    return replacements.get(model, model)


def _provider_spans(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    start = 0
    while start < len(rows):
        provider = str(rows[start]["upstream_provider"])
        end = start + 1
        while end < len(rows) and rows[end]["upstream_provider"] == provider:
            end += 1
        spans.append((provider, start, end))
        start = end
    return spans


def _decorate_provider_groups(ax: plt.Axes, rows: Sequence[Mapping[str, Any]]) -> None:
    for provider, start, end in _provider_spans(rows):
        if start:
            ax.axvline(start - 0.5, color=MUTED, linewidth=0.65, alpha=0.65)
        ax.text(
            (start + end - 1) / 2,
            -0.38,
            PROVIDER_DISPLAY_NAMES.get(provider, provider),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.7,
            fontweight="bold",
            color=_provider_color(rows[start]),
            clip_on=False,
        )


def _atomic_save(fig: plt.Figure, directory: Path, stem: str, title: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in ("pdf", "png"):
        descriptor, temporary_name = tempfile.mkstemp(
            dir=directory, prefix=f".{stem}.", suffix=f".{suffix}"
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            if suffix == "pdf":
                fig.savefig(
                    temporary,
                    format="pdf",
                    bbox_inches="tight",
                    metadata={
                        "Title": title,
                        "Author": "Safety Beyond Refusal asset generator",
                        "Creator": "Matplotlib",
                    },
                )
            else:
                fig.savefig(temporary, format="png", dpi=300, bbox_inches="tight")
            output = directory / f"{stem}.{suffix}"
            os.replace(temporary, output)
            outputs.append(output)
        finally:
            temporary.unlink(missing_ok=True)
    plt.close(fig)
    return outputs


def _bar_chart(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_field: str,
    low_field: str,
    high_field: str,
    ylabel: str,
    title: str,
    subtitle: str,
    stem: str,
    output_dir: Path,
) -> list[Path]:
    ordered = list(_provider_grouped_rows(rows))
    values = np.asarray([float(row[value_field]) for row in ordered])
    lows = np.asarray([float(row[low_field]) for row in ordered])
    highs = np.asarray([float(row[high_field]) for row in ordered])
    if np.any(lows > values) or np.any(values > highs):
        raise OriginalViewFigureError(f"{stem} contains an interval excluding its estimate.")
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    fig.patch.set_facecolor("white")
    bars = ax.bar(
        x,
        values * 100.0,
        width=0.72,
        color=[_provider_color(row) for row in ordered],
        edgecolor="white",
        linewidth=0.45,
        zorder=2,
    )
    ax.errorbar(
        x,
        values * 100.0,
        yerr=np.vstack(((values - lows) * 100.0, (highs - values) * 100.0)),
        fmt="none",
        ecolor=INK,
        elinewidth=0.9,
        capsize=2.3,
        capthick=0.9,
        zorder=4,
    )
    # Labels sit above the upper interval cap, never on top of a whisker.
    for bar, value, high in zip(bars, values * 100.0, highs * 100.0):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(value, high) + 1.8,
            f"{value:.1f}%",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=6.0,
            color=INK,
            clip_on=False,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_short_label(str(row["model"])) for row in ordered],
        rotation=52,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xlim(-0.6, len(ordered) - 0.4)
    ax.set_ylim(0.0, 122.0)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=27)
    ax.text(0.0, 1.045, subtitle, transform=ax.transAxes, color=MUTED, fontsize=7.4)
    ax.grid(axis="y", color=GRID, linewidth=0.65, alpha=0.85, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    _decorate_provider_groups(ax, ordered)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.84, bottom=0.36)
    return _atomic_save(fig, output_dir, stem, title)


def _validated_curves(
    manifest_path: Path,
    public_trajectory_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], int, int, int]:
    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path, "Part 2 private manifest")
    if manifest.get("evidence_sha256") != _self_hash(manifest) or manifest.get("complete") is not True:
        raise OriginalViewFigureError("Part 2 manifest is incomplete or fails its self-hash.")
    subjects, contract, environment_seeds = _validate_frozen_inputs(manifest)
    references = manifest.get("journals")
    if not isinstance(references, Mapping):
        raise OriginalViewFigureError("Part 2 journal references are absent.")
    public_by_key = {
        (str(row["target_id"]), int(row["trajectory_index"])): row
        for row in public_trajectory_rows
    }
    curves: dict[str, list[dict[str, Any]]] = defaultdict(list)
    execution = manifest.get("execution_contract")
    if not isinstance(execution, Mapping):
        raise OriginalViewFigureError("Part 2 execution contract is absent.")
    private_dir = manifest_path.parent
    for subject in subjects:
        target_id = str(subject["target_id"])
        expected_directory = private_dir / "trajectories" / _safe_file_stem(target_id)
        for trajectory_index, environment_seed in enumerate(environment_seeds):
            key = f"{target_id}::{trajectory_index}"
            reference = references.get(key)
            if not isinstance(reference, Mapping):
                raise OriginalViewFigureError(f"Missing trajectory binding: {key}.")
            path = Path(str(reference.get("path", ""))).resolve()
            if path != (expected_directory / f"seed-{trajectory_index:03d}.jsonl").resolve():
                raise OriginalViewFigureError(f"Trajectory path changed: {key}.")
            journal = _ChainedJournal(path)
            _validate_checkpoint_reference(journal, reference, label=f"trajectory {key}")
            _validate_journal_identity(
                journal.records,
                subject=subject,
                trajectory_index=trajectory_index,
                environment_seed=environment_seed,
            )
            replayed = _offline_replay(
                subject=subject,
                trajectory_index=trajectory_index,
                environment_seed=environment_seed,
                contract=contract,
                records=journal.records,
                execution_contract=execution,
            )
            if replayed is None or replayed.get("operationally_eligible") is not True:
                raise OriginalViewFigureError(f"Trajectory is incomplete or ineligible: {key}.")
            results, _attempts = _result_index(journal, subject, trajectory_index)
            living = list(range(contract.society_size))
            reserve = contract.capacity
            reserve_curve: list[int] = []
            population_curve: list[int] = []
            agent_actions = [
                ["NO_ACTIVE_DECISION"] * contract.days
                for _ in range(contract.society_size)
            ]
            restraint = scheduled = 0
            for day in range(1, contract.days + 1):
                if not living:
                    reserve_curve.append(0)
                    population_curve.append(0)
                    continue
                day_rows = [results[(day, slot)] for slot in living]
                actions = [str(row["action"]) for row in day_rows]
                for slot, action in zip(living, actions):
                    if action not in {"OPTION_A", "OPTION_B", "INVALID"}:
                        raise OriginalViewFigureError(
                            f"Unexpected retained Part 2 action {action!r}: {key}."
                        )
                    agent_actions[slot][day - 1] = action
                scheduled += len(actions)
                restraint += actions.count("OPTION_A")
                reserve = max(0, reserve - contract.reserve_cost * actions.count("OPTION_B"))
                deaths = _collapse_deaths(len(living), reserve, contract.collapse_death_rate)
                dead, _attrition_seed = _matched_attrition(living, deaths, environment_seed, day)
                dead_set = set(dead)
                living = [slot for slot in living if slot not in dead_set]
                reserve_curve.append(reserve)
                population_curve.append(len(living))
            derived = {
                "final_reserve": reserve_curve[-1],
                "final_population": population_curve[-1],
                "aurc": sum(reserve_curve) / (contract.capacity * contract.days),
                "aupc": sum(population_curve) / (contract.society_size * contract.days),
                "restraint_rate": restraint / scheduled if scheduled else 0.0,
            }
            for field, observed in derived.items():
                if not math.isclose(float(replayed[field]), float(observed), abs_tol=1e-12):
                    raise OriginalViewFigureError(f"Replayed {field} does not reconcile: {key}.")
            public = public_by_key.get((target_id, trajectory_index))
            if public is None:
                raise OriginalViewFigureError(f"Public trajectory aggregate is absent: {key}.")
            for field, observed in derived.items():
                if not math.isclose(float(public[field]), float(observed), abs_tol=1e-12):
                    raise OriginalViewFigureError(f"Public {field} does not reconcile: {key}.")
            curves[target_id].append(
                {
                    "trajectory_index": trajectory_index,
                    "reserve": reserve_curve,
                    "population": population_curve,
                    "agent_actions": agent_actions,
                }
            )
    if set(curves) != {str(row["target_id"]) for row in subjects}:
        raise OriginalViewFigureError("Trajectory replay target set changed.")
    return subjects, curves, contract.days, contract.capacity, contract.society_size


def _line_chart(
    rows: Sequence[Mapping[str, Any]],
    curves: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    metric: str,
    days: int,
    maximum: int,
    ylabel: str,
    title: str,
    stem: str,
    output_dir: Path,
) -> list[Path]:
    ordered = list(_provider_grouped_rows(rows))
    fig, ax = plt.subplots(figsize=(10.8, 6.25))
    fig.patch.set_facecolor("white")
    provider_counts: dict[str, int] = defaultdict(int)
    x = np.arange(1, days + 1)
    for row in ordered:
        target_id = str(row["target_id"])
        provider = str(row["upstream_provider"])
        provider_index = provider_counts[provider]
        provider_counts[provider] += 1
        matrix = np.asarray([curve[metric] for curve in curves[target_id]], dtype=float)
        if matrix.shape != (12, days):
            raise OriginalViewFigureError(f"{target_id} lacks the 12 x {days} curve matrix.")
        mean = matrix.mean(axis=0)
        ax.plot(
            x,
            mean,
            label=f"{PROVIDER_DISPLAY_NAMES.get(provider, provider)}: {_short_label(str(row['model']))}",
            color=_provider_color(row),
            linestyle=LINE_STYLES[provider_index % len(LINE_STYLES)],
            marker=MARKERS[provider_index % len(MARKERS)],
            markevery=(provider_index % 3, 3),
            markersize=2.7,
            linewidth=1.45,
            alpha=0.94,
        )
    ax.set_xlim(1, days)
    ax.set_xticks(x)
    ax.set_ylim(0, maximum * 1.04)
    ax.set_xlabel("Simulation day")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=20)
    ax.text(
        0.0,
        1.025,
        "Each curve is the day-wise mean of 12 common-seed trajectories; legend order is provider-grouped, newest route first.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=7.4,
    )
    ax.grid(color=GRID, linewidth=0.65, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.87, bottom=0.31)
    return _atomic_save(fig, output_dir, stem, title)


def _agent_day_raster(
    rows: Sequence[Mapping[str, Any]],
    curves: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    days: int,
    society_size: int,
    output_dir: Path,
) -> list[Path]:
    """Restore the original micro-level raster on one common, fixed seed."""

    ordered = list(_provider_grouped_rows(rows))
    code = {
        "OPTION_B": 0,
        "OPTION_A": 1,
        "NO_ACTIVE_DECISION": 2,
        "INVALID": 3,
    }
    raster = np.full((len(ordered) * society_size, days), 2, dtype=int)
    for model_index, row in enumerate(ordered):
        target_id = str(row["target_id"])
        seed_zero = [
            curve for curve in curves[target_id]
            if int(curve["trajectory_index"]) == 0
        ]
        if len(seed_zero) != 1:
            raise OriginalViewFigureError(
                f"{target_id} does not have exactly one seed-index-0 trajectory."
            )
        actions = seed_zero[0]["agent_actions"]
        if len(actions) != society_size or any(len(agent) != days for agent in actions):
            raise OriginalViewFigureError(f"{target_id} has a malformed action raster.")
        start = model_index * society_size
        raster[start : start + society_size, :] = np.asarray(
            [[code[action] for action in agent] for agent in actions],
            dtype=int,
        )

    if np.any(raster == code["INVALID"]):
        raise OriginalViewFigureError(
            "The prespecified seed-index-0 raster unexpectedly contains an INVALID action."
        )
    cmap = ListedColormap([RED, GREEN, "#E5E7EB", ORANGE])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(figsize=(8.6, 11.3))
    fig.patch.set_facecolor("white")
    ax.imshow(raster, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    centers = [
        index * society_size + (society_size - 1) / 2
        for index in range(len(ordered))
    ]
    ax.set_yticks(centers)
    ax.set_yticklabels([_short_label(str(row["model"])) for row in ordered])
    ax.set_xticks(np.arange(days))
    ax.set_xticklabels([str(day) for day in range(1, days + 1)])
    ax.set_xlabel("Simulation day")
    ax.set_title(
        "Agent-day actions in one prespecified common-seed trajectory",
        loc="left",
        fontsize=12.5,
        fontweight="bold",
        pad=25,
    )
    ax.text(
        0.0,
        1.012,
        "All 19 routes; seed index 0; providers contiguous and newest route first.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=7.4,
    )
    spans = _provider_spans(ordered)
    for provider, start_model, end_model in spans:
        if start_model:
            ax.axhline(
                start_model * society_size - 0.5,
                color=INK,
                linewidth=1.25,
            )
        ax.text(
            -0.48,
            1.0
            - (((start_model + end_model) * society_size / 2 - 0.5) / raster.shape[0]),
            PROVIDER_DISPLAY_NAMES.get(provider, provider),
            transform=ax.transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=6.0,
            fontweight="bold",
            color=_provider_color(ordered[start_model]),
            clip_on=False,
        )
    provider_boundaries = {start for _provider, start, _end in spans if start}
    for boundary in range(1, len(ordered)):
        if boundary not in provider_boundaries:
            ax.axhline(
                boundary * society_size - 0.5,
                color="white",
                linewidth=0.8,
            )
    ax.set_xticks(np.arange(-0.5, days, 1), minor=True)
    ax.grid(which="minor", axis="x", color="white", linewidth=0.45, alpha=0.7)
    ax.tick_params(which="minor", bottom=False)
    ax.legend(
        handles=[
            Patch(facecolor=GREEN, label="Restraint (OPTION_A)"),
            Patch(facecolor=RED, label="Overuse (OPTION_B)"),
            Patch(facecolor="#E5E7EB", label="No active decision after attrition"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.055),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(left=0.30, right=0.99, top=0.93, bottom=0.105)
    return _atomic_save(
        fig,
        output_dir,
        "part2_agent_day_raster_current",
        "Agent-day actions in one prespecified common-seed trajectory",
    )


def build_original_view_figures(
    analysis_dir: Path,
    part2_manifest: Path,
    output_dir: Path,
) -> list[Path]:
    data = _load_and_validate(analysis_dir.resolve())
    aggregates = json.loads((analysis_dir / "figure_aggregates.json").read_text(encoding="utf-8"))
    outputs: list[Path] = []
    outputs.extend(
        _bar_chart(
            data["part0"],
            value_field="refusal_rate_all_scheduled",
            low_field="refusal_rate_all_scheduled_finite_bank_sensitivity_low",
            high_field="refusal_rate_all_scheduled_finite_bank_sensitivity_high",
            ylabel="Material-refusal rate (%)",
            title="Safety refusal by current exact model route",
            subtitle="Whiskers: 5,000-replicate harmful-root sensitivity intervals; higher means less harmful-request assistance.",
            stem="part0_refusal_rate_by_model",
            output_dir=output_dir,
        )
    )
    outputs.extend(
        _bar_chart(
            data["part2"],
            value_field="mean_trajectory_restraint_rate_all_scheduled",
            low_field="mean_trajectory_restraint_rate_all_scheduled_t95_low",
            high_field="mean_trajectory_restraint_rate_all_scheduled_t95_high",
            ylabel="Mean trajectory restraint rate (%)",
            title="Commons restraint by current exact model route",
            subtitle="Whiskers: Student-t 95% intervals over 12 matched trajectories; higher means less commons overuse.",
            stem="part2_restraint_rate_by_model",
            output_dir=output_dir,
        )
    )
    subjects, curves, days, capacity, society_size = _validated_curves(
        part2_manifest,
        aggregates["part2_trajectories"],
    )
    subject_ids = {str(row["target_id"]) for row in subjects}
    part2_rows = [row for row in data["part2"] if str(row["target_id"]) in subject_ids]
    if len(part2_rows) != len(subjects) != 0:
        raise OriginalViewFigureError("Public and private Part 2 target panels differ.")
    outputs.extend(
        _line_chart(
            part2_rows,
            curves,
            metric="reserve",
            days=days,
            maximum=capacity,
            ylabel="Shared reserve units",
            title="Shared reserve over time by current model route",
            stem="part2_shared_reserve_over_time",
            output_dir=output_dir,
        )
    )
    outputs.extend(
        _line_chart(
            part2_rows,
            curves,
            metric="population",
            days=days,
            maximum=society_size,
            ylabel="Living population",
            title="Living population over time by current model route",
            stem="part2_population_over_time",
            output_dir=output_dir,
        )
    )
    outputs.extend(
        _agent_day_raster(
            part2_rows,
            curves,
            days=days,
            society_size=society_size,
            output_dir=output_dir,
        )
    )
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--part2-manifest", type=Path, default=DEFAULT_PART2_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        paths = build_original_view_figures(
            args.analysis_dir,
            args.part2_manifest,
            args.output_dir,
        )
    except (OriginalViewFigureError, OSError, ValueError) as error:
        print(f"Original-view figure generation failed: {error}")
        return 1
    for path in paths:
        print(f"wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
