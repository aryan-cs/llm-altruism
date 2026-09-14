"""Rebuild the original paper's bar, line, and raster views on current routes.

The two bar charts use only the sealed public aggregate graph.  The two
longitudinal charts and agent-day raster validate the definitive three-shard
Part 2 composition, select each effective source or operational-repair
journal, and replay it without network dispatch.  Every retained action and
transition must reconcile with both the simulator and the public aggregate.
Longitudinal environmental means exclude an entire trajectory when any action
is semantically invalid; the all-scheduled restraint bar and prespecified-seed
raster retain those invalid actions under their separate contracts.  This
keeps the visual grammar of the earlier ``Safety Beyond Refusal`` submission
while making the 23-route, 50-agent, 100-day panel readable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

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
    _sha256_file,
    _validate_checkpoint_reference,
)
from experiments.misc.inference_hub_part2_panel import _matched_attrition, _result_index
from experiments.part2.part_2 import _collapse_deaths


DEFAULT_ANALYSIS_DIR = Path("data/processed/provider-safe-v2-definitive-analysis")
DEFAULT_PART2_MANIFEST = Path(
    "data/private/inference_hub/definitive-part2-n12-main19-v3/private/manifest.json"
)
DEFAULT_PART2_SOURCE_OVERLAY_PAIRS = (
    (
        Path(
            "data/private/inference_hub/full-part2-n12-n50-d100-main21-v5/"
            "private/manifest.json"
        ),
        Path(
            "data/private/inference_hub/full-part2-n12-n50-d100-main21-v5-"
            "operational-completion-capability-v4/private/manifest.json"
        ),
    ),
    (
        Path(
            "data/private/inference_hub/full-part2-n12-n50-d100-"
            "nemotron-3-ultra-recovered-v1/private/manifest.json"
        ),
        Path(
            "data/private/inference_hub/full-part2-n12-n50-d100-"
            "nemotron-3-ultra-operational-repair-v1/private/manifest.json"
        ),
    ),
    (
        Path(
            "data/private/inference_hub/full-part2-n12-n50-d100-"
            "deepseek-v4-flash-recovered-v1/private/manifest.json"
        ),
        Path(
            "data/private/inference_hub/full-part2-n12-n50-d100-"
            "deepseek-v4-flash-operational-repair-v1/private/manifest.json"
        ),
    ),
)
DEFAULT_OUTPUT_DIR = Path("docs/conference_submission/figures")

EXPECTED_PART2_ROUTE_COUNT = 23
EXPECTED_PART2_TRAJECTORIES_PER_ROUTE = 12
EXPECTED_PART2_TRAJECTORY_COUNT = (
    EXPECTED_PART2_ROUTE_COUNT * EXPECTED_PART2_TRAJECTORIES_PER_ROUTE
)
EXPECTED_PART2_SOCIETY_SIZE = 50
EXPECTED_PART2_DAYS = 100
EXPECTED_PART2_SINGLETON_SHARDS = (
    "nvidia/nemotron-3-ultra",
    "deepseek-ai/deepseek-v4-flash",
)
EXPECTED_OUTPUT_BASENAMES = (
    "part0_refusal_rate_by_model.pdf",
    "part0_refusal_rate_by_model.png",
    "part2_restraint_rate_by_model.pdf",
    "part2_restraint_rate_by_model.png",
    "part2_shared_reserve_over_time.pdf",
    "part2_shared_reserve_over_time.png",
    "part2_population_over_time.pdf",
    "part2_population_over_time.png",
    "part2_agent_day_raster_current.pdf",
    "part2_agent_day_raster_current.png",
)

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
    "minimaxai": "#CC79A7",
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


def _portable_private_manifest_binding(
    path: Path,
    *,
    label: str,
) -> tuple[Path, dict[str, Any], dict[str, str]]:
    """Return the exact public projection of one private manifest."""

    try:
        from analysis import (
            validate_inference_hub_part2_operational_overlays as overlay_validation,
        )
    except ImportError as error:
        raise OriginalViewFigureError(
            "The operational-overlay validator is required for manifest binding."
        ) from error
    manifest_path = overlay_validation._private_manifest_path(path).resolve()
    manifest = _read_json(manifest_path, label)
    evidence_sha256 = manifest.get("evidence_sha256")
    if (
        not isinstance(evidence_sha256, str)
        or len(evidence_sha256) != 64
        or any(character not in "0123456789abcdef" for character in evidence_sha256)
    ):
        raise OriginalViewFigureError(f"{label} lacks a canonical evidence digest.")
    return (
        manifest_path,
        manifest,
        {
            "basename": manifest_path.name,
            "file_sha256": _sha256_file(manifest_path),
            "evidence_sha256": evidence_sha256,
        },
    )


def _validate_analysis_composition_binding(
    pairs: Sequence[tuple[Path, Path]],
    topology: Mapping[str, Any],
) -> None:
    """Bind the private replay inputs to the exact public analysis topology."""

    if topology.get("mode") != "three_pair_operational_repair_composition":
        raise OriginalViewFigureError(
            "Original 100-day paper views require the composed three-pair analysis."
        )
    expected_pairs = topology.get("pairs")
    if not isinstance(expected_pairs, list) or len(expected_pairs) != 3:
        raise OriginalViewFigureError(
            "The public analysis does not bind exactly three Part 2 pairs."
        )
    if len(pairs) != 3:
        raise OriginalViewFigureError(
            "Original 100-day paper views require exactly three source/overlay pairs."
        )

    actual_pairs: list[dict[str, Any]] = []
    for index, (source_value, overlay_value) in enumerate(pairs, start=1):
        source_path, source_manifest, source = _portable_private_manifest_binding(
            Path(source_value), label=f"Part 2 source manifest {index}"
        )
        overlay_path, overlay_manifest, overlay = _portable_private_manifest_binding(
            Path(overlay_value), label=f"Part 2 overlay manifest {index}"
        )
        del source_manifest
        overlay.update(
            {
                "source_manifest_file_sha256": source["file_sha256"],
                "source_manifest_evidence_sha256": source["evidence_sha256"],
            }
        )
        parent_reference = overlay_manifest.get("parent_overlay_manifest")
        if parent_reference is not None:
            if (
                not isinstance(parent_reference, Mapping)
                or set(parent_reference)
                != {"path", "file_sha256", "evidence_sha256"}
                or not isinstance(parent_reference.get("path"), str)
            ):
                raise OriginalViewFigureError(
                    f"Part 2 overlay manifest {index} has malformed parent lineage."
                )
            parent_path, _parent_manifest, parent = _portable_private_manifest_binding(
                Path(str(parent_reference["path"])),
                label=f"Part 2 parent overlay manifest {index}",
            )
            if (
                parent_reference.get("file_sha256") != parent["file_sha256"]
                or parent_reference.get("evidence_sha256")
                != parent["evidence_sha256"]
            ):
                raise OriginalViewFigureError(
                    f"Part 2 overlay manifest {index} parent binding changed."
                )
            overlay["parent_operational_repair_overlay"] = parent
            del parent_path
        actual_pairs.append({"source": source, "overlay": overlay})
        del source_path, overlay_path

    if actual_pairs != expected_pairs:
        raise OriginalViewFigureError(
            "Private Part 2 replay inputs do not match the analysis manifest bindings."
        )


def _publish_staged_outputs(staging_dir: Path, output_dir: Path) -> list[Path]:
    """Publish all ten views as one rollback-safe output transaction."""

    staging_dir = staging_dir.resolve()
    output_dir = output_dir.resolve()
    staged = [staging_dir / basename for basename in EXPECTED_OUTPUT_BASENAMES]
    if any(not path.is_file() for path in staged):
        raise OriginalViewFigureError(
            "The staged original-view output set is incomplete."
        )
    unexpected = {
        path.name for path in staging_dir.iterdir() if path.is_file()
    } - set(EXPECTED_OUTPUT_BASENAMES)
    if unexpected:
        raise OriginalViewFigureError(
            "The staged original-view output set contains unexpected files."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir_preexisted = output_dir.exists()
    output_dir.mkdir(parents=False, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.backup.",
    ) as backup_name:
        backup_dir = Path(backup_name)
        moved_old: list[tuple[Path, Path]] = []
        published: list[Path] = []
        try:
            for basename in EXPECTED_OUTPUT_BASENAMES:
                target = output_dir / basename
                if target.exists():
                    if not target.is_file():
                        raise OriginalViewFigureError(
                            f"Original-view target is not a regular file: {target}"
                        )
                    backup = backup_dir / basename
                    os.replace(target, backup)
                    moved_old.append((target, backup))
            for source, basename in zip(
                staged, EXPECTED_OUTPUT_BASENAMES, strict=True
            ):
                target = output_dir / basename
                os.replace(source, target)
                published.append(target)
        except Exception as publish_error:
            rollback_errors: list[Exception] = []
            for target in reversed(published):
                try:
                    target.unlink(missing_ok=True)
                except OSError as error:
                    rollback_errors.append(error)
            for target, backup in reversed(moved_old):
                try:
                    os.replace(backup, target)
                except OSError as error:
                    rollback_errors.append(error)
            if not output_dir_preexisted:
                try:
                    output_dir.rmdir()
                except OSError:
                    pass
            if rollback_errors:
                raise OriginalViewFigureError(
                    "Original-view publication failed and rollback was incomplete."
                ) from publish_error
            raise OriginalViewFigureError(
                "Original-view publication failed; prior outputs were restored."
            ) from publish_error
    return [output_dir / basename for basename in EXPECTED_OUTPUT_BASENAMES]


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
        "minimax-m2.7": "MiniMax M2.7",
        "glm-5.1": "GLM 5.1",
    }
    return replacements.get(model, model)


def _day_ticks(days: int) -> list[int]:
    """Return stable, sparse day labels including both endpoints."""

    if isinstance(days, bool) or not isinstance(days, int) or days < 1:
        raise OriginalViewFigureError("The simulation horizon must be a positive integer.")
    if days <= 6:
        return list(range(1, days + 1))
    ticks = [1]
    ticks.extend(round(days * fraction / 4) for fraction in range(1, 4))
    ticks.append(days)
    return list(dict.fromkeys(ticks))


def _route_legend_label(row: Mapping[str, Any]) -> str:
    provider = str(row["upstream_provider"])
    return (
        f"{PROVIDER_DISPLAY_NAMES.get(provider, provider)} · "
        f"{_short_label(str(row['model']))}"
    )


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
    try:
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
    finally:
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


def _public_trajectory_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise OriginalViewFigureError("A public Part 2 trajectory row is malformed.")
        target_id = row.get("target_id")
        trajectory_index = row.get("trajectory_index")
        if (
            not isinstance(target_id, str)
            or not target_id
            or isinstance(trajectory_index, bool)
            or not isinstance(trajectory_index, int)
            or trajectory_index < 0
        ):
            raise OriginalViewFigureError("A public Part 2 trajectory key is malformed.")
        key = (target_id, trajectory_index)
        if key in indexed:
            raise OriginalViewFigureError(
                f"A public Part 2 trajectory key is duplicated: {target_id}::{trajectory_index}."
            )
        indexed[key] = row
    return indexed


def _effective_row_without_lineage(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key
        not in {
            "operational_repair_round",
            "source_replaced_for_operational_failure",
        }
    }


def _curve_from_replayed_records(
    records: Sequence[Mapping[str, Any]],
    *,
    subject: Mapping[str, Any],
    trajectory_index: int,
    environment_seed: int,
    contract: Any,
    execution_contract: Mapping[str, Any],
    effective_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay one already hash-bound selected journal into plot-ready curves."""

    target_id = str(subject["target_id"])
    key = f"{target_id}::{trajectory_index}"
    try:
        _validate_journal_identity(
            records,
            subject=subject,
            trajectory_index=trajectory_index,
            environment_seed=environment_seed,
        )
        replayed = _offline_replay(
            subject=subject,
            trajectory_index=trajectory_index,
            environment_seed=environment_seed,
            contract=contract,
            records=records,
            execution_contract=execution_contract,
        )
        results, _attempts = _result_index(
            SimpleNamespace(records=records), subject, trajectory_index
        )
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as error:
        raise OriginalViewFigureError(
            f"Selected Part 2 journal cannot be replayed offline: {key}."
        ) from error
    if replayed is None or replayed.get("operationally_eligible") is not True:
        raise OriginalViewFigureError(
            f"Selected Part 2 journal is incomplete or ineligible: {key}."
        )
    if replayed != _effective_row_without_lineage(effective_row):
        raise OriginalViewFigureError(
            f"Selected Part 2 journal does not reproduce its effective row: {key}."
        )

    living = list(range(contract.society_size))
    reserve = contract.capacity
    reserve_curve: list[int] = []
    population_curve: list[int] = []
    agent_actions = (
        [
            ["NO_ACTIVE_DECISION"] * contract.days
            for _ in range(contract.society_size)
        ]
        if trajectory_index == 0
        else None
    )
    restraint = scheduled = semantic_invalid = 0
    for day in range(1, contract.days + 1):
        if not living:
            reserve_curve.append(0)
            population_curve.append(0)
            continue
        try:
            day_rows = [results[(day, slot)] for slot in living]
        except KeyError as error:
            raise OriginalViewFigureError(
                f"Selected Part 2 journal omits a scheduled participant: {key}."
            ) from error
        actions = [str(row["action"]) for row in day_rows]
        if any(action not in {"OPTION_A", "OPTION_B", "INVALID"} for action in actions):
            raise OriginalViewFigureError(
                f"Selected Part 2 journal contains an unknown retained action: {key}."
            )
        if agent_actions is not None:
            for slot, action in zip(living, actions, strict=True):
                agent_actions[slot][day - 1] = action
        scheduled += len(actions)
        restraint += actions.count("OPTION_A")
        semantic_invalid += actions.count("INVALID")
        reserve = max(0, reserve - contract.reserve_cost * actions.count("OPTION_B"))
        deaths = _collapse_deaths(
            len(living), reserve, contract.collapse_death_rate
        )
        dead, _attrition_seed = _matched_attrition(
            living, deaths, environment_seed, day
        )
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
        "invalid_count": semantic_invalid,
    }
    for field, observed in derived.items():
        if not math.isclose(float(replayed[field]), float(observed), abs_tol=1e-12):
            raise OriginalViewFigureError(
                f"Selected Part 2 replayed {field} does not reconcile: {key}."
            )
    curve = {
        "trajectory_index": trajectory_index,
        "reserve": reserve_curve,
        "population": population_curve,
        "invalid_count": semantic_invalid,
    }
    if agent_actions is not None:
        curve["agent_actions"] = agent_actions
    return curve


@contextmanager
def _validated_pair_snapshot(
    pairs: Sequence[tuple[Path, Path]],
) -> Iterator[tuple[list[Any], Any]]:
    """Yield the fully replay-validated three-pair snapshot under shared locks."""

    if len(pairs) != 3 or any(
        not isinstance(pair, (tuple, list)) or len(pair) != 2
        for pair in pairs
    ):
        raise OriginalViewFigureError(
            "Part 2 original views require exactly three source/overlay pairs."
        )
    try:
        from analysis import (
            validate_inference_hub_part2_operational_overlays as overlay_validation,
        )
    except ImportError as error:
        raise OriginalViewFigureError(
            "The operational-overlay validator is required for three-pair Part 2 views."
        ) from error
    try:
        # The validator discovers and locks a cascading parent before opening
        # any journals, then recursively replays the incomplete parent and its
        # complete child.  Direct singleton overlays retain their original
        # strict validation path inside the same shared snapshot.
        with overlay_validation.validated_operational_overlay_pair_snapshot(
            [(Path(source), Path(overlay)) for source, overlay in pairs]
        ) as (validated_snapshot, audit, tracker):
            validated = list(validated_snapshot)
            if (
                audit.get("route_count") != EXPECTED_PART2_ROUTE_COUNT
                or audit.get("trajectory_count") != EXPECTED_PART2_TRAJECTORY_COUNT
                or audit.get("common_environment_seed_count")
                != EXPECTED_PART2_TRAJECTORIES_PER_ROUTE
            ):
                raise OriginalViewFigureError(
                    "Validated Part 2 evidence is not the exact 23-route/276-trajectory panel."
                )
            shard_sizes = sorted(len(pair.subjects) for pair in validated)
            singleton_ids = {
                str(pair.subjects[0]["target_id"])
                for pair in validated
                if len(pair.subjects) == 1
            }
            if shard_sizes != [1, 1, 21] or singleton_ids != set(
                EXPECTED_PART2_SINGLETON_SHARDS
            ):
                raise OriginalViewFigureError(
                    "Part 2 source/overlay pairs are not the exact main21 plus two singleton shards."
                )
            yield validated, tracker
    except OriginalViewFigureError:
        raise
    except (
        overlay_validation.Part2OperationalOverlayValidationError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise OriginalViewFigureError(str(error)) from error


def _read_bound_journal_reference(
    reference: Mapping[str, Any],
    *,
    expected_path: Path,
    label: str,
    tracker: Any,
) -> list[dict[str, Any]]:
    try:
        from analysis import (
            validate_inference_hub_part2_operational_overlays as overlay_validation,
        )
    except ImportError as error:
        raise OriginalViewFigureError(
            "The operational-overlay validator is required for selected-journal replay."
        ) from error
    return overlay_validation._read_journal_reference(
        reference,
        expected_path=expected_path,
        label=label,
        tracker=tracker,
    )


def _selected_journal_records(
    pair: Any,
    effective_row: Mapping[str, Any],
    tracker: Any,
) -> list[dict[str, Any]]:
    target_id = str(effective_row["target_id"])
    trajectory_index = int(effective_row["trajectory_index"])
    trajectory_key = (target_id, trajectory_index)
    replaced = effective_row.get("source_replaced_for_operational_failure")
    repair_round = effective_row.get("operational_repair_round")

    parent_reference = pair.overlay_manifest.get("parent_overlay_manifest")
    parent_manifest = getattr(pair, "parent_overlay_manifest", None)
    parent_path_value = getattr(pair, "parent_overlay_manifest_path", None)
    selected_cascading_key: tuple[str, int] | None = None
    if parent_reference is None:
        if parent_manifest is not None or parent_path_value is not None:
            raise OriginalViewFigureError(
                "Direct operational-repair lineage contains an unexpected parent."
            )
    else:
        if (
            not isinstance(parent_reference, Mapping)
            or set(parent_reference)
            != {"path", "file_sha256", "evidence_sha256"}
            or not isinstance(parent_reference.get("path"), str)
            or not isinstance(parent_manifest, Mapping)
            or parent_path_value is None
            or Path(str(parent_reference["path"])).resolve()
            != Path(parent_path_value).resolve()
        ):
            raise OriginalViewFigureError(
                "Cascading operational-repair parent lineage is unavailable or changed."
            )
        selected = pair.overlay_manifest.get("selected_trajectory")
        selected_target = (
            selected.get("target_id") if isinstance(selected, Mapping) else None
        )
        selected_index = (
            selected.get("trajectory_index")
            if isinstance(selected, Mapping)
            else None
        )
        if (
            not isinstance(selected_target, str)
            or isinstance(selected_index, bool)
            or not isinstance(selected_index, int)
            or selected_index < 0
        ):
            raise OriginalViewFigureError(
                "Cascading operational-repair selected lineage is malformed."
            )
        selected_cascading_key = (selected_target, selected_index)

    if replaced is True:
        repair_manifest = pair.overlay_manifest
        repair_manifest_path = pair.overlay_manifest_path
        label = "selected operational-repair trajectory journal"
        if (
            selected_cascading_key is not None
            and trajectory_key != selected_cascading_key
        ):
            repair_manifest = parent_manifest
            repair_manifest_path = Path(parent_path_value)
            label = "selected parent operational-repair trajectory journal"
        if (
            isinstance(repair_round, bool)
            or not isinstance(repair_round, int)
            or repair_round < 1
            or repair_round > int(repair_manifest["maximum_rounds"])
        ):
            raise OriginalViewFigureError(
                f"Effective repair lineage is malformed: {target_id}::{trajectory_index}."
            )
        reference_key = f"{target_id}::{trajectory_index}::{repair_round}"
        references = repair_manifest.get("journals")
        expected_path = (
            repair_manifest_path.parent
            / "trajectories"
            / _safe_file_stem(target_id)
            / f"seed-{trajectory_index:03d}-round-{repair_round:02d}.jsonl"
        )
    elif replaced is False and repair_round is None:
        reference_key = f"{target_id}::{trajectory_index}"
        references = pair.source_manifest.get("journals")
        expected_path = (
            pair.source_manifest_path.parent
            / "trajectories"
            / _safe_file_stem(target_id)
            / f"seed-{trajectory_index:03d}.jsonl"
        )
        label = "selected source trajectory journal"
    else:
        raise OriginalViewFigureError(
            f"Effective source lineage is malformed: {target_id}::{trajectory_index}."
        )
    if not isinstance(references, Mapping) or not isinstance(
        references.get(reference_key), Mapping
    ):
        raise OriginalViewFigureError(
            f"Effective selected journal binding is absent: {reference_key}."
        )
    records = _read_bound_journal_reference(
        references[reference_key],
        expected_path=expected_path,
        label=label,
        tracker=tracker,
    )
    if not records:
        raise OriginalViewFigureError(
            f"Effective selected journal is empty: {reference_key}."
        )
    return records


def _validated_composed_curves(
    pairs: Sequence[tuple[Path, Path]],
    public_trajectory_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], int, int, int]:
    public_by_key = _public_trajectory_index(public_trajectory_rows)
    if len(public_by_key) != EXPECTED_PART2_TRAJECTORY_COUNT:
        raise OriginalViewFigureError(
            "Public Part 2 aggregates are not exactly 276 unique trajectories."
        )

    subjects: list[dict[str, Any]] = []
    curves: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with _validated_pair_snapshot(pairs) as (validated_pairs, tracker):
        effective_rows = [
            dict(row)
            for pair in validated_pairs
            for row in pair.effective_trajectories
        ]
        effective_by_key = {
            (str(row["target_id"]), int(row["trajectory_index"])): row
            for row in effective_rows
        }
        if (
            len(effective_rows) != EXPECTED_PART2_TRAJECTORY_COUNT
            or len(effective_by_key) != EXPECTED_PART2_TRAJECTORY_COUNT
            or set(public_by_key) != set(effective_by_key)
        ):
            raise OriginalViewFigureError(
                "Public and private Part 2 trajectory panels differ from the exact 23 x 12 union."
            )
        for key, effective in effective_by_key.items():
            if dict(public_by_key[key]) != effective:
                raise OriginalViewFigureError(
                    f"Public effective Part 2 row does not match private replay binding: {key[0]}::{key[1]}."
                )

        for pair in validated_pairs:
            if (
                pair.contract.society_size != EXPECTED_PART2_SOCIETY_SIZE
                or pair.contract.days != EXPECTED_PART2_DAYS
                or pair.contract.trajectories
                != EXPECTED_PART2_TRAJECTORIES_PER_ROUTE
                or len(pair.environment_seeds)
                != EXPECTED_PART2_TRAJECTORIES_PER_ROUTE
            ):
                raise OriginalViewFigureError(
                    "Part 2 curves require the frozen 50-agent/100-day/12-seed contract."
                )
            execution = pair.source_manifest.get("execution_contract")
            if not isinstance(execution, Mapping):
                raise OriginalViewFigureError("A Part 2 source execution contract is absent.")
            pair_subjects = {
                str(subject["target_id"]): dict(subject)
                for subject in pair.subjects
            }
            subjects.extend(pair_subjects.values())
            for target_id, subject in pair_subjects.items():
                for trajectory_index, environment_seed in enumerate(
                    pair.environment_seeds
                ):
                    effective = effective_by_key[(target_id, trajectory_index)]
                    records = _selected_journal_records(pair, effective, tracker)
                    curves[target_id].append(
                        _curve_from_replayed_records(
                            records,
                            subject=subject,
                            trajectory_index=trajectory_index,
                            environment_seed=environment_seed,
                            contract=pair.contract,
                            execution_contract=execution,
                            effective_row=effective,
                        )
                    )

        if (
            len(subjects) != EXPECTED_PART2_ROUTE_COUNT
            or len({str(row["target_id"]) for row in subjects})
            != EXPECTED_PART2_ROUTE_COUNT
            or set(curves) != {str(row["target_id"]) for row in subjects}
            or any(
                len(group) != EXPECTED_PART2_TRAJECTORIES_PER_ROUTE
                for group in curves.values()
            )
        ):
            raise OriginalViewFigureError(
                "Replayed Part 2 curves are not exactly 23 routes by 12 trajectories."
            )
        contract = validated_pairs[0].contract
    return (
        subjects,
        curves,
        contract.days,
        contract.capacity,
        contract.society_size,
    )


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
    public_by_key = _public_trajectory_index(public_trajectory_rows)
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
            restraint = scheduled = semantic_invalid = 0
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
                semantic_invalid += actions.count("INVALID")
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
                "invalid_count": semantic_invalid,
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
                    "invalid_count": semantic_invalid,
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
        route_curves = list(curves[target_id])
        matrix = np.asarray([curve[metric] for curve in route_curves], dtype=float)
        if matrix.shape != (
            EXPECTED_PART2_TRAJECTORIES_PER_ROUTE,
            days,
        ) or not np.all(np.isfinite(matrix)):
            raise OriginalViewFigureError(
                f"{target_id} lacks the 12 x {days} curve matrix."
            )
        eligible: list[bool] = []
        for curve in route_curves:
            invalid_count = curve.get("invalid_count")
            if (
                isinstance(invalid_count, bool)
                or not isinstance(invalid_count, int)
                or invalid_count < 0
            ):
                raise OriginalViewFigureError(
                    f"{target_id} has a malformed semantic-invalid trajectory count."
                )
            eligible.append(invalid_count == 0)
        eligible_count = sum(eligible)
        label = _route_legend_label(row)
        if eligible_count == 0:
            label += " · NE (0 valid trajectories)"
        elif eligible_count < EXPECTED_PART2_TRAJECTORIES_PER_ROUTE:
            label += f" · n={eligible_count} valid"
        mean = (
            matrix[np.asarray(eligible, dtype=bool)].mean(axis=0)
            if eligible_count
            else None
        )
        ax.plot(
            x if mean is not None else [],
            mean if mean is not None else [],
            label=label,
            color=_provider_color(row),
            linestyle=LINE_STYLES[provider_index % len(LINE_STYLES)],
            marker=MARKERS[
                (provider_index // len(LINE_STYLES)) % len(MARKERS)
            ],
            markevery=(
                provider_index % max(1, days // 10),
                max(1, days // 10),
            ),
            markersize=2.9,
            linewidth=1.45,
            alpha=0.94,
        )
    ax.set_xlim(1, days)
    ax.set_xticks(_day_ticks(days))
    ax.set_ylim(0, maximum * 1.04)
    ax.set_xlabel("Simulation day")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=20)
    ax.text(
        0.0,
        1.025,
        "Means use fully valid trajectories only; semantic-invalid trajectories "
        "are excluded. Reduced n or NE appears in the legend.",
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
        bbox_to_anchor=(0.5, -0.15),
        ncol=5,
        frameon=False,
        fontsize=6.6,
        handlelength=2.35,
        columnspacing=1.0,
        labelspacing=0.75,
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.87, bottom=0.33)
    return _atomic_save(fig, output_dir, stem, title)


ACTION_CODES = {
    "OPTION_B": 0,
    "OPTION_A": 1,
    "NO_ACTIVE_DECISION": 2,
    "INVALID": 3,
}


def _agent_day_raster_data(
    rows: Sequence[Mapping[str, Any]],
    curves: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    days: int,
    society_size: int,
) -> tuple[list[Mapping[str, Any]], np.ndarray]:
    """Build the deterministic route/agent/day matrix for fixed seed index 0."""

    ordered = list(_provider_grouped_rows(rows))
    raster = np.full(
        (len(ordered) * society_size, days),
        ACTION_CODES["NO_ACTIVE_DECISION"],
        dtype=np.uint8,
    )
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
        try:
            raster[start : start + society_size, :] = np.asarray(
                [[ACTION_CODES[action] for action in agent] for agent in actions],
                dtype=np.uint8,
            )
        except KeyError as error:
            raise OriginalViewFigureError(
                f"{target_id} has an unknown action in its seed-index-0 raster."
            ) from error
    return ordered, raster


def _agent_day_raster(
    rows: Sequence[Mapping[str, Any]],
    curves: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    days: int,
    society_size: int,
    output_dir: Path,
) -> list[Path]:
    """Render seed zero as readable route-level 50-by-100 small multiples."""

    ordered, raster = _agent_day_raster_data(
        rows, curves, days=days, society_size=society_size
    )
    cmap = ListedColormap([RED, GREEN, "#E5E7EB", ORANGE])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    columns = min(4, max(1, len(ordered)))
    panel_rows = math.ceil(len(ordered) / columns)
    fig, axes = plt.subplots(
        panel_rows,
        columns,
        figsize=(11.2, 1.72 * panel_rows + 1.85),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    fig.patch.set_facecolor("white")
    day_ticks = _day_ticks(days)
    tick_positions = [day - 1 for day in day_ticks]
    last_row_by_column = {
        column: max(
            index // columns
            for index in range(len(ordered))
            if index % columns == column
        )
        for column in range(min(columns, len(ordered)))
    }
    agent_ticks = list(
        dict.fromkeys((0, (society_size - 1) // 2, society_size - 1))
    )
    for index, row in enumerate(ordered):
        panel_row, column = divmod(index, columns)
        ax = axes[panel_row, column]
        start = index * society_size
        ax.imshow(
            raster[start : start + society_size],
            cmap=cmap,
            norm=norm,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(
            _route_legend_label(row),
            loc="left",
            fontsize=7.4,
            fontweight="bold",
            color=_provider_color(row),
            pad=3.0,
        )
        ax.set_xticks(tick_positions)
        show_days = panel_row == last_row_by_column[column]
        ax.tick_params(
            axis="x",
            labelbottom=show_days,
            bottom=show_days,
            length=2.0,
            pad=1.5,
            labelsize=5.8,
        )
        if show_days:
            ax.set_xticklabels([str(day) for day in day_ticks])
        ax.set_yticks(agent_ticks)
        ax.tick_params(
            axis="y",
            labelleft=column == 0,
            left=column == 0,
            length=2.0,
            pad=1.5,
            labelsize=5.8,
        )
        if column == 0:
            ax.set_yticklabels([str(slot + 1) for slot in agent_ticks])
        for position in tick_positions:
            ax.axvline(position - 0.5, color="white", linewidth=0.35, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color(_provider_color(row))
            spine.set_linewidth(0.7)
    for index in range(len(ordered), panel_rows * columns):
        axes.flat[index].set_visible(False)

    fig.suptitle(
        "Agent-day actions in one prespecified common-seed trajectory",
        x=0.075,
        ha="left",
        y=0.985,
        fontsize=12.5,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.955,
        (
            f"All {len(ordered)} routes × {society_size} initially living agents "
            f"× {days} days; seed index 0; each tile uses the same scales."
        ),
        color=MUTED,
        fontsize=7.4,
    )
    fig.supxlabel("Simulation day", fontsize=8.0, y=0.072)
    fig.supylabel("Initial agent slot", fontsize=8.0, x=0.025)
    fig.legend(
        handles=[
            Patch(facecolor=GREEN, label="Restraint (OPTION_A)"),
            Patch(facecolor=RED, label="Overuse (OPTION_B)"),
            Patch(facecolor="#E5E7EB", label="No active decision after attrition"),
            Patch(facecolor=ORANGE, label="Semantic invalid (zero state effect)"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        fontsize=7.0,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.992,
        top=0.925,
        bottom=0.115,
        wspace=0.11,
        hspace=0.48,
    )
    return _atomic_save(
        fig,
        output_dir,
        "part2_agent_day_raster_current",
        "Agent-day actions in one prespecified common-seed trajectory",
    )


def build_original_view_figures(
    analysis_dir: Path,
    part2_manifest: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    part2_source_overlay_pairs: Sequence[tuple[Path, Path]] = (),
) -> list[Path]:
    analysis_dir = analysis_dir.resolve()
    output_dir = output_dir.resolve()
    pairs = list(part2_source_overlay_pairs)
    if part2_manifest is not None:
        raise OriginalViewFigureError(
            "Legacy Part 2 manifests cannot generate current 100-day paper views."
        )
    if not pairs:
        pairs = list(DEFAULT_PART2_SOURCE_OVERLAY_PAIRS)

    data = _load_and_validate(analysis_dir)
    topology = data.get("topology")
    if not isinstance(topology, Mapping):
        raise OriginalViewFigureError(
            "The analysis lacks a validated Part 2 composition topology."
        )
    _validate_analysis_composition_binding(pairs, topology)
    aggregates = json.loads(
        (analysis_dir / "figure_aggregates.json").read_text(encoding="utf-8")
    )
    public_trajectories = aggregates.get("part2_trajectories")
    if not isinstance(public_trajectories, list):
        raise OriginalViewFigureError("Public Part 2 trajectory aggregates are absent.")
    subjects, curves, days, capacity, society_size = _validated_composed_curves(
        pairs, public_trajectories
    )
    # Recheck the six direct bindings and optional cascading parent after the
    # private replay snapshot, before rendering or replacing any public file.
    _validate_analysis_composition_binding(pairs, topology)

    subject_by_id = {str(row["target_id"]): row for row in subjects}
    part2_by_id: dict[str, Mapping[str, Any]] = {}
    for row in data["part2"]:
        target_id = str(row["target_id"])
        if target_id in part2_by_id:
            raise OriginalViewFigureError(
                f"A public Part 2 model row is duplicated: {target_id}."
            )
        part2_by_id[target_id] = row
    if not subject_by_id or set(part2_by_id) != set(subject_by_id):
        raise OriginalViewFigureError("Public and private Part 2 target panels differ.")
    for target_id, row in part2_by_id.items():
        subject = subject_by_id[target_id]
        if (
            row.get("upstream_provider") != subject.get("upstream_provider")
            or row.get("model") != subject.get("model")
        ):
            raise OriginalViewFigureError(
                f"Public and private Part 2 route identity differs: {target_id}."
            )
        for metric in ("reserve", "population"):
            matrix = np.asarray(
                [curve[metric] for curve in curves[target_id]], dtype=float
            )
            if matrix.shape != (EXPECTED_PART2_TRAJECTORIES_PER_ROUTE, days):
                raise OriginalViewFigureError(
                    f"{target_id} lacks the 12 x {days} {metric} curve matrix."
                )
    part2_rows = list(part2_by_id.values())
    # Materialize the raster matrix before any output is replaced so malformed
    # route/seed/action topology fails the build transactionally.
    _agent_day_raster_data(
        part2_rows, curves, days=days, society_size=society_size
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.stage.",
    ) as staging_name:
        staging_dir = Path(staging_name)
        staged_outputs: list[Path] = []
        staged_outputs.extend(
            _bar_chart(
                data["part0"],
                value_field="refusal_rate_all_scheduled",
                low_field="refusal_rate_all_scheduled_finite_bank_sensitivity_low",
                high_field="refusal_rate_all_scheduled_finite_bank_sensitivity_high",
                ylabel="Material-refusal rate (%)",
                title="Safety refusal by current exact model route",
                subtitle="Whiskers: 5,000-replicate harmful-root sensitivity intervals; higher means less harmful-request assistance.",
                stem="part0_refusal_rate_by_model",
                output_dir=staging_dir,
            )
        )
        staged_outputs.extend(
            _bar_chart(
                part2_rows,
                value_field="mean_trajectory_restraint_rate_all_scheduled",
                low_field="mean_trajectory_restraint_rate_all_scheduled_t95_low",
                high_field="mean_trajectory_restraint_rate_all_scheduled_t95_high",
                ylabel="Mean trajectory restraint rate (%)",
                title="Commons restraint by current exact model route",
                subtitle="Whiskers: Student-t 95% intervals over 12 matched trajectories; higher means less commons overuse.",
                stem="part2_restraint_rate_by_model",
                output_dir=staging_dir,
            )
        )
        staged_outputs.extend(
            _line_chart(
                part2_rows,
                curves,
                metric="reserve",
                days=days,
                maximum=capacity,
                ylabel="Shared reserve units",
                title="Shared reserve over time by current model route",
                stem="part2_shared_reserve_over_time",
                output_dir=staging_dir,
            )
        )
        staged_outputs.extend(
            _line_chart(
                part2_rows,
                curves,
                metric="population",
                days=days,
                maximum=society_size,
                ylabel="Living population",
                title="Living population over time by current model route",
                stem="part2_population_over_time",
                output_dir=staging_dir,
            )
        )
        staged_outputs.extend(
            _agent_day_raster(
                part2_rows,
                curves,
                days=days,
                society_size=society_size,
                output_dir=staging_dir,
            )
        )
        if [path.name for path in staged_outputs] != list(
            EXPECTED_OUTPUT_BASENAMES
        ) or any(path.parent.resolve() != staging_dir.resolve() for path in staged_outputs):
            raise OriginalViewFigureError(
                "Original-view renderers returned an unexpected staged output set."
            )
        return _publish_staged_outputs(staging_dir, output_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument(
        "--part2-manifest",
        type=Path,
        help=(
            "Legacy single COMPLETE Part 2 manifest. If omitted, the definitive "
            "three source/overlay pairs are used."
        ),
    )
    parser.add_argument(
        "--part2-source-overlay",
        "--part2-source-overlay-pair",
        dest="part2_source_overlay_pairs",
        action="append",
        nargs=2,
        type=Path,
        default=[],
        metavar=("SOURCE_MANIFEST", "OVERLAY_MANIFEST"),
        help=(
            "Terminalized Part 2 source and its COMPLETE operational overlay; "
            "repeat exactly three times. Pair order does not affect route order."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        paths = build_original_view_figures(
            args.analysis_dir,
            args.part2_manifest,
            args.output_dir,
            part2_source_overlay_pairs=[
                (source, overlay)
                for source, overlay in args.part2_source_overlay_pairs
            ],
        )
    except (OriginalViewFigureError, OSError, ValueError) as error:
        print(f"Original-view figure generation failed: {error}")
        return 1
    for path in paths:
        print(f"wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
