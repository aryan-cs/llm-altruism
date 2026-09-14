from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import threading
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
import pytest

from analysis import build_original_view_figures as original_views
from experiments.misc import inference_hub_part2_panel as runner
from experiments.misc.inference_hub_part1_panel import _sha256_file, _sha256_json


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "experiments/sota_cross_axis_part2_100day_panel.json"


def _target_ids() -> list[str]:
    panel = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    return [
        target_id
        for target_id in panel["subject_target_ids"]
        if target_id != "anthropic/claude-opus-4-5"
    ]


def _subject(target_id: str) -> dict[str, Any]:
    provider, model = target_id.split("/", 1)
    return {
        "target_id": target_id,
        "upstream_provider": provider,
        "model": model,
        "route": f"region/{model}",
    }


def _effective_row(
    subject: dict[str, Any], trajectory_index: int, *, repaired: bool = False
) -> dict[str, Any]:
    return {
        "target_id": subject["target_id"],
        "upstream_provider": subject["upstream_provider"],
        "model": subject["model"],
        "trajectory_index": trajectory_index,
        "environment_seed_index": trajectory_index,
        "environment_seed": 10_000 + trajectory_index,
        "operationally_eligible": True,
        "operational_repair_round": 2 if repaired else None,
        "source_replaced_for_operational_failure": repaired,
    }


def _production_shaped_pairs() -> list[SimpleNamespace]:
    target_ids = _target_ids()
    singleton_ids = set(original_views.EXPECTED_PART2_SINGLETON_SHARDS)
    shards = [
        [target_id for target_id in target_ids if target_id not in singleton_ids],
        ["nvidia/nemotron-3-ultra"],
        ["deepseek-ai/deepseek-v4-flash"],
    ]
    contract = runner.Part2Contract(50, 100, 12, 2500, 2, 2, 5, 0.2)
    pairs: list[SimpleNamespace] = []
    for shard_index, target_shard in enumerate(shards):
        subjects = tuple(_subject(target_id) for target_id in target_shard)
        effective = tuple(
            _effective_row(
                subject,
                trajectory_index,
                repaired=(
                    subject["target_id"] == "nvidia/nemotron-3-ultra"
                    and trajectory_index == 4
                ),
            )
            for subject in subjects
            for trajectory_index in range(12)
        )
        pairs.append(
            SimpleNamespace(
                source_manifest_path=Path(
                    f"/source-{shard_index}/private/manifest.json"
                ),
                overlay_manifest_path=Path(
                    f"/overlay-{shard_index}/private/manifest.json"
                ),
                source_manifest={"execution_contract": {"offline": True}},
                overlay_manifest={"maximum_rounds": 8},
                subjects=subjects,
                contract=contract,
                environment_seeds=tuple(10_000 + index for index in range(12)),
                effective_trajectories=effective,
            )
        )
    return pairs


def _synthetic_view_data() -> tuple[
    list[dict[str, Any]], dict[str, list[dict[str, Any]]]
]:
    rows = [_subject(target_id) for target_id in _target_ids()]
    curves: dict[str, list[dict[str, Any]]] = {}
    for route_index, row in enumerate(rows):
        target_id = row["target_id"]
        route_curves: list[dict[str, Any]] = []
        for trajectory_index in range(12):
            curve: dict[str, Any] = {
                "trajectory_index": trajectory_index,
                "invalid_count": 0,
                "reserve": [
                    max(0, 2500 - 2 * (day + route_index + trajectory_index))
                    for day in range(100)
                ],
                "population": [
                    max(0, 50 - (day // 25 if route_index % 5 == 0 else 0))
                    for day in range(100)
                ],
            }
            if trajectory_index == 0:
                actions: list[list[str]] = []
                for slot in range(50):
                    agent: list[str] = []
                    for day in range(100):
                        if target_id == "deepseek-ai/deepseek-v4-flash" and (
                            slot + day
                        ) % 3:
                            action = "INVALID"
                        elif slot >= 45 and day >= 90:
                            action = "NO_ACTIVE_DECISION"
                        elif (route_index + slot + day) % 11 == 0:
                            action = "OPTION_B"
                        else:
                            action = "OPTION_A"
                        agent.append(action)
                    actions.append(agent)
                curve["agent_actions"] = actions
                curve["invalid_count"] = sum(
                    action == "INVALID"
                    for agent in actions
                    for action in agent
                )
            route_curves.append(curve)
        curves[target_id] = route_curves
    return rows, curves


def test_composed_curves_require_exact_23_by_12_and_replay_all_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = _production_shaped_pairs()
    # Pair ordering is deliberately different from the default CLI order.
    pairs = [pairs[2], pairs[0], pairs[1]]
    public_rows = [
        dict(row) for pair in pairs for row in pair.effective_trajectories
    ]

    @contextmanager
    def snapshot(
        _paths: Any,
    ) -> Iterator[tuple[list[SimpleNamespace], object]]:
        yield pairs, object()

    selected: list[tuple[str, int, bool]] = []

    def records(pair: Any, row: dict[str, Any], tracker: Any) -> list[dict[str, Any]]:
        del pair, tracker
        selected.append(
            (
                row["target_id"],
                row["trajectory_index"],
                row["source_replaced_for_operational_failure"],
            )
        )
        return [{"selected": True}]

    def replay(
        _records: Any,
        *,
        subject: dict[str, Any],
        trajectory_index: int,
        environment_seed: int,
        contract: runner.Part2Contract,
        execution_contract: dict[str, Any],
        effective_row: dict[str, Any],
    ) -> dict[str, Any]:
        del environment_seed, execution_contract, effective_row
        curve: dict[str, Any] = {
            "trajectory_index": trajectory_index,
            "reserve": [contract.capacity] * contract.days,
            "population": [contract.society_size] * contract.days,
        }
        if trajectory_index == 0:
            curve["agent_actions"] = [
                ["OPTION_A"] * contract.days
                for _ in range(contract.society_size)
            ]
        assert subject["target_id"] in _target_ids()
        return curve

    monkeypatch.setattr(original_views, "_validated_pair_snapshot", snapshot)
    monkeypatch.setattr(original_views, "_selected_journal_records", records)
    monkeypatch.setattr(original_views, "_curve_from_replayed_records", replay)

    subjects, curves, days, capacity, society_size = (
        original_views._validated_composed_curves(
            [(Path("source"), Path("overlay"))] * 3,
            public_rows,
        )
    )

    assert len(subjects) == 23
    assert len(curves) == 23
    assert sum(len(group) for group in curves.values()) == 276
    assert (days, capacity, society_size) == (100, 2500, 50)
    assert len(selected) == 276
    assert selected.count(("nvidia/nemotron-3-ultra", 4, True)) == 1
    assert sum(repaired for _, _, repaired in selected) == 1


def test_composed_curves_reject_duplicate_missing_and_wrong_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = _production_shaped_pairs()
    public_rows = [
        dict(row) for pair in pairs for row in pair.effective_trajectories
    ]

    with pytest.raises(
        original_views.OriginalViewFigureError, match="duplicated"
    ):
        original_views._validated_composed_curves(
            [(Path("source"), Path("overlay"))] * 3,
            [*public_rows, dict(public_rows[0])],
        )
    with pytest.raises(
        original_views.OriginalViewFigureError, match="exactly 276"
    ):
        original_views._validated_composed_curves(
            [(Path("source"), Path("overlay"))] * 3,
            public_rows[:-1],
        )

    pairs[0] = SimpleNamespace(
        **{
            **vars(pairs[0]),
            "contract": runner.Part2Contract(5, 100, 12, 2500, 2, 2, 5, 0.2),
        }
    )

    @contextmanager
    def snapshot(
        _paths: Any,
    ) -> Iterator[tuple[list[SimpleNamespace], object]]:
        yield pairs, object()

    monkeypatch.setattr(original_views, "_validated_pair_snapshot", snapshot)
    with pytest.raises(
        original_views.OriginalViewFigureError, match="50-agent/100-day/12-seed"
    ):
        original_views._validated_composed_curves(
            [(Path("source"), Path("overlay"))] * 3,
            public_rows,
        )


def test_pair_gate_rejects_any_count_other_than_three() -> None:
    with pytest.raises(
        original_views.OriginalViewFigureError, match="exactly three"
    ):
        with original_views._validated_pair_snapshot(
            [(Path("source"), Path("overlay"))] * 2
        ):
            raise AssertionError("invalid pair count must never yield")


def test_cli_accepts_three_repeatable_pairs_and_rejects_legacy_paper_mode(
    tmp_path: Path,
) -> None:
    arguments: list[str] = []
    expected: list[list[Path]] = []
    for index in range(3):
        source = Path(f"source-{index}/private/manifest.json")
        overlay = Path(f"overlay-{index}/private/manifest.json")
        arguments.extend(
            ["--part2-source-overlay", str(source), str(overlay)]
        )
        expected.append([source, overlay])
    parsed = original_views._parser().parse_args(arguments)

    assert parsed.part2_manifest is None
    assert parsed.part2_source_overlay_pairs == expected
    with pytest.raises(
        original_views.OriginalViewFigureError, match="Legacy Part 2"
    ):
        original_views.build_original_view_figures(
            tmp_path,
            Path("legacy/private/manifest.json"),
            tmp_path / "figures",
            part2_source_overlay_pairs=[
                (Path("source"), Path("overlay"))
            ]
            * 3,
        )


def test_effective_lineage_selects_source_or_declared_repair_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_ref = {"which": "source"}
    repair_ref = {"which": "repair"}
    pair = SimpleNamespace(
        source_manifest_path=tmp_path / "source/private/manifest.json",
        overlay_manifest_path=tmp_path / "overlay/private/manifest.json",
        source_manifest={"journals": {"route/model::3": source_ref}},
        overlay_manifest={
            "maximum_rounds": 4,
            "journals": {"route/model::3::2": repair_ref},
        },
    )
    observed: list[tuple[dict[str, str], Path, str]] = []

    def read(
        reference: dict[str, str],
        *,
        expected_path: Path,
        label: str,
        tracker: object,
    ) -> list[dict[str, Any]]:
        del tracker
        observed.append((reference, expected_path, label))
        return [{"record": True}]

    monkeypatch.setattr(original_views, "_read_bound_journal_reference", read)
    base = {
        "target_id": "route/model",
        "trajectory_index": 3,
    }
    original_views._selected_journal_records(
        pair,
        {
            **base,
            "source_replaced_for_operational_failure": False,
            "operational_repair_round": None,
        },
        object(),
    )
    original_views._selected_journal_records(
        pair,
        {
            **base,
            "source_replaced_for_operational_failure": True,
            "operational_repair_round": 2,
        },
        object(),
    )

    assert observed[0] == (
        source_ref,
        tmp_path / "source/private/trajectories/route_model/seed-003.jsonl",
        "selected source trajectory journal",
    )
    assert observed[1] == (
        repair_ref,
        tmp_path
        / "overlay/private/trajectories/route_model/seed-003-round-02.jsonl",
        "selected operational-repair trajectory journal",
    )

    with pytest.raises(
        original_views.OriginalViewFigureError, match="lineage is malformed"
    ):
        original_views._selected_journal_records(
            pair,
            {
                **base,
                "source_replaced_for_operational_failure": True,
                "operational_repair_round": 5,
            },
            object(),
        )


def test_cascading_effective_lineage_selects_parent_and_child_journals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_manifest_path = tmp_path / "parent/private/manifest.json"
    child_manifest_path = tmp_path / "child/private/manifest.json"
    parent_ref = {"which": "parent"}
    child_ref = {"which": "child"}
    pair = SimpleNamespace(
        source_manifest_path=tmp_path / "source/private/manifest.json",
        overlay_manifest_path=child_manifest_path,
        source_manifest={"journals": {}},
        overlay_manifest={
            "maximum_rounds": 8,
            "parent_overlay_manifest": {
                "path": str(parent_manifest_path),
                "file_sha256": "1" * 64,
                "evidence_sha256": "2" * 64,
            },
            "selected_trajectory": {
                "target_id": "route/selected",
                "trajectory_index": 1,
            },
            "journals": {"route/selected::1::3": child_ref},
        },
        parent_overlay_manifest_path=parent_manifest_path,
        parent_overlay_manifest={
            "maximum_rounds": 8,
            "journals": {"route/inherited::2::4": parent_ref},
        },
    )
    observed: list[tuple[dict[str, str], Path, str]] = []

    def read(
        reference: dict[str, str],
        *,
        expected_path: Path,
        label: str,
        tracker: object,
    ) -> list[dict[str, Any]]:
        del tracker
        observed.append((reference, expected_path, label))
        return [{"record": True}]

    monkeypatch.setattr(original_views, "_read_bound_journal_reference", read)
    original_views._selected_journal_records(
        pair,
        {
            "target_id": "route/inherited",
            "trajectory_index": 2,
            "source_replaced_for_operational_failure": True,
            "operational_repair_round": 4,
        },
        object(),
    )
    original_views._selected_journal_records(
        pair,
        {
            "target_id": "route/selected",
            "trajectory_index": 1,
            "source_replaced_for_operational_failure": True,
            "operational_repair_round": 3,
        },
        object(),
    )

    assert observed == [
        (
            parent_ref,
            tmp_path
            / "parent/private/trajectories/route_inherited/seed-002-round-04.jsonl",
            "selected parent operational-repair trajectory journal",
        ),
        (
            child_ref,
            tmp_path
            / "child/private/trajectories/route_selected/seed-001-round-03.jsonl",
            "selected operational-repair trajectory journal",
        ),
    ]


@pytest.mark.parametrize("tamper", ["parent_path", "selected_trajectory"])
def test_cascading_effective_lineage_rejects_adversarial_binding(
    tmp_path: Path, tamper: str
) -> None:
    parent_manifest_path = tmp_path / "parent/private/manifest.json"
    overlay_manifest: dict[str, Any] = {
        "maximum_rounds": 8,
        "parent_overlay_manifest": {
            "path": str(parent_manifest_path),
            "file_sha256": "1" * 64,
            "evidence_sha256": "2" * 64,
        },
        "selected_trajectory": {
            "target_id": "route/selected",
            "trajectory_index": 1,
        },
        "journals": {},
    }
    if tamper == "parent_path":
        overlay_manifest["parent_overlay_manifest"]["path"] = str(
            tmp_path / "other/private/manifest.json"
        )
        expected = "parent lineage is unavailable or changed"
    else:
        overlay_manifest["selected_trajectory"]["trajectory_index"] = True
        expected = "selected lineage is malformed"
    pair = SimpleNamespace(
        source_manifest_path=tmp_path / "source/private/manifest.json",
        overlay_manifest_path=tmp_path / "child/private/manifest.json",
        source_manifest={"journals": {}},
        overlay_manifest=overlay_manifest,
        parent_overlay_manifest_path=parent_manifest_path,
        parent_overlay_manifest={"maximum_rounds": 8, "journals": {}},
    )

    with pytest.raises(original_views.OriginalViewFigureError, match=expected):
        original_views._selected_journal_records(
            pair,
            {
                "target_id": "route/inherited",
                "trajectory_index": 2,
                "source_replaced_for_operational_failure": True,
                "operational_repair_round": 4,
            },
            object(),
        )


def test_100_day_ticks_line_legend_and_provider_encodings_are_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, curves = _synthetic_view_data()
    captured: list[Any] = []

    def capture(
        fig: Any, directory: Path, stem: str, title: str
    ) -> list[Path]:
        del title
        captured.append(fig)
        return [directory / f"{stem}.pdf", directory / f"{stem}.png"]

    monkeypatch.setattr(original_views, "_atomic_save", capture)
    paths = original_views._line_chart(
        rows,
        curves,
        metric="reserve",
        days=100,
        maximum=2500,
        ylabel="Shared reserve units",
        title="Shared reserve",
        stem="part2_shared_reserve_over_time",
        output_dir=tmp_path,
    )

    assert [path.name for path in paths] == [
        "part2_shared_reserve_over_time.pdf",
        "part2_shared_reserve_over_time.png",
    ]
    assert original_views._day_ticks(100) == [1, 25, 50, 75, 100]
    fig = captured[0]
    fig.canvas.draw()
    ax = fig.axes[0]
    assert ax.get_xticks().tolist() == [1, 25, 50, 75, 100]
    legend = ax.get_legend()
    labels = [text.get_text() for text in legend.get_texts()]
    assert len(labels) == len(set(labels)) == 23
    renderer = fig.canvas.get_renderer()
    assert not legend.get_window_extent(renderer).overlaps(
        ax.get_window_extent(renderer)
    )
    ordered = list(original_views._provider_grouped_rows(rows))
    openai_encodings = {
        (line.get_linestyle(), line.get_marker())
        for line, row in zip(ax.lines, ordered, strict=True)
        if row["upstream_provider"] == "openai"
    }
    assert len(openai_encodings) == 7
    plt.close(fig)


def test_line_curves_exclude_invalid_trajectories_and_mark_zero_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {
            "target_id": "openai/partially-valid",
            "upstream_provider": "openai",
            "model": "partially-valid",
        },
        {
            "target_id": "deepseek-ai/no-valid-trajectories",
            "upstream_provider": "deepseek-ai",
            "model": "no-valid-trajectories",
        },
    ]
    curves = {
        "openai/partially-valid": [
            {
                "trajectory_index": trajectory_index,
                "invalid_count": int(trajectory_index == 11),
                "reserve": [999.0] * 4 if trajectory_index == 11 else [100.0] * 4,
                "population": [1.0] * 4,
            }
            for trajectory_index in range(12)
        ],
        "deepseek-ai/no-valid-trajectories": [
            {
                "trajectory_index": trajectory_index,
                "invalid_count": 1,
                "reserve": [500.0 + trajectory_index] * 4,
                "population": [1.0] * 4,
            }
            for trajectory_index in range(12)
        ],
    }
    captured: list[Any] = []

    def capture(
        fig: Any, directory: Path, stem: str, title: str
    ) -> list[Path]:
        del title
        captured.append(fig)
        return [directory / f"{stem}.pdf", directory / f"{stem}.png"]

    monkeypatch.setattr(original_views, "_atomic_save", capture)
    original_views._line_chart(
        rows,
        curves,
        metric="reserve",
        days=4,
        maximum=1000,
        ylabel="Shared reserve units",
        title="Shared reserve",
        stem="part2_shared_reserve_over_time",
        output_dir=tmp_path,
    )

    ax = captured[0].axes[0]
    lines = {line.get_label(): line for line in ax.lines}
    eligible_label = next(label for label in lines if "n=11 valid" in label)
    zero_label = next(label for label in lines if "NE (0 valid trajectories)" in label)
    np.testing.assert_allclose(lines[eligible_label].get_ydata(), [100.0] * 4)
    assert len(lines[zero_label].get_xdata()) == 0
    assert len(lines[zero_label].get_ydata()) == 0
    assert any(
        "semantic-invalid trajectories are excluded" in text.get_text()
        for text in ax.texts
    )
    plt.close(captured[0])


def test_line_curves_require_explicit_semantic_invalid_counts(
    tmp_path: Path,
) -> None:
    row = {
        "target_id": "openai/missing-validity",
        "upstream_provider": "openai",
        "model": "missing-validity",
    }
    curves = {
        row["target_id"]: [
            {
                "trajectory_index": trajectory_index,
                "reserve": [100.0] * 4,
            }
            for trajectory_index in range(12)
        ]
    }

    with pytest.raises(
        original_views.OriginalViewFigureError,
        match="malformed semantic-invalid trajectory count",
    ):
        original_views._line_chart(
            [row],
            curves,
            metric="reserve",
            days=4,
            maximum=100,
            ylabel="Shared reserve units",
            title="Shared reserve",
            stem="part2_shared_reserve_over_time",
            output_dir=tmp_path,
        )


def test_agent_day_matrix_is_exact_deterministic_and_retains_invalids() -> None:
    rows, curves = _synthetic_view_data()
    ordered_a, raster_a = original_views._agent_day_raster_data(
        list(reversed(rows)), curves, days=100, society_size=50
    )
    ordered_b, raster_b = original_views._agent_day_raster_data(
        rows, curves, days=100, society_size=50
    )

    assert [row["target_id"] for row in ordered_a] == [
        row["target_id"] for row in ordered_b
    ]
    assert raster_a.shape == (1150, 100)
    assert raster_a.dtype == np.uint8
    assert np.array_equal(raster_a, raster_b)
    assert set(np.unique(raster_a)) == set(original_views.ACTION_CODES.values())
    deepseek_index = next(
        index
        for index, row in enumerate(ordered_a)
        if row["target_id"] == "deepseek-ai/deepseek-v4-flash"
    )
    deepseek = raster_a[deepseek_index * 50 : (deepseek_index + 1) * 50]
    assert np.count_nonzero(deepseek == original_views.ACTION_CODES["INVALID"]) > 0


def test_agent_day_small_multiples_have_23_legible_50_by_100_tiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, curves = _synthetic_view_data()
    captured: list[Any] = []

    def capture(
        fig: Any, directory: Path, stem: str, title: str
    ) -> list[Path]:
        del title
        captured.append(fig)
        return [directory / f"{stem}.pdf", directory / f"{stem}.png"]

    monkeypatch.setattr(original_views, "_atomic_save", capture)
    outputs = original_views._agent_day_raster(
        rows, curves, days=100, society_size=50, output_dir=tmp_path
    )

    assert [path.name for path in outputs] == [
        "part2_agent_day_raster_current.pdf",
        "part2_agent_day_raster_current.png",
    ]
    fig = captured[0]
    fig.canvas.draw()
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible_axes) == 23
    assert all(len(ax.images) == 1 for ax in visible_axes)
    assert all(ax.images[0].get_array().shape == (50, 100) for ax in visible_axes)
    assert any(
        [text.get_text() for text in ax.get_xticklabels()]
        == ["1", "25", "50", "75", "100"]
        for ax in visible_axes
    )
    legend_labels = [text.get_text() for text in fig.legends[0].get_texts()]
    assert legend_labels == [
        "Restraint (OPTION_A)",
        "Overuse (OPTION_B)",
        "No active decision after attrition",
        "Semantic invalid (zero state effect)",
    ]
    plt.close(fig)


def test_build_validates_private_evidence_before_writing_any_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "figure_aggregates.json").write_text(
        json.dumps({"part2_trajectories": []}) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        original_views,
        "_load_and_validate",
        lambda _path: {
            "part0": [],
            "part2": [],
            "topology": {"mode": "three_pair_operational_repair_composition"},
        },
    )
    monkeypatch.setattr(
        original_views, "_validate_analysis_composition_binding", lambda *_args: None
    )
    monkeypatch.setattr(
        original_views,
        "_validated_composed_curves",
        lambda *_args: (_ for _ in ()).throw(
            original_views.OriginalViewFigureError("overlay incomplete")
        ),
    )
    writes: list[str] = []
    monkeypatch.setattr(
        original_views,
        "_bar_chart",
        lambda *_args, **_kwargs: writes.append("bar") or [],
    )
    output_dir = tmp_path / "figures"

    with pytest.raises(
        original_views.OriginalViewFigureError, match="overlay incomplete"
    ):
        original_views.build_original_view_figures(
            analysis_dir,
            None,
            output_dir,
            part2_source_overlay_pairs=[
                (Path("source"), Path("overlay"))
            ]
            * 3,
        )

    assert writes == []
    assert not output_dir.exists()


def test_build_preserves_all_ten_original_view_filenames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "figure_aggregates.json").write_text(
        json.dumps({"part2_trajectories": []}) + "\n", encoding="utf-8"
    )
    part2_rows, curves = _synthetic_view_data()
    part0_rows = [{"target_id": "part0/model"}]
    monkeypatch.setattr(
        original_views,
        "_load_and_validate",
        lambda _path: {
            "part0": part0_rows,
            "part2": part2_rows,
            "topology": {"mode": "three_pair_operational_repair_composition"},
        },
    )
    monkeypatch.setattr(
        original_views, "_validate_analysis_composition_binding", lambda *_args: None
    )
    monkeypatch.setattr(
        original_views,
        "_validated_composed_curves",
        lambda *_args: (part2_rows, curves, 100, 2500, 50),
    )

    bar_value_fields: list[str] = []

    def two_outputs(*_args: Any, **kwargs: Any) -> list[Path]:
        stem = kwargs["stem"]
        directory = kwargs["output_dir"]
        return [directory / f"{stem}.pdf", directory / f"{stem}.png"]

    def two_bar_outputs(*_args: Any, **kwargs: Any) -> list[Path]:
        bar_value_fields.append(kwargs["value_field"])
        return two_outputs(*_args, **kwargs)

    monkeypatch.setattr(original_views, "_bar_chart", two_bar_outputs)
    monkeypatch.setattr(original_views, "_line_chart", two_outputs)
    monkeypatch.setattr(
        original_views,
        "_agent_day_raster",
        lambda *_args, **kwargs: [
            kwargs["output_dir"] / "part2_agent_day_raster_current.pdf",
            kwargs["output_dir"] / "part2_agent_day_raster_current.png",
        ],
    )
    monkeypatch.setattr(
        original_views,
        "_publish_staged_outputs",
        lambda _staging, output: [
            output / basename for basename in original_views.EXPECTED_OUTPUT_BASENAMES
        ],
    )

    outputs = original_views.build_original_view_figures(
        analysis_dir,
        None,
        tmp_path / "figures",
        part2_source_overlay_pairs=[(Path("source"), Path("overlay"))] * 3,
    )

    assert [path.name for path in outputs] == [
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
    ]
    assert bar_value_fields == [
        "refusal_rate_all_scheduled",
        "mean_trajectory_restraint_rate_all_scheduled",
    ]


def test_private_pair_inputs_must_match_public_analysis_bindings(
    tmp_path: Path,
) -> None:
    pairs: list[tuple[Path, Path]] = []
    expected: list[dict[str, Any]] = []
    for index in range(3):
        source = tmp_path / f"source-{index}" / "private" / "manifest.json"
        overlay = tmp_path / f"overlay-{index}" / "private" / "manifest.json"
        source.parent.mkdir(parents=True)
        overlay.parent.mkdir(parents=True)
        source.parent.chmod(0o700)
        overlay.parent.chmod(0o700)
        source_manifest = {"evidence_sha256": f"{index + 1:064x}"}
        source.write_text(json.dumps(source_manifest) + "\n", encoding="utf-8")
        source.chmod(0o600)
        overlay_manifest: dict[str, Any] = {
            "evidence_sha256": f"{index + 11:064x}"
        }
        parent_binding: dict[str, str] | None = None
        if index == 0:
            parent = tmp_path / "parent" / "private" / "manifest.json"
            parent.parent.mkdir(parents=True)
            parent.parent.chmod(0o700)
            parent_manifest = {"evidence_sha256": f"{21:064x}"}
            parent.write_text(json.dumps(parent_manifest) + "\n", encoding="utf-8")
            parent.chmod(0o600)
            parent_binding = {
                "basename": parent.name,
                "file_sha256": _sha256_file(parent),
                "evidence_sha256": parent_manifest["evidence_sha256"],
            }
            overlay_manifest["parent_overlay_manifest"] = {
                "path": str(parent),
                "file_sha256": parent_binding["file_sha256"],
                "evidence_sha256": parent_binding["evidence_sha256"],
            }
        overlay.write_text(json.dumps(overlay_manifest) + "\n", encoding="utf-8")
        overlay.chmod(0o600)
        source_binding = {
            "basename": source.name,
            "file_sha256": _sha256_file(source),
            "evidence_sha256": source_manifest["evidence_sha256"],
        }
        overlay_binding = {
            "basename": overlay.name,
            "file_sha256": _sha256_file(overlay),
            "evidence_sha256": overlay_manifest["evidence_sha256"],
            "source_manifest_file_sha256": source_binding["file_sha256"],
            "source_manifest_evidence_sha256": source_binding["evidence_sha256"],
        }
        if parent_binding is not None:
            overlay_binding["parent_operational_repair_overlay"] = parent_binding
        pairs.append((source, overlay))
        expected.append({"source": source_binding, "overlay": overlay_binding})

    topology = {
        "mode": "three_pair_operational_repair_composition",
        "pairs": expected,
    }
    original_views._validate_analysis_composition_binding(pairs, topology)

    tampered = json.loads(json.dumps(topology))
    tampered["pairs"][2]["overlay"]["file_sha256"] = "0" * 64
    with pytest.raises(
        original_views.OriginalViewFigureError, match="do not match"
    ):
        original_views._validate_analysis_composition_binding(pairs, tampered)


def test_ten_file_publication_rolls_back_after_late_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "figures"
    staging.mkdir()
    output.mkdir()
    for basename in original_views.EXPECTED_OUTPUT_BASENAMES:
        (staging / basename).write_bytes(f"new:{basename}".encode())
        (output / basename).write_bytes(f"old:{basename}".encode())
    unrelated = output / "unrelated.txt"
    unrelated.write_text("keep", encoding="utf-8")

    real_replace = original_views.os.replace
    calls = 0

    def fail_once_late(source: Any, target: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == len(original_views.EXPECTED_OUTPUT_BASENAMES) + 6:
            raise OSError("injected late publication failure")
        real_replace(source, target)

    monkeypatch.setattr(original_views.os, "replace", fail_once_late)
    with pytest.raises(
        original_views.OriginalViewFigureError, match="prior outputs were restored"
    ):
        original_views._publish_staged_outputs(staging, output)

    for basename in original_views.EXPECTED_OUTPUT_BASENAMES:
        assert (output / basename).read_bytes() == f"old:{basename}".encode()
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_ten_file_publication_commits_exact_set_and_preserves_unrelated(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "figures"
    staging.mkdir()
    output.mkdir()
    for basename in original_views.EXPECTED_OUTPUT_BASENAMES:
        (staging / basename).write_bytes(f"new:{basename}".encode())
        (output / basename).write_bytes(f"old:{basename}".encode())
    unrelated = output / "unrelated.txt"
    unrelated.write_text("keep", encoding="utf-8")

    published = original_views._publish_staged_outputs(staging, output)

    assert [path.name for path in published] == list(
        original_views.EXPECTED_OUTPUT_BASENAMES
    )
    for basename in original_views.EXPECTED_OUTPUT_BASENAMES:
        assert (output / basename).read_bytes() == f"new:{basename}".encode()
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_replayed_curve_carries_semantic_invalid_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = runner.Part2Contract(1, 1, 1, 10, 2, 2, 5, 0.2)
    replayed = {
        "operationally_eligible": True,
        "final_reserve": 10,
        "final_population": 1,
        "aurc": 1.0,
        "aupc": 1.0,
        "restraint_rate": 0.0,
        "invalid_count": 1,
    }
    effective = {
        **replayed,
        "operational_repair_round": None,
        "source_replaced_for_operational_failure": False,
    }
    monkeypatch.setattr(
        original_views,
        "_validate_journal_identity",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        original_views,
        "_offline_replay",
        lambda **_kwargs: replayed,
    )
    monkeypatch.setattr(
        original_views,
        "_result_index",
        lambda *_args, **_kwargs: ({(1, 0): {"action": "INVALID"}}, []),
    )

    curve = original_views._curve_from_replayed_records(
        [{"record": True}],
        subject={"target_id": "openai/semantic-invalid"},
        trajectory_index=0,
        environment_seed=123,
        contract=contract,
        execution_contract={},
        effective_row=effective,
    )

    assert curve == {
        "trajectory_index": 0,
        "reserve": [10],
        "population": [1],
        "invalid_count": 1,
        "agent_actions": [["INVALID"]],
    }


class _Client:
    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        assert path == "/chat/completions"
        return {
            "id": "fixture-response",
            "model": body["model"],
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "action": "OPTION_A",
                                "reasoning": "Preserve the shared reserve.",
                            }
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }


class _MemoryJournal:
    """Thread-safe chained journal without per-record disk fsync test overhead."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.tail: str | None = None
        self._lock = threading.Lock()

    def append(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            row = {**payload, "previous_record_sha256": self.tail}
            row["record_sha256"] = _sha256_json(row)
            self.records.append(row)
            self.tail = row["record_sha256"]
            return row


def _real_100_day_journal(
    tmp_path: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    runner.Part2Contract,
    dict[str, Any],
    Path,
]:
    subject = {
        "target_id": "subject.alpha",
        "upstream_provider": "developer",
        "model": "alpha-model",
        "route": "region/alpha-model",
        "supported_controls": [
            "seed",
            "temperature",
            "top_p",
            "structured_response",
        ],
        "compatibility_max_tokens": 64,
    }
    contract = runner.Part2Contract(50, 100, 1, 2500, 2, 2, 5, 0.2)
    environment_seed = 20260911
    journal = _MemoryJournal()
    effective = runner._run_trajectory(
        subject=subject,
        trajectory_index=0,
        environment_seed=environment_seed,
        contract=contract,
        journal=journal,
        client=_Client(),
        participant_workers=50,
        max_attempts=1,
        initial_backoff_seconds=0,
        sleep_fn=lambda _seconds: None,
    )
    journal_path = tmp_path / "seed-000.jsonl"
    journal_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in journal.records
        ),
        encoding="utf-8",
    )
    return journal.records, subject, contract, effective, journal_path


def test_real_50_agent_100_day_replay_never_dispatches_or_mutates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records, subject, contract, effective, journal_path = _real_100_day_journal(
        tmp_path
    )
    before = (
        _sha256_file(journal_path),
        journal_path.stat().st_mtime_ns,
        journal_path.stat().st_size,
    )

    def forbidden_dispatch(**_kwargs: Any) -> Any:
        raise AssertionError("offline figure replay attempted model dispatch")

    monkeypatch.setattr(runner, "_dispatch_unit", forbidden_dispatch)
    curve = original_views._curve_from_replayed_records(
        records,
        subject=subject,
        trajectory_index=0,
        environment_seed=effective["environment_seed"],
        contract=contract,
        execution_contract={
            "participant_workers": 50,
            "max_transport_attempts": 1,
            "initial_exponential_backoff_seconds": 0,
        },
        effective_row=effective,
    )

    assert curve["reserve"] == [2500] * 100
    assert curve["population"] == [50] * 100
    assert curve["invalid_count"] == 0
    assert len(curve["agent_actions"]) == 50
    assert all(actions == ["OPTION_A"] * 100 for actions in curve["agent_actions"])
    assert before == (
        _sha256_file(journal_path),
        journal_path.stat().st_mtime_ns,
        journal_path.stat().st_size,
    )


def _production_ready() -> bool:
    try:
        manifests = [
            json.loads(Path(path).read_text(encoding="utf-8"))
            for pair in original_views.DEFAULT_PART2_SOURCE_OVERLAY_PAIRS
            for path in pair
        ]
        public = json.loads(
            (
                ROOT
                / "data/processed/provider-safe-v2-definitive-analysis/figure_aggregates.json"
            ).read_text(encoding="utf-8")
        )["part2_trajectories"]
    except (OSError, KeyError, json.JSONDecodeError):
        return False
    overlays = manifests[1::2]
    return all(manifest.get("complete") is True for manifest in overlays) and len(
        public
    ) == 276


@pytest.mark.skipif(
    not _production_ready(),
    reason="The private three-pair 23-route production snapshot is not COMPLETE/published.",
)
def test_completed_production_three_pair_curves_replay_end_to_end() -> None:
    main_overlay = json.loads(
        original_views.DEFAULT_PART2_SOURCE_OVERLAY_PAIRS[0][1].read_text(
            encoding="utf-8"
        )
    )
    assert (
        main_overlay["artifact_type"]
        == "inference_hub_part2_cascading_operational_trajectory_repair_v1"
    )
    assert (
        Path(main_overlay["parent_overlay_manifest"]["path"]).parents[1].name
        == "full-part2-n12-n50-d100-main21-v5-operational-repair-multikey-v3"
    )

    aggregate_path = (
        ROOT
        / "data/processed/provider-safe-v2-definitive-analysis/figure_aggregates.json"
    )
    public_rows = json.loads(aggregate_path.read_text(encoding="utf-8"))[
        "part2_trajectories"
    ]

    subjects, curves, days, capacity, society_size = (
        original_views._validated_composed_curves(
            original_views.DEFAULT_PART2_SOURCE_OVERLAY_PAIRS,
            public_rows,
        )
    )
    rows = [
        {
            "target_id": subject["target_id"],
            "upstream_provider": subject["upstream_provider"],
            "model": subject["model"],
        }
        for subject in subjects
    ]
    _ordered, raster = original_views._agent_day_raster_data(
        rows, curves, days=days, society_size=society_size
    )

    assert len(subjects) == 23
    assert sum(len(group) for group in curves.values()) == 276
    assert len(curves["google/gemini-3.5-flash"]) == 12
    assert (days, capacity, society_size) == (100, 2500, 50)
    assert raster.shape == (1150, 100)
