from argparse import Namespace
from pathlib import Path

from data.graphs.part_0_graphs import default_out_prefix, resolve_csv_paths


def test_default_out_prefix_uses_single_stem_or_combined_label() -> None:
    single = [Path("results/alignment/04-11-2026_13_04_37.csv")]
    multiple = [
        Path("results/alignment/04-11-2026_13_04_37.csv"),
        Path("results/alignment/04-12-2026_10_00_00.csv"),
    ]

    assert default_out_prefix(single) == "04-11-2026_13_04_37"
    assert default_out_prefix(multiple) == "alignment_combined_2files"


def test_resolve_csv_paths_returns_multiple_explicit_alignment_paths() -> None:
    args = Namespace(
        alignment_dir="results/alignment",
        latest=False,
        csv=["04-11-2026_13_04_37.csv", "04-12-2026_10_00_00.csv"],
    )

    assert resolve_csv_paths(args) == [
        Path("results/alignment/04-11-2026_13_04_37.csv"),
        Path("results/alignment/04-12-2026_10_00_00.csv"),
    ]
