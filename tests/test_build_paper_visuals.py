from pathlib import Path

from PIL import Image

from analysis.build_paper_visuals import build_paper_visuals


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_builds_all_axis_specific_visuals_from_sealed_results(tmp_path: Path) -> None:
    paths = build_paper_visuals(
        PROJECT_ROOT / "data/analysis/final_results/final_results.json",
        tmp_path,
    )

    assert {path.name for path in paths} == {
        "part0_response_language_conditions.pdf",
        "part0_response_language_conditions.png",
        "part1_scope_distributions.pdf",
        "part1_scope_distributions.png",
        "part2_corrected_outcomes.pdf",
        "part2_corrected_outcomes.png",
    }
    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 1_000
    for path in (value for value in paths if value.suffix == ".png"):
        with Image.open(path) as image:
            assert image.width >= 2_000
            assert image.height >= 700
