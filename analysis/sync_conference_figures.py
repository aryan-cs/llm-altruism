from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PAPER_VISUALS_DIR = Path("data") / "graphs" / "paper_visuals"
CROSS_PART_DIR = Path("data") / "graphs" / "cross_part" / "individual-plots"
DEFAULT_OUTPUT_DIR = Path("docs") / "conference_submission" / "figures"

PAPER_VISUALS = (
    "behavioral_fingerprint_heatmap.png",
    "frame_sensitivity_heatmap.png",
    "model_behavior_pca.png",
    "part0_refusal_by_language_heatmap.png",
    "part0_refusal_rate_by_model.png",
    "part1_cooperation_by_game_heatmap.png",
    "part2_agent_day_raster.png",
    "part2_population_over_time.png",
    "part2_restraint_choice_over_time.png",
    "part2_restraint_rate_by_model.png",
    "part2_shared_reserve_over_time.png",
)

CROSS_PART_VISUALS = (
    "safety_refusal_vs_restraint.png",
    "restraint_vs_final_population.png",
)


def _copy_required_files(source_dir: Path, names: tuple[str, ...], output_dir: Path) -> list[Path]:
    copied: list[Path] = []
    missing: list[Path] = []
    for name in names:
        source = source_dir / name
        destination = output_dir / name
        if not source.exists():
            missing.append(source)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)

    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing required figure files:\n{formatted}")
    return copied


def sync_conference_figures(
    *,
    paper_visuals_dir: Path = PAPER_VISUALS_DIR,
    cross_part_dir: Path = CROSS_PART_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    copied = _copy_required_files(paper_visuals_dir, PAPER_VISUALS, output_dir)
    copied.extend(_copy_required_files(cross_part_dir, CROSS_PART_VISUALS, output_dir))
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy generated paper figures into the LaTeX figure directory.")
    parser.add_argument("--paper-visuals-dir", default=str(PAPER_VISUALS_DIR))
    parser.add_argument("--cross-part-dir", default=str(CROSS_PART_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    copied = sync_conference_figures(
        paper_visuals_dir=Path(args.paper_visuals_dir),
        cross_part_dir=Path(args.cross_part_dir),
        output_dir=Path(args.output_dir),
    )
    for path in copied:
        print(f"synced: {path}")
    print(f"Synced {len(copied)} figures to {args.output_dir}")


if __name__ == "__main__":
    main()
