#!/usr/bin/env python3
"""Compare one or more llm-altruism experiment outputs."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.report import comparison_table  # noqa: E402

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare experiment result files.")
    parser.add_argument(
        "results",
        nargs="+",
        help="Paths or globs pointing to result JSON/JSONL files or result directories.",
    )
    parser.add_argument(
        "--output",
        help="Optional CSV path to write the comparison table.",
    )
    return parser.parse_args()


def expand_paths(inputs: list[str]) -> list[Path]:
    """Expand any glob patterns into concrete paths."""
    expanded: list[Path] = []
    for item in inputs:
        matches = [Path(match) for match in glob.glob(item)]
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(Path(item))
    return expanded


def render_frame(frame) -> None:
    """Render a pandas dataframe as a colored rich table."""
    table = Table(title="Experiment Comparison", box=box.ROUNDED, header_style="bold cyan")
    for column in frame.columns:
        justify = "right" if column.endswith("_usd") or column.endswith("_seconds") else "left"
        table.add_column(str(column), style="white", justify=justify)

    for _, row in frame.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        table.add_row(*values)

    console.print(table)


def main() -> int:
    """Run the comparison CLI."""
    args = parse_args()
    paths = expand_paths(args.results)
    try:
        frame = comparison_table(paths)
    except (FileNotFoundError, ValueError) as exc:
        console.print(Panel(str(exc), title="Comparison Failed", border_style="red"))
        return 1
    if frame.empty:
        console.print(Panel("No comparable experiment outputs found.", border_style="yellow"))
        return 1

    render_frame(frame)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        console.print(
            Panel(
                f"Wrote comparison CSV to {output_path}",
                title="Export Complete",
                border_style="green",
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
