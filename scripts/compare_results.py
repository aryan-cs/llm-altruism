#!/usr/bin/env python3
"""Compare one or more llm-altruism experiment outputs."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.report import comparison_table


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


def main() -> int:
    """Run the comparison CLI."""
    args = parse_args()
    paths = expand_paths(args.results)
    frame = comparison_table(paths)
    if frame.empty:
        print("No comparable experiment outputs found.")
        return 1

    print(frame.to_string(index=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        print(f"\nWrote comparison CSV to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
