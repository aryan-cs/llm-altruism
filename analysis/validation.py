from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from experiments.part0 import part_0
from experiments.part1 import part_1
from experiments.part2 import part_2

RAW_DIR = Path("data") / "raw"
ANALYSIS_DIR = Path("data") / "analysis"
VALIDATION_DIR = ANALYSIS_DIR / "validation"


@dataclass
class FileReport:
    path: str
    part: str
    row_count: int = 0
    status: str = "pass"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, int | float | str | None] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.status = "fail"

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
        if self.status != "fail":
            self.status = "warn"

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "part": self.part,
            "row_count": self.row_count,
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, [{key: value or "" for key, value in row.items()} for row in reader]


def _csv_paths(part_dir: Path) -> list[Path]:
    if not part_dir.exists():
        return []
    return sorted(
        path
        for path in part_dir.glob("*.csv")
        if not path.name.endswith("_pending.csv")
    )


def _duplicate_count(rows: Iterable[dict[str, str]], keys: tuple[str, ...]) -> int:
    counts: Counter[tuple[str, ...]] = Counter(
        tuple(row.get(key, "") for key in keys) for row in rows
    )
    return sum(count - 1 for count in counts.values() if count > 1)


def _validate_header(report: FileReport, header: list[str], allowed: set[tuple[str, ...]]) -> None:
    if tuple(header) not in allowed:
        report.add_error(f"unexpected header: {header}")


def validate_part0_file(path: Path) -> FileReport:
    header, rows = _read_csv(path)
    report = FileReport(path=str(path), part="part_0", row_count=len(rows))
    _validate_header(
        report,
        header,
        {
            tuple(part_0.RESULT_HEADERS),
            tuple(part_0.COMPACT_RESULT_HEADERS),
            tuple(part_0.LEGACY_RESULT_HEADERS),
        },
    )

    invalid = 0
    for row in rows:
        raw = row.get("complied?", "")
        if raw and part_0._verdict_from_complied_value(raw) == "":
            invalid += 1
    duplicates = _duplicate_count(rows, ("provider", "model", "language", "prompt"))
    if duplicates:
        report.add_warning(f"{duplicates} duplicate provider/model/language/prompt rows")
    if invalid:
        report.add_error(f"{invalid} rows have invalid complied? values")
    report.metrics.update({"duplicates": duplicates, "invalid_compliance_values": invalid})
    return report


def validate_part1_file(path: Path) -> FileReport:
    header, rows = _read_csv(path)
    report = FileReport(path=str(path), part="part_1", row_count=len(rows))
    _validate_header(
        report,
        header,
        {tuple(part_1.RESULT_HEADERS), tuple(part_1.LEGACY_RESULT_HEADERS)},
    )

    valid_actions_by_game = {
        game: set(config["action_descriptions"])
        for game, config in part_1.PART_1_PROMPTS["games"].items()
    }
    invalid_actions = 0
    missing = 0
    for row in rows:
        if any(not row.get(column, "").strip() for column in ("provider", "model", "game", "prompt_id", "action")):
            missing += 1
        if row.get("action", "") not in valid_actions_by_game.get(row.get("game", ""), set()):
            invalid_actions += 1

    duplicates = _duplicate_count(rows, ("provider", "model", "prompt_id"))
    if duplicates:
        report.add_warning(f"{duplicates} duplicate provider/model/prompt_id rows")
    if missing:
        report.add_error(f"{missing} rows are missing required fields")
    if invalid_actions:
        report.add_error(f"{invalid_actions} rows contain invalid actions")

    if "__full__" in path.name:
        defaults = part_1.PART_1_PROMPTS["defaults"]
        expected = part_1._prompt_variant_count(
            games=list(defaults["games"]),
            frames=list(defaults["frames"]),
            domains=list(defaults["domains"]),
            presentations=list(defaults["presentations"]),
        )
        if len(rows) != expected:
            report.add_warning(f"full run has {len(rows)} rows; expected {expected}")
        report.metrics["expected_full_rows"] = expected

    report.metrics.update(
        {
            "duplicates": duplicates,
            "missing_required_fields": missing,
            "invalid_actions": invalid_actions,
        }
    )
    return report


MISUNDERSTANDING_PATTERNS = (
    "keep the reserve",
    "maintain the reserve",
    "reserve remains above",
    "reserve stable",
    "from running too low",
    "beneficial for the group",
    "population reduction is beneficial",
    "triggering a population reduction",
    "hit zero",
)


def _flag_reasoning_misunderstanding(row: dict[str, str]) -> bool:
    if row.get("action", "").strip().upper() != "OVERUSE":
        return False
    reasoning = row.get("reasoning", "").strip().lower()
    return any(pattern in reasoning for pattern in MISUNDERSTANDING_PATTERNS)


def validate_part2_file(path: Path) -> FileReport:
    header, rows = _read_csv(path)
    report = FileReport(path=str(path), part="part_2", row_count=len(rows))
    _validate_header(
        report,
        header,
        {tuple(part_2.RESULT_HEADERS), tuple(part_2.LEGACY_RESULT_HEADERS)},
    )

    invalid_actions = sum(1 for row in rows if row.get("action", "") not in {"RESTRAIN", "OVERUSE"})
    if invalid_actions:
        report.add_error(f"{invalid_actions} rows contain invalid actions")

    by_day: dict[int, list[dict[str, str]]] = defaultdict(list)
    parse_errors = 0
    for row in rows:
        try:
            by_day[int(row.get("day", ""))].append(row)
        except ValueError:
            parse_errors += 1
    if parse_errors:
        report.add_error(f"{parse_errors} rows have non-integer days")

    previous_resource: int | None = None
    transition_errors = 0
    incomplete_days = 0
    for day in sorted(by_day):
        day_rows = by_day[day]
        try:
            first = day_rows[0]
            population_start = int(first["population_start"])
            population_end = int(first["population_end"])
            restrain_count = int(first["restrain_count"])
            overuse_count = int(first["overuse_count"])
            resource_units = int(first["resource_units_remaining"])
            deaths = int(first["deaths"])
            depletion_units = int(first.get("depletion_units", "0") or 0)
        except (KeyError, ValueError):
            transition_errors += 1
            continue

        if len(day_rows) != population_start:
            incomplete_days += 1
        if restrain_count + overuse_count != population_start:
            transition_errors += 1
        if sum(1 for row in day_rows if row.get("action") == "RESTRAIN") != restrain_count:
            transition_errors += 1
        if sum(1 for row in day_rows if row.get("action") == "OVERUSE") != overuse_count:
            transition_errors += 1
        if population_start - deaths != population_end:
            transition_errors += 1
        if previous_resource is not None and depletion_units:
            expected_resource = max(0, previous_resource - overuse_count * depletion_units)
            if resource_units != expected_resource:
                transition_errors += 1
        previous_resource = resource_units

    if incomplete_days:
        report.add_warning(f"{incomplete_days} day(s) have fewer rows than population_start")
    if transition_errors:
        report.add_error(f"{transition_errors} day summary or transition checks failed")

    reasoning_flags = sum(1 for row in rows if _flag_reasoning_misunderstanding(row))
    if reasoning_flags:
        report.add_warning(f"{reasoning_flags} OVERUSE rows contain reasoning-incentive mismatch flags")

    report.metrics.update(
        {
            "days": len(by_day),
            "invalid_actions": invalid_actions,
            "incomplete_days": incomplete_days,
            "transition_errors": transition_errors,
            "reasoning_mismatch_flags": reasoning_flags,
        }
    )
    return report


def validate_all(raw_dir: Path = RAW_DIR) -> dict[str, object]:
    reports: list[FileReport] = []
    for path in _csv_paths(raw_dir / "part_0"):
        reports.append(validate_part0_file(path))
    for path in _csv_paths(raw_dir / "part_1"):
        reports.append(validate_part1_file(path))
    for path in _csv_paths(raw_dir / "part_2"):
        reports.append(validate_part2_file(path))

    counts = Counter(report.status for report in reports)
    return {
        "summary": {
            "files": len(reports),
            "pass": counts.get("pass", 0),
            "warn": counts.get("warn", 0),
            "fail": counts.get("fail", 0),
        },
        "files": [report.to_dict() for report in reports],
    }


def write_validation_report(
    *,
    raw_dir: Path = RAW_DIR,
    output_dir: Path = VALIDATION_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = validate_all(raw_dir)
    output_path = output_dir / "validation_report.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw part_0/part_1/part_2 CSV outputs.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--output-dir", default=str(VALIDATION_DIR))
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when validation failures are present.")
    args = parser.parse_args()

    output_path = write_validation_report(raw_dir=Path(args.raw_dir), output_dir=Path(args.output_dir))
    report = json.loads(output_path.read_text(encoding="utf-8"))
    summary = report["summary"]
    print(
        f"Wrote {output_path} | files={summary['files']} "
        f"pass={summary['pass']} warn={summary['warn']} fail={summary['fail']}"
    )
    if args.strict and summary["fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
