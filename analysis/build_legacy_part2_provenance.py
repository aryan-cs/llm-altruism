"""Build the deterministic provenance seal for the April 2026 Part 2 pilot.

The original pilot sidecars predate explicit recording of
``collapse_death_rate``.  This builder does not guess or edit those sidecars.
It binds their exact bytes to a portable, immutable copy of the historical
execution source and prompt, then replays every recorded population transition
under the archived divisor-of-five rule.  Verification deliberately requires
no Git repository so it also works inside the anonymous supplement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any


ARTIFACT_TYPE = "legacy_part2_structural_provenance"
SCHEMA_VERSION = 1
RECOVERY_PROTOCOL = "archived_source_and_full_transition_replay_v1"
SOURCE_PATH = "experiments/part2/part_2.py"
SOURCE_SHA256 = "e4351e8a18f0faa6cc289f261efa4bfe71a370dd46935c629901f8890d994f4f"
PROMPT_PATH = "experiments/part2/part_2_prompt.json"
PROMPT_SHA256 = "26cabafbf3bc03c6d0287c4b82f1037db8c43a0b410b3d5f7debd4ff220580c5"
PROMPT_CONFIG_SHA256 = "b8ab4a3611f1f7ea9e360b821dbfec778f671c0c01840629abfc3a58cd2c5fdb"
ALLOWED_SOURCE_COMMITS = frozenset(
    {
        "69712b522357a935d3be15ba7a71d23d9a43e590",
        "5cf6595d37f0a926895c1571062ea0e3d94fee62",
    }
)
COLLAPSE_ATTRITION_DIVISOR = 5
COLLAPSE_DEATH_RATE = 0.2
DEFAULT_RAW_DIR = Path("data/raw/part_2")
DEFAULT_OUTPUT = DEFAULT_RAW_DIR / "legacy_structural_provenance.json"
ARCHIVE_DIRNAME = "legacy_execution_archive"
ARCHIVE_MANIFEST_FILENAME = "manifest.json"
ARCHIVE_ARTIFACT_TYPE = "legacy_part2_execution_archive"
ARCHIVE_SOURCE_FILENAME = "part_2.py"
ARCHIVE_PROMPT_FILENAME = "part_2_prompt.json"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _stable_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _git_file(commit: str, source_path: str, project_root: Path) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{source_path}"],
        cwd=project_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"Could not read archived Part 2 file {source_path} at {commit}: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    return completed.stdout


def _validate_source(source: bytes) -> None:
    if _sha256_bytes(source) != SOURCE_SHA256:
        raise ValueError("Portable archived Part 2 source hash disagrees")
    text = source.decode("utf-8")
    required_fragments = (
        "COLLAPSE_ATTRITION_DIVISOR = 5",
        "def _collapse_deaths(population: int, resource_units: int) -> int:",
        "ceil(population / COLLAPSE_ATTRITION_DIVISOR)",
    )
    if any(fragment not in text for fragment in required_fragments):
        raise ValueError("Portable archived Part 2 source is not the frozen rule")


def _validate_prompt(prompt: bytes) -> None:
    if _sha256_bytes(prompt) != PROMPT_SHA256:
        raise ValueError("Portable archived Part 2 prompt hash disagrees")
    try:
        payload = json.loads(prompt.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Portable archived Part 2 prompt is not valid UTF-8 JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("experiment_name") != "Part 2: Society Starter"
        or not isinstance(payload.get("agent"), dict)
        or "commons_prompt_template" not in payload["agent"]
    ):
        raise ValueError("Portable archived Part 2 prompt is not the frozen prompt")
    if _stable_hash(payload) != PROMPT_CONFIG_SHA256:
        raise ValueError("Portable archived Part 2 prompt canonical hash disagrees")


def _archive_manifest_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": ARCHIVE_ARTIFACT_TYPE,
        "recorded_commits": sorted(ALLOWED_SOURCE_COMMITS),
        "assets": [
            {
                "archive_filename": ARCHIVE_SOURCE_FILENAME,
                "original_path": SOURCE_PATH,
                "sha256": SOURCE_SHA256,
            },
            {
                "archive_filename": ARCHIVE_PROMPT_FILENAME,
                "original_path": PROMPT_PATH,
                "sha256": PROMPT_SHA256,
                "canonical_json_sha256": PROMPT_CONFIG_SHA256,
            },
        ],
    }


def _render_archive_manifest() -> str:
    payload = _archive_manifest_payload()
    payload["artifact_sha256"] = _stable_hash(payload)
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def materialize_execution_archive(
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    project_root: Path | None = None,
) -> Path:
    """Write the immutable historical files once, using the local Git object store."""

    project_root = (
        Path(__file__).resolve().parents[1]
        if project_root is None
        else project_root.resolve()
    )
    archive_dir = raw_dir / ARCHIVE_DIRNAME
    if archive_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {archive_dir}")
    portable_raw_dir = project_root / DEFAULT_RAW_DIR
    portable_archive = portable_raw_dir / ARCHIVE_DIRNAME
    if portable_archive.is_dir():
        _verify_execution_archive(portable_raw_dir)
        archive_dir.mkdir(parents=True)
        for filename in (
            ARCHIVE_SOURCE_FILENAME,
            ARCHIVE_PROMPT_FILENAME,
            ARCHIVE_MANIFEST_FILENAME,
        ):
            (archive_dir / filename).write_bytes(
                (portable_archive / filename).read_bytes()
            )
        return archive_dir
    commit = sorted(ALLOWED_SOURCE_COMMITS)[0]
    source = _git_file(commit, SOURCE_PATH, project_root)
    prompt = _git_file(commit, PROMPT_PATH, project_root)
    _validate_source(source)
    _validate_prompt(prompt)
    # Verify both recorded commits contain byte-identical source and prompt.
    for recorded_commit in sorted(ALLOWED_SOURCE_COMMITS)[1:]:
        if _git_file(recorded_commit, SOURCE_PATH, project_root) != source:
            raise ValueError(f"Part 2 source differs at {recorded_commit}")
        if _git_file(recorded_commit, PROMPT_PATH, project_root) != prompt:
            raise ValueError(f"Part 2 prompt differs at {recorded_commit}")
    archive_dir.mkdir(parents=True)
    (archive_dir / ARCHIVE_SOURCE_FILENAME).write_bytes(source)
    (archive_dir / ARCHIVE_PROMPT_FILENAME).write_bytes(prompt)
    (archive_dir / ARCHIVE_MANIFEST_FILENAME).write_text(
        _render_archive_manifest(), encoding="utf-8"
    )
    return archive_dir


def _verify_execution_archive(raw_dir: Path) -> None:
    archive_dir = raw_dir / ARCHIVE_DIRNAME
    manifest_path = archive_dir / ARCHIVE_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ValueError(f"Portable legacy Part 2 archive is missing: {manifest_path}")
    expected_manifest = _render_archive_manifest()
    if manifest_path.read_text(encoding="utf-8") != expected_manifest:
        raise ValueError(f"Portable legacy Part 2 archive manifest is stale: {manifest_path}")
    source = (archive_dir / ARCHIVE_SOURCE_FILENAME).read_bytes()
    prompt = (archive_dir / ARCHIVE_PROMPT_FILENAME).read_bytes()
    _validate_source(source)
    _validate_prompt(prompt)


def _transition_payload(csv_path: Path) -> tuple[list[dict[str, int]], int]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Legacy Part 2 CSV is empty: {csv_path}")
    by_day: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        day = int(row["day"])
        by_day.setdefault(day, []).append(row)
    if sorted(by_day) != list(range(1, max(by_day) + 1)):
        raise ValueError(f"Legacy Part 2 day sequence is incomplete: {csv_path}")

    transitions: list[dict[str, int]] = []
    collapsed_days = 0
    previous_population_end: int | None = None
    for day, day_rows in sorted(by_day.items()):
        first = day_rows[0]
        fields = (
            "population_start",
            "population_end",
            "resource_units_remaining",
            "deaths",
        )
        if any(
            any(row.get(field) != first.get(field) for row in day_rows[1:])
            for field in fields
        ):
            raise ValueError(f"Legacy Part 2 day summary changes within day {day}: {csv_path}")
        population_start = int(first["population_start"])
        population_end = int(first["population_end"])
        resource_units = int(first["resource_units_remaining"])
        deaths = int(first["deaths"])
        if previous_population_end is not None and population_start != previous_population_end:
            raise ValueError(f"Legacy Part 2 population discontinuity on day {day}: {csv_path}")
        expected_deaths = (
            min(
                population_start,
                max(1, math.ceil(population_start / COLLAPSE_ATTRITION_DIVISOR)),
            )
            if resource_units == 0
            else 0
        )
        if deaths != expected_deaths or population_end != population_start - deaths:
            raise ValueError(
                f"Legacy Part 2 transition disagrees with archived collapse rule on day {day}: "
                f"{csv_path}"
            )
        collapsed_days += int(resource_units == 0)
        transitions.append(
            {
                "day": day,
                "population_start": population_start,
                "population_end": population_end,
                "resource_units_remaining": resource_units,
                "deaths": deaths,
            }
        )
        previous_population_end = population_end
    return transitions, collapsed_days


def build_provenance(
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    project_root: Path | None = None,
) -> dict[str, Any]:
    # Keep the keyword for API compatibility with earlier releases.  The
    # portable check intentionally does not consult project_root or `.git`.
    del project_root
    _verify_execution_archive(raw_dir)
    entries: list[dict[str, Any]] = []
    observed_commits: set[str] = set()
    for csv_path in sorted(raw_dir.glob("*.csv")):
        metadata_path = csv_path.with_name(f"{csv_path.stem}_meta.json")
        if not metadata_path.is_file():
            raise ValueError(f"Missing legacy Part 2 sidecar: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"Legacy Part 2 sidecar is not an object: {metadata_path}")
        parameters = metadata.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError(f"Legacy Part 2 sidecar has no parameters object: {metadata_path}")
        if "collapse_death_rate" in parameters:
            continue
        if metadata.get("prompt_config_hash") != PROMPT_CONFIG_SHA256:
            raise ValueError(
                f"Legacy Part 2 sidecar is not bound to the archived prompt: {metadata_path}"
            )
        commit = str(metadata.get("git_commit", ""))
        if commit not in ALLOWED_SOURCE_COMMITS:
            raise ValueError(f"Unrecognized legacy Part 2 execution commit {commit!r}")
        command = metadata.get("command")
        if not isinstance(command, list) or "--collapse-death-rate" in command:
            raise ValueError(f"Legacy Part 2 command is not the default-rate command: {metadata_path}")
        observed_commits.add(commit)
        transitions, collapsed_days = _transition_payload(csv_path)
        entries.append(
            {
                "csv_filename": csv_path.name,
                "csv_sha256": _sha256_file(csv_path),
                "metadata_filename": metadata_path.name,
                "metadata_sha256": _sha256_file(metadata_path),
                "recorded_git_commit": commit,
                "recorded_git_dirty": bool(metadata.get("git_dirty")),
                "collapse_death_rate": COLLAPSE_DEATH_RATE,
                "days_replayed": len(transitions),
                "collapsed_days_replayed": collapsed_days,
                "transition_replay_sha256": _stable_hash(transitions),
            }
        )
    if not entries:
        raise ValueError(f"No legacy Part 2 artifacts requiring provenance were found in {raw_dir}")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "recovery_protocol": RECOVERY_PROTOCOL,
        "source_contract": {
            "path": SOURCE_PATH,
            "commits": sorted(observed_commits),
            "sha256": SOURCE_SHA256,
            "collapse_attrition_divisor": COLLAPSE_ATTRITION_DIVISOR,
            "collapse_death_rate": COLLAPSE_DEATH_RATE,
        },
        "entries": entries,
    }
    payload["artifact_sha256"] = _stable_hash(payload)
    return payload


def write_or_check_provenance(
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    check: bool = False,
) -> Path:
    artifact = build_provenance(raw_dir=raw_dir)
    rendered = json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if check:
        if not output_path.is_file() or output_path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"Legacy Part 2 provenance is missing or stale: {output_path}")
        return output_path
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing provenance: {output_path}")
    output_path.write_text(rendered, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the deterministic legacy Part 2 provenance seal."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--materialize-execution-archive",
        action="store_true",
        help="write the immutable source/prompt archive once from local Git history",
    )
    args = parser.parse_args()
    if args.materialize_execution_archive:
        archive_dir = materialize_execution_archive(raw_dir=args.raw_dir)
        print(f"Wrote {archive_dir}")
        return
    output = write_or_check_provenance(
        raw_dir=args.raw_dir,
        output_path=args.output,
        check=args.check,
    )
    print(f"{'Verified' if args.check else 'Wrote'} {output}")


if __name__ == "__main__":
    main()
