"""Recorded Part 2 dynamics, structural cells, and analytic baselines.

This module deliberately has no dependency on an inference provider.  It is the
single analysis-side definition of the environmental parameters that make two
Part 2 trajectories comparable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Mapping, Sequence


STRUCTURAL_CELL_SCHEMA_VERSION = 2
LEGACY_PROVENANCE_FILENAME = "legacy_structural_provenance.json"
LEGACY_PROVENANCE_ARTIFACT_TYPE = "legacy_part2_structural_provenance"
LEGACY_PROVENANCE_PROTOCOL = "archived_source_and_full_transition_replay_v1"
LEGACY_PART2_SOURCE_SHA256 = (
    "e4351e8a18f0faa6cc289f261efa4bfe71a370dd46935c629901f8890d994f4f"
)
STRICT_PART2_ROW_IDENTITY_FIELDS = frozenset(
    {
        "run_id",
        "trajectory_id",
        "structural_cell_id",
        "environment_seed",
        "generation_seed",
        "anonymous_agent_slot",
        "died_today",
        "attrition_rank",
        "attrition_seed",
        "death_selected_slots_json",
    }
)


class StructuralMetadataError(ValueError):
    """Raised when a trajectory cannot be assigned to a defensible cell."""


@dataclass(frozen=True)
class Part2StructuralCell:
    """All model and environmental fields that define a Part 2 cell.

    Generation seeds are intentionally excluded: they distinguish replicate
    trajectories within a cell rather than defining different treatments.
    """

    provider: str
    model: str
    society_size: int
    horizon_days: int
    resource: str
    resource_capacity: int
    selfish_gain: int
    depletion_units: int
    community_benefit: int
    collapse_death_rate: float
    structural_cell_id: str = ""
    dynamics_sha256: str = ""
    part_2_schema_version: int = 0

    @property
    def key(self) -> str:
        payload = {
            "schema_version": STRUCTURAL_CELL_SCHEMA_VERSION,
            **asdict(self),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def output_fields(self) -> dict[str, object]:
        return {
            "structural_cell_key": self.key,
            "society_size": self.society_size,
            "horizon_days": self.horizon_days,
            "resource": self.resource,
            "resource_capacity": self.resource_capacity,
            "selfish_gain": self.selfish_gain,
            "depletion_units": self.depletion_units,
            "community_benefit": self.community_benefit,
            "collapse_death_rate": self.collapse_death_rate,
            "recorded_structural_cell_id": self.structural_cell_id,
            "dynamics_sha256": self.dynamics_sha256,
            "part_2_schema_version": self.part_2_schema_version,
        }


STRUCTURAL_OUTPUT_FIELDS = [
    "structural_cell_key",
    "society_size",
    "horizon_days",
    "resource",
    "resource_capacity",
    "selfish_gain",
    "depletion_units",
    "community_benefit",
    "collapse_death_rate",
    "recorded_structural_cell_id",
    "dynamics_sha256",
    "part_2_schema_version",
]


@dataclass(frozen=True)
class Part2RunIdentity:
    """Recorded replicate identity; seeds are never structural-cell factors."""

    run_id: str
    trajectory_id: str
    structural_cell_id: str
    environment_seed: int | None
    generation_seed: int | None
    strict_schema: bool


def stable_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def structural_cell_id_for_contract(
    *,
    society_config: Mapping[str, object],
    resource_capacity: int,
    collapse_death_rate: float,
    dynamics: Mapping[str, object],
) -> str:
    payload = {
        "society_config": dict(society_config),
        "resource_capacity": resource_capacity,
        "collapse_death_rate": collapse_death_rate,
        "dynamics": dict(dynamics),
    }
    return f"p2cell_{stable_json_sha256(payload)[:20]}"


def trajectory_id_for_contract(*, structural_cell_id: str, environment_seed: int) -> str:
    return f"p2traj_{stable_json_sha256([structural_cell_id, environment_seed])[:20]}"


def _metadata_for_csv(path: Path) -> dict[str, object]:
    metadata_path = path.with_name(f"{path.stem}_meta.json")
    if not metadata_path.exists():
        raise StructuralMetadataError(
            f"missing Part 2 metadata sidecar {metadata_path.name}; "
            "collapse_death_rate and the configured horizon cannot be verified"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StructuralMetadataError(
            f"could not read Part 2 metadata sidecar {metadata_path.name}: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise StructuralMetadataError(
            f"Part 2 metadata sidecar {metadata_path.name} is not a JSON object"
        )
    return metadata


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise StructuralMetadataError(
                f"legacy Part 2 provenance contains duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _legacy_transition_payload(
    rows: Sequence[Mapping[str, str]],
    *,
    divisor: int,
) -> tuple[list[dict[str, int]], int]:
    by_day: dict[int, list[Mapping[str, str]]] = {}
    try:
        for row in rows:
            by_day.setdefault(int(row["day"]), []).append(row)
    except (KeyError, ValueError) as exc:
        raise StructuralMetadataError(
            "legacy Part 2 provenance cannot replay an invalid day field"
        ) from exc
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
            raise StructuralMetadataError(
                f"legacy Part 2 day summary changes within day {day}"
            )
        try:
            population_start = int(first["population_start"])
            population_end = int(first["population_end"])
            resource_units = int(first["resource_units_remaining"])
            deaths = int(first["deaths"])
        except (KeyError, ValueError) as exc:
            raise StructuralMetadataError(
                f"legacy Part 2 transition fields are invalid on day {day}"
            ) from exc
        if previous_population_end is not None and population_start != previous_population_end:
            raise StructuralMetadataError(
                f"legacy Part 2 population is discontinuous on day {day}"
            )
        expected_deaths = (
            min(population_start, max(1, math.ceil(population_start / divisor)))
            if resource_units == 0
            else 0
        )
        if deaths != expected_deaths or population_end != population_start - deaths:
            raise StructuralMetadataError(
                f"legacy Part 2 transition disagrees with archived collapse rule on day {day}"
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


def _legacy_collapse_death_rate(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    metadata: Mapping[str, object],
) -> float:
    """Recover a legacy rate only from an exact, self-hashed provenance seal."""

    provenance_path = path.parent / LEGACY_PROVENANCE_FILENAME
    if not provenance_path.is_file():
        raise StructuralMetadataError(
            "Part 2 metadata is missing parameters.collapse_death_rate and no exact "
            f"{LEGACY_PROVENANCE_FILENAME} seal exists"
        )
    try:
        provenance = json.loads(
            provenance_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                StructuralMetadataError(
                    f"legacy Part 2 provenance contains nonfinite constant {value}"
                )
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise StructuralMetadataError(
            f"could not read legacy Part 2 provenance {provenance_path.name}: {exc}"
        ) from exc
    if not isinstance(provenance, dict):
        raise StructuralMetadataError("legacy Part 2 provenance is not a JSON object")
    expected_top_keys = {
        "schema_version",
        "artifact_type",
        "recovery_protocol",
        "source_contract",
        "entries",
        "artifact_sha256",
    }
    if set(provenance) != expected_top_keys:
        raise StructuralMetadataError("legacy Part 2 provenance has an unexpected schema")
    recorded_hash = provenance.get("artifact_sha256")
    payload = {key: value for key, value in provenance.items() if key != "artifact_sha256"}
    if recorded_hash != stable_json_sha256(payload):
        raise StructuralMetadataError("legacy Part 2 provenance artifact hash is invalid")
    if (
        provenance.get("schema_version") != 1
        or provenance.get("artifact_type") != LEGACY_PROVENANCE_ARTIFACT_TYPE
        or provenance.get("recovery_protocol") != LEGACY_PROVENANCE_PROTOCOL
    ):
        raise StructuralMetadataError("legacy Part 2 provenance protocol is not supported")
    source_contract = provenance.get("source_contract")
    if not isinstance(source_contract, dict) or set(source_contract) != {
        "path",
        "commits",
        "sha256",
        "collapse_attrition_divisor",
        "collapse_death_rate",
    }:
        raise StructuralMetadataError("legacy Part 2 source contract is invalid")
    divisor = source_contract.get("collapse_attrition_divisor")
    rate = source_contract.get("collapse_death_rate")
    if (
        source_contract.get("path") != "experiments/part2/part_2.py"
        or source_contract.get("sha256") != LEGACY_PART2_SOURCE_SHA256
        or divisor != 5
        or rate != 0.2
    ):
        raise StructuralMetadataError("legacy Part 2 source contract is not the archived rule")
    commit = str(metadata.get("git_commit", ""))
    commits = source_contract.get("commits")
    if not isinstance(commits, list) or commit not in commits:
        raise StructuralMetadataError("legacy Part 2 execution commit is not source-bound")

    entries = provenance.get("entries")
    if not isinstance(entries, list):
        raise StructuralMetadataError("legacy Part 2 provenance entries must be a list")
    entry_keys = {
        "csv_filename",
        "csv_sha256",
        "metadata_filename",
        "metadata_sha256",
        "recorded_git_commit",
        "recorded_git_dirty",
        "collapse_death_rate",
        "days_replayed",
        "collapsed_days_replayed",
        "transition_replay_sha256",
    }
    names = [entry.get("csv_filename") for entry in entries if isinstance(entry, dict)]
    if len(names) != len(entries) or len(names) != len(set(names)):
        raise StructuralMetadataError("legacy Part 2 provenance entries are invalid or duplicated")
    matches = [entry for entry in entries if entry.get("csv_filename") == path.name]
    if len(matches) != 1 or set(matches[0]) != entry_keys:
        raise StructuralMetadataError("legacy Part 2 artifact has no exact provenance entry")
    entry = matches[0]
    metadata_path = path.with_name(f"{path.stem}_meta.json")
    if (
        entry.get("csv_sha256") != _sha256_file(path)
        or entry.get("metadata_filename") != metadata_path.name
        or entry.get("metadata_sha256") != _sha256_file(metadata_path)
        or entry.get("recorded_git_commit") != commit
        or entry.get("recorded_git_dirty") != bool(metadata.get("git_dirty"))
        or entry.get("collapse_death_rate") != rate
    ):
        raise StructuralMetadataError("legacy Part 2 provenance bytes or metadata do not match")
    transitions, collapsed_days = _legacy_transition_payload(rows, divisor=divisor)
    if (
        entry.get("days_replayed") != len(transitions)
        or entry.get("collapsed_days_replayed") != collapsed_days
        or entry.get("transition_replay_sha256") != stable_json_sha256(transitions)
    ):
        raise StructuralMetadataError("legacy Part 2 transition replay seal does not match")
    return float(rate)


def _constant_row_value(
    rows: Sequence[Mapping[str, str]],
    field: str,
) -> str | None:
    values = {str(row.get(field, "")).strip() for row in rows}
    values.discard("")
    if len(values) > 1:
        raise StructuralMetadataError(
            f"CSV field {field!r} changes within a trajectory: {sorted(values)}"
        )
    return next(iter(values)) if values else None


def _uses_strict_schema(rows: Sequence[Mapping[str, str]]) -> bool:
    return bool(rows) and STRICT_PART2_ROW_IDENTITY_FIELDS.issubset(rows[0])


def _strict_run_identity(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    metadata: Mapping[str, object],
    parameters: Mapping[str, object],
) -> Part2RunIdentity:
    if not _uses_strict_schema(rows):
        legacy_cell = _constant_row_value(rows, "structural_cell_id") or ""
        return Part2RunIdentity(
            run_id=path.stem,
            trajectory_id="",
            structural_cell_id=legacy_cell,
            environment_seed=None,
            generation_seed=None,
            strict_schema=False,
        )

    row_values = {
        field: _constant_row_value(rows, field)
        for field in (
            "run_id",
            "trajectory_id",
            "structural_cell_id",
            "environment_seed",
            "generation_seed",
        )
    }
    missing = [field for field, value in row_values.items() if value is None]
    if missing:
        raise StructuralMetadataError(
            "strict Part 2 CSV is missing constant run identity field(s): "
            + ", ".join(missing)
        )
    environment_seed = _parse_int(row_values["environment_seed"], "environment_seed")
    generation_seed = _parse_int(row_values["generation_seed"], "generation_seed")
    for field, csv_value in row_values.items():
        recorded = parameters.get(field, metadata.get(field))
        if recorded in (None, ""):
            raise StructuralMetadataError(f"strict Part 2 metadata is missing {field}")
        expected: object = (
            _parse_int(recorded, field)
            if field in {"environment_seed", "generation_seed"}
            else str(recorded).strip()
        )
        parsed_csv: object = (
            _parse_int(csv_value, field)
            if field in {"environment_seed", "generation_seed"}
            else str(csv_value).strip()
        )
        if parsed_csv != expected:
            raise StructuralMetadataError(
                f"metadata {field}={expected!r} disagrees with CSV {field}={parsed_csv!r}"
            )
    structural_cell_id = str(row_values["structural_cell_id"])
    trajectory_id = str(row_values["trajectory_id"])
    expected_trajectory_id = trajectory_id_for_contract(
        structural_cell_id=structural_cell_id,
        environment_seed=environment_seed,
    )
    if trajectory_id != expected_trajectory_id:
        raise StructuralMetadataError(
            "recorded trajectory_id does not bind structural_cell_id and environment_seed"
        )
    return Part2RunIdentity(
        run_id=str(row_values["run_id"]),
        trajectory_id=trajectory_id,
        structural_cell_id=structural_cell_id,
        environment_seed=environment_seed,
        generation_seed=generation_seed,
        strict_schema=True,
    )


def _required_mapping_value(mapping: Mapping[str, object], field: str, location: str) -> object:
    if field not in mapping or mapping[field] in (None, ""):
        raise StructuralMetadataError(f"Part 2 metadata is missing {location}.{field}")
    return mapping[field]


def _parse_int(value: object, name: str, *, positive: bool = False) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise StructuralMetadataError(f"{name} must be an integer; recorded {value!r}") from exc
    if positive and parsed <= 0:
        raise StructuralMetadataError(f"{name} must be greater than zero; recorded {parsed}")
    return parsed


def _parse_rate(value: object, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise StructuralMetadataError(f"{name} must be numeric; recorded {value!r}") from exc
    if not 0 < parsed <= 1:
        raise StructuralMetadataError(f"{name} must be in (0, 1]; recorded {parsed}")
    return parsed


def load_part2_structural_cell(
    path: Path,
    rows: Sequence[Mapping[str, str]],
) -> Part2StructuralCell:
    """Load and cross-check the recorded structural cell for one trajectory.

    The death rate is required in the sidecar because it is not present in the
    CSV schema.  Strict artifacts have no fallback.  A legacy artifact may use
    the adjacent provenance seal only when that seal binds the exact CSV and
    sidecar bytes, the archived execution source, and a full transition replay;
    the current runtime default is never consulted.
    """

    if not rows:
        raise StructuralMetadataError("cannot identify a structural cell for an empty CSV")
    metadata = _metadata_for_csv(path)
    parameters = metadata.get("parameters", metadata)
    if not isinstance(parameters, dict):
        raise StructuralMetadataError("Part 2 metadata.parameters must be an object")
    society = parameters.get("society_config")
    if not isinstance(society, dict):
        society = metadata.get("society_config")
    if not isinstance(society, dict):
        raise StructuralMetadataError("Part 2 metadata is missing society_config")

    identity = _strict_run_identity(path, rows, metadata, parameters)
    strict_schema = identity.strict_schema
    dynamics = parameters.get("dynamics", metadata.get("dynamics"))
    if strict_schema and not isinstance(dynamics, dict):
        raise StructuralMetadataError("strict Part 2 metadata is missing dynamics")
    dynamics_sha256 = stable_json_sha256(dynamics) if isinstance(dynamics, dict) else ""
    schema_version_value = parameters.get(
        "part_2_schema_version", metadata.get("part_2_schema_version", 0)
    )
    schema_version = _parse_int(schema_version_value, "part_2_schema_version")
    if strict_schema and schema_version <= 0:
        raise StructuralMetadataError("strict Part 2 schema version must be positive")

    provider = _constant_row_value(rows, "provider") or str(metadata.get("provider", "")).strip()
    model = _constant_row_value(rows, "model") or str(metadata.get("model", "")).strip()
    if not provider or not model:
        raise StructuralMetadataError("Part 2 provider/model is missing from both CSV and metadata")
    for field, csv_value in (("provider", provider), ("model", model)):
        metadata_value = str(metadata.get(field, "")).strip()
        if metadata_value and metadata_value != csv_value:
            raise StructuralMetadataError(
                f"metadata {field}={metadata_value!r} disagrees with CSV {field}={csv_value!r}"
            )

    recorded_collapse_rate = parameters.get("collapse_death_rate")
    if recorded_collapse_rate in (None, ""):
        if strict_schema:
            raise StructuralMetadataError(
                "Part 2 metadata is missing parameters.collapse_death_rate"
            )
        recorded_collapse_rate = _legacy_collapse_death_rate(path, rows, metadata)

    cell = Part2StructuralCell(
        provider=provider,
        model=model,
        society_size=_parse_int(
            _required_mapping_value(society, "society_size", "society_config"),
            "society_size",
            positive=True,
        ),
        horizon_days=_parse_int(
            _required_mapping_value(society, "days", "society_config"),
            "days",
            positive=True,
        ),
        resource=str(
            _required_mapping_value(society, "resource", "society_config")
        ).strip(),
        resource_capacity=_parse_int(
            _required_mapping_value(parameters, "resource_capacity", "parameters"),
            "resource_capacity",
            positive=True,
        ),
        selfish_gain=_parse_int(
            _required_mapping_value(society, "selfish_gain", "society_config"),
            "selfish_gain",
        ),
        depletion_units=_parse_int(
            _required_mapping_value(society, "depletion_units", "society_config"),
            "depletion_units",
            positive=True,
        ),
        community_benefit=_parse_int(
            _required_mapping_value(society, "community_benefit", "society_config"),
            "community_benefit",
        ),
        collapse_death_rate=_parse_rate(
            recorded_collapse_rate,
            "collapse_death_rate",
        ),
        structural_cell_id=identity.structural_cell_id,
        dynamics_sha256=dynamics_sha256,
        part_2_schema_version=schema_version,
    )
    if not cell.resource:
        raise StructuralMetadataError("resource must not be empty")

    if strict_schema:
        assert isinstance(dynamics, dict)
        expected_cell_id = structural_cell_id_for_contract(
            society_config=society,
            resource_capacity=cell.resource_capacity,
            collapse_death_rate=cell.collapse_death_rate,
            dynamics=dynamics,
        )
        if identity.structural_cell_id != expected_cell_id:
            raise StructuralMetadataError(
                "recorded structural_cell_id does not bind the complete dynamics contract"
            )

    row_checks: tuple[tuple[str, object], ...] = (
        ("resource", cell.resource),
        ("resource_capacity", cell.resource_capacity),
        ("selfish_gain", cell.selfish_gain),
        ("depletion_units", cell.depletion_units),
        ("community_benefit", cell.community_benefit),
    )
    for field, recorded in row_checks:
        row_value = _constant_row_value(rows, field)
        if row_value is None:
            raise StructuralMetadataError(f"CSV is missing recorded structural field {field}")
        parsed: object = row_value if isinstance(recorded, str) else _parse_int(row_value, field)
        if parsed != recorded:
            raise StructuralMetadataError(
                f"metadata {field}={recorded!r} disagrees with CSV {field}={parsed!r}"
            )

    first_day_populations = {
        _parse_int(row.get("population_start", ""), "population_start")
        for row in rows
        if str(row.get("day", "")).strip() == "1"
    }
    if first_day_populations and first_day_populations != {cell.society_size}:
        raise StructuralMetadataError(
            "metadata society_size disagrees with day-1 population_start: "
            f"{sorted(first_day_populations)}"
        )

    filename_match = re.search(r"__n(?P<size>\d+)__d(?P<days>\d+)__", path.name)
    if filename_match:
        filename_config = (
            int(filename_match.group("size")),
            int(filename_match.group("days")),
        )
        if filename_config != (cell.society_size, cell.horizon_days):
            raise StructuralMetadataError(
                "metadata society_size/days disagrees with the CSV filename: "
                f"{filename_config}"
            )
    return cell


def load_part2_run_identity(
    path: Path,
    rows: Sequence[Mapping[str, str]],
) -> Part2RunIdentity:
    """Load and cross-check run, trajectory, and seed identity for one CSV."""

    if not rows:
        raise StructuralMetadataError("cannot identify an empty Part 2 CSV")
    metadata = _metadata_for_csv(path)
    parameters = metadata.get("parameters", metadata)
    if not isinstance(parameters, dict):
        raise StructuralMetadataError("Part 2 metadata.parameters must be an object")
    return _strict_run_identity(path, rows, metadata, parameters)


def normalized_part2_auc(
    day_rows: Mapping[int, Sequence[Mapping[str, str]]],
    *,
    horizon: int,
    society_size: int,
    resource_capacity: int,
) -> tuple[float, float]:
    """Compute run-level normalized reserve/population AUC without analysis imports."""

    if horizon <= 0 or society_size <= 0 or resource_capacity <= 0:
        return float("nan"), float("nan")
    if day_rows:
        last_day = max(day_rows)
        last_group = day_rows[last_day]
        last_population = (
            int(last_group[-1].get("population_end") or 0) if last_group else 0
        )
        if last_day < horizon and last_population > 0:
            return float("nan"), float("nan")
    reserve_area = 0.0
    population_area = 0.0
    for day in range(1, horizon + 1):
        grouped = day_rows.get(day, ())
        if grouped:
            row = grouped[-1]
            reserve_area += (
                int(row.get("resource_units_remaining") or 0) / resource_capacity
            )
            population_area += int(row.get("population_end") or 0) / society_size
    return reserve_area / horizon, population_area / horizon


def collapse_deaths(population: int, resource_units: int, death_rate: float) -> int:
    """Analysis-side closed-form collapse transition."""

    if population <= 0 or resource_units > 0:
        return 0
    return min(population, max(1, ceil(population * death_rate)))


@dataclass(frozen=True)
class AnalyticBaseline:
    policy: str
    days_completed: int
    depletion_day: int | None
    final_population: int
    final_resource_units: int
    total_deaths: int
    restraint_rate: float
    normalized_aurc: float
    normalized_aupc: float


@dataclass(frozen=True)
class MechanicalSurvivalThreshold:
    """Exact no-collapse action budget when every noncollapsed agent survives."""

    total_agent_days: int
    max_safe_overuse_actions: int
    max_safe_overuse_rate: float
    min_restraint_actions: int
    min_restraint_rate: float
    resource_floor_units: int


def mechanical_survival_threshold(
    cell: Part2StructuralCell,
) -> MechanicalSurvivalThreshold:
    """Calculate the strict resource-positive threshold through the horizon.

    Depletion occurs at zero, so the safe budget is the largest integer number
    of overuse actions whose cumulative depletion leaves at least one resource
    unit.  The denominator is the no-collapse agent-day count.
    """

    total_agent_days = cell.society_size * cell.horizon_days
    max_safe = min(
        total_agent_days,
        max(0, (cell.resource_capacity - 1) // cell.depletion_units),
    )
    min_restraint = total_agent_days - max_safe
    return MechanicalSurvivalThreshold(
        total_agent_days=total_agent_days,
        max_safe_overuse_actions=max_safe,
        max_safe_overuse_rate=(max_safe / total_agent_days if total_agent_days else 0.0),
        min_restraint_actions=min_restraint,
        min_restraint_rate=(
            min_restraint / total_agent_days if total_agent_days else 0.0
        ),
        resource_floor_units=cell.resource_capacity - max_safe * cell.depletion_units,
    )


def deterministic_policy_baseline(
    cell: Part2StructuralCell,
    policy: str,
) -> AnalyticBaseline:
    """Exact always-restrain or always-overuse trajectory for a cell."""

    normalized_policy = policy.strip().lower().replace("_", "-")
    if normalized_policy not in {"always-restrain", "always-overuse"}:
        raise ValueError("policy must be 'always-restrain' or 'always-overuse'")

    population = cell.society_size
    resource_units = cell.resource_capacity
    total_deaths = 0
    depletion_day: int | None = None
    reserve_area = 0.0
    population_area = 0.0
    days_completed = 0
    for day in range(1, cell.horizon_days + 1):
        if population <= 0:
            break
        overuses = population if normalized_policy == "always-overuse" else 0
        resource_units = max(0, resource_units - overuses * cell.depletion_units)
        if resource_units == 0 and depletion_day is None:
            depletion_day = day
        deaths = collapse_deaths(population, resource_units, cell.collapse_death_rate)
        population -= deaths
        total_deaths += deaths
        reserve_area += resource_units / cell.resource_capacity
        population_area += population / cell.society_size
        days_completed = day

    return AnalyticBaseline(
        policy=normalized_policy,
        days_completed=days_completed,
        depletion_day=depletion_day,
        final_population=population,
        final_resource_units=resource_units,
        total_deaths=total_deaths,
        restraint_rate=1.0 if normalized_policy == "always-restrain" else 0.0,
        normalized_aurc=reserve_area / cell.horizon_days,
        normalized_aupc=population_area / cell.horizon_days,
    )


def threshold_policy_baseline(cell: Part2StructuralCell) -> AnalyticBaseline:
    """Exact safe-reserve policy: spend the overuse budget, then restrain.

    Within each day, anonymous slots up to the remaining strict-positive
    resource budget overuse and all other slots restrain.  This prespecified
    tie-break makes the aggregate policy deterministic while guaranteeing that
    the resource never reaches zero.
    """

    population = cell.society_size
    resource_units = cell.resource_capacity
    reserve_area = 0.0
    population_area = 0.0
    restraints = 0
    overuses = 0
    days_completed = 0
    for day in range(1, cell.horizon_days + 1):
        safe_overuse_budget = max(0, (resource_units - 1) // cell.depletion_units)
        overuse_count = min(population, safe_overuse_budget)
        restraint_count = population - overuse_count
        resource_units -= overuse_count * cell.depletion_units
        overuses += overuse_count
        restraints += restraint_count
        reserve_area += resource_units / cell.resource_capacity
        population_area += population / cell.society_size
        days_completed = day
    total_actions = restraints + overuses
    return AnalyticBaseline(
        policy="safe-reserve-threshold",
        days_completed=days_completed,
        depletion_day=None,
        final_population=population,
        final_resource_units=resource_units,
        total_deaths=0,
        restraint_rate=restraints / total_actions if total_actions else 0.0,
        normalized_aurc=reserve_area / cell.horizon_days,
        normalized_aupc=population_area / cell.horizon_days,
    )


_BERNOULLI_OVERUSE_PROBABILITIES = (0.25, 0.5, 0.75)


def _bernoulli_overuse(
    *,
    probability: float,
    environment_seed: int,
    day: int,
    anonymous_slot: int,
) -> bool:
    payload = json.dumps(
        [
            "part2-no-call-bernoulli-v1",
            environment_seed,
            day,
            anonymous_slot,
            format(probability, ".17g"),
        ],
        separators=(",", ":"),
    ).encode("utf-8")
    draw = int.from_bytes(hashlib.sha256(payload).digest(), "big")
    threshold = int(probability * (1 << 256))
    return draw < threshold


def bernoulli_policy_baseline(
    cell: Part2StructuralCell,
    overuse_probability: float,
    *,
    environment_seed: int,
) -> AnalyticBaseline:
    """Run an exact hash-seeded no-call Bernoulli policy trajectory.

    The only allowed probabilities are the three protocol values.  Hash-based
    draws make the simulation invariant to Python's PRNG implementation.
    """

    probability = float(overuse_probability)
    if probability not in _BERNOULLI_OVERUSE_PROBABILITIES:
        raise ValueError("Bernoulli overuse probability must be 0.25, 0.50, or 0.75")
    if isinstance(environment_seed, bool) or not isinstance(environment_seed, int):
        raise ValueError("environment_seed must be an integer")

    population = cell.society_size
    resource_units = cell.resource_capacity
    reserve_area = 0.0
    population_area = 0.0
    total_deaths = 0
    restraints = 0
    overuses = 0
    depletion_day: int | None = None
    days_completed = 0
    for day in range(1, cell.horizon_days + 1):
        if population <= 0:
            break
        overuse_count = sum(
            _bernoulli_overuse(
                probability=probability,
                environment_seed=environment_seed,
                day=day,
                anonymous_slot=anonymous_slot,
            )
            for anonymous_slot in range(1, population + 1)
        )
        restraint_count = population - overuse_count
        overuses += overuse_count
        restraints += restraint_count
        resource_units = max(
            0, resource_units - overuse_count * cell.depletion_units
        )
        if resource_units == 0 and depletion_day is None:
            depletion_day = day
        deaths = collapse_deaths(population, resource_units, cell.collapse_death_rate)
        population -= deaths
        total_deaths += deaths
        reserve_area += resource_units / cell.resource_capacity
        population_area += population / cell.society_size
        days_completed = day
    total_actions = restraints + overuses
    return AnalyticBaseline(
        policy=f"bernoulli-overuse-p{probability:.2f}-seed-{environment_seed}",
        days_completed=days_completed,
        depletion_day=depletion_day,
        final_population=population,
        final_resource_units=resource_units,
        total_deaths=total_deaths,
        restraint_rate=restraints / total_actions if total_actions else 0.0,
        normalized_aurc=reserve_area / cell.horizon_days,
        normalized_aupc=population_area / cell.horizon_days,
    )


def no_call_baseline_suite(
    cell: Part2StructuralCell,
    *,
    environment_seeds: Sequence[int],
) -> list[AnalyticBaseline]:
    """Build all prespecified no-call policies without fabricating model data."""

    seeds = [int(seed) for seed in environment_seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("no-call baseline environment seeds must be unique")
    output = [
        deterministic_policy_baseline(cell, "always-restrain"),
        deterministic_policy_baseline(cell, "always-overuse"),
        threshold_policy_baseline(cell),
    ]
    output.extend(
        bernoulli_policy_baseline(cell, probability, environment_seed=seed)
        for probability in _BERNOULLI_OVERUSE_PROBABILITIES
        for seed in seeds
    )
    return output
