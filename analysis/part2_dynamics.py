"""Recorded Part 2 dynamics, structural cells, and analytic baselines.

This module deliberately has no dependency on an inference provider.  It is the
single analysis-side definition of the environmental parameters that make two
Part 2 trajectories comparable.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Mapping, Sequence


STRUCTURAL_CELL_SCHEMA_VERSION = 2
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
    CSV schema.  There is intentionally no legacy/default fallback: silently
    applying the current default would make sensitivity trajectories appear
    valid under dynamics that may not have generated them.
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
            _required_mapping_value(parameters, "collapse_death_rate", "parameters"),
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
