"""Build fail-closed paper assets from definitive provider-safe-v2 outputs.

The input is only the completed, sanitized output directory produced by
``analysis.analyze_provider_safe_v2_definitive``.  This module never opens the
private manifests named in that output and never generates human labels.  It
publishes six within-task figure/table families plus one side-by-side
cross-phase display atomically into a new output directory.  The display does
not pool axes, compute a composite, or fill unexecuted cells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_hex
from matplotlib.ticker import PercentFormatter


SCHEMA_VERSION = 1
SOURCE_ARTIFACT_TYPE = "provider_safe_v2_definitive_descriptive_analysis"
OUTPUT_ARTIFACT_TYPE = "provider_safe_v2_definitive_paper_assets"
INVALID_POLICY = (
    "first_attempt_invalids_retained_in_all_primary_scheduled_unit_denominators;"
    "repairs_reported_separately"
)
EXPECTED_ROW_COUNTS = {
    "part0_models": 22,
    "part1_models": 75,
    "part2_models": 19,
    "role_calibration_model_frames": 18,
    "sensitivity_models": 6,
    "sensitivity_main_effects": 30,
}
PART0_LANGUAGES = ("english", "chinese", "russian")
ROLE_FRAMES = ("advice", "observer_evaluation", "prediction")
SENSITIVITY_LEVELS: dict[str, tuple[int | float, int | float]] = {
    "capacity_per_initial_agent": (5, 15),
    "depletion_units": (1, 2),
    "collapse_death_rate": (0.1, 0.4),
    "society_size": (4, 8),
    "horizon_days": (10, 20),
}
SENSITIVITY_FACTORS = tuple(SENSITIVITY_LEVELS)
DEFAULT_LOCAL_CONTROLS_PATH = (
    Path(__file__).resolve().parents[1] / "data/analysis/local_hf_part1_controls.json"
)

INK = "#20252B"
MUTED = "#66707A"
GRID = "#D9DEE3"
TURBO = matplotlib.colormaps["turbo"]
BLUE = to_hex(TURBO(0.10))
GREEN = to_hex(TURBO(0.42))
RED = to_hex(TURBO(0.90))
RATE_CMAP = LinearSegmentedColormap.from_list(
    "rate_turbo",
    tuple(TURBO(stop) for stop in (0.08, 0.20, 0.31, 0.42)),
)
VALID_CMAP = LinearSegmentedColormap.from_list(
    "valid_turbo",
    tuple(TURBO(stop) for stop in (0.08, 0.20, 0.31, 0.42)),
)
SIGNED_CMAP = LinearSegmentedColormap.from_list(
    "signed_turbo", (BLUE, "#FAFAF8", RED)
)


class PaperAssetsError(RuntimeError):
    """Sanitized analysis output cannot support the requested paper assets."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _self_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_bytes({key: item for key, item in value.items() if key != "evidence_sha256"})
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PaperAssetsError(f"{label} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise PaperAssetsError(f"{label} must be a JSON object: {path}")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
        if not text or not text.endswith("\n"):
            raise ValueError("missing rows or final JSONL delimiter")
        values = [json.loads(line) for line in text[:-1].split("\n")]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PaperAssetsError(f"{label} is not valid complete JSONL: {path}") from error
    if any(not isinstance(row, dict) for row in values):
        raise PaperAssetsError(f"{label} contains a non-object row.")
    return [dict(row) for row in values]


def _identity(row: Mapping[str, Any], field: str, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value or value.strip() != value:
        raise PaperAssetsError(f"{label}.{field} must be a nonempty exact identifier.")
    return value


def _integer(
    row: Mapping[str, Any], field: str, label: str, *, minimum: int = 0
) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PaperAssetsError(f"{label}.{field} must be an integer >= {minimum}.")
    return value


def _number(
    row: Mapping[str, Any],
    field: str,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PaperAssetsError(f"{label}.{field} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise PaperAssetsError(f"{label}.{field} must be finite.")
    if minimum is not None and numeric < minimum:
        raise PaperAssetsError(f"{label}.{field} is below {minimum}.")
    if maximum is not None and numeric > maximum:
        raise PaperAssetsError(f"{label}.{field} is above {maximum}.")
    return numeric


def _rate(row: Mapping[str, Any], field: str, label: str) -> float:
    return _number(row, field, label, minimum=0.0, maximum=1.0)


def _same_rate(observed: float, numerator: int | float, denominator: int) -> bool:
    return denominator > 0 and math.isclose(
        observed, float(numerator) / denominator, rel_tol=1e-12, abs_tol=1e-12
    )


def _optional_reconciled_rate(
    row: Mapping[str, Any], field: str, label: str, numerator: int, denominator: int
) -> float | None:
    """Require a numeric rate when estimable and JSON null only at denominator zero."""

    if denominator == 0:
        if row.get(field) is not None:
            raise PaperAssetsError(f"{label}.{field} must be null at denominator zero.")
        return None
    value = _rate(row, field, label)
    if not _same_rate(value, numerator, denominator):
        raise PaperAssetsError(f"{label}.{field} does not reconcile.")
    return value


def _required(row: Mapping[str, Any], fields: Sequence[str], label: str) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise PaperAssetsError(f"{label} lacks required fields: {', '.join(missing)}.")


def _model_index(
    rows: Sequence[Mapping[str, Any]], label: str, expected_count: int
) -> dict[str, dict[str, Any]]:
    if len(rows) != expected_count:
        raise PaperAssetsError(f"{label} requires exactly {expected_count} rows.")
    output: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = dict(raw)
        row_label = f"{label}[{index}]"
        _required(row, ("target_id", "upstream_provider", "model"), row_label)
        target = _identity(row, "target_id", row_label)
        _identity(row, "upstream_provider", row_label)
        _identity(row, "model", row_label)
        if target in output:
            raise PaperAssetsError(f"{label} duplicates exact target_id {target!r}.")
        output[target] = row
    return output


def _validate_manifest(input_dir: Path) -> dict[str, Any]:
    manifest = _read_json(input_dir / "analysis_manifest.json", "analysis manifest")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("artifact_type") != SOURCE_ARTIFACT_TYPE
        or manifest.get("evidence_sha256") != _self_hash(manifest)
    ):
        raise PaperAssetsError("Analysis manifest type, schema, or self-hash failed.")
    if manifest.get("row_counts") != EXPECTED_ROW_COUNTS:
        raise PaperAssetsError("Analysis manifest does not declare the production row matrix.")
    if (
        manifest.get("invalid_policy") != INVALID_POLICY
        or manifest.get("human_labels_generated") is not False
        or manifest.get("exploratory_only") is not True
        or manifest.get("confirmatory_or_paper_promotion_permitted") is not False
    ):
        raise PaperAssetsError("Analysis manifest safety/status contract changed.")
    return manifest


def _validate_part0(
    rows: Sequence[Mapping[str, Any]], aggregates: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    models = _model_index(rows, "part0_models", 22)
    for target, row in models.items():
        label = f"part0_models[{target}]"
        _required(
            row,
            (
                "scheduled_units", "refusal_count", "compliance_count", "unclear_count",
                "first_attempt_invalid_count", "repaired_invalid_count",
                "refusal_rate_all_scheduled", "primary_denominator", "exploratory_only",
            ),
            label,
        )
        scheduled = _integer(row, "scheduled_units", label, minimum=1)
        if scheduled != 144:
            raise PaperAssetsError(f"{label} does not contain 48 roots x 3 languages.")
        counts = [
            _integer(row, field, label)
            for field in (
                "refusal_count", "compliance_count", "unclear_count",
                "first_attempt_invalid_count",
            )
        ]
        if _integer(row, "repaired_invalid_count", label) != 0:
            raise PaperAssetsError("Part 0 repaired outcomes are outside this frozen asset contract.")
        refusal_rate = _rate(row, "refusal_rate_all_scheduled", label)
        if sum(counts) != scheduled or not _same_rate(refusal_rate, counts[0], scheduled):
            raise PaperAssetsError(f"{label} counts or scheduled-unit refusal rate do not reconcile.")
        if row.get("primary_denominator") != "all_scheduled_units" or row.get("exploratory_only") is not True:
            raise PaperAssetsError(f"{label} changed its denominator/status contract.")

    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if len(aggregates) != 22 * len(PART0_LANGUAGES):
        raise PaperAssetsError("Part 0 language aggregate matrix must contain exactly 66 rows.")
    for index, raw in enumerate(aggregates):
        row = dict(raw)
        label = f"part0_by_model_language[{index}]"
        _required(
            row,
            (
                "target_id", "model", "language", "scheduled_units", "refusal_count",
                "compliance_count", "unclear_count", "invalid_count",
                "refusal_rate_all_scheduled",
            ),
            label,
        )
        target = _identity(row, "target_id", label)
        model = _identity(row, "model", label)
        language = _identity(row, "language", label)
        if target not in models or model != models[target]["model"] or language not in PART0_LANGUAGES:
            raise PaperAssetsError(f"{label} has an unknown route/model/language identity.")
        key = (target, language)
        if key in by_key:
            raise PaperAssetsError(f"Part 0 language matrix duplicates {key!r}.")
        scheduled = _integer(row, "scheduled_units", label, minimum=1)
        if scheduled != 48:
            raise PaperAssetsError(f"{label} does not contain exactly 48 scheduled roots.")
        counts = [
            _integer(row, field, label)
            for field in ("refusal_count", "compliance_count", "unclear_count", "invalid_count")
        ]
        refusal_rate = _rate(row, "refusal_rate_all_scheduled", label)
        if sum(counts) != scheduled or not _same_rate(refusal_rate, counts[0], scheduled):
            raise PaperAssetsError(f"{label} counts or rate do not reconcile.")
        by_key[key] = row
    expected_keys = {(target, language) for target in models for language in PART0_LANGUAGES}
    if set(by_key) != expected_keys:
        raise PaperAssetsError("Part 0 language aggregate matrix is incomplete.")
    for target, model_row in models.items():
        grouped = [by_key[(target, language)] for language in PART0_LANGUAGES]
        if (
            sum(int(row["scheduled_units"]) for row in grouped) != model_row["scheduled_units"]
            or sum(int(row["refusal_count"]) for row in grouped) != model_row["refusal_count"]
            or sum(int(row["invalid_count"]) for row in grouped)
            != model_row["first_attempt_invalid_count"]
        ):
            raise PaperAssetsError(f"Part 0 language rows do not reconcile for {target!r}.")
    ordered = sorted(
        models.values(),
        key=lambda row: (-float(row["refusal_rate_all_scheduled"]), str(row["target_id"])),
    )
    return ordered, by_key


def _validate_part1(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    models = _model_index(rows, "part1_models", 75)
    for target, row in models.items():
        label = f"part1_models[{target}]"
        _required(
            row,
            (
                "scheduled_units", "format_valid_first_attempt_count",
                "first_attempt_invalid_count", "repaired_invalid_count",
                "welfare_preserving_count_first_attempt",
                "welfare_preserving_rate_all_scheduled",
                "welfare_preserving_rate_among_first_attempt_valid",
                "primary_denominator", "exploratory_only",
            ),
            label,
        )
        scheduled = _integer(row, "scheduled_units", label, minimum=1)
        valid = _integer(row, "format_valid_first_attempt_count", label)
        invalid = _integer(row, "first_attempt_invalid_count", label)
        welfare = _integer(row, "welfare_preserving_count_first_attempt", label)
        if scheduled != 384 or valid + invalid != scheduled or welfare > valid:
            raise PaperAssetsError(f"{label} does not reconcile its 384 scheduled units.")
        if _integer(row, "repaired_invalid_count", label) != 0:
            raise PaperAssetsError("Part 1 repaired outcomes are outside this frozen asset contract.")
        all_rate = _rate(row, "welfare_preserving_rate_all_scheduled", label)
        _optional_reconciled_rate(
            row, "welfare_preserving_rate_among_first_attempt_valid", label, welfare, valid
        )
        if not _same_rate(all_rate, welfare, scheduled):
            raise PaperAssetsError(f"{label} welfare rates do not reconcile.")
        if row.get("primary_denominator") != "all_scheduled_units" or row.get("exploratory_only") is not True:
            raise PaperAssetsError(f"{label} changed its denominator/status contract.")
    return sorted(
        models.values(),
        key=lambda row: (-float(row["welfare_preserving_rate_all_scheduled"]), str(row["target_id"])),
    )


def _validate_part2(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    models = _model_index(rows, "part2_models", 19)
    for target, row in models.items():
        label = f"part2_models[{target}]"
        _required(
            row,
            (
                "trajectory_count", "operationally_eligible_trajectory_count",
                "scheduled_agent_days", "restraint_count", "overuse_count",
                "first_attempt_invalid_count", "repaired_invalid_count",
                "restraint_rate_all_scheduled", "restraint_rate_among_valid",
                "mean_aurc_eligible", "primary_denominator", "exploratory_only",
            ),
            label,
        )
        trajectories = _integer(row, "trajectory_count", label, minimum=1)
        eligible = _integer(row, "operationally_eligible_trajectory_count", label)
        scheduled = _integer(row, "scheduled_agent_days", label, minimum=1)
        restraint = _integer(row, "restraint_count", label)
        overuse = _integer(row, "overuse_count", label)
        invalid = _integer(row, "first_attempt_invalid_count", label)
        if trajectories != 12 or eligible > trajectories or restraint + overuse + invalid != scheduled:
            raise PaperAssetsError(f"{label} trajectory/action accounting does not reconcile.")
        if _integer(row, "repaired_invalid_count", label) != 0:
            raise PaperAssetsError("Part 2 repaired outcomes are outside this frozen asset contract.")
        all_rate = _rate(row, "restraint_rate_all_scheduled", label)
        _optional_reconciled_rate(
            row, "restraint_rate_among_valid", label, restraint, scheduled - invalid
        )
        if eligible == 0:
            if row.get("mean_aurc_eligible") is not None:
                raise PaperAssetsError(f"{label}.mean_aurc_eligible must be null with no eligible trajectories.")
        else:
            _number(row, "mean_aurc_eligible", label, minimum=0.0, maximum=1.0)
        if not _same_rate(all_rate, restraint, scheduled):
            raise PaperAssetsError(f"{label} scheduled-unit restraint rate does not reconcile.")
        if row.get("primary_denominator") != "all_scheduled_agent_days" or row.get("exploratory_only") is not True:
            raise PaperAssetsError(f"{label} changed its denominator/status contract.")
    return sorted(
        models.values(),
        key=lambda row: (-float(row["restraint_rate_all_scheduled"]), str(row["target_id"])),
    )


def _validate_role(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], dict[tuple[str, str], dict[str, Any]]]:
    if len(rows) != 18:
        raise PaperAssetsError("Role calibration requires exactly 18 model-frame rows.")
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    identity: dict[str, tuple[str, str]] = {}
    for index, raw in enumerate(rows):
        row = dict(raw)
        label = f"role_calibration_model_frames[{index}]"
        _required(
            row,
            (
                "target_id", "upstream_provider", "model", "frame_id", "scheduled_draws",
                "format_valid_first_attempt_count", "first_attempt_invalid_count",
                "repaired_invalid_count", "welfare_preserving_count_first_attempt",
                "welfare_preserving_rate_all_scheduled",
                "welfare_preserving_rate_among_first_attempt_valid",
                "primary_denominator", "ancillary_only", "frames_pooled",
            ),
            label,
        )
        target = _identity(row, "target_id", label)
        provider = _identity(row, "upstream_provider", label)
        model = _identity(row, "model", label)
        frame = _identity(row, "frame_id", label)
        if frame not in ROLE_FRAMES:
            raise PaperAssetsError(f"{label} has unknown frame {frame!r}.")
        previous_identity = identity.setdefault(target, (provider, model))
        if previous_identity != (provider, model):
            raise PaperAssetsError(f"Role calibration changes identity for {target!r}.")
        key = (target, frame)
        if key in by_key:
            raise PaperAssetsError(f"Role calibration duplicates {key!r}.")
        scheduled = _integer(row, "scheduled_draws", label, minimum=1)
        valid = _integer(row, "format_valid_first_attempt_count", label)
        invalid = _integer(row, "first_attempt_invalid_count", label)
        welfare = _integer(row, "welfare_preserving_count_first_attempt", label)
        if scheduled != 384 or valid + invalid != scheduled or welfare > valid:
            raise PaperAssetsError(f"{label} does not reconcile its 384 scheduled draws.")
        if _integer(row, "repaired_invalid_count", label) != 0:
            raise PaperAssetsError("Role repaired outcomes are outside this frozen asset contract.")
        _optional_reconciled_rate(
            row, "welfare_preserving_rate_among_first_attempt_valid", label, welfare, valid
        )
        if not _same_rate(_rate(row, "welfare_preserving_rate_all_scheduled", label), welfare, scheduled):
            raise PaperAssetsError(f"{label} welfare rates do not reconcile.")
        if (
            row.get("primary_denominator") != "all_scheduled_draws"
            or row.get("ancillary_only") is not True
            or row.get("frames_pooled") is not False
        ):
            raise PaperAssetsError(f"{label} changed its separate ancillary-frame contract.")
        by_key[key] = row
    if len(identity) != 6 or set(by_key) != {(target, frame) for target in identity for frame in ROLE_FRAMES}:
        raise PaperAssetsError("Role calibration is not a complete six-model x three-frame matrix.")
    return sorted(identity), by_key


def _validate_sensitivity(
    model_rows: Sequence[Mapping[str, Any]], effect_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[str], dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    models = _model_index(model_rows, "sensitivity_models", 6)
    for target, row in models.items():
        label = f"sensitivity_models[{target}]"
        _required(
            row,
            (
                "trajectory_count", "scheduled_agent_days", "first_attempt_invalid_count",
                "repaired_invalid_count", "inference_scope", "confirmatory", "exploratory_only",
            ),
            label,
        )
        if _integer(row, "trajectory_count", label) != 32:
            raise PaperAssetsError(f"{label} does not contain 16 cells x 2 seeds.")
        scheduled = _integer(row, "scheduled_agent_days", label, minimum=1)
        if scheduled != 2880:
            raise PaperAssetsError(
                f"{label} does not contain the frozen 2,880 scheduled agent-days."
            )
        invalid = _integer(row, "first_attempt_invalid_count", label)
        if invalid > scheduled or _integer(row, "repaired_invalid_count", label) != 0:
            raise PaperAssetsError(f"{label} has invalid sensitivity coverage accounting.")
        if row.get("inference_scope") != "deadline_exploratory" or row.get("confirmatory") is not False or row.get("exploratory_only") is not True:
            raise PaperAssetsError(f"{label} changed its exploratory sensitivity scope.")

    if len(effect_rows) != 30:
        raise PaperAssetsError("Sensitivity requires exactly 30 main-effect rows.")
    effects: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw in enumerate(effect_rows):
        row = dict(raw)
        label = f"sensitivity_main_effects[{index}]"
        _required(
            row,
            (
                "sentinel_id", "factor", "low_level", "high_level",
                "effect_high_minus_low", "standard_error", "raw_exact_p",
                "holm_adjusted_p", "within_sentinel_holm_adjusted_p",
                "within_sentinel_max_t_adjusted_p", "cell_count", "common_seed_count",
                "permutation_count", "design", "analysis_unit", "holm_family",
                "holm_family_size", "max_t_family", "inference_scope", "confirmatory",
            ),
            label,
        )
        sentinel = _identity(row, "sentinel_id", label)
        factor = _identity(row, "factor", label)
        if sentinel not in models or factor not in SENSITIVITY_FACTORS:
            raise PaperAssetsError(f"{label} has an unknown sentinel/factor.")
        key = (sentinel, factor)
        if key in effects:
            raise PaperAssetsError(f"Sensitivity duplicates {key!r}.")
        low, high = SENSITIVITY_LEVELS[factor]
        if row.get("low_level") != low or row.get("high_level") != high:
            raise PaperAssetsError(f"{label} changed the frozen low/high levels.")
        _number(row, "effect_high_minus_low", label, minimum=-1.0, maximum=1.0)
        _number(row, "standard_error", label, minimum=0.0)
        for field in (
            "raw_exact_p", "holm_adjusted_p", "within_sentinel_holm_adjusted_p",
            "within_sentinel_max_t_adjusted_p",
        ):
            _number(row, field, label, minimum=0.0, maximum=1.0)
        if (
            _integer(row, "cell_count", label) != 16
            or _integer(row, "common_seed_count", label) != 2
            or _integer(row, "permutation_count", label) != 4
            or row.get("design") != "2^(5-1)_resolution_V_I=ABCDE"
            or row.get("analysis_unit") != "environment_seed_block"
            or row.get("holm_family") != "30_prespecified_sentinel_by_factor_main_effects"
            or _integer(row, "holm_family_size", label) != 30
            or row.get("max_t_family") != "five_main_effects_within_sentinel_diagnostic"
            or row.get("inference_scope") != "deadline_exploratory"
            or row.get("confirmatory") is not False
        ):
            raise PaperAssetsError(f"{label} changed the frozen sensitivity/Holm contract.")
        effects[key] = row
    expected = {(target, factor) for target in models for factor in SENSITIVITY_FACTORS}
    if set(effects) != expected:
        raise PaperAssetsError("Sensitivity is not a complete six-model x five-factor matrix.")
    return sorted(models), models, effects


def _parameter_millions(value: str, label: str) -> float:
    if value.endswith("M"):
        multiplier = 1.0
    elif value.endswith("B"):
        multiplier = 1000.0
    else:
        raise PaperAssetsError(f"{label}.parameter_scale must end in M or B.")
    try:
        numeric = float(value[:-1]) * multiplier
    except ValueError as error:
        raise PaperAssetsError(f"{label}.parameter_scale is malformed.") from error
    if not math.isfinite(numeric) or numeric <= 0:
        raise PaperAssetsError(f"{label}.parameter_scale must be positive and finite.")
    return numeric


def _validate_local_controls(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate only the sanitized four-model aggregate; never follow its bindings."""

    artifact = _read_json(path, "sanitized local Part 1 controls")
    if (
        artifact.get("schema_version") != 1
        or artifact.get("artifact_type") != "local_hf_part1_exploratory_private_panel_analysis"
        or artifact.get("evidence_sha256") != _self_hash(artifact)
    ):
        raise PaperAssetsError("Local-control artifact type, schema, or self-hash failed.")
    if (
        artifact.get("analysis_role")
        != "exploratory_only_no_confirmatory_or_paper_promotion"
        or artifact.get("draft_bank_human_approved") is not False
        or artifact.get("confirmatory_or_paper_promotion_permitted") is not False
        or artifact.get("frontier_route_substitution_permitted") is not False
    ):
        raise PaperAssetsError("Local-control exploratory/nonpromotion contract changed.")
    coverage = artifact.get("coverage")
    if not isinstance(coverage, Mapping) or dict(coverage) != {
        "all_selected_models_complete": True,
        "game_domain_strata": 12,
        "model_count": 4,
        "retained_model_trial_count": 1536,
        "roots_per_model": 384,
        "roots_per_stratum_per_model": 32,
    }:
        raise PaperAssetsError("Local-control artifact is not the complete four-model x 384 matrix.")
    privacy = artifact.get("privacy_contract")
    if not isinstance(privacy, Mapping) or dict(privacy) != {
        "contains_prompt_text": False,
        "contains_raw_response_or_response_text": False,
        "contains_reasoning": False,
        "contains_only_identifiers_hash_bindings_and_derived_aggregates": True,
    }:
        raise PaperAssetsError("Local-control sanitized privacy contract changed.")
    parameters = artifact.get("parameters")
    if (
        not isinstance(parameters, Mapping)
        or parameters.get("invalid_handling")
        != "format_invalid_responses_retained_in_primary_denominator_as_non_x_and_noncooperation"
    ):
        raise PaperAssetsError("Local-control invalid-denominator contract changed.")
    bindings = artifact.get("bindings")
    if not isinstance(bindings, Mapping) or bindings.get("path_policy") != "portable_input_basename_only_no_host_absolute_paths":
        raise PaperAssetsError("Local-control portable sanitized binding contract changed.")
    # Paths are provenance strings only. Reject host-absolute values and never
    # dereference any binding from this sanitized artifact.
    for key, value in bindings.items():
        if key.endswith("_path") and (not isinstance(value, str) or Path(value).is_absolute()):
            raise PaperAssetsError("Local-control artifact contains a host-absolute binding path.")

    models = artifact.get("models")
    if not isinstance(models, list) or len(models) != 4:
        raise PaperAssetsError("Local controls require exactly four model rows.")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(models):
        if not isinstance(raw, Mapping):
            raise PaperAssetsError(f"local_controls.models[{index}] must be an object.")
        row = dict(raw)
        label = f"local_controls.models[{index}]"
        _required(row, ("model_id", "parameter_scale", "counts", "inference_status"), label)
        model_id = _identity(row, "model_id", label)
        parameter_scale = _identity(row, "parameter_scale", label)
        if model_id in seen:
            raise PaperAssetsError(f"Local controls duplicate exact model_id {model_id!r}.")
        seen.add(model_id)
        parameter_millions = _parameter_millions(parameter_scale, label)
        counts = row.get("counts")
        if not isinstance(counts, Mapping):
            raise PaperAssetsError(f"{label}.counts must be an object.")
        _required(
            counts,
            (
                "planned_and_retained_count", "format_valid_count", "format_invalid_count",
                "format_valid_rate",
                "primary_cooperation_rate_format_invalid_retained_as_noncooperation",
            ),
            f"{label}.counts",
        )
        scheduled = _integer(counts, "planned_and_retained_count", f"{label}.counts", minimum=1)
        valid = _integer(counts, "format_valid_count", f"{label}.counts")
        invalid = _integer(counts, "format_invalid_count", f"{label}.counts")
        validity = _rate(counts, "format_valid_rate", f"{label}.counts")
        welfare = _rate(
            counts,
            "primary_cooperation_rate_format_invalid_retained_as_noncooperation",
            f"{label}.counts",
        )
        if scheduled != 384 or valid + invalid != scheduled or not _same_rate(validity, valid, scheduled):
            raise PaperAssetsError(f"{label} does not reconcile n=384 and format validity.")
        welfare_count = welfare * scheduled
        if not math.isclose(welfare_count, round(welfare_count), abs_tol=1e-10):
            raise PaperAssetsError(f"{label} welfare rate is not an integer count over n=384.")
        if row.get("inference_status") != "exploratory_descriptive_fixed_local_model":
            raise PaperAssetsError(f"{label} changed its exploratory local-control status.")
        rows.append(
            {
                "model_id": model_id,
                "parameter_scale": parameter_scale,
                "parameter_millions": parameter_millions,
                "scheduled": scheduled,
                "welfare_rate": welfare,
                "welfare_count": int(round(welfare_count)),
                "valid_count": valid,
                "invalid_count": invalid,
                "validity_rate": validity,
            }
        )
    return artifact, sorted(rows, key=lambda row: (float(row["parameter_millions"]), str(row["model_id"])))


def _load_and_validate(input_dir: Path) -> dict[str, Any]:
    if not input_dir.is_dir():
        raise PaperAssetsError(f"Input analysis directory does not exist: {input_dir}")
    manifest = _validate_manifest(input_dir)
    tables = {
        name: _read_jsonl(input_dir / f"{name}.jsonl", name)
        for name in EXPECTED_ROW_COUNTS
    }
    for name, expected in EXPECTED_ROW_COUNTS.items():
        if len(tables[name]) != expected:
            raise PaperAssetsError(f"{name} row count differs from its manifest declaration.")
    aggregates = _read_json(input_dir / "figure_aggregates.json", "figure aggregates")
    required_aggregates = {
        "part0_by_model_language", "part1_by_model_game_domain", "part2_trajectories",
        "role_calibration_by_model_frame", "sensitivity_main_effects",
    }
    if not required_aggregates <= set(aggregates):
        raise PaperAssetsError("Figure aggregates are missing a required schema member.")
    if not all(isinstance(aggregates[key], list) for key in required_aggregates):
        raise PaperAssetsError("Figure aggregate members must be arrays.")
    if _canonical_bytes(aggregates["role_calibration_by_model_frame"]) != _canonical_bytes(tables["role_calibration_model_frames"]):
        raise PaperAssetsError("Role table and figure aggregate rows disagree.")
    if _canonical_bytes(aggregates["sensitivity_main_effects"]) != _canonical_bytes(tables["sensitivity_main_effects"]):
        raise PaperAssetsError("Sensitivity table and figure aggregate rows disagree.")

    part0, part0_matrix = _validate_part0(
        tables["part0_models"], aggregates["part0_by_model_language"]
    )
    part1 = _validate_part1(tables["part1_models"])
    part2 = _validate_part2(tables["part2_models"])
    role_targets, role = _validate_role(tables["role_calibration_model_frames"])
    sensitivity_targets, sensitivity_models, sensitivity = _validate_sensitivity(
        tables["sensitivity_models"], tables["sensitivity_main_effects"]
    )
    if role_targets != sensitivity_targets:
        raise PaperAssetsError("Role and sensitivity sentinel target sets differ.")
    for target in role_targets:
        role_row = role[(target, ROLE_FRAMES[0])]
        sensitivity_row = sensitivity_models[target]
        if (
            role_row["upstream_provider"] != sensitivity_row["upstream_provider"]
            or role_row["model"] != sensitivity_row["model"]
        ):
            raise PaperAssetsError(f"Role/sensitivity identity differs for {target!r}.")
    return {
        "manifest": manifest,
        "part0": part0,
        "part0_matrix": part0_matrix,
        "part1": part1,
        "part2": part2,
        "role_targets": role_targets,
        "role": role,
        "sensitivity_targets": sensitivity_targets,
        "sensitivity_models": sensitivity_models,
        "sensitivity": sensitivity,
    }


def _label(row: Mapping[str, Any]) -> str:
    return f"{row['target_id']} | {row['model']}"


def _style_axes(ax: plt.Axes) -> None:
    ax.tick_params(colors=INK, labelcolor=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.7)


def _save_figure(fig: plt.Figure, directory: Path, stem: str, title: str) -> list[Path]:
    pdf = directory / f"{stem}.pdf"
    png = directory / f"{stem}.png"
    fig.savefig(
        pdf,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.08,
        metadata={"Title": title, "Author": "Safety Beyond Refusal asset generator", "Creator": "Matplotlib"},
    )
    fig.savefig(png, format="png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return [pdf, png]


def _heatmap(
    ax: plt.Axes,
    matrix: Sequence[Sequence[float]],
    xlabels: Sequence[str],
    ylabels: Sequence[str],
    *,
    title: str,
    cmap: LinearSegmentedColormap,
    vmin: float,
    vmax: float,
    percent: bool = True,
) -> None:
    mesh = ax.pcolormesh(
        matrix, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat",
        edgecolors="white", linewidth=0.8, rasterized=False,
    )
    ax.set_xlim(0, len(xlabels))
    ax.set_ylim(0, len(ylabels))
    ax.invert_yaxis()
    ax.set_xticks([index + 0.5 for index in range(len(xlabels))], labels=xlabels)
    ax.set_yticks([index + 0.5 for index in range(len(ylabels))], labels=ylabels)
    ax.tick_params(axis="x", labelrotation=25, labelsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    for row_index, values in enumerate(matrix):
        for column_index, value in enumerate(values):
            label = f"{value:.0%}" if percent else f"{value:+.3f}"
            midpoint = (vmin + vmax) / 2.0
            color = "white" if abs(value - midpoint) > (vmax - vmin) * 0.34 else INK
            ax.text(column_index + 0.5, row_index + 0.5, label, ha="center", va="center", fontsize=6, color=color)
    colorbar = ax.figure.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
    # Matplotlib rasterizes colorbar solids above its internal segment count by
    # default.  These paper PDFs promise vector-only marks, so override that
    # backend optimization explicitly.
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    if percent:
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    colorbar.ax.tick_params(labelsize=7, colors=INK)
    _style_axes(ax)


def _plot_part0(data: Mapping[str, Any], directory: Path) -> list[Path]:
    rows = data["part0"]
    by_key = data["part0_matrix"]
    labels = [_label(row) for row in rows]
    refusal = [
        [float(by_key[(row["target_id"], language)]["refusal_rate_all_scheduled"]) for language in PART0_LANGUAGES]
        for row in rows
    ]
    validity = [
        [1.0 - int(by_key[(row["target_id"], language)]["invalid_count"]) / int(by_key[(row["target_id"], language)]["scheduled_units"]) for language in PART0_LANGUAGES]
        for row in rows
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 10.6), sharey=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Part 0 response-language outcomes by exact model route", x=0.08, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.08, 0.955, "Each cell uses 48 scheduled harmful-request roots; rows are ordered by overall within-task refusal rate.", fontsize=9, color=MUTED)
    _heatmap(axes[0], refusal, PART0_LANGUAGES, labels, title="Refusal rate / all scheduled roots", cmap=RATE_CMAP, vmin=0.0, vmax=1.0)
    _heatmap(axes[1], validity, PART0_LANGUAGES, labels, title="Valid-output coverage", cmap=VALID_CMAP, vmin=0.0, vmax=1.0)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.text(0.08, 0.018, "Higher refusal means less assistance on this harmful-request task; higher validity means fewer invalid outputs. Neither panel is a general safety score.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0.06, 0.045, 0.99, 0.94), w_pad=2.2)
    return _save_figure(fig, directory, "part0_model_language", "Part 0 model by language outcomes")


def _lollipop_panel(
    ax: plt.Axes,
    values: Sequence[float | None],
    labels: Sequence[str],
    *,
    title: str,
    color: str,
    show_labels: bool,
) -> None:
    positions = list(range(len(values)))
    estimable = [(position, value) for position, value in zip(positions, values, strict=True) if value is not None]
    ax.hlines(
        [position for position, _ in estimable], 0.0,
        [value for _, value in estimable], color=GRID, linewidth=1.0, zorder=1,
    )
    ax.scatter(
        [value for _, value in estimable], [position for position, _ in estimable],
        s=19, color=color, edgecolor=INK, linewidth=0.35, zorder=2,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.8, len(values) - 0.2)
    ax.invert_yaxis()
    if show_labels:
        ax.set_yticks(positions, labels=labels)
        ax.tick_params(axis="y", labelsize=5.6, length=0, labelleft=True)
    else:
        ax.set_yticks(positions)
        ax.tick_params(axis="y", labelsize=5.6, length=0, labelleft=False)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    for position, value in zip(positions, values, strict=True):
        if value is None:
            ax.scatter([0.015], [position], marker="x", s=17, color=MUTED, linewidth=0.8, zorder=2)
            ax.text(0.03, position, "NE", va="center", ha="left", fontsize=5.4, color=MUTED)
        else:
            ax.text(min(value + 0.012, 0.985), position, f"{value:.1%}", va="center", ha="left" if value < 0.95 else "right", fontsize=5.4, color=INK)
    _style_axes(ax)


def _plot_part1(data: Mapping[str, Any], directory: Path) -> list[Path]:
    rows = data["part1"]
    labels = [_label(row) for row in rows]
    welfare = [float(row["welfare_preserving_rate_all_scheduled"]) for row in rows]
    validity = [1.0 - int(row["first_attempt_invalid_count"]) / int(row["scheduled_units"]) for row in rows]
    # Keep separate y-axis formatters: sharing the formatter allows an unlabeled
    # right panel to erase the exact route/model labels on the left panel.
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 20.5), sharey=False)
    fig.patch.set_facecolor("white")
    fig.suptitle("Part 1 self-choice outcomes for all 75 exact model routes", x=0.08, y=0.995, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.08, 0.973, "One row per route; 384 scheduled units per route; ordered by within-task welfare-preserving rate.", fontsize=9, color=MUTED)
    _lollipop_panel(axes[0], welfare, labels, title="Welfare-preserving / all scheduled units", color=BLUE, show_labels=True)
    _lollipop_panel(axes[1], validity, labels, title="Valid first-attempt coverage", color=GREEN, show_labels=False)
    fig.text(0.08, 0.012, "Higher welfare preservation means fewer counterpart costs in this self-choice task; higher validity means fewer invalid outputs. Neither is a general safety ranking.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0.055, 0.028, 0.99, 0.965), w_pad=2.0)
    return _save_figure(fig, directory, "part1_all_models", "Part 1 all-model outcomes")


def _plot_part2(data: Mapping[str, Any], directory: Path) -> list[Path]:
    rows = data["part2"]
    labels = [_label(row) for row in rows]
    restraint = [float(row["restraint_rate_all_scheduled"]) for row in rows]
    aurc = [
        None if row["mean_aurc_eligible"] is None else float(row["mean_aurc_eligible"])
        for row in rows
    ]
    validity = [1.0 - int(row["first_attempt_invalid_count"]) / int(row["scheduled_agent_days"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 9.2), sharey=False)
    fig.patch.set_facecolor("white")
    fig.suptitle("Part 2 commons outcomes for 19 exact model routes", x=0.075, y=0.995, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.075, 0.953, "One row per route; 12 trajectories per route; ordered by within-task restraint rate.", fontsize=9, color=MUTED)
    _lollipop_panel(axes[0], restraint, labels, title="Restraint / scheduled agent-days", color=BLUE, show_labels=True)
    _lollipop_panel(axes[1], aurc, labels, title="Mean normalized AURC / eligible trajectories", color=BLUE, show_labels=False)
    _lollipop_panel(axes[2], validity, labels, title="Valid first-attempt coverage", color=GREEN, show_labels=False)
    fig.text(0.075, 0.018, "Higher restraint and AURC mean more reserve preservation in this simulator; higher validity means fewer invalid actions. These are not general safety scores.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0.055, 0.045, 0.995, 0.94), w_pad=1.8)
    return _save_figure(fig, directory, "part2_all_models", "Part 2 all-model outcomes")


def _plot_role(data: Mapping[str, Any], directory: Path) -> list[Path]:
    targets = data["role_targets"]
    by_key = data["role"]
    labels = [_label(by_key[(target, ROLE_FRAMES[0])]) for target in targets]
    welfare = [[float(by_key[(target, frame)]["welfare_preserving_rate_all_scheduled"]) for frame in ROLE_FRAMES] for target in targets]
    validity = [[1.0 - int(by_key[(target, frame)]["first_attempt_invalid_count"]) / int(by_key[(target, frame)]["scheduled_draws"]) for frame in ROLE_FRAMES] for target in targets]
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.6), sharey=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Part 1 role-calibration outcomes by exact sentinel route and frame", x=0.08, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.08, 0.91, "Six sentinels x three separate frames; 384 scheduled draws per route-frame; frames are not pooled.", fontsize=9, color=MUTED)
    _heatmap(axes[0], welfare, ROLE_FRAMES, labels, title="Welfare-preserving / all scheduled draws", cmap=RATE_CMAP, vmin=0.0, vmax=1.0)
    _heatmap(axes[1], validity, ROLE_FRAMES, labels, title="Valid first-attempt coverage", cmap=VALID_CMAP, vmin=0.0, vmax=1.0)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.text(0.08, 0.025, "Higher welfare preservation means fewer counterpart costs within that role-conditioned task; higher validity means fewer invalid outputs. Frame differences are descriptive, not causal.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0.055, 0.07, 0.995, 0.87), w_pad=2.2)
    return _save_figure(fig, directory, "part1_role_calibration", "Part 1 role-calibration outcomes")


def _plot_sensitivity(data: Mapping[str, Any], directory: Path) -> list[Path]:
    targets = data["sensitivity_targets"]
    models = data["sensitivity_models"]
    effects = data["sensitivity"]
    labels = [_label(models[target]) for target in targets]
    matrix = [[float(effects[(target, factor)]["effect_high_minus_low"]) for factor in SENSITIVITY_FACTORS] for target in targets]
    limit = max(0.02, max(abs(value) for row in matrix for value in row))
    fig, ax = plt.subplots(figsize=(14.8, 6.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Part 2 exploratory sensitivity: high-minus-low normalized-AURC effects", x=0.09, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.09, 0.91, "Six sentinels x five prespecified factors; 16 resolution-V cells and two common seeds per sentinel; Holm family = 30.", fontsize=9, color=MUTED)
    mesh = ax.pcolormesh(
        matrix, cmap=SIGNED_CMAP, norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        shading="flat", edgecolors="white", linewidth=1.0, rasterized=False,
    )
    ax.set_xlim(0, len(SENSITIVITY_FACTORS))
    ax.set_ylim(0, len(targets))
    ax.invert_yaxis()
    ax.set_xticks([index + 0.5 for index in range(len(SENSITIVITY_FACTORS))], labels=SENSITIVITY_FACTORS)
    ax.set_yticks([index + 0.5 for index in range(len(targets))], labels=labels)
    ax.tick_params(axis="x", labelrotation=22, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    for row_index, target in enumerate(targets):
        for column_index, factor in enumerate(SENSITIVITY_FACTORS):
            effect = effects[(target, factor)]
            value = float(effect["effect_high_minus_low"])
            status = "H" if float(effect["holm_adjusted_p"]) <= 0.05 else "n.s."
            color = "white" if abs(value) > limit * 0.58 else INK
            ax.text(column_index + 0.5, row_index + 0.5, f"{value:+.3f}\n{status}", ha="center", va="center", fontsize=7, color=color)
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.027, pad=0.025)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    colorbar.set_label("High - low normalized AURC", fontsize=8, color=INK)
    colorbar.ax.tick_params(labelsize=7, colors=INK)
    _style_axes(ax)
    fig.text(0.09, 0.025, "H: global Holm-adjusted p <= 0.05; n.s.: otherwise. Positive/negative indicates effect direction only and is not automatically good/bad for a parameter factor.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0.06, 0.075, 0.99, 0.87))
    return _save_figure(fig, directory, "part2_sensitivity_effects", "Part 2 sensitivity main effects")


def _plot_local_controls(data: Mapping[str, Any], directory: Path) -> list[Path]:
    rows = data["local_controls"]
    labels = [f"{row['model_id']} | {row['parameter_scale']}" for row in rows]
    welfare = [float(row["welfare_rate"]) for row in rows]
    validity = [float(row["validity_rate"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8), sharey=False)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Part 1 exploratory local execution-scale controls", x=0.085, y=0.99,
        ha="left", fontsize=15, fontweight="bold", color=INK,
    )
    fig.text(
        0.085, 0.91,
        "Four fixed offline HF models; one row per exact model ID and parameter scale; n = 384 scheduled units each.",
        fontsize=9, color=MUTED,
    )
    _lollipop_panel(
        axes[0], welfare, labels,
        title="Welfare-preserving / all 384 scheduled units", color=BLUE, show_labels=True,
    )
    _lollipop_panel(
        axes[1], validity, labels,
        title="Format-valid / all 384 scheduled units", color=GREEN, show_labels=False,
    )
    fig.text(
        0.085, 0.025,
        "Invalid outputs remain nonsuccesses for welfare preservation. Higher values are preferable only within this task; these exploratory scale controls are separate from, and not substitutes for, hosted-route or confirmatory evidence.",
        fontsize=8, color=MUTED,
    )
    fig.tight_layout(rect=(0.055, 0.095, 0.99, 0.84), w_pad=2.0)
    return _save_figure(
        fig, directory, "part1_local_controls", "Part 1 exploratory local execution-scale controls"
    )


def _tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}\\%"


def _table_tex(
    *,
    caption: str,
    label: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    column_spec: str,
    chunk_size: int,
) -> str:
    chunks = [rows[index : index + chunk_size] for index in range(0, len(rows), chunk_size)]
    blocks: list[str] = ["% Generated file. Requires booktabs and graphicx. Do not edit by hand.\n"]
    for chunk_index, chunk in enumerate(chunks, start=1):
        continued = f" Display block {chunk_index} of {len(chunks)}." if len(chunks) > 1 else ""
        blocks.extend(
            [
                "\\par\\addvspace{15pt}\n",
                "\\begin{table*}[tbp]\n",
                "\\centering\n",
                f"\\caption{{{caption}{continued}}}\n",
            ]
        )
        if chunk_index == 1:
            blocks.append(f"\\label{{{label}}}\n")
        blocks.extend(
            [
                "\\scriptsize\n",
                "\\setlength{\\tabcolsep}{3.5pt}\n",
                "\\renewcommand{\\arraystretch}{1.08}\n",
                "\\resizebox{\\textwidth}{!}{%\n",
                f"\\begin{{tabular}}{{{column_spec}}}\n",
                "\\toprule\n",
                " & ".join(headers) + " \\\\\n",
                "\\midrule\n",
            ]
        )
        blocks.extend(" & ".join(row) + " \\\\\n" for row in chunk)
        blocks.extend(
            [
                "\\bottomrule\n",
                "\\end{tabular}%\n",
                "}\n",
                "\\end{table*}\n",
                "\\par\\addvspace{15pt}\n",
            ]
        )
    return "".join(blocks)


def _write_tables(data: Mapping[str, Any], directory: Path) -> list[Path]:
    output: list[Path] = []

    phase_indices = {
        "part0": {str(row["target_id"]): row for row in data["part0"]},
        "part1": {str(row["target_id"]): row for row in data["part1"]},
        "part2": {str(row["target_id"]): row for row in data["part2"]},
    }
    cross_phase_rows = []
    all_targets = sorted(set().union(*(set(index) for index in phase_indices.values())))
    for target in all_targets:
        present = [index[target] for index in phase_indices.values() if target in index]
        identities = {
            (str(row["upstream_provider"]), str(row["model"])) for row in present
        }
        if len(identities) != 1:
            raise PaperAssetsError(
                f"Cross-phase exact identity differs for target {target!r}."
            )
        provider, model = next(iter(identities))
        p0 = phase_indices["part0"].get(target)
        p1 = phase_indices["part1"].get(target)
        p2 = phase_indices["part2"].get(target)
        p0_cell = "-- (not in panel)"
        if p0 is not None:
            scheduled = int(p0["scheduled_units"])
            invalid = int(p0["first_attempt_invalid_count"])
            p0_cell = (
                f"R {_pct(float(p0['refusal_rate_all_scheduled']))}; "
                f"V {_pct(1.0 - invalid / scheduled)}"
            )
        p1_cell = "-- (not in panel)"
        if p1 is not None:
            scheduled = int(p1["scheduled_units"])
            invalid = int(p1["first_attempt_invalid_count"])
            p1_cell = (
                f"W {_pct(float(p1['welfare_preserving_rate_all_scheduled']))}; "
                f"V {_pct(1.0 - invalid / scheduled)}"
            )
        p2_cell = "-- (not in panel)"
        if p2 is not None:
            scheduled = int(p2["scheduled_agent_days"])
            invalid = int(p2["first_attempt_invalid_count"])
            aurc = (
                "NE"
                if p2["mean_aurc_eligible"] is None
                else f"{float(p2['mean_aurc_eligible']):.3f}"
            )
            p2_cell = (
                f"R {_pct(float(p2['restraint_rate_all_scheduled']))}; "
                f"A {aurc}; V {_pct(1.0 - invalid / scheduled)}"
            )
        cross_phase_rows.append(
            [
                _tex_escape(target),
                _tex_escape(provider),
                _tex_escape(model),
                p0_cell,
                p1_cell,
                p2_cell,
            ]
        )
    path = directory / "all_models_cross_phase_table.tex"
    path.write_text(
        _table_tex(
            caption=(
                "Exact-route cross-phase result matrix. Each row is one exact target route in the union of the three primary hosted panels; Provider and Model ID preserve the authenticated upstream identity. Part 0 reports material refusal R and valid first-attempt coverage V over 144 scheduled responses; higher R means less harmful-request assistance and higher V means fewer invalid outputs. Part 1 reports welfare-preserving self-choice W and V over 384 scheduled roots; higher W means fewer counterpart costs and higher V means fewer invalid outputs. Part 2 reports all-scheduled restraint R, eligible-trajectory normalized AURC A, and V over 12 common-seed trajectories; higher restraint and AURC mean greater reserve preservation and higher V means fewer invalid actions. NE means no operationally eligible AURC trajectory, while -- means the exact route was not in that phase panel. The columns are juxtaposed for coverage and inspection only: they remain different estimands, no cell is imputed, and no composite or general safety ranking is computed."
            ),
            label="tab:provider-safe-v2-all-models-cross-phase",
            headers=("Target route ID", "Provider", "Model ID", "Part 0: R; V", "Part 1: W; V", "Part 2: R; A; V"),
            rows=cross_phase_rows,
            column_spec="lllrrr",
            chunk_size=24,
        ),
        encoding="utf-8",
    )
    output.append(path)

    part0_rows = []
    for order, row in enumerate(data["part0"], start=1):
        values = [str(order), _tex_escape(row["target_id"]), _tex_escape(row["upstream_provider"]), _tex_escape(row["model"])]
        for language in PART0_LANGUAGES:
            aggregate = data["part0_matrix"][(row["target_id"], language)]
            values.extend(
                [
                    _pct(float(aggregate["refusal_rate_all_scheduled"])),
                    f"{aggregate['invalid_count']}/{aggregate['scheduled_units']}",
                ]
            )
        part0_rows.append(values)
    path = directory / "part0_model_language_table.tex"
    path.write_text(
        _table_tex(
            caption=(
                "Part 0 exact-route response-language outcomes. Each row is one target route, shown in descending overall refusal-rate display order; Provider and Model are the exact upstream identifiers. For each response-language column (english, chinese, russian), R is REFUSAL divided by all 48 scheduled harmful-request roots and Invalid/scheduled is invalid-output coverage. Higher R means less assistance on this harmful-request task; lower R means more compliance, unclear, or invalid outcomes. Lower invalid coverage is operationally preferable. These directions apply only within Part 0 and are not a general safety ranking."
            ),
            label="tab:provider-safe-v2-part0-model-language",
            headers=("Order", "Target route ID", "Provider", "Model ID", "english R", "english invalid/scheduled", "chinese R", "chinese invalid/scheduled", "russian R", "russian invalid/scheduled"),
            rows=part0_rows,
            column_spec="rlllrrrrrr",
            chunk_size=22,
        ),
        encoding="utf-8",
    )
    output.append(path)

    local_rows = []
    for row in data["local_controls"]:
        local_rows.append(
            [
                _tex_escape(row["model_id"]), _tex_escape(row["parameter_scale"]),
                str(row["scheduled"]),
                f"{row['welfare_count']}/{row['scheduled']} ({_pct(float(row['welfare_rate']))})",
                f"{row['invalid_count']}/{row['scheduled']}",
                f"{row['valid_count']}/{row['scheduled']} ({_pct(float(row['validity_rate']))})",
            ]
        )
    path = directory / "part1_local_controls_table.tex"
    path.write_text(
        _table_tex(
            caption=(
                "Part 1 exploratory local execution-scale controls, reported separately from hosted model routes. Each row is one exact offline HF Model ID; Parameter scale is the advertised model parameter count; n is 384 scheduled trial units per model. Welfare/scheduled is the welfare-preserving count divided by all 384 units, with format-invalid outputs retained as nonsuccesses; Invalid/scheduled is the invalid count and denominator; Format validity is the valid-format count divided by 384. Higher welfare preservation means fewer counterpart costs in this task and higher format validity means fewer malformed outputs; lower values mean the converse only within these operational measures. These four fixed local models are exploratory execution-scale controls, not substitutes for hosted routes and not confirmatory or general safety evidence."
            ),
            label="tab:provider-safe-v2-part1-local-controls",
            headers=("Model ID", "Parameter scale", "n", "Welfare/scheduled", "Invalid/scheduled", "Format validity"),
            rows=local_rows,
            column_spec="llrrrr",
            chunk_size=4,
        ),
        encoding="utf-8",
    )
    output.append(path)

    part1_rows = []
    for order, row in enumerate(data["part1"], start=1):
        scheduled = int(row["scheduled_units"])
        invalid = int(row["first_attempt_invalid_count"])
        part1_rows.append(
            [
                str(order), _tex_escape(row["target_id"]), _tex_escape(row["upstream_provider"]),
                _tex_escape(row["model"]), str(scheduled),
                _pct(float(row["welfare_preserving_rate_all_scheduled"])),
                f"{invalid}/{scheduled}", _pct(1.0 - invalid / scheduled),
            ]
        )
    path = directory / "part1_all_models_table.tex"
    path.write_text(
        _table_tex(
            caption=(
                "Part 1 self-choice outcomes for all 75 exact model routes. Each row is one target route, shown in descending within-task welfare-preserving-rate display order; Provider and Model are exact upstream identifiers. Scheduled is the number of retained trial units. Welfare/scheduled is the first-attempt welfare-preserving count divided by all scheduled units; Invalid/scheduled gives the invalid count and denominator; Valid coverage is one minus that invalid fraction. Higher welfare preservation means fewer counterpart costs in this task, and higher validity means fewer invalid outputs. These are separate operational/behavioral measures within Part 1, not a general safety or cross-axis ranking."
            ),
            label="tab:provider-safe-v2-part1-all-models",
            headers=("Order", "Target route ID", "Provider", "Model ID", "Scheduled", "Welfare/scheduled", "Invalid/scheduled", "Valid coverage"),
            rows=part1_rows,
            column_spec="rlllrrrr",
            chunk_size=25,
        ),
        encoding="utf-8",
    )
    output.append(path)

    part2_rows = []
    for order, row in enumerate(data["part2"], start=1):
        scheduled = int(row["scheduled_agent_days"])
        invalid = int(row["first_attempt_invalid_count"])
        part2_rows.append(
            [
                str(order), _tex_escape(row["target_id"]), _tex_escape(row["upstream_provider"]),
                _tex_escape(row["model"]), str(row["trajectory_count"]), str(scheduled),
                _pct(float(row["restraint_rate_all_scheduled"])),
                "NE" if row["mean_aurc_eligible"] is None else f"{float(row['mean_aurc_eligible']):.3f}",
                f"{invalid}/{scheduled}", _pct(1.0 - invalid / scheduled),
            ]
        )
    path = directory / "part2_all_models_table.tex"
    path.write_text(
        _table_tex(
            caption=(
                "Part 2 commons outcomes for all 19 exact model routes. Each row is one target route, shown in descending within-task restraint-rate display order; Provider and Model are exact upstream identifiers. Traj. is the trajectory count; Agent-days is the scheduled action denominator; Restraint/agent-days retains first-attempt invalid actions in that denominator; Mean AURC is normalized reserve area over operationally eligible trajectories and NE means no eligible trajectory; Invalid/agent-days gives invalid count and denominator; Valid coverage is one minus that invalid fraction. Higher restraint and AURC mean more reserve preservation in this simulator, while higher validity means fewer invalid actions. These directions apply only within Part 2 and do not imply general safety."
            ),
            label="tab:provider-safe-v2-part2-all-models",
            headers=("Order", "Target route ID", "Provider", "Model ID", "Traj.", "Agent-days", "Restraint/agent-days", "Mean AURC", "Invalid/agent-days", "Valid coverage"),
            rows=part2_rows,
            column_spec="rlllrrrrrr",
            chunk_size=19,
        ),
        encoding="utf-8",
    )
    output.append(path)

    role_rows = []
    for target in data["role_targets"]:
        base = data["role"][(target, ROLE_FRAMES[0])]
        values = [_tex_escape(target), _tex_escape(base["upstream_provider"]), _tex_escape(base["model"])]
        for frame in ROLE_FRAMES:
            row = data["role"][(target, frame)]
            values.extend(
                [
                    _pct(float(row["welfare_preserving_rate_all_scheduled"])),
                    f"{row['first_attempt_invalid_count']}/{row['scheduled_draws']}",
                ]
            )
        role_rows.append(values)
    path = directory / "part1_role_calibration_table.tex"
    path.write_text(
        _table_tex(
            caption=(
                "Part 1 exploratory role calibration for six exact sentinel routes and three separate frames. Each row is one target route; Provider and Model are exact upstream identifiers. For advice, observer\\_evaluation, and prediction, W/scheduled is the first-attempt welfare-preserving count divided by all 384 scheduled draws and Invalid/scheduled is the invalid count and denominator. Higher W/scheduled means fewer counterpart costs within that role-conditioned task; lower invalid coverage means better operational validity. The frames ask different questions, are not pooled, and their differences are descriptive rather than causal. Neither high nor low values imply general safety outside this task."
            ),
            label="tab:provider-safe-v2-part1-role-calibration",
            headers=("Target route ID", "Provider", "Model ID", "advice W/scheduled", "advice invalid/scheduled", "observer\\_evaluation W/scheduled", "observer\\_evaluation invalid/scheduled", "prediction W/scheduled", "prediction invalid/scheduled"),
            rows=role_rows,
            column_spec="lllrrrrrr",
            chunk_size=6,
        ),
        encoding="utf-8",
    )
    output.append(path)

    sensitivity_rows = []
    for target in data["sensitivity_targets"]:
        model = data["sensitivity_models"][target]
        for factor in SENSITIVITY_FACTORS:
            row = data["sensitivity"][(target, factor)]
            holm = float(row["holm_adjusted_p"])
            sensitivity_rows.append(
                [
                    _tex_escape(target), _tex_escape(model["model"]), _tex_escape(factor),
                    _tex_escape(row["low_level"]), _tex_escape(row["high_level"]),
                    f"{float(row['effect_high_minus_low']):+.4f}", f"{holm:.4f}",
                    "H" if holm <= 0.05 else "n.s.",
                ]
            )
    path = directory / "part2_sensitivity_effects_table.tex"
    path.write_text(
        _table_tex(
            caption=(
                "Part 2 deadline-exploratory sensitivity main effects. Each row is one exact sentinel route-factor estimate; Model is the exact upstream model identifier; Factor names the varied parameter; Low and High are the frozen numeric levels; Effect is mean normalized AURC at High minus mean normalized AURC at Low over the resolution-V design and two common environment-seed blocks; Holm p is the adjustment over all 30 sentinel-by-factor tests; Status is H when Holm p is at most 0.05 and n.s. otherwise. Positive effects mean the high factor level increased reserve preservation in this simulator and negative effects mean it decreased preservation. Positive/negative is not automatically good/bad for a parameter factor, especially depletion and death rate, and no cell is a general safety score. With only two common seeds, the panel is underpowered and descriptive; Holm values document the prespecified family rather than support confirmatory claims."
            ),
            label="tab:provider-safe-v2-part2-sensitivity",
            headers=("Target route ID", "Model ID", "Factor", "Low", "High", "Effect (high-low AURC)", "Holm p", "Status"),
            rows=sensitivity_rows,
            column_spec="lllrrrrl",
            chunk_size=15,
        ),
        encoding="utf-8",
    )
    output.append(path)
    return output


def _summary(values: Sequence[float]) -> tuple[float, float, float] | None:
    numeric = [float(value) for value in values]
    if not numeric:
        return None
    if any(not math.isfinite(value) for value in numeric):
        raise PaperAssetsError("Headline summary received a nonfinite value.")
    return min(numeric), float(statistics.median(numeric)), max(numeric)


def _percent_headline(value: float) -> str:
    return f"{100.0 * value:.1f}"


def _decimal_headline(value: float, places: int) -> str:
    return f"{value:.{places}f}"


def _summary_macros(
    values: Sequence[float],
    *,
    prefix: str,
    formatter,
) -> list[tuple[str, str]]:
    summary = _summary(values)
    if summary is None:
        return [
            (f"{prefix}Minimum", "NE"),
            (f"{prefix}Median", "NE"),
            (f"{prefix}Maximum", "NE"),
        ]
    minimum, median, maximum = summary
    return [
        (f"{prefix}Minimum", formatter(minimum)),
        (f"{prefix}Median", formatter(median)),
        (f"{prefix}Maximum", formatter(maximum)),
    ]


def _headline_values(data: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return deterministic, within-task TeX macro names and scalar values."""

    part0 = data["part0"]
    part1 = data["part1"]
    part2 = data["part2"]
    role_targets = data["role_targets"]
    role = data["role"]
    sensitivity_models = data["sensitivity_models"]
    sensitivity = data["sensitivity"]

    part0_scheduled = sum(int(row["scheduled_units"]) for row in part0)
    part0_refusal = sum(int(row["refusal_count"]) for row in part0)
    part0_compliance = sum(int(row["compliance_count"]) for row in part0)
    part0_unclear = sum(int(row["unclear_count"]) for row in part0)
    part0_invalid = sum(int(row["first_attempt_invalid_count"]) for row in part0)
    if part0_refusal + part0_compliance + part0_unclear + part0_invalid != part0_scheduled:
        raise PaperAssetsError("Part 0 headline totals do not reconcile to scheduled responses.")

    part1_scheduled = sum(int(row["scheduled_units"]) for row in part1)
    part1_welfare = sum(int(row["welfare_preserving_count_first_attempt"]) for row in part1)
    part1_invalid = sum(int(row["first_attempt_invalid_count"]) for row in part1)
    if part1_welfare > part1_scheduled - part1_invalid:
        raise PaperAssetsError("Part 1 headline welfare count exceeds valid scheduled units.")

    part2_trajectories = sum(int(row["trajectory_count"]) for row in part2)
    part2_eligible_trajectories = sum(
        int(row["operationally_eligible_trajectory_count"]) for row in part2
    )
    part2_scheduled_agent_days = sum(int(row["scheduled_agent_days"]) for row in part2)
    part2_invalid_agent_days = sum(int(row["first_attempt_invalid_count"]) for row in part2)
    part2_valid_agent_days = part2_scheduled_agent_days - part2_invalid_agent_days
    part2_nonestimable = sum(row["mean_aurc_eligible"] is None for row in part2)
    if part2_eligible_trajectories > part2_trajectories or part2_valid_agent_days < 0:
        raise PaperAssetsError("Part 2 headline trajectory/agent-day totals do not reconcile.")
    part2_aurc = [
        float(row["mean_aurc_eligible"])
        for row in part2
        if row["mean_aurc_eligible"] is not None
    ]

    values: list[tuple[str, str]] = [
        ("ProviderSafePartZeroModelCount", str(len(part0))),
        ("ProviderSafePartZeroScheduledResponseCount", str(part0_scheduled)),
        ("ProviderSafePartZeroRefusalCount", str(part0_refusal)),
        ("ProviderSafePartZeroComplianceCount", str(part0_compliance)),
        ("ProviderSafePartZeroUnclearCount", str(part0_unclear)),
        ("ProviderSafePartZeroInvalidCount", str(part0_invalid)),
        *_summary_macros(
            [float(row["refusal_rate_all_scheduled"]) for row in part0],
            prefix="ProviderSafePartZeroModelRefusalRatePct",
            formatter=_percent_headline,
        ),
        ("ProviderSafePartOneModelCount", str(len(part1))),
        ("ProviderSafePartOneScheduledUnitCount", str(part1_scheduled)),
        ("ProviderSafePartOneWelfarePreservingCount", str(part1_welfare)),
        ("ProviderSafePartOneInvalidCount", str(part1_invalid)),
        *_summary_macros(
            [float(row["welfare_preserving_rate_all_scheduled"]) for row in part1],
            prefix="ProviderSafePartOneModelWelfareRatePct",
            formatter=_percent_headline,
        ),
        ("ProviderSafePartTwoModelCount", str(len(part2))),
        ("ProviderSafePartTwoTrajectoryCount", str(part2_trajectories)),
        ("ProviderSafePartTwoOperationallyEligibleTrajectoryCount", str(part2_eligible_trajectories)),
        ("ProviderSafePartTwoOperationallyIneligibleTrajectoryCount", str(part2_trajectories - part2_eligible_trajectories)),
        ("ProviderSafePartTwoScheduledAgentDayCount", str(part2_scheduled_agent_days)),
        ("ProviderSafePartTwoValidAgentDayCount", str(part2_valid_agent_days)),
        ("ProviderSafePartTwoInvalidAgentDayCount", str(part2_invalid_agent_days)),
        ("ProviderSafePartTwoNonestimableModelCount", str(part2_nonestimable)),
        *_summary_macros(
            part2_aurc,
            prefix="ProviderSafePartTwoModelNormalizedAURC",
            formatter=lambda value: _decimal_headline(value, 3),
        ),
        *_summary_macros(
            [float(row["restraint_rate_all_scheduled"]) for row in part2],
            prefix="ProviderSafePartTwoModelRestraintRatePct",
            formatter=_percent_headline,
        ),
    ]

    frame_macro_names = {
        "advice": "Advice",
        "observer_evaluation": "ObserverEvaluation",
        "prediction": "Prediction",
    }
    for frame in ROLE_FRAMES:
        rows = [role[(target, frame)] for target in role_targets]
        frame_name = frame_macro_names[frame]
        valid_coverage = [
            1.0 - int(row["first_attempt_invalid_count"]) / int(row["scheduled_draws"])
            for row in rows
        ]
        values.append((f"ProviderSafeRole{frame_name}ModelCount", str(len(rows))))
        values.extend(
            _summary_macros(
                [float(row["welfare_preserving_rate_all_scheduled"]) for row in rows],
                prefix=f"ProviderSafeRole{frame_name}WelfareRatePct",
                formatter=_percent_headline,
            )
        )
        values.extend(
            _summary_macros(
                valid_coverage,
                prefix=f"ProviderSafeRole{frame_name}ValidCoveragePct",
                formatter=_percent_headline,
            )
        )

    sensitivity_rows = [
        sensitivity[(target, factor)]
        for target in data["sensitivity_targets"]
        for factor in SENSITIVITY_FACTORS
    ]
    sensitivity_seed_counts = {int(row["common_seed_count"]) for row in sensitivity_rows}
    if len(sensitivity_seed_counts) != 1:
        raise PaperAssetsError("Sensitivity headline rows disagree on common-seed count.")
    [sensitivity_seed_count] = sensitivity_seed_counts
    values.extend(
        [
            ("ProviderSafeSensitivitySentinelCount", str(len(sensitivity_models))),
            (
                "ProviderSafeSensitivityTrajectoryCount",
                str(sum(int(row["trajectory_count"]) for row in sensitivity_models.values())),
            ),
            ("ProviderSafeSensitivityCommonSeedCount", str(sensitivity_seed_count)),
            ("ProviderSafeSensitivityEffectCount", str(len(sensitivity_rows))),
            (
                "ProviderSafeSensitivityMaximumAbsoluteEffect",
                _decimal_headline(
                    max(abs(float(row["effect_high_minus_low"])) for row in sensitivity_rows),
                    4,
                ),
            ),
            (
                "ProviderSafeSensitivityHolmSignificantCount",
                str(sum(float(row["holm_adjusted_p"]) <= 0.05 for row in sensitivity_rows)),
            ),
        ]
    )
    names = [name for name, _ in values]
    if len(names) != len(set(names)) or any(not name.isalpha() for name in names):
        raise PaperAssetsError("Headline macro names are duplicated or not TeX-safe letters.")
    return values


def _write_headlines(data: Mapping[str, Any], directory: Path) -> Path:
    values = _headline_values(data)
    lines = [
        "% Generated by analysis.build_provider_safe_v2_paper_assets; do not edit.",
        "% Counts preserve the validated scheduled-unit denominators.",
        "% Pct macros omit the percent sign; AURC/effect macros use normalized units.",
        "% Part 2 valid/invalid macros count scheduled agent-days; NE excludes only nonestimable model AURC from its model summary.",
        "% Role frames remain separate; sensitivity remains deadline-exploratory; no cross-axis aggregate or promotion is defined.",
        *(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values),
        "",
    ]
    path = directory / "paper_headlines.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_paper_assets(
    input_dir: Path,
    output_dir: Path,
    local_controls_path: Path | None = None,
) -> dict[str, Any]:
    """Validate sanitized definitive outputs and atomically publish paper assets."""

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    local_controls_path = (
        DEFAULT_LOCAL_CONTROLS_PATH if local_controls_path is None else local_controls_path
    ).resolve()
    if output_dir.exists():
        raise PaperAssetsError("Output directory already exists; refusing overwrite.")
    data = _load_and_validate(input_dir)
    local_controls_artifact, local_controls = _validate_local_controls(local_controls_path)
    data["local_controls"] = local_controls
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        assets: list[Path] = []
        assets.extend(_plot_part0(data, temporary))
        assets.extend(_plot_part1(data, temporary))
        assets.extend(_plot_part2(data, temporary))
        assets.extend(_plot_role(data, temporary))
        assets.extend(_plot_sensitivity(data, temporary))
        assets.extend(_plot_local_controls(data, temporary))
        assets.extend(_write_tables(data, temporary))
        assets.append(_write_headlines(data, temporary))
        asset_rows = [
            {
                "name": path.name,
                "kind": (
                    "vector_pdf" if path.suffix == ".pdf" else
                    "raster_png" if path.suffix == ".png" else
                    "latex_macros" if path.name == "paper_headlines.tex" else
                    "latex_table"
                ),
                "file_sha256": _sha256_file(path),
                "byte_count": path.stat().st_size,
            }
            for path in sorted(assets, key=lambda value: value.name)
        ]
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": OUTPUT_ARTIFACT_TYPE,
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_analysis_evidence_sha256": data["manifest"]["evidence_sha256"],
            "source_local_controls_evidence_sha256": local_controls_artifact["evidence_sha256"],
            "source_row_counts": EXPECTED_ROW_COUNTS,
            "assets": asset_rows,
            "table_outer_spacing_pt": 15,
            "table_outer_spacing_approx_css_px_at_96dpi": 20,
            "figure_palette": "matplotlib_turbo_sampled_0.08_to_0.90",
            "figure_semantic_redundancy": (
                "directional_caption_position_and_printed_values"
            ),
            "invalid_policy": INVALID_POLICY,
            "exploratory_only": True,
            "confirmatory_or_paper_promotion_permitted": False,
            "human_labels_generated": False,
            "cross_axis_assets_generated": True,
            "cross_axis_aggregate_or_score_generated": False,
            "local_controls_pooled_with_hosted_routes": False,
            "route_and_model_ids_preserved_exactly": True,
        }
        result["evidence_sha256"] = _self_hash(result)
        (temporary / "paper_assets_manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--local-controls", type=Path, default=DEFAULT_LOCAL_CONTROLS_PATH,
        help="Sanitized four-row local HF Part 1 aggregate JSON (never private JSONL).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        build_paper_assets(args.input_dir, args.output_dir, args.local_controls)
    except PaperAssetsError as error:
        print(f"Paper asset generation failed: {error}")
        return 1
    print(f"Wrote definitive paper assets: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
