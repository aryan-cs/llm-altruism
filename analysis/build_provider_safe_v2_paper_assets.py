"""Build fail-closed paper assets from definitive provider-safe-v2 outputs.

The input is only the completed, sanitized output directory produced by
``analysis.analyze_provider_safe_v2_definitive``.  This module never opens the
private manifests named in that output and never generates human labels.  It
publishes nine figure families, including provider-grouped model comparisons,
plus full exact-route tables atomically into a new output
directory.  The displays do not pool axes, compute a composite, or fill
unexecuted cells.
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
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter


SCHEMA_VERSION = 1
SOURCE_ARTIFACT_TYPE = "provider_safe_v2_definitive_descriptive_analysis"
OUTPUT_ARTIFACT_TYPE = "provider_safe_v2_definitive_paper_assets"
INVALID_POLICY = (
    "first_attempt_invalids_retained_in_all_primary_scheduled_unit_denominators;"
    "repairs_reported_separately"
)
ROLE_SENTINEL_COUNT = 6
SENSITIVITY_SENTINEL_COUNT = 5
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
SENSITIVITY_HOLM_FAMILY_SIZE = SENSITIVITY_SENTINEL_COUNT * len(
    SENSITIVITY_FACTORS
)
SENSITIVITY_HOLM_FAMILY = (
    f"{SENSITIVITY_HOLM_FAMILY_SIZE}_prespecified_sentinel_by_factor_main_effects"
)
EXPECTED_ROW_COUNTS = {
    "part0_models": 22,
    "part1_models": 75,
    "part2_models": 19,
    "role_calibration_model_frames": ROLE_SENTINEL_COUNT * len(ROLE_FRAMES),
    "sensitivity_models": SENSITIVITY_SENTINEL_COUNT,
    "sensitivity_main_effects": SENSITIVITY_HOLM_FAMILY_SIZE,
}
DEFAULT_LOCAL_CONTROLS_PATH = (
    Path(__file__).resolve().parents[1] / "data/analysis/local_hf_part1_controls.json"
)

INK = "#20252B"
MUTED = "#66707A"
GRID = "#D9DEE3"
# Original submission palette (Okabe-Ito): refusal blue, one-shot-choice
# orange, commons green, and adverse-outcome vermillion.  Keep this explicit
# so a Matplotlib style change cannot silently recolor the paper.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
P0_CMAP = LinearSegmentedColormap.from_list("part0_original_blue", ("#F7FBFF", BLUE))
P1_CMAP = LinearSegmentedColormap.from_list("part1_original_orange", ("#FFF9E8", ORANGE))
SIGNED_CMAP = LinearSegmentedColormap.from_list("signed_original", (BLUE, "#FAFAF8", RED))

PROVIDER_DISPLAY_ORDER = (
    "openai",
    "anthropic",
    "google",
    "nvidia",
    "qwen",
    "meta",
    "deepseek-ai",
    "zai-org",
    "perplexity",
)
PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "nvidia": "NVIDIA",
    "qwen": "Qwen",
    "meta": "Meta",
    "deepseek-ai": "DeepSeek",
    "zai-org": "Z.ai",
    "perplexity": "Perplexity",
}


def _relative_luminance(rgb: Sequence[float]) -> float:
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in rgb[:3]
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _annotation_color(rgba: Sequence[float]) -> str:
    """Choose black or white by WCAG contrast against the rendered cell."""

    background = _relative_luminance(rgba)
    contrast_black = (background + 0.05) / 0.05
    contrast_white = 1.05 / (background + 0.05)
    return "black" if contrast_black >= contrast_white else "white"

# The NeurIPS template sets ptm (Times) as its Roman default.  Times New Roman
# is the installed metric-compatible plotting font on the submission host.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "figure.titlesize": 10.0,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
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


def _interval(
    row: Mapping[str, Any], low_field: str, high_field: str, label: str,
    *, estimate: float | None, bounded: bool, allow_nonestimable: bool = False,
) -> tuple[float | None, float | None]:
    """Validate an interval pair, including the explicit nonestimable case."""

    low_raw, high_raw = row.get(low_field), row.get(high_field)
    if estimate is None:
        if low_raw is not None or high_raw is not None:
            raise PaperAssetsError(f"{label} interval must be null when its estimate is null.")
        return None, None
    if low_raw is None and high_raw is None and allow_nonestimable:
        return None, None
    low = _number(
        row, low_field, label, minimum=0.0 if bounded else None,
        maximum=1.0 if bounded else None,
    )
    high = _number(
        row, high_field, label, minimum=0.0 if bounded else None,
        maximum=1.0 if bounded else None,
    )
    if low > estimate or estimate > high:
        raise PaperAssetsError(f"{label} interval does not contain its estimate.")
    return low, high


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
    if manifest.get("path_policy") != "portable_basenames_only_no_host_absolute_paths_in_public_manifest":
        raise PaperAssetsError("Analysis manifest path policy is not release-safe.")
    if manifest.get("privacy_policy") != {
        "contains_prompt_text": False,
        "contains_response_text_or_reasoning": False,
        "contains_private_journal_paths": False,
        "contains_only_identifiers_hash_bindings_and_derived_aggregates": True,
    }:
        raise PaperAssetsError("Analysis manifest privacy policy changed.")
    inputs = manifest.get("input_manifests")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "part0", "part1", "part2", "role", "sensitivity"
    }:
        raise PaperAssetsError("Analysis manifest input bindings are incomplete.")
    for phase, binding in inputs.items():
        if not isinstance(binding, Mapping) or set(binding) != {
            "basename", "file_sha256", "evidence_sha256"
        }:
            raise PaperAssetsError(f"Analysis input binding is malformed: {phase}.")
        basename = binding.get("basename")
        if (
            not isinstance(basename, str)
            or Path(basename).name != basename
            or Path(basename).is_absolute()
        ):
            raise PaperAssetsError("Analysis manifest contains a nonportable input path.")
    expected_public = {
        *(f"{name}.jsonl" for name in EXPECTED_ROW_COUNTS),
        *(f"{name}.csv" for name in EXPECTED_ROW_COUNTS),
        "figure_aggregates.json",
    }
    outputs = manifest.get("public_outputs")
    if manifest.get("public_output_inventory_scope") != (
        "all_nonmanifest_outputs_created_before_manifest_self_seal"
    ):
        raise PaperAssetsError("Analysis public-output inventory scope changed.")
    if not isinstance(outputs, list) or len(outputs) != len(expected_public):
        raise PaperAssetsError("Analysis manifest public-output inventory is incomplete.")
    seen: set[str] = set()
    for binding in outputs:
        if not isinstance(binding, Mapping):
            raise PaperAssetsError("Analysis public-output binding is malformed.")
        basename = binding.get("basename")
        if (
            not isinstance(basename, str)
            or basename not in expected_public
            or Path(basename).name != basename
            or basename in seen
        ):
            raise PaperAssetsError("Analysis public-output basename is invalid or duplicated.")
        seen.add(basename)
        path = input_dir / basename
        expected_kind = (
            "machine_readable_table_jsonl" if path.suffix == ".jsonl" else
            "machine_readable_table_csv" if path.suffix == ".csv" else
            "machine_readable_figure_aggregates_json"
        )
        if binding.get("kind") != expected_kind:
            raise PaperAssetsError(f"Analysis public-output kind failed: {basename}.")
        if binding.get("file_sha256") != _sha256_file(path):
            raise PaperAssetsError(f"Analysis public-output hash failed: {basename}.")
        expected_rows = EXPECTED_ROW_COUNTS.get(path.stem)
        if expected_rows is None:
            declared_rows = binding.get("row_count")
            if isinstance(declared_rows, bool) or not isinstance(declared_rows, int) or declared_rows < 0:
                raise PaperAssetsError(f"Analysis public-output row count failed: {basename}.")
        elif binding.get("row_count") != expected_rows:
            raise PaperAssetsError(f"Analysis public-output row count failed: {basename}.")
    if seen != expected_public:
        raise PaperAssetsError("Analysis public-output inventory changed.")
    if (
        manifest.get("invalid_policy") != INVALID_POLICY
        or manifest.get("human_labels_generated") is not False
        or manifest.get("exploratory_only") is not True
        or manifest.get("confirmatory_or_paper_promotion_permitted") is not False
    ):
        raise PaperAssetsError("Analysis manifest safety/status contract changed.")
    uncertainty = manifest.get("uncertainty_policy")
    if not isinstance(uncertainty, Mapping) or uncertainty.get("bootstrap_replicates") != 5_000 or uncertainty.get("finite_bank_scope") != (
        "part0_and_part1_bootstrap_intervals_are_descriptive_frozen_bank_sensitivity_intervals_not_population_confidence_intervals"
    ):
        raise PaperAssetsError("Analysis uncertainty/scope contract changed.")
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
                "semantic_invalid_count", "operational_invalid_count",
                "visible_subject_response_count",
                "refusal_rate_all_scheduled", "primary_denominator", "exploratory_only",
                "refusal_rate_all_scheduled_finite_bank_sensitivity_low",
                "refusal_rate_all_scheduled_finite_bank_sensitivity_high",
                "finite_bank_sensitivity_method", "finite_bank_sensitivity_unit",
                "finite_bank_sensitivity_replicates", "finite_bank_sensitivity_seed",
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
        semantic_invalid = _integer(row, "semantic_invalid_count", label)
        operational_invalid = _integer(row, "operational_invalid_count", label)
        visible = _integer(row, "visible_subject_response_count", label)
        if (
            semantic_invalid + operational_invalid != counts[3]
            or visible + operational_invalid != scheduled
        ):
            raise PaperAssetsError(
                f"{label} semantic/operational invalid accounting does not reconcile."
            )
        refusal_rate = _rate(row, "refusal_rate_all_scheduled", label)
        if sum(counts) != scheduled or not _same_rate(refusal_rate, counts[0], scheduled):
            raise PaperAssetsError(f"{label} counts or scheduled-unit refusal rate do not reconcile.")
        _interval(
            row,
            "refusal_rate_all_scheduled_finite_bank_sensitivity_low",
            "refusal_rate_all_scheduled_finite_bank_sensitivity_high",
            label,
            estimate=refusal_rate,
            bounded=True,
        )
        if (
            row.get("finite_bank_sensitivity_method")
            != "percentile_root_cluster_bootstrap_95"
            or row.get("finite_bank_sensitivity_unit")
            != "harmful_request_root_with_three_languages_retained"
            or _integer(row, "finite_bank_sensitivity_replicates", label) != 5_000
            or _integer(row, "finite_bank_sensitivity_seed", label) < 1
        ):
            raise PaperAssetsError(f"{label} changed its root-cluster sensitivity interval.")
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
                "refusal_rate_all_scheduled_wilson95_low",
                "refusal_rate_all_scheduled_wilson95_high",
                "interval_method", "interval_unit",
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
        _interval(
            row, "refusal_rate_all_scheduled_wilson95_low",
            "refusal_rate_all_scheduled_wilson95_high", label,
            estimate=refusal_rate, bounded=True,
        )
        if (
            row.get("interval_method") != "wilson_score_binomial_95"
            or row.get("interval_unit")
            != "harmful_request_root_within_response_language"
        ):
            raise PaperAssetsError(f"{label} changed its Wilson interval contract.")
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
                "welfare_preserving_rate_all_scheduled_finite_bank_sensitivity_low",
                "welfare_preserving_rate_all_scheduled_finite_bank_sensitivity_high",
                "finite_bank_sensitivity_method", "finite_bank_sensitivity_unit",
                "finite_bank_sensitivity_strata",
                "finite_bank_sensitivity_roots_per_stratum",
                "finite_bank_sensitivity_replicates", "finite_bank_sensitivity_seed",
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
        _interval(
            row,
            "welfare_preserving_rate_all_scheduled_finite_bank_sensitivity_low",
            "welfare_preserving_rate_all_scheduled_finite_bank_sensitivity_high",
            label,
            estimate=all_rate,
            bounded=True,
        )
        if (
            row.get("finite_bank_sensitivity_method")
            != "percentile_root_bootstrap_stratified_by_game_domain_95"
            or row.get("finite_bank_sensitivity_unit") != "one_shot_scenario_root"
            or _integer(row, "finite_bank_sensitivity_strata", label) != 12
            or _integer(row, "finite_bank_sensitivity_roots_per_stratum", label) != 32
            or _integer(row, "finite_bank_sensitivity_replicates", label) != 5_000
            or _integer(row, "finite_bank_sensitivity_seed", label) < 1
        ):
            raise PaperAssetsError(f"{label} changed its stratified-root sensitivity interval.")
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
                "environmentally_estimable_trajectory_count",
                "semantic_invalid_trajectory_count",
                "scheduled_agent_days", "restraint_count", "overuse_count",
                "first_attempt_invalid_count", "repaired_invalid_count",
                "restraint_rate_all_scheduled", "restraint_rate_among_valid",
                "mean_trajectory_restraint_rate_all_scheduled",
                "mean_trajectory_restraint_rate_all_scheduled_t95_low",
                "mean_trajectory_restraint_rate_all_scheduled_t95_high",
                "mean_aurc_eligible", "mean_aupc_eligible",
                "mean_aurc_eligible_t95_low", "mean_aurc_eligible_t95_high",
                "mean_aupc_eligible_t95_low", "mean_aupc_eligible_t95_high",
                "reserve_nondepletion_rate_eligible",
                "reserve_nondepletion_rate_eligible_wilson95_low",
                "reserve_nondepletion_rate_eligible_wilson95_high",
                "mean_population_retention_eligible",
                "mean_population_retention_eligible_t95_low",
                "mean_population_retention_eligible_t95_high",
                "trajectory_interval_method", "trajectory_interval_unit",
                "restraint_interval_trajectory_count",
                "environmental_interval_trajectory_count",
                "nondepletion_interval_method",
                "primary_denominator", "exploratory_only",
            ),
            label,
        )
        trajectories = _integer(row, "trajectory_count", label, minimum=1)
        operational = _integer(
            row, "operationally_eligible_trajectory_count", label
        )
        eligible = _integer(
            row, "environmentally_estimable_trajectory_count", label
        )
        semantic_invalid_trajectories = _integer(
            row, "semantic_invalid_trajectory_count", label
        )
        scheduled = _integer(row, "scheduled_agent_days", label, minimum=1)
        restraint = _integer(row, "restraint_count", label)
        overuse = _integer(row, "overuse_count", label)
        invalid = _integer(row, "first_attempt_invalid_count", label)
        if (
            trajectories != 12
            or operational > trajectories
            or eligible > operational
            or semantic_invalid_trajectories > operational
            or eligible + semantic_invalid_trajectories != operational
            or restraint + overuse + invalid != scheduled
        ):
            raise PaperAssetsError(f"{label} trajectory/action accounting does not reconcile.")
        if _integer(row, "repaired_invalid_count", label) != 0:
            raise PaperAssetsError("Part 2 repaired outcomes are outside this frozen asset contract.")
        all_rate = _rate(row, "restraint_rate_all_scheduled", label)
        mean_trajectory_restraint = _rate(
            row, "mean_trajectory_restraint_rate_all_scheduled", label
        )
        _optional_reconciled_rate(
            row, "restraint_rate_among_valid", label, restraint, scheduled - invalid
        )
        if eligible == 0:
            for field in (
                "mean_aurc_eligible",
                "mean_aurc_eligible_t95_low",
                "mean_aurc_eligible_t95_high",
                "mean_aupc_eligible",
                "mean_aupc_eligible_t95_low",
                "mean_aupc_eligible_t95_high",
                "reserve_nondepletion_rate_eligible",
                "reserve_nondepletion_rate_eligible_wilson95_low",
                "reserve_nondepletion_rate_eligible_wilson95_high",
                "mean_population_retention_eligible",
                "mean_population_retention_eligible_t95_low",
                "mean_population_retention_eligible_t95_high",
            ):
                if row.get(field) is not None:
                    raise PaperAssetsError(
                        f"{label}.{field} must be null with no eligible trajectories."
                    )
        else:
            aurc = _rate(row, "mean_aurc_eligible", label)
            aupc = _rate(row, "mean_aupc_eligible", label)
            nondepletion = _rate(row, "reserve_nondepletion_rate_eligible", label)
            population = _rate(row, "mean_population_retention_eligible", label)
            _interval(
                row, "mean_aurc_eligible_t95_low", "mean_aurc_eligible_t95_high",
                label, estimate=aurc, bounded=True, allow_nonestimable=eligible < 2,
            )
            _interval(
                row, "mean_aupc_eligible_t95_low", "mean_aupc_eligible_t95_high",
                label, estimate=aupc, bounded=True, allow_nonestimable=eligible < 2,
            )
            _interval(
                row, "reserve_nondepletion_rate_eligible_wilson95_low",
                "reserve_nondepletion_rate_eligible_wilson95_high", label,
                estimate=nondepletion, bounded=True,
            )
            _interval(
                row, "mean_population_retention_eligible_t95_low",
                "mean_population_retention_eligible_t95_high", label,
                estimate=population, bounded=True, allow_nonestimable=eligible < 2,
            )
        _interval(
            row, "mean_trajectory_restraint_rate_all_scheduled_t95_low",
            "mean_trajectory_restraint_rate_all_scheduled_t95_high", label,
            estimate=mean_trajectory_restraint, bounded=True,
        )
        if (
            row.get("trajectory_interval_method")
            != "student_t_95_over_independent_trajectories"
            or row.get("trajectory_interval_unit")
            != "matched_environment_seed_trajectory"
            or _integer(row, "restraint_interval_trajectory_count", label)
            != trajectories
            or _integer(row, "environmental_interval_trajectory_count", label)
            != eligible
            or row.get("nondepletion_interval_method")
            != "wilson_score_binomial_95"
        ):
            raise PaperAssetsError(f"{label} changed its trajectory interval contract.")
        if not _same_rate(all_rate, restraint, scheduled):
            raise PaperAssetsError(f"{label} scheduled-unit restraint rate does not reconcile.")
        if row.get("primary_denominator") != "all_scheduled_agent_days" or row.get("exploratory_only") is not True:
            raise PaperAssetsError(f"{label} changed its denominator/status contract.")
    return sorted(
        models.values(),
        key=lambda row: (-float(row["restraint_rate_all_scheduled"]), str(row["target_id"])),
    )


def _validate_role(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], dict[tuple[str, str], dict[str, Any]]]:
    expected_rows = ROLE_SENTINEL_COUNT * len(ROLE_FRAMES)
    if len(rows) != expected_rows:
        raise PaperAssetsError(
            f"Role calibration requires exactly {expected_rows} model-frame rows."
        )
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
    if len(identity) != ROLE_SENTINEL_COUNT or set(by_key) != {
        (target, frame) for target in identity for frame in ROLE_FRAMES
    }:
        raise PaperAssetsError("Role calibration is not a complete six-model x three-frame matrix.")
    return sorted(identity), by_key


def _validate_sensitivity(
    model_rows: Sequence[Mapping[str, Any]], effect_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[str], dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    models = _model_index(
        model_rows, "sensitivity_models", SENSITIVITY_SENTINEL_COUNT
    )
    for target, row in models.items():
        label = f"sensitivity_models[{target}]"
        _required(
            row,
            (
                "trajectory_count", "cell_count", "common_seed_count",
                "execution_ceiling_agent_days", "scheduled_agent_days",
                "responses_received", "transport_failure_count",
                "identity_mismatch_count", "first_attempt_invalid_count",
                "repaired_invalid_count", "inference_scope", "confirmatory", "exploratory_only",
                "primary_denominator", "schedule_semantics",
            ),
            label,
        )
        if (
            _integer(row, "trajectory_count", label) != 32
            or _integer(row, "cell_count", label) != 16
            or _integer(row, "common_seed_count", label) != 2
        ):
            raise PaperAssetsError(f"{label} does not contain 16 cells x 2 seeds.")
        ceiling = _integer(row, "execution_ceiling_agent_days", label, minimum=1)
        if ceiling != 2880:
            raise PaperAssetsError(
                f"{label} changed the frozen 2,880 agent-day execution ceiling."
            )
        scheduled = _integer(row, "scheduled_agent_days", label, minimum=1)
        responses = _integer(row, "responses_received", label)
        transport = _integer(row, "transport_failure_count", label)
        identity = _integer(row, "identity_mismatch_count", label)
        invalid = _integer(row, "first_attempt_invalid_count", label)
        if (
            scheduled > ceiling
            or responses + transport != scheduled
            or identity + transport > invalid
        ):
            raise PaperAssetsError(
                f"{label} realized living-agent schedule does not reconcile."
            )
        if invalid > scheduled or _integer(row, "repaired_invalid_count", label) != 0:
            raise PaperAssetsError(f"{label} has invalid sensitivity coverage accounting.")
        if (
            row.get("primary_denominator") != "all_scheduled_living_agent_days"
            or row.get("schedule_semantics")
            != "one_decision_per_living_agent_per_day_dead_agents_have_no_future_scheduled_days"
        ):
            raise PaperAssetsError(
                f"{label} changed its realized living-agent denominator contract."
            )
        if row.get("inference_scope") != "deadline_exploratory" or row.get("confirmatory") is not False or row.get("exploratory_only") is not True:
            raise PaperAssetsError(f"{label} changed its exploratory sensitivity scope.")

    expected_effect_count = len(models) * len(SENSITIVITY_FACTORS)
    if len(effect_rows) != expected_effect_count:
        raise PaperAssetsError(
            f"Sensitivity requires exactly {expected_effect_count} main-effect rows."
        )
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
            or row.get("holm_family") != SENSITIVITY_HOLM_FAMILY
            or _integer(row, "holm_family_size", label)
            != expected_effect_count
            or row.get("max_t_family") != "five_main_effects_within_sentinel_diagnostic"
            or row.get("inference_scope") != "deadline_exploratory"
            or row.get("confirmatory") is not False
        ):
            raise PaperAssetsError(f"{label} changed the frozen sensitivity/Holm contract.")
        effects[key] = row
    expected = {(target, factor) for target in models for factor in SENSITIVITY_FACTORS}
    if set(effects) != expected:
        raise PaperAssetsError(
            "Sensitivity is not a complete five-compatible-sentinel x five-factor matrix."
        )
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
    figure_binding = next(
        binding
        for binding in manifest["public_outputs"]
        if binding["basename"] == "figure_aggregates.json"
    )
    if figure_binding["row_count"] != sum(len(aggregates[key]) for key in aggregates):
        raise PaperAssetsError("Figure aggregate row count differs from its hash binding.")
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


def _provider_order_key(provider: str) -> tuple[int, str]:
    try:
        return PROVIDER_DISPLAY_ORDER.index(provider), provider
    except ValueError:
        return len(PROVIDER_DISPLAY_ORDER), provider


def _provider_grouped_rows(
    rows: Sequence[Mapping[str, Any]], *, score_key: str | None = None
) -> list[Mapping[str, Any]]:
    """Group exact routes by provider, with optional within-provider ranking."""

    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        provider = str(row["upstream_provider"])
        score = 0.0 if score_key is None else -float(row[score_key])
        return (*_provider_order_key(provider), score, str(row["model"]), str(row["target_id"]))

    return sorted(rows, key=key)


def _provider_prefixed_labels(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    labels: list[str] = []
    previous: str | None = None
    for row in rows:
        provider = str(row["upstream_provider"])
        model = str(row["model"])
        if provider != previous:
            labels.append(f"{PROVIDER_DISPLAY_NAMES.get(provider, provider)}: {model}")
        else:
            labels.append(f"    {model}")
        previous = provider
    return labels


def _draw_provider_separators(
    ax: plt.Axes, rows: Sequence[Mapping[str, Any]]
) -> None:
    for index in range(1, len(rows)):
        if rows[index]["upstream_provider"] != rows[index - 1]["upstream_provider"]:
            ax.axhline(index - 0.5, color=MUTED, linewidth=0.75, alpha=0.65, zorder=4)


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
        metadata={"Title": title, "Author": "Safety Beyond Refusal asset generator", "Creator": "Matplotlib"},
    )
    fig.savefig(png, format="png", dpi=300)
    plt.close(fig)
    return [pdf, png]


def _figure_footer(
    fig: plt.Figure,
    text: str,
    *,
    x: float = 0.08,
    y: float = 0.018,
    width: int = 145,
) -> None:
    """Place a bounded multi-line footer inside the physical figure canvas."""

    fig.text(
        x,
        y,
        textwrap.fill(text, width=width),
        fontsize=8,
        color=MUTED,
        va="bottom",
        linespacing=1.15,
    )


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
    intervals: Sequence[Sequence[tuple[float, float]]] | None = None,
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
    ax.tick_params(axis="y", labelsize=7.2)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    for row_index, values in enumerate(matrix):
        for column_index, value in enumerate(values):
            label = f"{value:.0%}" if percent else f"{value:+.3f}"
            if intervals is not None:
                low, high = intervals[row_index][column_index]
                label += f"\n[{low:.0%}, {high:.0%}]"
            # Select annotation ink from the rendered cell luminance, rather
            # than distance from the scale midpoint.  Sequential maps are
            # intentionally near-white at their low end, where white labels
            # would disappear in print.
            normalized = 0.5 if vmax == vmin else min(1.0, max(0.0, (value - vmin) / (vmax - vmin)))
            color = _annotation_color(cmap(normalized))
            ax.text(
                column_index + 0.5, row_index + 0.5, label,
                ha="center", va="center", fontsize=6.8 if intervals is not None else 7.2,
                color=color, linespacing=0.9,
            )
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
    rows = _provider_grouped_rows(data["part0"], score_key="refusal_rate_all_scheduled")
    by_key = data["part0_matrix"]
    labels = _provider_prefixed_labels(rows)
    refusal = [
        [float(by_key[(row["target_id"], language)]["refusal_rate_all_scheduled"]) for language in PART0_LANGUAGES]
        for row in rows
    ]
    refusal_intervals = [
        [
            (
                float(by_key[(row["target_id"], language)]["refusal_rate_all_scheduled_wilson95_low"]),
                float(by_key[(row["target_id"], language)]["refusal_rate_all_scheduled_wilson95_high"]),
            )
            for language in PART0_LANGUAGES
        ]
        for row in rows
    ]
    # Match the portrait appendix footprint so the exported figure does not
    # require a rotated conference page or reduced annotations.
    fig, ax = plt.subplots(figsize=(7.2, 9.2))
    fig.patch.set_facecolor("white")
    fig.suptitle("Part 0 response-language outcomes by exact model route", x=0.08, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.08, 0.93, "Each cell uses 48 scheduled harmful-request roots; provider families are contiguous and ranked within family.", fontsize=9, color=MUTED)
    _heatmap(
        ax, refusal, PART0_LANGUAGES, labels,
        title="Refusal rate [Wilson 95%] / 48 roots", cmap=P0_CMAP,
        vmin=0.0, vmax=1.0, intervals=refusal_intervals,
    )
    _draw_provider_separators(ax, rows)
    _figure_footer(fig, "Brackets are condition-specific Wilson 95% intervals over 48 roots. Higher refusal means less assistance on this harmful-request task. Invalid outputs remain in the scheduled denominator but are reported in the reproducibility artifacts rather than as a separate argument-facing column.")
    fig.tight_layout(rect=(0.06, 0.085, 0.99, 0.90))
    return _save_figure(fig, directory, "part0_model_language", "Part 0 model by language outcomes")


def _plot_refusal_and_cooperation_overview(
    data: Mapping[str, Any], directory: Path
) -> list[Path]:
    """Recreate the original bar/line visual grammar from current evidence.

    The upper panel is an exact-route comparison, while the lower panel is a
    distributional rank profile.  Keeping these roles separate avoids making
    the ordered Part 1 line look like a longitudinal trajectory.
    """

    part0 = _provider_grouped_rows(
        data["part0"], score_key="refusal_rate_all_scheduled"
    )
    part1 = data["part1"]
    refusal = [float(row["refusal_rate_all_scheduled"]) for row in part0]
    refusal_intervals = [
        (
            float(row["refusal_rate_all_scheduled_finite_bank_sensitivity_low"]),
            float(row["refusal_rate_all_scheduled_finite_bank_sensitivity_high"]),
        )
        for row in part0
    ]
    welfare = [float(row["welfare_preserving_rate_all_scheduled"]) for row in part1]
    ranks = list(range(1, len(welfare) + 1))
    welfare_median = statistics.median(welfare)

    fig = plt.figure(figsize=(7.2, 9.4))
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(2, 1, height_ratios=(3.1, 1.35), hspace=0.38)
    refusal_ax = fig.add_subplot(grid[0])
    rank_ax = fig.add_subplot(grid[1])
    fig.suptitle(
        "Explicit refusal compresses; dyadic cooperation separates",
        x=0.075,
        y=0.993,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.075,
        0.958,
        "Current authenticated routes only; panels retain their own task and denominator.",
        fontsize=9,
        color=MUTED,
    )

    _lollipop_panel(
        refusal_ax,
        refusal,
        _provider_prefixed_labels(part0),
        title="A  Harmful-request refusal grouped by provider [root sensitivity 95%]",
        color=BLUE,
        show_labels=True,
        intervals=refusal_intervals,
    )
    _draw_provider_separators(refusal_ax, part0)
    refusal_ax.tick_params(axis="y", labelsize=5.9)
    refusal_ax.set_xlabel("Refusal over 144 scheduled responses")

    rank_ax.plot(
        ranks,
        welfare,
        color=ORANGE,
        linewidth=1.6,
        marker="o",
        markersize=2.7,
        markeredgecolor=INK,
        markeredgewidth=0.25,
        zorder=2,
    )
    rank_ax.axhline(
        welfare_median,
        color=INK,
        linestyle=(0, (4, 3)),
        linewidth=0.8,
        zorder=1,
    )
    rank_ax.text(
        len(welfare) - 0.5,
        min(0.98, welfare_median + 0.055),
        f"median {welfare_median:.1%}",
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=INK,
    )
    rank_ax.set_xlim(1, len(welfare))
    rank_ax.set_ylim(-0.02, 1.02)
    rank_ax.set_xticks([1, 15, 30, 45, 60, len(welfare)])
    rank_ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    rank_ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    rank_ax.grid(axis="y", color=GRID, linewidth=0.6)
    rank_ax.set_axisbelow(True)
    rank_ax.set_title(
        "B  Welfare-preserving self-choice across all 75 routes, ordered high to low",
        fontsize=10,
        fontweight="bold",
        loc="left",
    )
    rank_ax.set_xlabel("Within-task route rank (exact-route lookup in Appendix tables)")
    rank_ax.set_ylabel("Welfare-preserving / 384 roots")
    _style_axes(rank_ax)

    _figure_footer(
        fig,
        "Panel A groups provider families contiguously in the order OpenAI, Anthropic, Google, NVIDIA, then the remaining providers; routes are ranked by refusal only within family. Blue bars end at the all-scheduled point estimate, dots repeat it, and whiskers are 5,000-replicate harmful-root sensitivity intervals. Farther right means less harmful assistance. Panel B remains a global ordered cross-sectional rank profile, not a time series: every orange point is one exact route over the same balanced 384-root bank. Higher means fewer counterpart costs. Part 1 intervals and exact identities appear in the complete appendix bars and table.",
        x=0.075,
        y=0.012,
        width=125,
    )
    fig.subplots_adjust(left=0.32, right=0.985, top=0.91, bottom=0.155, hspace=0.38)
    return _save_figure(
        fig,
        directory,
        "refusal_cooperation_overview",
        "Current-model refusal bars and cooperation rank profile",
    )


def _plot_matched_current_route_profile(
    data: Mapping[str, Any], directory: Path
) -> list[Path]:
    """Show the exact-route rank shifts behind the three reported correlations."""

    indices = {
        "part0": {str(row["target_id"]): row for row in data["part0"]},
        "part1": {str(row["target_id"]): row for row in data["part1"]},
        "part2": {str(row["target_id"]): row for row in data["part2"]},
    }
    shared = set(indices["part0"]) & set(indices["part1"]) & set(indices["part2"])
    targets = sorted(shared)
    for target in targets:
        identities = {
            (
                str(indices[phase][target]["upstream_provider"]),
                str(indices[phase][target]["model"]),
            )
            for phase in ("part0", "part1", "part2")
        }
        if len(identities) != 1:
            raise PaperAssetsError(
                f"Matched current-route profile found inconsistent exact identity for {target!r}."
            )
    targets.sort(
        key=lambda target: (
            *_provider_order_key(str(indices["part0"][target]["upstream_provider"])),
            -float(indices["part0"][target]["refusal_rate_all_scheduled"]),
            str(indices["part0"][target]["model"]),
            target,
        )
    )
    if not targets:
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        fig.patch.set_facecolor("white")
        ax.axis("off")
        ax.text(
            0.5,
            0.66,
            "No authenticated exact route is shared across all three task panels",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=INK,
        )
        ax.text(
            0.5,
            0.43,
            "The generator leaves the matched profile empty instead of joining different routes or models.\n"
            "Within-task figures remain valid, but a cross-task row alignment is not estimable.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color=MUTED,
            linespacing=1.5,
        )
        return _save_figure(
            fig,
            directory,
            "matched_current_route_profile",
            "No exact routes shared across all three task panels",
        )
    panels = (
        (
            "Part 0: refusal",
            "part0",
            "refusal_rate_all_scheduled",
            "refusal_rate_all_scheduled_finite_bank_sensitivity_low",
            "refusal_rate_all_scheduled_finite_bank_sensitivity_high",
            BLUE,
        ),
        (
            "Part 1: cooperation",
            "part1",
            "welfare_preserving_rate_all_scheduled",
            "welfare_preserving_rate_all_scheduled_finite_bank_sensitivity_low",
            "welfare_preserving_rate_all_scheduled_finite_bank_sensitivity_high",
            ORANGE,
        ),
        (
            "Part 2: restraint",
            "part2",
            "mean_trajectory_restraint_rate_all_scheduled",
            "mean_trajectory_restraint_rate_all_scheduled_t95_low",
            "mean_trajectory_restraint_rate_all_scheduled_t95_high",
            GREEN,
        ),
    )
    positions = list(range(len(targets)))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 7.7), sharey=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "The same 19 routes reorder beyond refusal",
        x=0.08,
        y=0.992,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.08,
        0.947,
        "Rows are aligned exact routes; provider families are contiguous and ranked by Part 0 only within family.",
        fontsize=8.8,
        color=MUTED,
    )
    for panel_index, (title, phase, value_key, low_key, high_key, color) in enumerate(panels):
        ax = axes[panel_index]
        values = [float(indices[phase][target][value_key]) for target in targets]
        lows = [float(indices[phase][target][low_key]) for target in targets]
        highs = [float(indices[phase][target][high_key]) for target in targets]
        ax.hlines(positions, 0.0, values, color=color, alpha=0.38, linewidth=2.4, zorder=1)
        ax.errorbar(
            values,
            positions,
            xerr=[
                [max(0.0, value - low) for value, low in zip(values, lows, strict=True)],
                [max(0.0, high - value) for value, high in zip(values, highs, strict=True)],
            ],
            fmt="o",
            markersize=4.0,
            markerfacecolor=color,
            markeredgecolor=INK,
            markeredgewidth=0.45,
            ecolor=INK,
            elinewidth=0.7,
            capsize=1.8,
            capthick=0.7,
            zorder=2,
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=9.2, fontweight="bold", loc="left")
        ax.set_xlabel(("higher is safer" if panel_index == 0 else "higher preserves more"), fontsize=7.2)
        _style_axes(ax)
    matched_rows = [indices["part0"][target] for target in targets]
    axes[0].set_yticks(positions, labels=_provider_prefixed_labels(matched_rows))
    axes[0].tick_params(axis="y", labelsize=5.9, length=0)
    axes[0].invert_yaxis()
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0, labelleft=False)
    for ax in axes:
        _draw_provider_separators(ax, matched_rows)
    _figure_footer(
        fig,
        "Blue, orange, and green dots are task-specific point estimates; horizontal whiskers are harmful-root sensitivity intervals, stratified scenario-root sensitivity intervals, and trajectory Student-t 95% intervals, respectively. Longer colored stems mean a higher rate within that panel. A row moving left from refusal to cooperation or restraint is a model-ordering disagreement, not a decline over time. The aligned profile visualizes why refusal has weak rank association with the two beyond-refusal outcomes; no values are averaged across panels.",
        x=0.08,
        y=0.012,
        width=110,
    )
    fig.tight_layout(rect=(0.025, 0.095, 0.995, 0.91), w_pad=0.9)
    return _save_figure(
        fig,
        directory,
        "matched_current_route_profile",
        "Matched current-model task-specific outcome profile",
    )


def _lollipop_panel(
    ax: plt.Axes,
    values: Sequence[float | None],
    labels: Sequence[str],
    *,
    title: str,
    color: str,
    show_labels: bool,
    intervals: Sequence[tuple[float | None, float | None]] | None = None,
) -> None:
    # Reserve a dedicated label gutter to the right of the bounded [0, 1]
    # outcome scale.  Labels must never sit on top of the point estimate or
    # uncertainty whisker, including for estimates close to either boundary.
    label_gutter_center = 1.115
    display_limit = 1.23
    positions = list(range(len(values)))
    estimable = [(position, value) for position, value in zip(positions, values, strict=True) if value is not None]
    ax.barh(
        [position for position, _ in estimable],
        [value for _, value in estimable],
        height=0.58, color=color, alpha=0.58, edgecolor="none", zorder=1,
    )
    ax.scatter(
        [value for _, value in estimable], [position for position, _ in estimable],
        s=19, color=color, edgecolor=INK, linewidth=0.35, zorder=2,
    )
    if intervals is not None:
        for position, value, interval in zip(positions, values, intervals, strict=True):
            low, high = interval
            if value is None or low is None or high is None:
                continue
            ax.errorbar(
                [value], [position],
                xerr=[[max(0.0, value - low)], [max(0.0, high - value)]],
                fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6,
                capthick=0.7, zorder=1.5,
            )
    ax.set_xlim(0.0, display_limit)
    ax.set_ylim(-0.8, len(values) - 0.2)
    ax.invert_yaxis()
    if show_labels:
        ax.set_yticks(positions, labels=labels)
        ax.tick_params(axis="y", labelsize=7.0, length=0, labelleft=True)
    else:
        ax.set_yticks(positions)
        ax.tick_params(axis="y", labelsize=7.0, length=0, labelleft=False)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.axvline(1.025, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    for position, value in zip(positions, values, strict=True):
        if value is None:
            ax.scatter([0.015], [position], marker="x", s=17, color=MUTED, linewidth=0.8, zorder=2)
            ax.text(
                label_gutter_center,
                position,
                "NE",
                va="center",
                ha="center",
                fontsize=6.6,
                color=MUTED,
            )
        else:
            ax.text(
                label_gutter_center,
                position,
                f"{value:.1%}",
                va="center",
                ha="center",
                fontsize=6.6,
                color=INK,
            )
    _style_axes(ax)


def _plot_cross_phase_outcome_profile(
    data: Mapping[str, Any], directory: Path
) -> list[Path]:
    """Restore the original red/green per-model visual without pooling axes."""

    phase_indices = {
        "Part 0: refusal / compliance": {
            str(row["target_id"]): row for row in data["part0"]
        },
        "Part 1: welfare / focal advantage": {
            str(row["target_id"]): row for row in data["part1"]
        },
        "Part 2: restraint / overuse": {
            str(row["target_id"]): row for row in data["part2"]
        },
    }
    all_targets = list(set().union(*(set(index) for index in phase_indices.values())))
    identity_by_target: dict[str, tuple[str, str]] = {}
    for target in all_targets:
        identities = {
            (str(row["upstream_provider"]), str(row["model"]))
            for index in phase_indices.values()
            if (row := index.get(target)) is not None
        }
        if len(identities) != 1:
            raise PaperAssetsError(
                f"Cross-phase exact identity differs for target {target!r}."
            )
        identity_by_target[target] = next(iter(identities))
    all_targets.sort(
        key=lambda target: (
            *_provider_order_key(identity_by_target[target][0]),
            identity_by_target[target][1],
            target,
        )
    )
    # The exact Model ID remains in the adjacent generated table.  Route IDs
    # are used here so the union panel can wrap into print-legible blocks.
    block_count = min(4, max(1, math.ceil(len(all_targets) / 24)))
    block_size = math.ceil(len(all_targets) / block_count)
    output: list[Path] = []
    for block_index in range(block_count):
        block_targets = all_targets[
            block_index * block_size : min(len(all_targets), (block_index + 1) * block_size)
        ]
        fig, axes = plt.subplots(3, 1, figsize=(7.2, 10.2), squeeze=False)
        axes_column = [axes[index][0] for index in range(3)]
        fig.patch.set_facecolor("white")
        fig.suptitle(
            f"Task-specific outcome profiles across exact model routes · block {block_index + 1}/{block_count}",
            x=0.07, y=0.995, ha="left", fontsize=15, fontweight="bold", color=INK,
        )
        fig.text(
            0.07, 0.93,
            "Provider families are contiguous; panels retain separate tasks and denominators.",
            fontsize=9, color=MUTED,
        )
        positions = list(range(len(block_targets)))
        for panel_index, (title, index) in enumerate(phase_indices.items()):
            ax = axes_column[panel_index]
            for position, target in zip(positions, block_targets, strict=True):
                row = index.get(target)
                if row is None:
                    ax.scatter(
                        [0.02], [position], marker="x", s=13, color=MUTED,
                        linewidth=0.7, zorder=2,
                    )
                    continue
                if panel_index == 0:
                    denominator = int(row["scheduled_units"])
                    preferable = int(row["refusal_count"]) / denominator
                    adverse = int(row["compliance_count"]) / denominator
                    unresolved = (
                        int(row["unclear_count"])
                        + int(row["first_attempt_invalid_count"])
                    ) / denominator
                elif panel_index == 1:
                    denominator = int(row["scheduled_units"])
                    preferable_count = int(row["welfare_preserving_count_first_attempt"])
                    invalid_count = int(row["first_attempt_invalid_count"])
                    preferable = preferable_count / denominator
                    adverse = (denominator - preferable_count - invalid_count) / denominator
                    unresolved = invalid_count / denominator
                else:
                    denominator = int(row["scheduled_agent_days"])
                    preferable = int(row["restraint_count"]) / denominator
                    adverse = int(row["overuse_count"]) / denominator
                    unresolved = int(row["first_attempt_invalid_count"]) / denominator
                if min(preferable, adverse, unresolved) < 0 or not math.isclose(
                    preferable + adverse + unresolved, 1.0, abs_tol=1e-9
                ):
                    raise PaperAssetsError(
                        f"Cross-phase outcome partition is invalid for {target!r}."
                    )
                # The task-preferable and task-adverse shares can be equal
                # (most commonly at zero).  Give the two semantic series
                # distinct vertical lanes so their markers never cover one
                # another while preserving the shared per-route row.
                ax.scatter(
                    [preferable], [position - 0.13], marker="o", s=18, color=GREEN,
                    edgecolor=INK, linewidth=0.3, zorder=3,
                )
                ax.scatter(
                    [adverse], [position + 0.13], marker="D", s=17, color=RED,
                    edgecolor=INK, linewidth=0.3, zorder=3,
                )
            # Keep boundary markers fully inside the axes while preserving the
            # bounded 0--100% task scale and its exact tick labels.
            ax.set_xlim(-0.02, 1.02)
            ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_ylim(-0.8, len(block_targets) - 0.2)
            ax.invert_yaxis()
            ax.xaxis.set_major_formatter(PercentFormatter(1.0))
            ax.grid(axis="x", color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.set_title(title, fontsize=8.5, fontweight="bold", loc="left")
            ax.set_yticks(positions)
            block_identity_rows = [
                {
                    "upstream_provider": identity_by_target[target][0],
                    "model": identity_by_target[target][1],
                }
                for target in block_targets
            ]
            ax.set_yticklabels(_provider_prefixed_labels(block_identity_rows), fontsize=5.8)
            _draw_provider_separators(ax, block_identity_rows)
            _style_axes(ax)
        axes_column[0].scatter([], [], marker="o", s=26, color=GREEN, edgecolor=INK, linewidth=0.3, label="task-preferable outcome")
        axes_column[0].scatter([], [], marker="D", s=24, color=RED, edgecolor=INK, linewidth=0.3, label="task-adverse outcome")
        axes_column[0].scatter([], [], marker="x", s=18, color=MUTED, linewidth=0.7, label="route not in panel")
        fig.legend(
            loc="lower center", bbox_to_anchor=(0.53, 0.063), ncol=3,
            frameon=False, fontsize=8,
        )
        _figure_footer(
            fig,
            "Green circles are refusal, welfare-preserving choice, and restraint; red diamonds are compliance, focal-advantage choice, and overuse. Positions use 144 Part 0 responses, 384 Part 1 roots, or Part 2 scheduled agent-days. Unclear or invalid outputs stay in denominators but are omitted as visual bookkeeping. Panels are not pooled.",
            x=0.07,
            y=0.008,
            width=95,
        )
        fig.tight_layout(rect=(0.025, 0.125, 0.995, 0.90), h_pad=0.8)
        output.extend(
            _save_figure(
                fig, directory,
                f"all_models_cross_phase_outcome_profile_block{block_index + 1}",
                f"Task-specific outcome profiles by exact model route, block {block_index + 1}",
            )
        )
    return output


def _plot_part1(data: Mapping[str, Any], directory: Path) -> list[Path]:
    rows = data["part1"]
    # The adjacent full table preserves each exact upstream Model ID.  Route
    # IDs alone keep the continuous ranking readable at portrait print scale.
    labels = [str(row["target_id"]) for row in rows]
    welfare = [float(row["welfare_preserving_rate_all_scheduled"]) for row in rows]
    welfare_intervals = [
        (
            float(row["welfare_preserving_rate_all_scheduled_finite_bank_sensitivity_low"]),
            float(row["welfare_preserving_rate_all_scheduled_finite_bank_sensitivity_high"]),
        )
        for row in rows
    ]
    # Treat this as one continuous 75-route ranking displayed in consecutive
    # portrait segments.  Compressing every row onto one physical page would
    # leave less than nine points per row and make exact labels unreadable.
    block_count = 3
    block_size = math.ceil(len(rows) / block_count)
    output: list[Path] = []
    for block_index in range(block_count):
        start = block_index * block_size
        stop = min(len(rows), start + block_size)
        fig, ax = plt.subplots(figsize=(7.2, 9.2))
        fig.patch.set_facecolor("white")
        fig.suptitle(
            "Part 1 self-choice outcomes across all 75 exact routes",
            x=0.08, y=0.995, ha="left", fontsize=15, fontweight="bold", color=INK,
        )
        fig.text(
            0.08, 0.93,
            f"Continuous ranking, rows {start + 1}-{stop} of {len(rows)}; 384 scheduled roots per route.",
            fontsize=9, color=MUTED,
        )
        _lollipop_panel(
            ax, welfare[start:stop], labels[start:stop],
            title="Welfare-preserving [root sensitivity 95%]",
            color=ORANGE, show_labels=True,
            intervals=welfare_intervals[start:stop],
        )
        ax.tick_params(axis="y", labelsize=6.2)
        _figure_footer(
            fig,
            "Bars are welfare-preserving first attempts over all 384 roots; whiskers are frozen-root-bank sensitivity intervals, not population CIs. Higher values mean fewer counterpart costs in this task.",
            width=95,
        )
        fig.tight_layout(rect=(0.035, 0.07, 0.995, 0.90))
        output.extend(
            _save_figure(
                fig, directory, f"part1_all_models_block{block_index + 1}",
                f"Part 1 all-model outcomes, block {block_index + 1}",
            )
        )
    return output


def _plot_part2(data: Mapping[str, Any], directory: Path) -> list[Path]:
    rows = _provider_grouped_rows(
        data["part2"], score_key="mean_trajectory_restraint_rate_all_scheduled"
    )
    labels = _provider_prefixed_labels(rows)
    restraint = [
        float(row["mean_trajectory_restraint_rate_all_scheduled"]) for row in rows
    ]
    restraint_intervals = [
        (
            row["mean_trajectory_restraint_rate_all_scheduled_t95_low"],
            row["mean_trajectory_restraint_rate_all_scheduled_t95_high"],
        )
        for row in rows
    ]
    aurc = [
        None if row["mean_aurc_eligible"] is None else float(row["mean_aurc_eligible"])
        for row in rows
    ]
    aurc_intervals = [
        (row["mean_aurc_eligible_t95_low"], row["mean_aurc_eligible_t95_high"])
        for row in rows
    ]
    population = [
        None
        if row["mean_population_retention_eligible"] is None
        else float(row["mean_population_retention_eligible"])
        for row in rows
    ]
    population_intervals = [
        (
            row["mean_population_retention_eligible_t95_low"],
            row["mean_population_retention_eligible_t95_high"],
        )
        for row in rows
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 10.2), sharey=False)
    fig.patch.set_facecolor("white")
    fig.suptitle("Part 2 commons outcomes for 19 exact model routes", x=0.075, y=0.995, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.075, 0.953, "One row per route; 12 trajectories per route; provider families are contiguous and ranked within family.", fontsize=9, color=MUTED)
    _lollipop_panel(axes[0], restraint, labels, title="Model action: mean trajectory restraint [t95]", color=GREEN, show_labels=True, intervals=restraint_intervals)
    _lollipop_panel(axes[1], aurc, labels, title="Resource consequence: mean AURC [t95] / env.", color=GREEN, show_labels=True, intervals=aurc_intervals)
    _lollipop_panel(axes[2], population, labels, title="Group consequence: population retained [t95] / env.", color=GREEN, show_labels=True, intervals=population_intervals)
    for ax in axes:
        ax.tick_params(axis="y", labelsize=5.7)
        _draw_provider_separators(ax, rows)
    _figure_footer(fig, "The three stacked panels connect model action (restraint), resource consequence (AURC), and group consequence (final population retained). Whiskers are trajectory-level Student-t 95% intervals. Higher values mean more preservation in this simulator. AUPC and nondepletion remain in the released diagnostics; invalid actions remain in denominators and eligibility checks rather than a separate argument-facing panel.", x=0.075, width=95)
    fig.tight_layout(rect=(0.025, 0.08, 0.995, 0.94), h_pad=1.0)
    return _save_figure(fig, directory, "part2_all_models", "Part 2 all-model outcomes")


def _plot_role(data: Mapping[str, Any], directory: Path) -> list[Path]:
    by_key = data["role"]
    role_rows = _provider_grouped_rows(
        [by_key[(target, ROLE_FRAMES[0])] for target in data["role_targets"]]
    )
    targets = [str(row["target_id"]) for row in role_rows]
    labels = _provider_prefixed_labels(role_rows)
    welfare = [[float(by_key[(target, frame)]["welfare_preserving_rate_all_scheduled"]) for frame in ROLE_FRAMES] for target in targets]
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    fig.patch.set_facecolor("white")
    fig.suptitle("Part 1 role-calibration outcomes by exact sentinel route and frame", x=0.08, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(
        0.08,
        0.91,
        f"{len(targets)} provider-grouped sentinels x three separate frames; 384 scheduled draws per route-frame; frames are not pooled.",
        fontsize=9,
        color=MUTED,
    )
    _heatmap(ax, welfare, ROLE_FRAMES, labels, title="Welfare-preserving / all scheduled draws", cmap=P1_CMAP, vmin=0.0, vmax=1.0)
    _draw_provider_separators(ax, role_rows)
    _figure_footer(fig, "Higher welfare preservation means fewer counterpart costs within that role-conditioned task. Frame differences are descriptive, not causal; invalid outputs remain in each scheduled denominator.", y=0.02)
    fig.tight_layout(rect=(0.055, 0.09, 0.995, 0.87))
    return _save_figure(fig, directory, "part1_role_calibration", "Part 1 role-calibration outcomes")


def _plot_sensitivity(data: Mapping[str, Any], directory: Path) -> list[Path]:
    models = data["sensitivity_models"]
    sensitivity_rows = _provider_grouped_rows(
        [models[target] for target in data["sensitivity_targets"]]
    )
    targets = [str(row["target_id"]) for row in sensitivity_rows]
    effects = data["sensitivity"]
    labels = _provider_prefixed_labels(sensitivity_rows)
    matrix = [[float(effects[(target, factor)]["effect_high_minus_low"]) for factor in SENSITIVITY_FACTORS] for target in targets]
    limit = max(0.02, max(abs(value) for row in matrix for value in row))
    fig, ax = plt.subplots(figsize=(7.2, 7.8))
    fig.patch.set_facecolor("white")
    fig.suptitle("Part 2 exploratory sensitivity: high-minus-low normalized-AURC effects", x=0.09, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(
        0.09,
        0.91,
        f"{len(targets)} compatible sentinels x five prespecified factors; 16 resolution-V cells and two common seeds per sentinel; Holm family = {len(targets) * len(SENSITIVITY_FACTORS)}.",
        fontsize=9,
        color=MUTED,
    )
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
    _draw_provider_separators(ax, sensitivity_rows)
    for row_index, target in enumerate(targets):
        for column_index, factor in enumerate(SENSITIVITY_FACTORS):
            effect = effects[(target, factor)]
            value = float(effect["effect_high_minus_low"])
            status = "H" if float(effect["holm_adjusted_p"]) <= 0.05 else "n.s."
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
            color = _annotation_color(SIGNED_CMAP(norm(value)))
            ax.text(column_index + 0.5, row_index + 0.5, f"{value:+.3f}\n{status}", ha="center", va="center", fontsize=7, color=color)
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.027, pad=0.025)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    colorbar.set_label("High - low normalized AURC", fontsize=8, color=INK)
    colorbar.ax.tick_params(labelsize=7, colors=INK)
    _style_axes(ax)
    _figure_footer(fig, "H: global Holm-adjusted p <= 0.05; n.s.: otherwise. Positive/negative indicates effect direction only and is not automatically good/bad for a parameter factor.", x=0.09, y=0.02, width=95)
    fig.tight_layout(rect=(0.06, 0.095, 0.99, 0.87))
    return _save_figure(fig, directory, "part2_sensitivity_effects", "Part 2 sensitivity main effects")


def _plot_local_controls(data: Mapping[str, Any], directory: Path) -> list[Path]:
    rows = data["local_controls"]
    labels = [f"{str(row['model_id']).rsplit('/', 1)[-1]} | {row['parameter_scale']}" for row in rows]
    welfare = [float(row["welfare_rate"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
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
        ax, welfare, labels,
        title="Welfare-preserving / all 384 scheduled units", color=ORANGE, show_labels=True,
    )
    _figure_footer(
        fig,
        "Bars show welfare-preserving choice with invalid outputs retained as nonsuccesses. Higher values are preferable only within this task; these exploratory scale controls are separate from, and not substitutes for, hosted-route or confirmatory evidence.",
        x=0.085,
        y=0.02,
        width=110,
    )
    fig.tight_layout(rect=(0.055, 0.12, 0.99, 0.84))
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


def _pct_interval(estimate: float, low: float, high: float) -> str:
    return f"{_pct(estimate)} [{_pct(low)}, {_pct(high)}]"


def _decimal_interval(estimate: float, low: float, high: float) -> str:
    return f"{estimate:.3f} [{low:.3f}, {high:.3f}]"


def _optional_interval_cell(
    estimate: object, low: object, high: object, *, percent: bool
) -> str:
    if estimate is None:
        return "NE"
    if low is None or high is None:
        return (_pct(float(estimate)) if percent else f"{float(estimate):.3f}") + " [NE]"
    if percent:
        return _pct_interval(float(estimate), float(low), float(high))
    return _decimal_interval(float(estimate), float(low), float(high))


def _markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


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
    cross_phase_markdown_rows = []
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
        _provider, model = next(iter(identities))
        p0 = phase_indices["part0"].get(target)
        p1 = phase_indices["part1"].get(target)
        p2 = phase_indices["part2"].get(target)
        p0_cell = "-- (not in panel)"
        if p0 is not None:
            p0_cell = (
                "R "
                + _pct_interval(
                    float(p0["refusal_rate_all_scheduled"]),
                    float(p0["refusal_rate_all_scheduled_finite_bank_sensitivity_low"]),
                    float(p0["refusal_rate_all_scheduled_finite_bank_sensitivity_high"]),
                )
            )
        p1_cell = "-- (not in panel)"
        if p1 is not None:
            p1_cell = (
                "W "
                + _pct_interval(
                    float(p1["welfare_preserving_rate_all_scheduled"]),
                    float(p1["welfare_preserving_rate_all_scheduled_finite_bank_sensitivity_low"]),
                    float(p1["welfare_preserving_rate_all_scheduled_finite_bank_sensitivity_high"]),
                )
            )
        p2_cell = "-- (not in panel)"
        if p2 is not None:
            aurc = (
                "NE"
                if p2["mean_aurc_eligible"] is None
                else _optional_interval_cell(
                    p2["mean_aurc_eligible"], p2["mean_aurc_eligible_t95_low"],
                    p2["mean_aurc_eligible_t95_high"], percent=False,
                )
            )
            p2_cell = (
                f"R {_pct(float(p2['restraint_rate_all_scheduled']))}; "
                f"A {aurc}"
            )
        cross_phase_rows.append(
            [
                _tex_escape(target),
                _tex_escape(model),
                p0_cell,
                p1_cell,
                p2_cell,
            ]
        )
        cross_phase_markdown_rows.append(
            [target, model, p0_cell.replace("\\%", "%"),
             p1_cell.replace("\\%", "%"), p2_cell.replace("\\%", "%")]
        )
    path = directory / "all_models_cross_phase_table.tex"
    path.write_text(
        _table_tex(
            caption=(
                "Exact-route cross-phase result matrix. Each row is one authenticated target route and exact upstream Model ID in the union of the three hosted panels. Part 0 reports refusal R as estimate [95\\% frozen-root-bank sensitivity interval] over 144 responses; higher R means less harmful-request assistance. Part 1 reports welfare-preserving self-choice W as estimate [95\\% stratified frozen-root-bank sensitivity interval] over 384 roots; higher W means fewer counterpart costs. Those bootstrap intervals describe fixed-bank sensitivity, not population confidence intervals. Part 2 reports all-scheduled restraint R and environmentally estimable-trajectory normalized AURC A as estimate [trajectory Student-t 95\\% interval] over 12 common-seed trajectories; higher R and A mean greater reserve preservation. NE means no environmentally estimable AURC trajectory, while -- means the route was not in that phase. Columns are centered for comparison but remain different estimands: no cell is imputed and no composite or general safety ranking is computed."
            ),
            label="tab:provider-safe-v2-all-models-cross-phase",
            headers=("Target route ID", "Model ID", "Part 0: R [95\\%]", "Part 1: W [95\\%]", "Part 2: R; A [95\\%]"),
            rows=cross_phase_rows,
            column_spec="llccc",
            chunk_size=24,
        ),
        encoding="utf-8",
    )
    output.append(path)
    markdown_path = directory / "all_models_cross_phase_table.md"
    markdown_headers = (
        "Target route ID", "Model ID", "Part 0: R [95%]",
        "Part 1: W [95%]", "Part 2: R; A [95%]",
    )
    markdown_lines = [
        "# Exact-route cross-phase results",
        "",
        (
            "Each row is one authenticated exact route. Part 0 R is refusal over all scheduled "
            "responses with a deterministic 5,000-replicate frozen-root-bank sensitivity interval; "
            "higher is less harmful-request assistance. Part 1 W is welfare preservation over all "
            "scheduled roots with the analogous 12-stratum frozen-bank sensitivity interval; higher "
            "means fewer counterpart costs. These Part 0/1 intervals describe sensitivity to the "
            "fixed prompt banks, not population confidence intervals. Part 2 R is all-scheduled "
            "restraint and A is mean environmentally estimable AURC with a trajectory-level Student-t "
            "95% interval; higher means more reserve preservation. NE means no estimable "
            "trajectory; -- means the route was not tested in that phase. Columns remain distinct "
            "estimands and are not a composite or general safety ranking."
        ),
        "",
        "| " + " | ".join(markdown_headers) + " |",
        "| " + " | ".join("---" for _ in markdown_headers) + " |",
        *(
            "| " + " | ".join(_markdown_escape(value) for value in row) + " |"
            for row in cross_phase_markdown_rows
        ),
        "",
    ]
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    output.append(markdown_path)

    part0_rows = []
    for row in data["part0"]:
        values = [_tex_escape(row["target_id"]), _tex_escape(row["model"])]
        values.append(
            _pct_interval(
                float(row["refusal_rate_all_scheduled"]),
                float(row["refusal_rate_all_scheduled_finite_bank_sensitivity_low"]),
                float(row["refusal_rate_all_scheduled_finite_bank_sensitivity_high"]),
            )
        )
        for language in PART0_LANGUAGES:
            aggregate = data["part0_matrix"][(row["target_id"], language)]
            values.extend(
                [
                    _pct_interval(
                        float(aggregate["refusal_rate_all_scheduled"]),
                        float(aggregate["refusal_rate_all_scheduled_wilson95_low"]),
                        float(aggregate["refusal_rate_all_scheduled_wilson95_high"]),
                    )
                ]
            )
        part0_rows.append(values)
    path = directory / "part0_model_language_table.tex"
    path.write_text(
        _table_tex(
            caption=(
                "Part 0 exact-route response-language outcomes. Each row is one authenticated target route and exact upstream Model ID, shown in descending overall refusal-rate display order. Overall R [95\\%] is refusal over all 144 scheduled responses followed by a deterministic 5,000-replicate interval that resamples the 48 harmful-request roots while retaining each root's three language conditions. English, Chinese, and Russian R [95\\%] are refusal over 48 scheduled roots followed by Wilson intervals. Every quantitative column is centered. Higher R means less harmful-request assistance on this harmful-only task; lower R means less refusal under the all-scheduled scoring rule. It is not a general safety score."
            ),
            label="tab:provider-safe-v2-part0-model-language",
            headers=("Target route ID", "Model ID", "Overall R [95\\%]", "English R [95\\%]", "Chinese R [95\\%]", "Russian R [95\\%]"),
            rows=part0_rows,
            column_spec="llcccc",
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
                f"{row['welfare_count']}/{row['scheduled']} ({_pct(float(row['welfare_rate']))})",
            ]
        )
    path = directory / "part1_local_controls_table.tex"
    path.write_text(
        _table_tex(
            caption=(
                "Part 1 exploratory local execution-scale controls, reported separately from hosted routes. Each row is one exact offline HF Model ID; Parameter scale is the advertised parameter count; Welfare/scheduled is the welfare-preserving count and percentage over all 384 scheduled roots. The result column is centered. Higher welfare preservation means fewer counterpart costs in this task; lower values mean the focal actor more often selects its unilateral advantage. These four fixed local models are execution-scale controls, not substitutes for hosted routes and not confirmatory or general safety evidence."
            ),
            label="tab:provider-safe-v2-part1-local-controls",
            headers=("Model ID", "Parameter scale", "Welfare/scheduled"),
            rows=local_rows,
            column_spec="llc",
            chunk_size=4,
        ),
        encoding="utf-8",
    )
    output.append(path)

    part1_rows = []
    for row in data["part1"]:
        part1_rows.append(
            [
                _tex_escape(row["target_id"]), _tex_escape(row["model"]),
                _pct_interval(
                    float(row["welfare_preserving_rate_all_scheduled"]),
                    float(row["welfare_preserving_rate_all_scheduled_finite_bank_sensitivity_low"]),
                    float(row["welfare_preserving_rate_all_scheduled_finite_bank_sensitivity_high"]),
                ),
            ]
        )
    path = directory / "part1_all_models_table.tex"
    path.write_text(
        _table_tex(
            caption=(
                "Part 1 self-choice outcomes for all 75 exact model routes. Each row is one authenticated target route and exact upstream Model ID, shown in descending welfare-preserving-rate display order. Welfare/scheduled [95\\%] is the welfare-preserving first-attempt share over all 384 scheduled roots followed by a deterministic 5,000-replicate interval that resamples roots separately within the 12 game-domain strata. The centered interval is frozen-bank sensitivity, not a population confidence interval. Higher welfare preservation means fewer counterpart costs in this task; lower values mean the focal actor more often selects its unilateral advantage. The value is not a general safety score."
            ),
            label="tab:provider-safe-v2-part1-all-models",
            headers=("Target route ID", "Model ID", "Welfare/scheduled [95\\%]"),
            rows=part1_rows,
            column_spec="llc",
            chunk_size=25,
        ),
        encoding="utf-8",
    )
    output.append(path)

    part2_rows = []
    for row in data["part2"]:
        part2_rows.append(
            [
                _tex_escape(row["target_id"]), _tex_escape(row["model"]),
                _pct(float(row["restraint_rate_all_scheduled"])),
                _optional_interval_cell(
                    row["mean_aurc_eligible"], row["mean_aurc_eligible_t95_low"],
                    row["mean_aurc_eligible_t95_high"], percent=False,
                ),
                _optional_interval_cell(
                    row["mean_population_retention_eligible"],
                    row["mean_population_retention_eligible_t95_low"],
                    row["mean_population_retention_eligible_t95_high"], percent=True,
                ),
            ]
        )
    path = directory / "part2_all_models_table.tex"
    path.write_text(
        _table_tex(
            caption=(
                "Part 2 commons outcomes for all 19 exact model routes. Each row is one authenticated target route and exact upstream Model ID, shown in descending all-scheduled restraint-rate order. The three centered result columns connect model action, resource consequence, and group consequence: Restraint is the restrained-action share over scheduled agent-days; Mean AURC [95\\%] is normalized reserve area with its trajectory Student-t interval; Population retained [95\\%] is final population divided by initial population with the corresponding interval. Higher values mean more preservation in this simulator; lower values mean more overuse, reserve depletion, or population loss. NE means no environmentally estimable trajectory, not zero. None of these columns is a general safety score."
            ),
            label="tab:provider-safe-v2-part2-all-models",
            headers=("Target route ID", "Model ID", "Restraint", "Mean AURC [95\\%]", "Population retained [95\\%]"),
            rows=part2_rows,
            column_spec="llccc",
            chunk_size=19,
        ),
        encoding="utf-8",
    )
    output.append(path)

    role_rows = []
    for target in data["role_targets"]:
        base = data["role"][(target, ROLE_FRAMES[0])]
        values = [_tex_escape(target), _tex_escape(base["model"])]
        for frame in ROLE_FRAMES:
            row = data["role"][(target, frame)]
            values.append(_pct(float(row["welfare_preserving_rate_all_scheduled"])))
        role_rows.append(values)
    path = directory / "part1_role_calibration_table.tex"
    path.write_text(
        _table_tex(
            caption=(
                f"Part 1 exploratory role calibration for {len(data['role_targets'])} exact sentinel routes. Each row is one authenticated target route and exact upstream Model ID; Advice, Observer evaluation, and Prediction report welfare-preserving first attempts over all 384 scheduled draws in that named frame. The three centered columns are separate estimands and are never pooled. Higher values mean fewer counterpart costs only within the named frame; lower values mean the response more often favors the focal actor's unilateral advantage. Differences are descriptive rather than causal, and no frame is a general safety score."
            ),
            label="tab:provider-safe-v2-part1-role-calibration",
            headers=("Target route ID", "Model ID", "Advice W/scheduled", "Observer evaluation W/scheduled", "Prediction W/scheduled"),
            rows=role_rows,
            column_spec="llccc",
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
                    f"{float(row['effect_high_minus_low']):+.4f}", f"{holm:.4f}",
                ]
            )
    path = directory / "part2_sensitivity_effects_table.tex"
    path.write_text(
        _table_tex(
            caption=(
                f"Part 2 deadline-exploratory sensitivity effects for {len(data['sensitivity_targets'])} exact compatible sentinel routes, with no route substitution. Each row is one sentinel route and prespecified factor; Model ID is the authenticated upstream model; Effect is mean normalized AURC at the factor's high level minus its low level over the resolution-V design and two common-seed blocks. The low-to-high pairs are capacity per initial agent 5 to 15, depletion units 1 to 2, collapse death rate 0.1 to 0.4, society size 4 to 8, and horizon days 10 to 20. Holm p adjusts the {len(sensitivity_rows)} sentinel-by-factor tests, with adjusted p at most 0.05 treated as significant. The two centered result columns directly assess effect size and evidence against the null without a redundant derived status column. Positive means the high level increased reserve preservation and negative means it decreased preservation, but sign is not automatically good or bad for the parameter. With two seeds this panel is underpowered and descriptive, not a general safety score."
            ),
            label="tab:provider-safe-v2-part2-sensitivity",
            headers=("Target route ID", "Model ID", "Factor", "Effect (high-low AURC)", "Holm p"),
            rows=sensitivity_rows,
            column_spec="lllcc",
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


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return deterministic one-based average ranks, including exact ties."""

    indexed = sorted(enumerate(float(value) for value in values), key=lambda item: item[1])
    ranks = [0.0] * len(indexed)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for offset in range(start, end):
            ranks[indexed[offset][0]] = average
        start = end
    return ranks


def _pearson(values_x: Sequence[float], values_y: Sequence[float]) -> float | None:
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return None
    mean_x = math.fsum(float(value) for value in values_x) / len(values_x)
    mean_y = math.fsum(float(value) for value in values_y) / len(values_y)
    centered_x = [float(value) - mean_x for value in values_x]
    centered_y = [float(value) - mean_y for value in values_y]
    denominator = math.sqrt(
        math.fsum(value * value for value in centered_x)
        * math.fsum(value * value for value in centered_y)
    )
    if denominator == 0:
        return None
    return math.fsum(
        value_x * value_y for value_x, value_y in zip(centered_x, centered_y)
    ) / denominator


def _spearman(values_x: Sequence[float], values_y: Sequence[float]) -> float | None:
    return _pearson(_average_ranks(values_x), _average_ranks(values_y))


def _kendall_tau_b(values_x: Sequence[float], values_y: Sequence[float]) -> float | None:
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return None
    concordant = discordant = tied_x = tied_y = 0
    for left in range(len(values_x)):
        for right in range(left + 1, len(values_x)):
            delta_x = float(values_x[left]) - float(values_x[right])
            delta_y = float(values_y[left]) - float(values_y[right])
            if delta_x == 0 and delta_y == 0:
                continue
            if delta_x == 0:
                tied_x += 1
            elif delta_y == 0:
                tied_y += 1
            elif delta_x * delta_y > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + tied_x)
        * (concordant + discordant + tied_y)
    )
    if denominator == 0:
        return None
    return (concordant - discordant) / denominator


def _fixed_panel_pair_diagnostic(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    left_field: str,
    right_field: str,
) -> dict[str, Any] | None:
    """Describe rank preservation over matched exact routes without population inference."""

    left = {str(row["target_id"]): float(row[left_field]) for row in left_rows}
    right = {str(row["target_id"]): float(row[right_field]) for row in right_rows}
    targets = sorted(set(left) & set(right))
    if len(targets) < 3:
        return None
    values_x = [left[target] for target in targets]
    values_y = [right[target] for target in targets]
    spearman = _spearman(values_x, values_y)
    kendall = _kendall_tau_b(values_x, values_y)
    if spearman is None or kendall is None:
        return None
    ranks_x = _average_ranks(values_x)
    ranks_y = _average_ranks(values_y)
    shifts = [abs(rank_x - rank_y) for rank_x, rank_y in zip(ranks_x, ranks_y)]
    leave_one_out = []
    for omitted in range(len(targets)):
        rho = _spearman(
            [value for index, value in enumerate(values_x) if index != omitted],
            [value for index, value in enumerate(values_y) if index != omitted],
        )
        if rho is not None:
            leave_one_out.append(rho)
    if not leave_one_out:
        return None
    return {
        "matched_route_count": len(targets),
        "spearman_rho": spearman,
        "kendall_tau_b": kendall,
        "leave_one_route_out_spearman_minimum": min(leave_one_out),
        "leave_one_route_out_spearman_maximum": max(leave_one_out),
        "maximum_absolute_rank_shift_positions": max(shifts),
        "median_absolute_rank_shift_positions": float(statistics.median(shifts)),
    }


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
    part0_operational_invalid = sum(
        int(row["operational_invalid_count"]) for row in part0
    )
    part0_visible = sum(int(row["visible_subject_response_count"]) for row in part0)
    if part0_refusal + part0_compliance + part0_unclear + part0_invalid != part0_scheduled:
        raise PaperAssetsError("Part 0 headline totals do not reconcile to scheduled responses.")
    if part0_visible + part0_operational_invalid != part0_scheduled:
        raise PaperAssetsError("Part 0 visible/operational headline totals do not reconcile.")

    part1_scheduled = sum(int(row["scheduled_units"]) for row in part1)
    part1_welfare = sum(int(row["welfare_preserving_count_first_attempt"]) for row in part1)
    part1_invalid = sum(int(row["first_attempt_invalid_count"]) for row in part1)
    if part1_welfare > part1_scheduled - part1_invalid:
        raise PaperAssetsError("Part 1 headline welfare count exceeds valid scheduled units.")

    part2_trajectories = sum(int(row["trajectory_count"]) for row in part2)
    part2_eligible_trajectories = sum(
        int(row["operationally_eligible_trajectory_count"]) for row in part2
    )
    part2_environmentally_estimable_trajectories = sum(
        int(row["environmentally_estimable_trajectory_count"])
        for row in part2
    )
    part2_semantic_invalid_trajectories = sum(
        int(row["semantic_invalid_trajectory_count"])
        for row in part2
    )
    part2_scheduled_agent_days = sum(int(row["scheduled_agent_days"]) for row in part2)
    part2_invalid_agent_days = sum(int(row["first_attempt_invalid_count"]) for row in part2)
    part2_valid_agent_days = part2_scheduled_agent_days - part2_invalid_agent_days
    part2_nonestimable = sum(row["mean_aurc_eligible"] is None for row in part2)
    if (
        part2_eligible_trajectories > part2_trajectories
        or part2_environmentally_estimable_trajectories
        + part2_semantic_invalid_trajectories
        != part2_eligible_trajectories
        or part2_valid_agent_days < 0
    ):
        raise PaperAssetsError("Part 2 headline trajectory/agent-day totals do not reconcile.")
    part2_aurc = [
        float(row["mean_aurc_eligible"])
        for row in part2
        if row["mean_aurc_eligible"] is not None
    ]
    part2_aupc = [
        float(row["mean_aupc_eligible"])
        for row in part2
        if row["mean_aupc_eligible"] is not None
    ]
    part2_nondepletion = [
        float(row["reserve_nondepletion_rate_eligible"])
        for row in part2
        if row["reserve_nondepletion_rate_eligible"] is not None
    ]
    part2_population_retention = [
        float(row["mean_population_retention_eligible"])
        for row in part2
        if row["mean_population_retention_eligible"] is not None
    ]

    values: list[tuple[str, str]] = [
        ("ProviderSafePartZeroModelCount", str(len(part0))),
        ("ProviderSafePartZeroScheduledResponseCount", str(part0_scheduled)),
        ("ProviderSafePartZeroRefusalCount", str(part0_refusal)),
        ("ProviderSafePartZeroComplianceCount", str(part0_compliance)),
        ("ProviderSafePartZeroUnclearCount", str(part0_unclear)),
        ("ProviderSafePartZeroInvalidCount", str(part0_invalid)),
        ("ProviderSafePartZeroOperationalInvalidCount", str(part0_operational_invalid)),
        ("ProviderSafePartZeroVisibleResponseCount", str(part0_visible)),
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
        ("ProviderSafePartTwoEnvironmentallyEstimableTrajectoryCount", str(part2_environmentally_estimable_trajectories)),
        ("ProviderSafePartTwoSemanticInvalidTrajectoryCount", str(part2_semantic_invalid_trajectories)),
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
            part2_aupc,
            prefix="ProviderSafePartTwoModelNormalizedAUPC",
            formatter=lambda value: _decimal_headline(value, 3),
        ),
        *_summary_macros(
            part2_nondepletion,
            prefix="ProviderSafePartTwoModelReserveNondepletionRatePct",
            formatter=_percent_headline,
        ),
        *_summary_macros(
            part2_population_retention,
            prefix="ProviderSafePartTwoModelPopulationRetentionPct",
            formatter=_percent_headline,
        ),
        *_summary_macros(
            [float(row["restraint_rate_all_scheduled"]) for row in part2],
            prefix="ProviderSafePartTwoModelRestraintRatePct",
            formatter=_percent_headline,
        ),
    ]

    pairwise_specs = (
        (
            "PartZeroPartOne",
            part0,
            part1,
            "refusal_rate_all_scheduled",
            "welfare_preserving_rate_all_scheduled",
        ),
        (
            "PartZeroPartTwo",
            part0,
            part2,
            "refusal_rate_all_scheduled",
            "restraint_rate_all_scheduled",
        ),
        (
            "PartOnePartTwo",
            part1,
            part2,
            "welfare_preserving_rate_all_scheduled",
            "restraint_rate_all_scheduled",
        ),
    )
    for pair_name, left_rows, right_rows, left_field, right_field in pairwise_specs:
        diagnostic = _fixed_panel_pair_diagnostic(
            left_rows, right_rows, left_field, right_field
        )
        if diagnostic is None:
            continue
        prefix = f"ProviderSafePairwise{pair_name}"
        values.extend(
            [
                (f"{prefix}MatchedRouteCount", str(diagnostic["matched_route_count"])),
                (f"{prefix}SpearmanRho", _decimal_headline(diagnostic["spearman_rho"], 3)),
                (f"{prefix}KendallTauB", _decimal_headline(diagnostic["kendall_tau_b"], 3)),
                (
                    f"{prefix}LeaveOneOutSpearmanMinimum",
                    _decimal_headline(
                        diagnostic["leave_one_route_out_spearman_minimum"], 3
                    ),
                ),
                (
                    f"{prefix}LeaveOneOutSpearmanMaximum",
                    _decimal_headline(
                        diagnostic["leave_one_route_out_spearman_maximum"], 3
                    ),
                ),
                (
                    f"{prefix}MaximumAbsoluteRankShift",
                    _decimal_headline(
                        diagnostic["maximum_absolute_rank_shift_positions"], 1
                    ),
                ),
                (
                    f"{prefix}MedianAbsoluteRankShift",
                    _decimal_headline(
                        diagnostic["median_absolute_rank_shift_positions"], 1
                    ),
                ),
            ]
        )

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
        "% Pairwise fixed-panel rank diagnostics remain separate; no cross-axis aggregate, score, population inference, or promotion is defined.",
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
        assets.extend(_plot_refusal_and_cooperation_overview(data, temporary))
        assets.extend(_plot_matched_current_route_profile(data, temporary))
        assets.extend(_plot_part1(data, temporary))
        assets.extend(_plot_part2(data, temporary))
        assets.extend(_plot_role(data, temporary))
        assets.extend(_plot_sensitivity(data, temporary))
        assets.extend(_plot_local_controls(data, temporary))
        assets.extend(_plot_cross_phase_outcome_profile(data, temporary))
        assets.extend(_write_tables(data, temporary))
        assets.append(_write_headlines(data, temporary))
        asset_rows = [
            {
                "name": path.name,
                "kind": (
                    "vector_pdf" if path.suffix == ".pdf" else
                    "raster_png" if path.suffix == ".png" else
                    "latex_macros" if path.name == "paper_headlines.tex" else
                    "markdown_table" if path.suffix == ".md" else
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
            "figure_palette": "original_submission_okabe_ito_blue_orange_green_vermillion",
            "figure_font_family": "Times New Roman (NeurIPS ptm-compatible serif)",
            "figure_row_order": (
                "provider_family_then_within_family_outcome;"
                "global_outcome_order_retained_only_for_rank_profile_figures"
            ),
            "figure_semantic_redundancy": (
                "directional_caption_position_printed_values_and_distinct_marker_shapes"
            ),
            "invalid_policy": INVALID_POLICY,
            "exploratory_only": True,
            "paper_use_status": "exploratory_descriptive_panels_only",
            "confirmatory_or_paper_promotion_permitted": False,
            "human_labels_generated": False,
            "side_by_side_nonpooled_axis_assets_generated": True,
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
