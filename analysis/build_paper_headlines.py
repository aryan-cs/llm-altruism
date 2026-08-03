"""Build manuscript-safe headline slots from sanitized final results.

The exporter is deliberately narrower than ``build_final_results``.  It reads
only that builder's sealed, text-free public artifact and derives coverage,
count, and within-axis distribution summaries.  It never computes model
rankings, family effects, significance tests, or cross-axis associations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
SOURCE_ARTIFACT_TYPE = "prosocial_readiness_final_sanitized_results"
OUTPUT_ARTIFACT_TYPE = "prosocial_readiness_paper_headlines"
LANGUAGES = ("english", "chinese", "russian")
PART1_PRE_EXECUTION_UNAVAILABLE_TARGET_IDS = frozenset({
    "moonshotai/kimi-k2.5",
    "moonshotai/kimi-k2.6",
    "zai-org/glm-5.2",
})
FORBIDDEN_KEYS = {
    "assistant_response",
    "completion",
    "content",
    "prompt",
    "prompt_text",
    "messages",
    "visible_response",
    "response_text",
    "visible_content",
    "reasoning",
    "raw_response",
    "request_body",
    "requested_route",
    "route",
    "journal",
    "journals",
    "private",
    "private_path",
    "raw",
    "request",
    "response",
    "system_prompt",
    "text",
    "user_prompt",
}


class PaperHeadlineError(RuntimeError):
    """The sanitized result graph cannot support exact manuscript slots."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    return _sha256_json(
        {key: item for key, item in value.items() if key != "evidence_sha256"}
    )


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    value.pop("evidence_sha256", None)
    value["evidence_sha256"] = _self_hash(value)
    return value


def _read_source(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PaperHeadlineError("Final results are not readable UTF-8 JSON.") from error
    if not isinstance(value, dict):
        raise PaperHeadlineError("Final results must be a JSON object.")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("artifact_type") != SOURCE_ARTIFACT_TYPE
        or value.get("evidence_sha256") != _self_hash(value)
    ):
        raise PaperHeadlineError("Final-results schema/type/self-hash validation failed.")
    _reject_forbidden_fields(value)
    privacy = value.get("privacy_contract")
    expected_privacy = {
        "contains_prompt_text": False,
        "contains_response_text": False,
        "contains_reasoning": False,
        "contains_raw_responses": False,
        "contains_routes": False,
    }
    if not isinstance(privacy, Mapping) or any(
        privacy.get(key) is not expected for key, expected in expected_privacy.items()
    ):
        raise PaperHeadlineError("Final-results privacy contract is absent or unsafe.")
    bindings = value.get("bindings")
    if not isinstance(bindings, Mapping) or any(
        key not in bindings or bindings.get(key) is None
        for key in ("part0", "part1", "part2")
    ):
        raise PaperHeadlineError("Required part bindings are missing.")
    parameters = value.get("parameters")
    if (
        not isinstance(parameters, Mapping)
        or parameters.get("part1_scopes_pooled") is not False
        or parameters.get("part1_full_root_count") != 384
        or parameters.get("part1_balanced_partial_root_counts") != [12, 96]
        or parameters.get("part2_uncertainty_unit") != "independent_trajectory"
        or parameters.get("part0_condition_interpretation")
        != (
            "reconstructed response-language conditions over English source requests; "
            "not translated-prompt conditions"
        )
    ):
        raise PaperHeadlineError("Final-results reporting parameters changed.")
    return value, hashlib.sha256(raw).hexdigest()


def _reject_forbidden_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PaperHeadlineError(f"Non-string key at {path}.")
            normalized = key.casefold()
            if normalized in FORBIDDEN_KEYS or normalized.startswith("private_"):
                raise PaperHeadlineError(f"Forbidden private/text field at {path}.{key}.")
            _reject_forbidden_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_fields(item, f"{path}[{index}]")


def _objects(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise PaperHeadlineError(f"{label} rows are absent.")
    if any(not isinstance(row, Mapping) for row in value):
        raise PaperHeadlineError(f"{label} contains a non-object row.")
    return list(value)


def _nonempty_string(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise PaperHeadlineError(f"{label} must be a nonempty string.")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise PaperHeadlineError(f"{label} must be an integer >= {minimum}.")
    return value


def _finite(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise PaperHeadlineError(f"{label} must be finite.")
    return float(value)


def _unit_rate(value: Any, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise PaperHeadlineError(f"{label} must be in [0, 1].")
    return result


def _interval(
    value: Any,
    label: str,
    *,
    center_key: str,
    expected_n: int,
    bounded_center: bool = True,
    bounded_interval: bool = False,
) -> tuple[float, float, float]:
    if not isinstance(value, Mapping):
        raise PaperHeadlineError(f"{label} interval is absent.")
    center = (
        _unit_rate(value.get(center_key), f"{label}.{center_key}")
        if bounded_center
        else _finite(value.get(center_key), f"{label}.{center_key}")
    )
    lower = _finite(value.get("lower"), f"{label}.lower")
    upper = _finite(value.get("upper"), f"{label}.upper")
    if bounded_interval and not (0.0 <= lower <= 1.0 and 0.0 <= upper <= 1.0):
        raise PaperHeadlineError(f"{label} interval must be in [0, 1].")
    if lower > center or center > upper:
        raise PaperHeadlineError(f"{label} interval does not contain its center.")
    if _integer(value.get("n"), f"{label}.n", minimum=1) != expected_n:
        raise PaperHeadlineError(f"{label} denominator is inconsistent.")
    return center, lower, upper


def _integer_successes(rate: float, denominator: int, label: str) -> int:
    successes = round(rate * denominator)
    if not math.isclose(
        rate, successes / denominator, rel_tol=0.0, abs_tol=1e-12
    ):
        raise PaperHeadlineError(f"{label} is not exactly derivable as an integer count.")
    return successes


def _percent(value: float) -> float:
    return round(100.0 * value, 1)


def _summary(values: Sequence[float], *, percent: bool) -> dict[str, float]:
    if not values:
        raise PaperHeadlineError("Cannot summarize an empty system set.")
    convert = _percent if percent else lambda item: round(item, 3)
    return {
        "median": convert(float(statistics.median(values))),
        "minimum": convert(min(values)),
        "maximum": convert(max(values)),
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _unavailable_ids(binding: Any, label: str) -> list[str]:
    containers: list[Mapping[str, Any]]
    if isinstance(binding, Mapping):
        containers = [binding]
    elif (
        isinstance(binding, list)
        and binding
        and all(isinstance(item, Mapping) for item in binding)
    ):
        containers = list(binding)
    else:
        raise PaperHeadlineError(f"{label} binding has an invalid structure.")
    result: list[str] = []
    for container in containers:
        bound = container.get("primary") if container.get("overlay_schema_version") == 1 else container
        if (
            not isinstance(bound, Mapping)
            or not isinstance(bound.get("manifest_path"), str)
            or not bound.get("manifest_path")
            or not _is_sha256(bound.get("file_sha256"))
            or not _is_sha256(bound.get("evidence_sha256"))
        ):
            raise PaperHeadlineError(f"{label} manifest binding is incomplete.")
        ids = container.get("unavailable_target_ids", [])
        if not isinstance(ids, list):
            raise PaperHeadlineError(f"{label} unavailable ids are not a list.")
        for target_id in ids:
            result.append(_nonempty_string(target_id, f"{label} unavailable target"))
    if len(result) != len(set(result)):
        raise PaperHeadlineError(f"{label} unavailable target is duplicated.")
    return sorted(result)


def _part0(source: Mapping[str, Any]) -> dict[str, Any]:
    rows = _objects(source.get("part0"), "Part 0")
    unavailable = _unavailable_ids(source["bindings"]["part0"], "Part 0")
    seen: set[str] = set()
    root_counts: set[int] = set()
    overall_rates: list[float] = []
    condition_rates: dict[str, list[float]] = defaultdict(list)
    totals = {key: 0 for key in ("refusal", "compliance", "unclear", "invalid")}
    scheduled = 0
    for index, row in enumerate(rows):
        target_id = _nonempty_string(row.get("target_id"), f"Part 0 row {index} target")
        if target_id in seen or target_id in unavailable:
            raise PaperHeadlineError("Part 0 included/unavailable identities overlap or duplicate.")
        seen.add(target_id)
        _nonempty_string(row.get("model"), f"Part 0 {target_id} model")
        roots = _integer(
            row.get("root_count_per_condition"),
            f"Part 0 {target_id} roots per condition", minimum=1,
        )
        root_counts.add(roots)
        conditions = row.get("conditions")
        if not isinstance(conditions, list) or len(conditions) != 3:
            raise PaperHeadlineError("Part 0 requires exactly three condition rows per system.")
        by_language: dict[str, Mapping[str, Any]] = {}
        system_refusal = 0
        for condition in conditions:
            if not isinstance(condition, Mapping):
                raise PaperHeadlineError("Part 0 condition is not an object.")
            language = _nonempty_string(
                condition.get("response_language_condition"),
                f"Part 0 {target_id} condition",
            )
            if language in by_language:
                raise PaperHeadlineError("Part 0 condition is duplicated.")
            by_language[language] = condition
        if set(by_language) != set(LANGUAGES):
            raise PaperHeadlineError("Part 0 conditions must be English, Chinese, and Russian.")
        for language in LANGUAGES:
            condition = by_language[language]
            rate, _, _ = _interval(
                condition.get("refusal"),
                f"Part 0 {target_id} {language} refusal",
                center_key="estimate", expected_n=roots, bounded_interval=True,
            )
            refusals = _integer_successes(rate, roots, "Part 0 refusal rate")
            unclear = _integer(
                condition.get("unclear_count"), "Part 0 unclear count"
            )
            invalid = _integer(
                condition.get("invalid_count"), "Part 0 invalid count"
            )
            compliance = roots - refusals - unclear - invalid
            if compliance < 0:
                raise PaperHeadlineError("Part 0 outcome counts exceed their denominator.")
            system_refusal += refusals
            totals["refusal"] += refusals
            totals["compliance"] += compliance
            totals["unclear"] += unclear
            totals["invalid"] += invalid
            condition_rates[language].append(rate)
        system_scheduled = 3 * roots
        overall = row.get("overall_refusal")
        if not isinstance(overall, Mapping):
            raise PaperHeadlineError("Part 0 overall interval is absent.")
        overall_rate = _unit_rate(overall.get("estimate"), "Part 0 overall estimate")
        if (
            _integer(overall.get("n_rows"), "Part 0 overall rows", minimum=1)
            != system_scheduled
            or _integer(overall.get("n_clusters"), "Part 0 overall clusters", minimum=1)
            != roots
            or not math.isclose(
                overall_rate, system_refusal / system_scheduled,
                rel_tol=0.0, abs_tol=1e-12,
            )
        ):
            raise PaperHeadlineError("Part 0 overall denominator or estimate is inconsistent.")
        overall_lower = _unit_rate(overall.get("lower"), "Part 0 overall lower")
        overall_upper = _unit_rate(overall.get("upper"), "Part 0 overall upper")
        if overall_lower > overall_rate or overall_rate > overall_upper:
            raise PaperHeadlineError(
                "Part 0 overall interval does not contain its estimate."
            )
        overall_rates.append(overall_rate)
        scheduled += system_scheduled
    if len(root_counts) != 1:
        raise PaperHeadlineError("Part 0 roots per condition differ across systems.")
    if sum(totals.values()) != scheduled:
        raise PaperHeadlineError("Part 0 aggregate outcome denominator is inconsistent.")
    return {
        "included_systems": len(rows),
        "unavailable_systems": len(unavailable),
        "targeted_systems": len(rows) + len(unavailable),
        "roots_per_condition": next(iter(root_counts)),
        "scheduled_responses": scheduled,
        "outcome_counts": totals,
        "system_refusal_percent": _summary(overall_rates, percent=True),
        "condition_system_median_percent": {
            language: _percent(float(statistics.median(condition_rates[language])))
            for language in LANGUAGES
        },
    }


def _part1(source: Mapping[str, Any]) -> dict[str, Any]:
    rows = _objects(source.get("part1"), "Part 1")
    unavailable = _unavailable_ids(source["bindings"]["part1"], "Part 1")
    seen_rows: set[tuple[str, int]] = set()
    included_ids: set[str] = set()
    preferred_counts: dict[str, int] = defaultdict(int)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        target_id = _nonempty_string(row.get("target_id"), f"Part 1 row {index} target")
        model = _nonempty_string(row.get("model"), f"Part 1 {target_id} model")
        roots = _integer(row.get("root_count"), f"Part 1 {target_id} roots", minimum=1)
        if roots not in {12, 96, 384}:
            raise PaperHeadlineError("Part 1 root count is outside the locked scopes.")
        scope = row.get("scope")
        if (roots == 384 and scope != "full_384") or (
            roots in {12, 96} and scope != "balanced_partial"
        ):
            raise PaperHeadlineError("Part 1 scope/root-count binding is inconsistent.")
        key = (target_id, roots)
        if key in seen_rows or target_id in unavailable:
            raise PaperHeadlineError("Part 1 included/unavailable identities overlap or duplicate.")
        seen_rows.add(key)
        included_ids.add(target_id)
        valid = _integer(row.get("format_valid_count"), "Part 1 valid count")
        invalid = _integer(row.get("format_invalid_count"), "Part 1 invalid count")
        if valid + invalid != roots:
            raise PaperHeadlineError("Part 1 format counts do not equal scheduled roots.")
        rate, lower, upper = _interval(
            row.get("cooperation"), f"Part 1 {target_id} cooperation",
            center_key="estimate", expected_n=roots,
        )
        _integer_successes(rate, roots, "Part 1 cooperation rate")
        preferred = row.get("preferred_for_descriptive_outputs")
        if not isinstance(preferred, bool):
            raise PaperHeadlineError("Part 1 preferred-row flag is absent.")
        preferred_counts[target_id] += int(preferred)
        grouped[roots].append({
            "target_id": target_id,
            "model": model,
            "rate": rate,
            "lower": lower,
            "upper": upper,
            "invalid": invalid,
        })
    if set(grouped) != {12, 96, 384} or len(grouped[384]) != 1:
        raise PaperHeadlineError("Part 1 requires n=12, n=96, and one n=384 scope.")
    if any(preferred_counts[target_id] != 1 for target_id in included_ids):
        raise PaperHeadlineError("Part 1 requires exactly one preferred row per reported system.")
    if set(unavailable) & included_ids:
        raise PaperHeadlineError("Part 1 included and unavailable systems overlap.")
    if (set(unavailable) | included_ids) & PART1_PRE_EXECUTION_UNAVAILABLE_TARGET_IDS:
        raise PaperHeadlineError(
            "A pre-execution-unavailable Part 1 registry target appears in the "
            "execution roster."
        )

    def scope_totals(root_count: int) -> dict[str, Any]:
        scope_rows = grouped[root_count]
        return {
            "systems": len(scope_rows),
            "scheduled_roots": root_count * len(scope_rows),
            "invalid_outputs": sum(row["invalid"] for row in scope_rows),
        }

    n96 = {
        **scope_totals(96),
        "system_choice_percent": _summary(
            [row["rate"] for row in grouped[96]], percent=True
        ),
    }
    n12_rows = sorted(
        grouped[12], key=lambda row: (row["model"].casefold(), row["target_id"])
    )
    n12 = {
        **scope_totals(12),
        "values_alphabetical": [
            {
                "system": row["model"],
                "target_id": row["target_id"],
                "choice_percent": _percent(row["rate"]),
            }
            for row in n12_rows
        ],
    }
    full = grouped[384][0]
    n384 = {
        **scope_totals(384),
        "system": full["model"],
        "target_id": full["target_id"],
        "choice_percent": _percent(full["rate"]),
        "ci95_percent": {
            "lower": _percent(full["lower"]),
            "upper": _percent(full["upper"]),
        },
    }
    return {
        "reported_systems": len(included_ids),
        "operational_unavailable_systems": len(unavailable),
        "pre_execution_unavailable_systems": len(
            PART1_PRE_EXECUTION_UNAVAILABLE_TARGET_IDS
        ),
        "unavailable_systems": (
            len(unavailable) + len(PART1_PRE_EXECUTION_UNAVAILABLE_TARGET_IDS)
        ),
        "targeted_systems": (
            len(included_ids)
            + len(unavailable)
            + len(PART1_PRE_EXECUTION_UNAVAILABLE_TARGET_IDS)
        ),
        "scheduled_roots": sum(row["root_count"] for row in rows),
        "scopes": {"n96": n96, "n12": n12, "n384": n384},
    }


def _part2(source: Mapping[str, Any]) -> dict[str, Any]:
    rows = _objects(source.get("part2"), "Part 2")
    unavailable = _unavailable_ids(source["bindings"]["part2"], "Part 2")
    seen: set[str] = set()
    trajectory_counts: set[int] = set()
    metrics: dict[str, list[float]] = defaultdict(list)
    scheduled_agent_days = 0
    invalid_agent_days = 0
    nondepleted = 0
    total_trajectories = 0
    for index, row in enumerate(rows):
        target_id = _nonempty_string(row.get("target_id"), f"Part 2 row {index} target")
        if target_id in seen or target_id in unavailable:
            raise PaperHeadlineError("Part 2 included/unavailable identities overlap or duplicate.")
        seen.add(target_id)
        _nonempty_string(row.get("model"), f"Part 2 {target_id} model")
        count = _integer(
            row.get("trajectory_count"), f"Part 2 {target_id} trajectories", minimum=1
        )
        trajectory_counts.add(count)
        intervals = row.get("trajectory_level_95_percent_t_intervals")
        if not isinstance(intervals, Mapping):
            raise PaperHeadlineError("Part 2 trajectory intervals are absent.")
        for metric in ("aurc", "restraint_rate", "aupc"):
            center, _, _ = _interval(
                intervals.get(metric), f"Part 2 {target_id} {metric}",
                center_key="mean", expected_n=count,
            )
            metrics[metric].append(center)
        nondepletion = intervals.get("reserve_nondepletion")
        mean, _, _ = _interval(
            nondepletion, f"Part 2 {target_id} reserve nondepletion",
            center_key="mean", expected_n=count, bounded_interval=True,
        )
        if not isinstance(nondepletion, Mapping):
            raise PaperHeadlineError("Part 2 nondepletion interval is absent.")
        successes = _integer(
            nondepletion.get("successes"), "Part 2 nondepletion successes"
        )
        if successes > count or not math.isclose(
            mean, successes / count, rel_tol=0.0, abs_tol=1e-12
        ):
            raise PaperHeadlineError("Part 2 nondepletion denominator is inconsistent.")
        scheduled = _integer(
            row.get("total_scheduled_agent_days"),
            "Part 2 scheduled agent-days", minimum=1,
        )
        invalid = _integer(row.get("total_invalid_count"), "Part 2 invalid agent-days")
        if invalid > scheduled:
            raise PaperHeadlineError("Part 2 invalid agent-days exceed scheduled agent-days.")
        scheduled_agent_days += scheduled
        invalid_agent_days += invalid
        nondepleted += successes
        total_trajectories += count
    if len(trajectory_counts) != 1:
        raise PaperHeadlineError("Part 2 trajectories per system are inconsistent.")
    if set(unavailable) & seen:
        raise PaperHeadlineError("Part 2 included and unavailable systems overlap.")
    return {
        "included_systems": len(rows),
        "unavailable_systems": len(unavailable),
        "targeted_systems": len(rows) + len(unavailable),
        "trajectories_per_system": next(iter(trajectory_counts)),
        "total_trajectories": total_trajectories,
        "scheduled_agent_days": scheduled_agent_days,
        "invalid_agent_days": invalid_agent_days,
        "system_aurc": _summary(metrics["aurc"], percent=False),
        "system_restraint_percent": _summary(
            metrics["restraint_rate"], percent=True
        ),
        "system_aupc": _summary(metrics["aupc"], percent=False),
        "nondepleted_trajectories": nondepleted,
    }


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _macro_lines(artifact: Mapping[str, Any]) -> list[str]:
    p0, p1, p2 = artifact["part0"], artifact["part1"], artifact["part2"]
    p0_summary = p0["system_refusal_percent"]
    p1_n96 = p1["scopes"]["n96"]
    p1_n12 = p1["scopes"]["n12"]
    p1_n384 = p1["scopes"]["n384"]
    p2_aurc = p2["system_aurc"]
    p2_restraint = p2["system_restraint_percent"]
    p2_aupc = p2["system_aupc"]
    n12_text = "; ".join(
        f"{_latex_escape(row['system'])}: {row['choice_percent']:.1f}\\%"
        for row in p1_n12["values_alphabetical"]
    )
    values: list[tuple[str, Any]] = [
        ("PaperHeadlinesEvidenceSha", artifact["evidence_sha256"]),
        ("PaperHeadlinesSourceEvidenceSha", artifact["source"]["evidence_sha256"]),
        ("PaperPartZeroIncludedSystems", p0["included_systems"]),
        ("PaperPartZeroUnavailableSystems", p0["unavailable_systems"]),
        ("PaperPartZeroScheduledResponses", p0["scheduled_responses"]),
        ("PaperPartZeroRefusalCount", p0["outcome_counts"]["refusal"]),
        ("PaperPartZeroComplianceCount", p0["outcome_counts"]["compliance"]),
        ("PaperPartZeroUnclearCount", p0["outcome_counts"]["unclear"]),
        ("PaperPartZeroInvalidCount", p0["outcome_counts"]["invalid"]),
        ("PaperPartZeroRefusalMedianPct", f"{p0_summary['median']:.1f}"),
        ("PaperPartZeroRefusalMinimumPct", f"{p0_summary['minimum']:.1f}"),
        ("PaperPartZeroRefusalMaximumPct", f"{p0_summary['maximum']:.1f}"),
        ("PaperPartZeroEnglishMedianPct", f"{p0['condition_system_median_percent']['english']:.1f}"),
        ("PaperPartZeroChineseMedianPct", f"{p0['condition_system_median_percent']['chinese']:.1f}"),
        ("PaperPartZeroRussianMedianPct", f"{p0['condition_system_median_percent']['russian']:.1f}"),
        ("PaperPartOneReportedSystems", p1["reported_systems"]),
        ("PaperPartOneUnavailableSystems", p1["unavailable_systems"]),
        (
            "PaperPartOneOperationalUnavailableSystems",
            p1["operational_unavailable_systems"],
        ),
        (
            "PaperPartOnePreExecutionUnavailableSystems",
            p1["pre_execution_unavailable_systems"],
        ),
        ("PaperPartOneTargetedSystems", p1["targeted_systems"]),
        ("PaperPartOneScheduledRoots", p1["scheduled_roots"]),
        ("PaperPartOneNNinetySixSystems", p1_n96["systems"]),
        ("PaperPartOneNNinetySixScheduledRoots", p1_n96["scheduled_roots"]),
        ("PaperPartOneNNinetySixInvalid", p1_n96["invalid_outputs"]),
        ("PaperPartOneNNinetySixMedianPct", f"{p1_n96['system_choice_percent']['median']:.1f}"),
        ("PaperPartOneNNinetySixMinimumPct", f"{p1_n96['system_choice_percent']['minimum']:.1f}"),
        ("PaperPartOneNNinetySixMaximumPct", f"{p1_n96['system_choice_percent']['maximum']:.1f}"),
        ("PaperPartOneNTwelveSystems", p1_n12["systems"]),
        ("PaperPartOneNTwelveScheduledRoots", p1_n12["scheduled_roots"]),
        ("PaperPartOneNTwelveInvalid", p1_n12["invalid_outputs"]),
        ("PaperPartOneNTwelveValues", n12_text),
        ("PaperPartOneNThreeEightyFourSystems", p1_n384["systems"]),
        ("PaperPartOneNThreeEightyFourScheduledRoots", p1_n384["scheduled_roots"]),
        ("PaperPartOneNThreeEightyFourSystem", _latex_escape(p1_n384["system"])),
        ("PaperPartOneNThreeEightyFourEstimatePct", f"{p1_n384['choice_percent']:.1f}"),
        ("PaperPartOneNThreeEightyFourLowerPct", f"{p1_n384['ci95_percent']['lower']:.1f}"),
        ("PaperPartOneNThreeEightyFourUpperPct", f"{p1_n384['ci95_percent']['upper']:.1f}"),
        ("PaperPartOneNThreeEightyFourInvalid", p1_n384["invalid_outputs"]),
        ("PaperPartTwoIncludedSystems", p2["included_systems"]),
        ("PaperPartTwoUnavailableSystems", p2["unavailable_systems"]),
        ("PaperPartTwoTrajectoriesPerSystem", p2["trajectories_per_system"]),
        ("PaperPartTwoTrajectories", p2["total_trajectories"]),
        ("PaperPartTwoScheduledAgentDays", p2["scheduled_agent_days"]),
        ("PaperPartTwoInvalidAgentDays", p2["invalid_agent_days"]),
        ("PaperPartTwoAURCMedian", f"{p2_aurc['median']:.3f}"),
        ("PaperPartTwoAURCMinimum", f"{p2_aurc['minimum']:.3f}"),
        ("PaperPartTwoAURCMaximum", f"{p2_aurc['maximum']:.3f}"),
        ("PaperPartTwoRestraintMedianPct", f"{p2_restraint['median']:.1f}"),
        ("PaperPartTwoRestraintMinimumPct", f"{p2_restraint['minimum']:.1f}"),
        ("PaperPartTwoRestraintMaximumPct", f"{p2_restraint['maximum']:.1f}"),
        ("PaperPartTwoAUPCMedian", f"{p2_aupc['median']:.3f}"),
        ("PaperPartTwoAUPCMinimum", f"{p2_aupc['minimum']:.3f}"),
        ("PaperPartTwoAUPCMaximum", f"{p2_aupc['maximum']:.3f}"),
        ("PaperPartTwoNondepleted", p2["nondepleted_trajectories"]),
    ]
    return [
        "% Generated by analysis.build_paper_headlines; do not edit.",
        "% Percent macros omit the percent sign so prose controls typography.",
        *(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values),
    ]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def build_paper_headlines(
    input_path: Path, output_json: Path, output_tex: Path
) -> dict[str, Any]:
    """Validate one sanitized graph and emit exact within-axis headline slots."""
    input_path = input_path.resolve()
    output_json = output_json.resolve()
    output_tex = output_tex.resolve()
    if output_json == output_tex:
        raise PaperHeadlineError("JSON and LaTeX outputs must be different paths.")
    if output_json.exists() or output_tex.exists():
        raise PaperHeadlineError("Headline outputs already exist; refusing to overwrite.")
    source, source_file_sha256 = _read_source(input_path)
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": OUTPUT_ARTIFACT_TYPE,
        "source": {
            "path": input_path.name,
            "path_scope": "input_basename_only",
            "file_sha256": source_file_sha256,
            "evidence_sha256": source["evidence_sha256"],
        },
        "reporting_contract": {
            "within_axis_only": True,
            "cross_axis_associations_computed": False,
            "rankings_computed": False,
            "family_effects_computed": False,
            "significance_tests_computed": False,
            "part1_scopes_pooled": False,
            "part0_conditions": (
                "response-language instructions over English source requests; "
                "not translated prompts"
            ),
        },
        "part0": _part0(source),
        "part1": _part1(source),
        "part2": _part2(source),
    }
    _reject_forbidden_fields(artifact)
    _seal(artifact)
    _atomic_text(
        output_json,
        json.dumps(
            artifact, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        ) + "\n",
    )
    _atomic_text(output_tex, "\n".join(_macro_lines(artifact)) + "\n")
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build fail-closed within-axis manuscript headline slots."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/analysis/final_results/final_results.json"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        artifact = build_paper_headlines(
            args.input, args.output_json, args.output_tex
        )
    except PaperHeadlineError as error:
        print(f"Paper-headline build failed: {error}", file=os.sys.stderr)
        return 1
    print(f"Paper headline JSON: {args.output_json}")
    print(f"Paper headline LaTeX: {args.output_tex}")
    print(f"Evidence SHA256: {artifact['evidence_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
