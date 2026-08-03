"""Export fail-closed, within-axis developer-route descriptives.

This module consumes only the sealed, sanitized ``final_results.json`` graph.
``upstream_provider`` is used solely as an operational developer-route grouping
label.  The output is descriptive: it contains no rankings, model-family
effects, significance tests, cross-axis joins, or causal/vendor claims.
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
OUTPUT_ARTIFACT_TYPE = "prosocial_readiness_developer_route_descriptives"
ALLOWED_PART1_ROOT_COUNTS = (12, 96, 384)
FORBIDDEN_KEYS = {
    "assistant_response", "completion", "content", "journal", "journals",
    "messages", "private", "private_path", "prompt", "prompt_text", "raw",
    "raw_response", "reasoning", "request", "request_body", "requested_route",
    "response", "response_text", "route", "system_prompt", "text",
    "user_prompt", "visible_content", "visible_response",
}


class DeveloperDescriptiveError(RuntimeError):
    """The sanitized source cannot support developer-route descriptives."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    return _sha256_json({key: item for key, item in value.items() if key != "evidence_sha256"})


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    value.pop("evidence_sha256", None)
    value["evidence_sha256"] = _self_hash(value)
    return value


def _reject_forbidden_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DeveloperDescriptiveError(f"Non-string key at {path}.")
            normalized = key.casefold()
            if normalized in FORBIDDEN_KEYS or normalized.startswith("private_"):
                raise DeveloperDescriptiveError(f"Forbidden private/text/route field at {path}.{key}.")
            _reject_forbidden_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_fields(item, f"{path}[{index}]")


def _string(value: Any, label: str) -> str:
    if (
        not isinstance(value, str) or not value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise DeveloperDescriptiveError(f"{label} must be a nonempty string.")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise DeveloperDescriptiveError(f"{label} must be an integer >= {minimum}.")
    return value


def _finite(value: Any, label: str, *, unit: bool = False) -> float:
    if (
        not isinstance(value, (int, float)) or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise DeveloperDescriptiveError(f"{label} must be finite.")
    result = float(value)
    if unit and not 0.0 <= result <= 1.0:
        raise DeveloperDescriptiveError(f"{label} must be in [0, 1].")
    return result


def _rows(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value or any(not isinstance(row, Mapping) for row in value):
        raise DeveloperDescriptiveError(f"{label} rows are absent or malformed.")
    return list(value)


def _interval(
    value: Any, label: str, *, center_key: str, expected_n: int | None,
    unit_center: bool,
) -> float:
    if not isinstance(value, Mapping):
        raise DeveloperDescriptiveError(f"{label} interval is absent.")
    center = _finite(value.get(center_key), f"{label}.{center_key}", unit=unit_center)
    lower = _finite(value.get("lower"), f"{label}.lower")
    upper = _finite(value.get("upper"), f"{label}.upper")
    if lower > center or center > upper:
        raise DeveloperDescriptiveError(f"{label} interval does not contain its center.")
    if expected_n is not None and _integer(value.get("n"), f"{label}.n", minimum=1) != expected_n:
        raise DeveloperDescriptiveError(f"{label} denominator is inconsistent.")
    return center


def _read_source(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise DeveloperDescriptiveError("Final results must be a JSON object.")
        if (
            value.get("schema_version") != SCHEMA_VERSION
            or value.get("artifact_type") != SOURCE_ARTIFACT_TYPE
            or value.get("evidence_sha256") != _self_hash(value)
        ):
            raise DeveloperDescriptiveError("Final-results schema/type/self-hash validation failed.")
    except DeveloperDescriptiveError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise DeveloperDescriptiveError("Final results are not valid finite UTF-8 JSON.") from error
    _reject_forbidden_fields(value)
    privacy = value.get("privacy_contract")
    expected = {
        "contains_prompt_text": False, "contains_response_text": False,
        "contains_reasoning": False, "contains_raw_responses": False,
        "contains_routes": False,
    }
    if not isinstance(privacy, Mapping) or any(privacy.get(key) is not state for key, state in expected.items()):
        raise DeveloperDescriptiveError("Final-results privacy contract is absent or unsafe.")
    parameters = value.get("parameters")
    if (
        not isinstance(parameters, Mapping)
        or parameters.get("part1_scopes_pooled") is not False
        or parameters.get("part1_full_root_count") != 384
        or parameters.get("part2_uncertainty_unit") != "independent_trajectory"
    ):
        raise DeveloperDescriptiveError("Final-results reporting parameters are inconsistent.")
    bindings = value.get("bindings")
    if not isinstance(bindings, Mapping) or any(part not in bindings for part in ("part0", "part1", "part2")):
        raise DeveloperDescriptiveError("Required source bindings are absent.")
    return value, hashlib.sha256(raw).hexdigest()


def _identity(row: Mapping[str, Any], label: str) -> tuple[str, str, str]:
    return (
        _string(row.get("target_id"), f"{label} target_id"),
        _string(row.get("upstream_provider"), f"{label} upstream_provider"),
        _string(row.get("model"), f"{label} model"),
    )


def _summary(values: Sequence[float]) -> dict[str, float]:
    if len(values) < 2:
        raise DeveloperDescriptiveError("A developer-route group requires at least two systems.")
    return {
        "median": round(float(statistics.median(values)), 6),
        "minimum": round(min(values), 6),
        "maximum": round(max(values), 6),
    }


def _group_rows(
    values: Mapping[str, Sequence[tuple[str, float]]], metric: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for provider in sorted(values, key=lambda item: (item.casefold(), item)):
        rows = values[provider]
        target_ids = [target_id for target_id, _ in rows]
        if len(rows) < 2:
            continue
        if len(target_ids) != len(set(target_ids)):
            raise DeveloperDescriptiveError("A developer-route group contains duplicate systems.")
        result.append({
            "developer_route_group": provider,
            "system_count": len(rows),
            metric: _summary([value for _, value in rows]),
        })
    return result


def _part0(source: Mapping[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    seen: set[str] = set()
    for index, row in enumerate(_rows(source.get("part0"), "Part 0")):
        target_id, provider, _ = _identity(row, f"Part 0 row {index}")
        if target_id in seen:
            raise DeveloperDescriptiveError("Part 0 target_id is duplicated.")
        seen.add(target_id)
        roots = _integer(row.get("root_count_per_condition"), "Part 0 roots", minimum=1)
        conditions = row.get("conditions")
        if not isinstance(conditions, list) or len(conditions) != 3:
            raise DeveloperDescriptiveError("Part 0 requires exactly three conditions.")
        refusal_count = 0
        languages: set[str] = set()
        for condition in conditions:
            if not isinstance(condition, Mapping):
                raise DeveloperDescriptiveError("Part 0 condition is malformed.")
            language = _string(condition.get("response_language_condition"), "Part 0 language")
            if language in languages:
                raise DeveloperDescriptiveError("Part 0 condition is duplicated.")
            languages.add(language)
            rate = _interval(
                condition.get("refusal"), "Part 0 refusal", center_key="estimate",
                expected_n=roots, unit_center=True,
            )
            count = round(rate * roots)
            if not math.isclose(rate, count / roots, rel_tol=0.0, abs_tol=1e-12):
                raise DeveloperDescriptiveError("Part 0 refusal rate is not count-consistent.")
            unclear = _integer(condition.get("unclear_count"), "Part 0 unclear")
            invalid = _integer(condition.get("invalid_count"), "Part 0 invalid")
            if count + unclear + invalid > roots:
                raise DeveloperDescriptiveError("Part 0 outcome counts exceed roots.")
            refusal_count += count
        if languages != {"english", "chinese", "russian"}:
            raise DeveloperDescriptiveError("Part 0 condition labels changed.")
        overall = row.get("overall_refusal")
        if not isinstance(overall, Mapping):
            raise DeveloperDescriptiveError("Part 0 overall refusal is absent.")
        rate = _finite(overall.get("estimate"), "Part 0 overall estimate", unit=True)
        if (
            _integer(overall.get("n_rows"), "Part 0 overall rows", minimum=1) != 3 * roots
            or _integer(overall.get("n_clusters"), "Part 0 overall clusters", minimum=1) != roots
            or not math.isclose(rate, refusal_count / (3 * roots), rel_tol=0.0, abs_tol=1e-12)
        ):
            raise DeveloperDescriptiveError("Part 0 overall refusal is inconsistent.")
        lower = _finite(overall.get("lower"), "Part 0 overall lower")
        upper = _finite(overall.get("upper"), "Part 0 overall upper")
        if lower > rate or rate > upper:
            raise DeveloperDescriptiveError("Part 0 overall interval is inconsistent.")
        grouped[provider].append((target_id, rate))
    return {
        "metric_scale": "proportion",
        "groups_alphabetical": _group_rows(grouped, "overall_refusal"),
    }


def _part1(source: Mapping[str, Any]) -> dict[str, Any]:
    grouped: dict[int, dict[str, list[tuple[str, float]]]] = {
        count: defaultdict(list) for count in ALLOWED_PART1_ROOT_COUNTS
    }
    seen: set[tuple[str, int]] = set()
    identities: dict[str, tuple[str, str]] = {}
    for index, row in enumerate(_rows(source.get("part1"), "Part 1")):
        target_id, provider, model = _identity(row, f"Part 1 row {index}")
        roots = _integer(row.get("root_count"), "Part 1 root_count", minimum=1)
        if roots not in ALLOWED_PART1_ROOT_COUNTS:
            raise DeveloperDescriptiveError("Part 1 root_count is outside 12/96/384.")
        expected_scope = "full_384" if roots == 384 else "balanced_partial"
        if row.get("scope") != expected_scope:
            raise DeveloperDescriptiveError("Part 1 scope/root_count is inconsistent.")
        key = (target_id, roots)
        if key in seen:
            raise DeveloperDescriptiveError("Part 1 target/scope row is duplicated.")
        seen.add(key)
        if target_id in identities and identities[target_id] != (provider, model):
            raise DeveloperDescriptiveError("Part 1 system identity changes across scopes.")
        identities[target_id] = (provider, model)
        valid = _integer(row.get("format_valid_count"), "Part 1 valid count")
        invalid = _integer(row.get("format_invalid_count"), "Part 1 invalid count")
        if valid + invalid != roots:
            raise DeveloperDescriptiveError("Part 1 format counts do not equal root_count.")
        rate = _interval(
            row.get("cooperation"), "Part 1 self-choice", center_key="estimate",
            expected_n=roots, unit_center=True,
        )
        successes = round(rate * roots)
        if not math.isclose(rate, successes / roots, rel_tol=0.0, abs_tol=1e-12):
            raise DeveloperDescriptiveError("Part 1 self-choice is not count-consistent.")
        grouped[roots][provider].append((target_id, rate))
    return {
        "metric_scale": "proportion",
        "scopes_alphabetical": [
            {
                "root_count": roots,
                "groups_alphabetical": _group_rows(grouped[roots], "self_choice"),
            }
            for roots in ALLOWED_PART1_ROOT_COUNTS
        ],
        "scopes_pooled": False,
    }


def _part2(source: Mapping[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[tuple[str, dict[str, float]]]] = defaultdict(list)
    seen: set[str] = set()
    for index, row in enumerate(_rows(source.get("part2"), "Part 2")):
        target_id, provider, _ = _identity(row, f"Part 2 row {index}")
        if target_id in seen:
            raise DeveloperDescriptiveError("Part 2 target_id is duplicated.")
        seen.add(target_id)
        count = _integer(row.get("trajectory_count"), "Part 2 trajectories", minimum=1)
        intervals = row.get("trajectory_level_95_percent_t_intervals")
        if not isinstance(intervals, Mapping):
            raise DeveloperDescriptiveError("Part 2 intervals are absent.")
        metrics = {
            metric: _interval(
                intervals.get(metric), f"Part 2 {metric}", center_key="mean",
                expected_n=count, unit_center=True,
            )
            for metric in ("aurc", "restraint_rate", "aupc")
        }
        grouped[provider].append((target_id, metrics))
    output: list[dict[str, Any]] = []
    for provider in sorted(grouped, key=lambda item: (item.casefold(), item)):
        rows = grouped[provider]
        if len(rows) < 2:
            continue
        ids = [target_id for target_id, _ in rows]
        if len(ids) != len(set(ids)):
            raise DeveloperDescriptiveError("A Part 2 developer-route group duplicates systems.")
        output.append({
            "developer_route_group": provider,
            "system_count": len(rows),
            **{
                metric: _summary([values[metric] for _, values in rows])
                for metric in ("aurc", "restraint_rate", "aupc")
            },
        })
    return {"metric_scale": "proportion", "groups_alphabetical": output}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
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


def build_developer_descriptives(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Validate sanitized results and emit alphabetical developer-route groups."""
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise DeveloperDescriptiveError("Output already exists; refusing to overwrite.")
    if input_path == output_path:
        raise DeveloperDescriptiveError("Input and output paths must differ.")
    source, file_sha256 = _read_source(input_path)
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": OUTPUT_ARTIFACT_TYPE,
        "source": {
            "path": input_path.name,
            "path_scope": "input_basename_only",
            "file_sha256": file_sha256,
            "evidence_sha256": source["evidence_sha256"],
        },
        "reporting_contract": {
            "grouping_label": "developer_route_group",
            "minimum_systems_per_emitted_group": 2,
            "groups_order": "alphabetical_by_exact_upstream_provider",
            "within_axis_only": True,
            "part1_scopes_pooled": False,
            "cross_axis_join_performed": False,
            "rankings_computed": False,
            "outcome_sorting_performed": False,
            "significance_tests_computed": False,
            "model_family_effects_computed": False,
            "causal_or_vendor_claims_supported": False,
        },
        "part0": _part0(source),
        "part1": _part1(source),
        "part2": _part2(source),
    }
    _reject_forbidden_fields(artifact)
    _seal(artifact)
    _atomic_json(output_path, artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build fail-closed developer-route descriptives.")
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/analysis/final_results/final_results.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        build_developer_descriptives(args.input, args.output)
    except DeveloperDescriptiveError as error:
        print(f"Developer-route descriptive build failed: {error}", file=os.sys.stderr)
        return 1
    print(f"Developer-route descriptives: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
