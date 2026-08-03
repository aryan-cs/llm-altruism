"""Fail-closed analysis adapter for the five definitive provider-safe-v2 campaigns.

This module intentionally emits no paper claim.  It validates completed private
manifests and their hash-bound evidence, then writes text-free, machine-readable
descriptives.  In particular, first-attempt invalid outcomes remain in every
primary scheduled-unit denominator; explicit repairs are only audit counts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiments.part1.confirmatory_design import COUNTERBALANCE_BY_ID


SCHEMA_VERSION = 1
EXPECTED_TYPES = {
    "part0": "inference_hub_part0_accelerated_private_panel",
    "part1": "inference_hub_part1_large_n_exploratory_panel",
    "part2": "inference_hub_part2_corrected_matched_panel",
    "role": "inference_hub_part1_role_calibration_private_v1",
    "sensitivity": "inference_hub_part2_sensitivity_campaign_v1",
}


class DefinitiveAnalysisError(RuntimeError):
    """An input cannot support the definitive descriptive adapter."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    return _sha256_json({key: item for key, item in value.items() if key != "evidence_sha256"})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DefinitiveAnalysisError(f"{label} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise DefinitiveAnalysisError(f"{label} must be a JSON object: {path}")
    return value


def _manifest_path(value: Path) -> Path:
    return value / "private" / "manifest.json" if value.is_dir() else value


def _private_mode(path: Path) -> None:
    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise DefinitiveAnalysisError(f"Private evidence must have mode 0600: {path}")


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


_CONSERVATIVE_POLICY = {
    "global_concurrency": 16,
    "provider_concurrency": 1,
    "global_requests_per_second": 8.0,
    "provider_requests_per_second": 1.0,
}
_MAIN_ACCELERATED_POLICY = {
    "global_concurrency": 12,
    "provider_concurrency": 2,
    "global_requests_per_second": 8.0,
    "provider_requests_per_second": 1.5,
}
_PART1_DEADLINE_POLICY = {
    "global_concurrency": 24,
    "provider_concurrency": 4,
    "global_requests_per_second": 12.0,
    "provider_requests_per_second": 2.5,
}
_PART0_DEADLINE_POLICY = {
    "global_concurrency": 16,
    "provider_concurrency": 3,
    "global_requests_per_second": 10.0,
    "provider_requests_per_second": 2.0,
}
_SENSITIVITY_DEADLINE_POLICY = {
    "global_concurrency": 24,
    "provider_concurrency": 3,
    "global_requests_per_second": 12.0,
    "provider_requests_per_second": 2.5,
}
_EXPLORATORY_ACCELERATED_POLICY = {
    "global_concurrency": 12,
    "provider_concurrency": 3,
    "global_requests_per_second": 8.0,
    "provider_requests_per_second": 2.0,
}


def _source_digest(sources: Mapping[str, Any], basename: str) -> str | None:
    matches = [digest for name, digest in sources.items() if Path(str(name)).name == basename]
    if len(matches) != 1 or not isinstance(matches[0], str):
        return None
    return matches[0]


def _provider_safe_contract(manifest: Mapping[str, Any], phase: str) -> None:
    sources = manifest.get("source_artifacts")
    if not isinstance(sources, Mapping):
        raise DefinitiveAnalysisError("Manifest lacks source_artifacts.")
    safe_digest = _source_digest(sources, "inference_hub_provider_safe_v2.py")
    if safe_digest is None or len(safe_digest) != 64:
        raise DefinitiveAnalysisError("Manifest is not hash-bound to provider-safe-v2.")
    try:
        int(safe_digest, 16)
    except ValueError as error:
        raise DefinitiveAnalysisError("provider-safe-v2 source digest is malformed.") from error
    contract = manifest.get("execution_contract")
    shared = contract.get("shared_rate_limit") if isinstance(contract, Mapping) else None
    if not isinstance(shared, Mapping):
        raise DefinitiveAnalysisError("Manifest lacks a shared rate-limit contract.")
    recorded_policy_sha = shared.get("policy_sha256")
    if recorded_policy_sha != _sha256_json(
        {key: value for key, value in shared.items() if key != "policy_sha256"}
    ):
        raise DefinitiveAnalysisError("Shared rate-limit policy hash failed.")
    main_digest = _source_digest(sources, "inference_hub_main_accelerated.py")
    part1_deadline_digest = _source_digest(
        sources, "inference_hub_part1_deadline_accelerated.py"
    )
    part0_deadline_digest = _source_digest(
        sources, "inference_hub_part0_deadline_retry.py"
    )
    sensitivity_deadline_digest = _source_digest(
        sources, "inference_hub_sensitivity_deadline_accelerated.py"
    )
    exploratory_digest = _source_digest(
        sources, "inference_hub_exploratory_accelerated.py"
    )
    repository = Path(__file__).resolve().parents[1]
    if (
        phase == "sensitivity"
        and sensitivity_deadline_digest is not None
        and part0_deadline_digest is None
        and part1_deadline_digest is None
        and main_digest is None
        and exploratory_digest is None
    ):
        launcher = (
            repository
            / "experiments/misc/inference_hub_sensitivity_deadline_accelerated.py"
        )
        if sensitivity_deadline_digest != _sha256_file(launcher):
            raise DefinitiveAnalysisError(
                "Sensitivity deadline launcher source binding failed."
            )
        expected_policy = _SENSITIVITY_DEADLINE_POLICY
    elif (
        phase == "part0"
        and part0_deadline_digest is not None
        and part1_deadline_digest is None
        and sensitivity_deadline_digest is None
        and main_digest is None
        and exploratory_digest is None
    ):
        launcher = repository / "experiments/misc/inference_hub_part0_deadline_retry.py"
        if part0_deadline_digest != _sha256_file(launcher):
            raise DefinitiveAnalysisError(
                "Part 0 deadline launcher source binding failed."
            )
        expected_policy = _PART0_DEADLINE_POLICY
    elif (
        phase == "part1"
        and part1_deadline_digest is not None
        and part0_deadline_digest is None
        and sensitivity_deadline_digest is None
        and main_digest is None
        and exploratory_digest is None
    ):
        launcher = (
            repository
            / "experiments/misc/inference_hub_part1_deadline_accelerated.py"
        )
        if part1_deadline_digest != _sha256_file(launcher):
            raise DefinitiveAnalysisError(
                "Part 1 deadline launcher source binding failed."
            )
        expected_policy = _PART1_DEADLINE_POLICY
    elif phase in {"part0", "part1", "part2"} and main_digest is not None:
        launcher = repository / "experiments/misc/inference_hub_main_accelerated.py"
        if (
            main_digest != _sha256_file(launcher)
            or exploratory_digest is not None
            or part1_deadline_digest is not None
            or part0_deadline_digest is not None
            or sensitivity_deadline_digest is not None
        ):
            raise DefinitiveAnalysisError("Main accelerated launcher source binding failed.")
        expected_policy = _MAIN_ACCELERATED_POLICY
    elif phase in {"role", "sensitivity"} and exploratory_digest is not None:
        launcher = repository / "experiments/misc/inference_hub_exploratory_accelerated.py"
        if (
            exploratory_digest != _sha256_file(launcher)
            or main_digest is not None
            or part0_deadline_digest is not None
            or part1_deadline_digest is not None
            or sensitivity_deadline_digest is not None
        ):
            raise DefinitiveAnalysisError("Exploratory accelerated launcher source binding failed.")
        expected_policy = _EXPLORATORY_ACCELERATED_POLICY
    elif (
        main_digest is None
        and exploratory_digest is None
        and part1_deadline_digest is None
        and part0_deadline_digest is None
        and sensitivity_deadline_digest is None
    ):
        expected_policy = _CONSERVATIVE_POLICY
    else:
        raise DefinitiveAnalysisError(f"Wrong accelerated launcher bound for {phase}.")
    if any(shared.get(key) != value for key, value in expected_policy.items()):
        raise DefinitiveAnalysisError(f"Wrong shared rate-limit policy for {phase}.")
    required = contract.get("provider_concurrency_required") if isinstance(contract, Mapping) else None
    if required is not None and required != expected_policy["provider_concurrency"]:
        raise DefinitiveAnalysisError("Required provider concurrency disagrees with policy.")


def _load_manifest(value: Path, phase: str) -> tuple[Path, Path, dict[str, Any]]:
    path = _manifest_path(value).resolve()
    _private_mode(path)
    manifest = _read_object(path, f"{phase} manifest")
    if manifest.get("schema_version") != 1 or manifest.get("artifact_type") != EXPECTED_TYPES[phase]:
        raise DefinitiveAnalysisError(f"Wrong {phase} manifest type or schema.")
    if manifest.get("evidence_sha256") != _self_hash(manifest):
        raise DefinitiveAnalysisError(f"{phase} manifest self-hash failed.")
    if manifest.get("complete") is not True or not manifest.get("completed_at_utc"):
        raise DefinitiveAnalysisError(f"{phase} manifest is not COMPLETE.")
    _provider_safe_contract(manifest, phase)
    run = path.parent.parent
    return run, path, manifest


def _read_journal(reference: object, private_root: Path, label: str) -> list[dict[str, Any]]:
    if not isinstance(reference, Mapping):
        raise DefinitiveAnalysisError(f"Missing journal reference: {label}.")
    path_value = reference.get("path")
    if not isinstance(path_value, str):
        raise DefinitiveAnalysisError(f"Journal path is invalid: {label}.")
    path = Path(path_value).resolve()
    if not _within(path, private_root):
        raise DefinitiveAnalysisError(f"Journal escaped private run directory: {label}.")
    if not path.exists():
        if reference.get("record_count") == 0 and reference.get("file_sha256") is None:
            return []
        raise DefinitiveAnalysisError(f"Journal is missing: {label}.")
    _private_mode(path)
    try:
        text = path.read_text(encoding="utf-8")
        if text and not text.endswith("\n"):
            raise ValueError("missing final delimiter")
        rows = [] if not text else [json.loads(line) for line in text[:-1].split("\n")]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise DefinitiveAnalysisError(f"Journal is not valid JSONL: {label}.") from error
    previous = None
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise DefinitiveAnalysisError(f"Journal record is not an object: {label}/{index}.")
        recorded = row.get("record_sha256")
        unhashed = {key: item for key, item in row.items() if key != "record_sha256"}
        if row.get("previous_record_sha256") != previous or recorded != _sha256_json(unhashed):
            raise DefinitiveAnalysisError(f"Journal hash chain failed: {label}/{index}.")
        previous = recorded
    if (
        reference.get("record_count") != len(rows)
        or reference.get("tail_record_sha256") != previous
        or reference.get("file_sha256") != _sha256_file(path)
    ):
        raise DefinitiveAnalysisError(f"Journal checkpoint failed: {label}.")
    return rows


def _validate_standard_journals(run: Path, manifest: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    refs = manifest.get("journals")
    if not isinstance(refs, Mapping) or not isinstance(refs.get("raw_responses"), Mapping):
        raise DefinitiveAnalysisError("Manifest lacks raw response journal references.")
    private = run / "private"
    _read_journal(refs.get("attempt_ledger"), private, "attempt ledger")
    return {
        str(target): _read_journal(ref, private, f"raw/{target}")
        for target, ref in refs["raw_responses"].items()
    }


def _validate_flat_journals(run: Path, manifest: Mapping[str, Any], *, sensitivity: bool) -> None:
    refs = manifest.get("journals")
    if not isinstance(refs, Mapping) or not refs:
        raise DefinitiveAnalysisError("Manifest lacks trajectory journal references.")
    private = run / "private"
    for key, ref in refs.items():
        _read_journal(ref, private, f"trajectory/{key}")
    if sensitivity:
        _read_journal(manifest.get("attempt_ledger"), private, "sensitivity attempt ledger")


def _load_sanitized(run: Path, manifest: Mapping[str, Any], key: str, expected_type: str) -> dict[str, Any]:
    refs = manifest.get("sanitized_artifacts")
    ref = refs.get(key) if isinstance(refs, Mapping) else None
    if not isinstance(ref, Mapping) or not isinstance(ref.get("path"), str):
        raise DefinitiveAnalysisError(f"Missing sanitized artifact: {key}.")
    path = Path(ref["path"]).resolve()
    if not _within(path, run / "sanitized"):
        raise DefinitiveAnalysisError(f"Sanitized artifact escaped run directory: {key}.")
    payload = _read_object(path, key)
    if (
        payload.get("artifact_type") != expected_type
        or payload.get("evidence_sha256") != _self_hash(payload)
        or ref.get("evidence_sha256") != payload.get("evidence_sha256")
        or ref.get("file_sha256") != _sha256_file(path)
    ):
        raise DefinitiveAnalysisError(f"Sanitized artifact integrity failed: {key}.")
    if not isinstance(payload.get("rows"), list):
        raise DefinitiveAnalysisError(f"Sanitized artifact lacks rows: {key}.")
    return payload


def _ident(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DefinitiveAnalysisError("Route identity component is empty.")
    return value.strip().casefold()


def _judge_audit(phase: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    judge = manifest.get("judge") if phase == "part0" else manifest.get("judge_reservation")
    subjects = manifest.get("selected_subject_routes") if phase == "sensitivity" else manifest.get("subject_routes")
    if not isinstance(judge, Mapping) or not isinstance(subjects, list) or not subjects:
        raise DefinitiveAnalysisError(f"{phase} lacks judge/subject route identities.")
    if phase != "part0" and judge.get("dispatch_permitted_in_this_runner") is not False:
        raise DefinitiveAnalysisError(f"{phase} did not prohibit judge dispatch.")
    judge_id = _ident(judge.get("target_id"))
    judge_route = _ident(judge.get("route"))
    judge_upstream = (_ident(judge.get("upstream_provider")), _ident(judge.get("model")))
    for subject in subjects:
        if not isinstance(subject, Mapping):
            raise DefinitiveAnalysisError(f"{phase} subject route is malformed.")
        if judge_id == _ident(subject.get("target_id")):
            raise DefinitiveAnalysisError(f"{phase} judge overlaps a subject target.")
        if judge_route == _ident(subject.get("route")):
            raise DefinitiveAnalysisError(f"{phase} judge overlaps a subject route.")
        if judge_upstream == (_ident(subject.get("upstream_provider")), _ident(subject.get("model"))):
            raise DefinitiveAnalysisError(f"{phase} judge overlaps a subject upstream identity.")
    if phase == "part1" and manifest.get("judge_dispatched") not in (False, None):
        raise DefinitiveAnalysisError("Part 1 dispatched its reserved judge.")
    return {
        "phase": phase, "judge_target_id": judge["target_id"],
        "subject_count": len(subjects), "target_disjoint": True,
        "route_disjoint": True, "upstream_identity_disjoint": True,
    }


def _subject_index(manifest: Mapping[str, Any], phase: str = "") -> dict[str, Mapping[str, Any]]:
    key = "selected_subject_routes" if phase == "sensitivity" else "subject_routes"
    rows = manifest.get(key)
    if not isinstance(rows, list):
        raise DefinitiveAnalysisError("Manifest lacks subject routes.")
    result = {str(row["target_id"]): row for row in rows if isinstance(row, Mapping) and row.get("target_id")}
    if len(result) != len(rows):
        raise DefinitiveAnalysisError("Subject target identifiers are invalid or duplicated.")
    return result


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _repair_marker(row: Mapping[str, Any]) -> bool:
    return row.get("repaired_from_invalid") is True or bool(row.get("repair_of_attempt_id")) or row.get("semantic_repair") is True


def _part0(run: Path, manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    journals = _validate_standard_journals(run, manifest)
    subjects = _subject_index(manifest)
    models, figure = [], []
    for target, subject in sorted(subjects.items()):
        terminals = [row for row in journals.get(target, []) if row.get("event") == "unit_completed"]
        retired = [row for row in journals.get(target, []) if row.get("event") == "unit_operationally_retired"]
        if retired:
            raise DefinitiveAnalysisError("COMPLETE Part 0 contains operationally retired units.")
        if len({str(row.get("trial_id")) for row in terminals}) != len(terminals):
            raise DefinitiveAnalysisError(f"Part 0 terminal units are duplicated for {target}.")
        planned_total = manifest.get("summary", {}).get("planned_units")
        planned = (
            int(planned_total) // len(subjects)
            if isinstance(planned_total, int) and not isinstance(planned_total, bool)
            else len(terminals)
        )
        if isinstance(planned_total, int) and planned * len(subjects) != planned_total:
            raise DefinitiveAnalysisError("Part 0 planned-unit total is not model-balanced.")
        if len(terminals) != planned:
            raise DefinitiveAnalysisError(f"Part 0 terminal accounting is incomplete for {target}.")
        counts = {label: sum(row.get("outcome") == label for row in terminals) for label in ("REFUSAL", "COMPLIANCE", "UNCLEAR", "INVALID")}
        if sum(counts.values()) != planned:
            raise DefinitiveAnalysisError(f"Part 0 contains an unknown outcome for {target}.")
        repaired = sum(_repair_marker(row) for row in terminals)
        models.append({
            "phase": "part0", "target_id": target, "upstream_provider": subject["upstream_provider"], "model": subject["model"],
            "scheduled_units": planned, "refusal_count": counts["REFUSAL"], "compliance_count": counts["COMPLIANCE"],
            "unclear_count": counts["UNCLEAR"], "first_attempt_invalid_count": counts["INVALID"] + repaired,
            "repaired_invalid_count": repaired, "refusal_rate_all_scheduled": _rate(counts["REFUSAL"], planned),
            "primary_denominator": "all_scheduled_units", "exploratory_only": True,
        })
        for language in sorted({str(row.get("language")) for row in terminals}):
            group = [row for row in terminals if str(row.get("language")) == language]
            lc = {label: sum(row.get("outcome") == label for row in group) for label in counts}
            figure.append({"target_id": target, "model": subject["model"], "language": language, "scheduled_units": len(group), **{f"{k.lower()}_count": v for k, v in lc.items()}, "refusal_rate_all_scheduled": _rate(lc["REFUSAL"], len(group))})
    return models, figure


def _part1(run: Path, manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    journals = _validate_standard_journals(run, manifest)
    subjects = _subject_index(manifest)
    models, figure = [], []
    expected = int(manifest.get("trial_limit") or 384)
    for target, subject in sorted(subjects.items()):
        rows = journals.get(target, [])
        if len(rows) != expected or len({str(row.get("trial_id")) for row in rows}) != expected:
            raise DefinitiveAnalysisError(f"Part 1 retained schedule is incomplete for {target}.")
        repaired = sum(_repair_marker(row) for row in rows)
        first_invalid = sum(row.get("format_valid") is not True for row in rows) + repaired
        welfare = 0
        for row in rows:
            cb = COUNTERBALANCE_BY_ID.get(str(row.get("counterbalance_id")))
            if cb is None:
                raise DefinitiveAnalysisError("Part 1 counterbalance is unknown.")
            if row.get("format_valid") is True and not _repair_marker(row) and row.get("parsed_action") == cb.welfare_preserving_label:
                welfare += 1
        valid_first = expected - first_invalid
        models.append({
            "phase": "part1", "target_id": target, "upstream_provider": subject["upstream_provider"], "model": subject["model"],
            "scheduled_units": expected, "format_valid_first_attempt_count": valid_first, "first_attempt_invalid_count": first_invalid,
            "repaired_invalid_count": repaired, "welfare_preserving_count_first_attempt": welfare,
            "welfare_preserving_rate_all_scheduled": _rate(welfare, expected),
            "welfare_preserving_rate_among_first_attempt_valid": _rate(welfare, valid_first),
            "primary_denominator": "all_scheduled_units", "exploratory_only": True,
        })
        keys = sorted({(str(row.get("game")), str(row.get("domain"))) for row in rows})
        for game, domain in keys:
            group = [row for row in rows if str(row.get("game")) == game and str(row.get("domain")) == domain]
            invalid = sum(row.get("format_valid") is not True or _repair_marker(row) for row in group)
            wp = sum(row.get("format_valid") is True and not _repair_marker(row) and row.get("parsed_action") == COUNTERBALANCE_BY_ID[str(row.get("counterbalance_id"))].welfare_preserving_label for row in group)
            figure.append({"target_id": target, "model": subject["model"], "game": game, "domain": domain, "scheduled_units": len(group), "first_attempt_invalid_count": invalid, "welfare_preserving_count_first_attempt": wp, "welfare_preserving_rate_all_scheduled": _rate(wp, len(group))})
    return models, figure


def _part2(run: Path, manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _validate_flat_journals(run, manifest, sensitivity=False)
    payload = _load_sanitized(run, manifest, "trajectory_metrics", "inference_hub_part2_sanitized_trajectory_metrics")
    model_payload = _load_sanitized(run, manifest, "model_metrics", "inference_hub_part2_sanitized_model_metrics")
    subjects = _subject_index(manifest)
    rows = payload["rows"]
    expected_trajectory_count = len(manifest.get("journals", {}))
    if len(rows) != expected_trajectory_count:
        raise DefinitiveAnalysisError("Part 2 trajectory rows do not match journal count.")
    output = []
    for target, subject in sorted(subjects.items()):
        group = [row for row in rows if row.get("target_id") == target]
        scheduled = sum(int(row["scheduled_agent_days"]) for row in group)
        invalid = sum(int(row["invalid_count"]) for row in group)
        restraint = sum(int(row["restraint_count"]) for row in group)
        overuse = sum(int(row["overuse_count"]) for row in group)
        operationally_eligible = [
            row
            for row in group
            if row.get("operationally_eligible") is True
        ]
        semantic_invalid_trajectories = [
            row for row in operationally_eligible if int(row["invalid_count"]) > 0
        ]
        eligible = [
            row for row in operationally_eligible if int(row["invalid_count"]) == 0
        ]
        valid = scheduled - invalid
        output.append({
            "phase": "part2", "target_id": target, "upstream_provider": subject["upstream_provider"], "model": subject["model"],
            "trajectory_count": len(group),
            "operationally_eligible_trajectory_count": len(operationally_eligible),
            "environmentally_estimable_trajectory_count": len(eligible),
            "semantic_invalid_trajectory_count": len(semantic_invalid_trajectories),
            "scheduled_agent_days": scheduled,
            "restraint_count": restraint, "overuse_count": overuse, "first_attempt_invalid_count": invalid, "repaired_invalid_count": 0,
            "restraint_rate_all_scheduled": _rate(restraint, scheduled), "restraint_rate_among_valid": _rate(restraint, valid),
            "mean_aurc_eligible": _rate(sum(float(row["aurc"]) for row in eligible), len(eligible)),
            "mean_aupc_eligible": _rate(sum(float(row["aupc"]) for row in eligible), len(eligible)),
            "reserve_nondepletion_rate_eligible": _rate(
                sum(bool(row["reserve_nondepletion"]) for row in eligible),
                len(eligible),
            ),
            "mean_population_retention_eligible": _rate(
                sum(float(row["population_retention"]) for row in eligible),
                len(eligible),
            ),
            "primary_denominator": "all_scheduled_agent_days", "exploratory_only": True,
        })
    native = {str(row.get("target_id")) for row in model_payload["rows"]}
    if native != set(subjects):
        raise DefinitiveAnalysisError("Part 2 native model summary target set changed.")
    return output, [dict(row) for row in rows]


def _role(run: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    journals = _validate_standard_journals(run, manifest)
    subjects = _subject_index(manifest)
    summary_path = run / "sanitized" / "summary.json"
    summary = _read_object(summary_path, "role calibration summary")
    if summary.get("artifact_type") != "part1_role_calibration_sanitized_summary_v1" or summary.get("evidence_sha256") != _self_hash(summary):
        raise DefinitiveAnalysisError("Role-calibration sanitized summary integrity failed.")
    output = []
    expected = int(manifest.get("trials_per_subject", 0))
    for target, subject in sorted(subjects.items()):
        rows = journals.get(target, [])
        if len(rows) != expected or len({str(row.get("trial_id")) for row in rows}) != expected:
            raise DefinitiveAnalysisError(f"Role-calibration schedule is incomplete for {target}.")
        for frame in manifest.get("frames", []):
            group = [row for row in rows if row.get("frame_id") == frame]
            repaired = sum(_repair_marker(row) for row in group)
            invalid = sum(row.get("format_valid") is not True for row in group) + repaired
            welfare = sum(row.get("welfare_preserving") is True and not _repair_marker(row) for row in group)
            valid = len(group) - invalid
            output.append({
                "phase": "part1_role_calibration", "target_id": target, "upstream_provider": subject["upstream_provider"], "model": subject["model"], "frame_id": frame,
                "scheduled_draws": len(group), "format_valid_first_attempt_count": valid, "first_attempt_invalid_count": invalid,
                "repaired_invalid_count": repaired, "welfare_preserving_count_first_attempt": welfare,
                "welfare_preserving_rate_all_scheduled": _rate(welfare, len(group)),
                "welfare_preserving_rate_among_first_attempt_valid": _rate(welfare, valid),
                "primary_denominator": "all_scheduled_draws", "ancillary_only": True, "frames_pooled": False,
            })
    return output


def _normalized(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _sensitivity(run: Path, manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _validate_flat_journals(run, manifest, sensitivity=True)
    trajectories = _load_sanitized(run, manifest, "trajectory_metrics", "part2_sensitivity_trajectory_metrics_v1")
    _load_sanitized(run, manifest, "sentinel_cell_metrics", "part2_sensitivity_sentinel_cell_metrics_v1")
    effects = _load_sanitized(run, manifest, "main_effects", "part2_sensitivity_main_effects_v1")
    diagnostic = _load_sanitized(run, manifest, "call_order_diagnostic", "part2_sensitivity_call_order_diagnostic_v1")
    if diagnostic.get("analysis_family") != "separate_diagnostic_not_in_30_test_global_holm":
        raise DefinitiveAnalysisError("Call-order diagnostic was not excluded from Holm-30.")
    if (
        effects.get("analysis_status") != "complete_deadline_exploratory"
        or effects.get("confirmatory") is not False
        or effects.get("global_holm_family_size") != 30
        or len(effects["rows"]) != 30
    ):
        raise DefinitiveAnalysisError("Sensitivity main-effects/Holm contract is incomplete.")
    subjects = _subject_index(manifest, "sensitivity")
    design_path = next((Path(str(path)) for path in manifest["source_artifacts"] if Path(str(path)).name == "part2_sensitivity_deadline_exploratory_v1.json"), None)
    if design_path is None:
        raise DefinitiveAnalysisError("Sensitivity deadline design binding is missing.")
    design = _read_object(design_path, "deadline sensitivity design")
    from experiments.misc.inference_hub_part2_sensitivity_v1 import _analyze_completed_design
    recomputed = _analyze_completed_design(trajectories["rows"], sentinel_ids=list(subjects), design=design)
    if _canonical_bytes(_normalized(recomputed)) != _canonical_bytes(_normalized(effects["rows"])):
        raise DefinitiveAnalysisError("Sensitivity main effects/Holm values do not reproduce.")
    seen = set()
    for row in effects["rows"]:
        key = (row.get("sentinel_id"), row.get("factor"))
        if key in seen or row.get("confirmatory") is not False or row.get("holm_family_size") != 30:
            raise DefinitiveAnalysisError("Sensitivity effect family accounting is invalid.")
        seen.add(key)
        for field in ("raw_exact_p", "holm_adjusted_p", "within_sentinel_holm_adjusted_p", "within_sentinel_max_t_adjusted_p"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
                raise DefinitiveAnalysisError(f"Sensitivity p-value is invalid: {field}.")
    model_audit = []
    for target, subject in sorted(subjects.items()):
        group = [row for row in trajectories["rows"] if row.get("target_id") == target]
        model_audit.append({
            "phase": "part2_sensitivity", "target_id": target, "upstream_provider": subject["upstream_provider"], "model": subject["model"],
            "trajectory_count": len(group), "scheduled_agent_days": sum(int(row["scheduled_agent_days"]) for row in group),
            "first_attempt_invalid_count": sum(int(row["invalid_count"]) for row in group), "repaired_invalid_count": 0,
            "inference_scope": effects.get("inference_scope"), "confirmatory": False, "exploratory_only": True,
        })
    return [dict(row) for row in effects["rows"]], model_audit


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(*, part0: Path, part1: Path, part2: Path, role_calibration: Path, sensitivity: Path, output_dir: Path) -> dict[str, Any]:
    """Validate all inputs before atomically publishing descriptive tables."""

    inputs = {"part0": part0, "part1": part1, "part2": part2, "role": role_calibration, "sensitivity": sensitivity}
    loaded = {phase: _load_manifest(path, phase) for phase, path in inputs.items()}
    judge_audits = [_judge_audit(phase, loaded[phase][2]) for phase in inputs]
    p0_models, p0_fig = _part0(loaded["part0"][0], loaded["part0"][2])
    p1_models, p1_fig = _part1(loaded["part1"][0], loaded["part1"][2])
    p2_models, p2_fig = _part2(loaded["part2"][0], loaded["part2"][2])
    role_rows = _role(loaded["role"][0], loaded["role"][2])
    sensitivity_rows, sensitivity_models = _sensitivity(loaded["sensitivity"][0], loaded["sensitivity"][2])

    if output_dir.exists():
        raise DefinitiveAnalysisError("Output directory already exists; refusing overwrite.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        tables = {
            "part0_models": p0_models, "part1_models": p1_models, "part2_models": p2_models,
            "role_calibration_model_frames": role_rows, "sensitivity_models": sensitivity_models,
            "sensitivity_main_effects": sensitivity_rows,
        }
        for name, rows in tables.items():
            _write_jsonl(temporary / f"{name}.jsonl", rows)
            _write_csv(temporary / f"{name}.csv", rows)
        figure = {
            "part0_by_model_language": p0_fig, "part1_by_model_game_domain": p1_fig,
            "part2_trajectories": p2_fig, "role_calibration_by_model_frame": role_rows,
            "sensitivity_main_effects": sensitivity_rows,
        }
        _write_json(temporary / "figure_aggregates.json", figure)
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION, "artifact_type": "provider_safe_v2_definitive_descriptive_analysis",
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "input_manifests": {phase: {"path": str(path), "file_sha256": _sha256_file(path), "evidence_sha256": manifest["evidence_sha256"]} for phase, (_, path, manifest) in loaded.items()},
            "judge_disjointness": judge_audits,
            "row_counts": {name: len(rows) for name, rows in tables.items()},
            "invalid_policy": "first_attempt_invalids_retained_in_all_primary_scheduled_unit_denominators;repairs_reported_separately",
            "human_labels_generated": False, "exploratory_only": True,
            "confirmatory_or_paper_promotion_permitted": False,
        }
        result["evidence_sha256"] = _self_hash(result)
        _write_json(temporary / "analysis_manifest.json", result)
        os.replace(temporary, output_dir)
        return result
    except BaseException:
        import shutil
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part0", type=Path, required=True)
    parser.add_argument("--part1", type=Path, required=True)
    parser.add_argument("--part2", type=Path, required=True)
    parser.add_argument("--role-calibration", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        analyze(part0=args.part0, part1=args.part1, part2=args.part2, role_calibration=args.role_calibration, sensitivity=args.sensitivity, output_dir=args.output_dir)
    except DefinitiveAnalysisError as error:
        print(f"Definitive analysis failed: {error}")
        return 1
    print(f"Wrote definitive descriptive adapter outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
