"""Build fail-closed, text-free paper artifacts from completed hosted panels.

This module never emits prompts, visible responses, reasoning, raw responses, or
routes.  Part 0's reconstructed conditions are labelled response-language
conditions, Part 1's 384-root and balanced-partial scopes are never pooled, and
Part 2 uses independent trajectories as its uncertainty unit.  Cross-axis
correlations are created only for the exact frozen 24-system panel when every
gate passes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from analysis.analyze_inference_hub_part1_panel import (
    _atomic_json,
    _read_chained_journal,
    _read_json,
    _seal,
    _self_hash,
    _sha256_file,
    _sha256_json,
)
from experiments.part1.confirmatory_design import (
    COUNTERBALANCE_BY_ID,
    DOMAINS,
    GAMES,
    WELFARE_PRESERVING,
)


SCHEMA_VERSION = 1
BOOTSTRAP_REPLICATES = 5_000
DEFAULT_BOOTSTRAP_SEED = 20_260_802
_FORBIDDEN_PUBLIC_KEYS = {
    "prompt", "prompt_text", "messages", "visible_response", "response_text",
    "visible_content", "reasoning", "raw_response", "request_body", "requested_route",
    "route",
}


class FinalResultsError(RuntimeError):
    """An input cannot support a sanitized final-results artifact."""


def _manifest(
    path: Path, artifact_type: str, *, require_complete: bool = True,
) -> dict[str, Any]:
    value = _read_json(path.resolve(), f"{artifact_type} manifest")
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != artifact_type
        or value.get("evidence_sha256") != _self_hash(value)
    ):
        raise FinalResultsError(f"{artifact_type} manifest schema/type/self-hash failed.")
    if require_complete and value.get("complete") is not True:
        raise FinalResultsError(f"{artifact_type} manifest is incomplete.")
    return value


def _subject_routes(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("subject_routes")
    if not isinstance(rows, list) or not rows:
        raise FinalResultsError("Manifest subject routes are absent.")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise FinalResultsError("Manifest subject route is not an object.")
        target_id = row.get("target_id")
        if (
            not isinstance(target_id, str) or not target_id or target_id in result
            or not all(isinstance(row.get(key), str) and row.get(key) for key in ("upstream_provider", "model", "route"))
        ):
            raise FinalResultsError("Manifest subject identity is incomplete or duplicated.")
        result[target_id] = dict(row)
    return result


def _manifest_binding(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "manifest_path": path.name,
        "path_scope": "input_manifest_basename_only",
        "file_sha256": _sha256_file(path),
        "evidence_sha256": manifest["evidence_sha256"],
    }


def _selected_subject_routes(
    manifest: Mapping[str, Any], target_ids: set[str] | None,
) -> dict[str, dict[str, Any]]:
    subjects = _subject_routes(manifest)
    if target_ids is None:
        return subjects
    unknown = target_ids - set(subjects)
    if unknown:
        raise FinalResultsError(
            "Selected replacement-overlay targets are absent from the manifest: "
            + ",".join(sorted(unknown))
        )
    return {target_id: row for target_id, row in subjects.items() if target_id in target_ids}


def _overlay_contract(
    manifest: Mapping[str, Any], *, part: str,
) -> dict[str, Any]:
    keys_by_part = {
        "part0": (
            "stimulus_reconstruction", "selected_roots_per_language", "languages",
            "schedule", "schedule_sha256", "executed_trial_count_per_subject",
            "base_seed", "judge", "judge_batch_size", "judge_scoring_input",
            "execution_contract", "input_artifacts", "source_artifacts",
        ),
        "part1": (
            "executed_trial_count_per_subject", "trial_limit",
            "executed_schedule_sha256", "full_primary_schedule_sha256",
            "full_primary_root_count", "base_seed", "execution_contract",
            "input_artifacts", "source_artifacts",
        ),
        "part2": (
            "panel_id", "base_seed", "common_environment_seeds", "part2_contract",
            "execution_contract", "input_artifacts", "source_artifacts",
        ),
    }
    try:
        keys = keys_by_part[part]
    except KeyError as error:
        raise FinalResultsError(f"Unknown overlay part: {part}.") from error
    contract = {key: manifest.get(key) for key in keys}
    if isinstance(contract["execution_contract"], Mapping):
        # Replacement runs may increase transport resilience or reduce worker
        # fan-out.  These fields affect availability and wall-clock time, not
        # prompts, decoding controls, seeds, parsing, model identity, or task
        # dynamics.  The unnormalized contracts remain hash-bound in each
        # manifest and are preserved in the output provenance.
        execution_contract = dict(contract["execution_contract"])
        operational_fields = {
            "part0": {
                "max_workers", "max_attempts_per_request",
                "initial_exponential_backoff_seconds",
            },
            "part1": {
                "configured_max_workers", "effective_max_workers",
                "max_attempts_per_trial", "initial_exponential_backoff_seconds",
            },
            "part2": {
                "trajectory_workers", "max_transport_attempts",
                "initial_exponential_backoff_seconds",
            },
        }[part]
        for key in operational_fields:
            execution_contract.pop(key, None)
        contract["execution_contract"] = execution_contract
    if part == "part0" and isinstance(contract["input_artifacts"], Mapping):
        # The cross-axis panel selects the primary roster but does not define a
        # Part 0 prompt, schedule, judge, or model identity.  A one-target repair
        # is selected explicitly and validated against the primary identity.
        input_artifacts = dict(contract["input_artifacts"])
        input_artifacts.pop("cross_axis_panel", None)
        contract["input_artifacts"] = input_artifacts
    return contract


def _validate_replacement_manifest(
    *, primary: Mapping[str, Any], replacement: Mapping[str, Any], part: str,
    seen_target_ids: set[str],
) -> set[str]:
    if replacement.get("complete") is not True:
        raise FinalResultsError(f"Part {part[-1]} replacement manifest is incomplete.")
    primary_subjects = _subject_routes(primary)
    replacement_subjects = _subject_routes(replacement)
    replacement_ids = set(replacement_subjects)
    duplicated = seen_target_ids & replacement_ids
    if duplicated:
        raise FinalResultsError(
            f"Part {part[-1]} replacement target is duplicated: "
            + ",".join(sorted(duplicated))
        )
    unknown = replacement_ids - set(primary_subjects)
    if unknown:
        raise FinalResultsError(
            f"Part {part[-1]} replacement target is absent from the primary manifest: "
            + ",".join(sorted(unknown))
        )
    for target_id in replacement_ids:
        primary_identity = primary_subjects[target_id]
        replacement_identity = replacement_subjects[target_id]
        if any(
            replacement_identity.get(key) != primary_identity.get(key)
            for key in ("target_id", "upstream_provider", "model", "route")
        ):
            raise FinalResultsError(
                f"Part {part[-1]} replacement identity differs for {target_id}."
            )
    if _overlay_contract(primary, part=part) != _overlay_contract(replacement, part=part):
        raise FinalResultsError(
            f"Part {part[-1]} replacement schedule or scientific contract differs."
        )
    seen_target_ids.update(replacement_ids)
    return replacement_ids


def _explicit_target_ids(values: Sequence[str], *, label: str) -> set[str]:
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise FinalResultsError(f"{label} target id is empty or invalid.")
        if value in result:
            raise FinalResultsError(f"{label} target is duplicated: {value}.")
        result.add(value)
    return result


def _failure_provenance(
    target_id: str, failure_counts: Mapping[str, int], *, part: str,
) -> dict[str, Any]:
    sanitized = {
        category: count for category, count in sorted(failure_counts.items())
        if isinstance(count, int) and not isinstance(count, bool) and count > 0
    }
    if not sanitized:
        raise FinalResultsError(
            f"Part {part[-1]} target {target_id} has no target-bound operational "
            "or identity failure and cannot be marked unavailable."
        )
    return {
        "target_id": target_id,
        "failure_counts": sanitized,
        "total_failure_count": sum(sanitized.values()),
        "provenance": "validated_target_bound_primary_evidence",
    }


def _validate_failure_summary_bounds(
    manifest: Mapping[str, Any], observed: Mapping[str, int],
    summary_fields: Mapping[str, str], *, part: str,
) -> None:
    summary = manifest.get("summary")
    if not isinstance(summary, Mapping):
        raise FinalResultsError(f"Part {part[-1]} failure summary is absent.")
    for category, count in observed.items():
        field = summary_fields[category]
        total = summary.get(field)
        if (
            not isinstance(total, int) or isinstance(total, bool) or total < count
        ):
            raise FinalResultsError(
                f"Part {part[-1]} target-bound {category} count exceeds its "
                "sanitized manifest summary."
            )


def _journal_rows(manifest: Mapping[str, Any], target_id: str) -> list[dict[str, Any]]:
    journals = manifest.get("journals")
    raw = journals.get("raw_responses") if isinstance(journals, Mapping) else None
    reference = raw.get(target_id) if isinstance(raw, Mapping) else None
    if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
        raise FinalResultsError(f"Raw journal binding is absent for {target_id}.")
    try:
        return _read_chained_journal(
            Path(str(reference["path"])), reference, label=f"raw responses for {target_id}"
        )
    except RuntimeError as error:
        raise FinalResultsError(str(error)) from error


def _wilson(successes: int, total: int) -> dict[str, Any]:
    if total <= 0 or not 0 <= successes <= total:
        raise FinalResultsError("A rate numerator/denominator is invalid.")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {
        "estimate": p, "lower": max(0.0, center - half), "upper": min(1.0, center + half),
        "n": total, "method": "Wilson score 95%; independent prompt roots within condition",
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _stratified_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    if not rows:
        raise FinalResultsError("Cannot bootstrap an empty Part 1 result.")
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        strata[(str(row["game"]), str(row["domain"]))].append(int(bool(row["cooperation"])))
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        successes = 0
        for values in strata.values():
            successes += sum(rng.choice(values) for _ in values)
        draws.append(successes / len(rows))
    estimate = sum(int(bool(row["cooperation"])) for row in rows) / len(rows)
    return {
        "estimate": estimate, "lower": _percentile(draws, 0.025),
        "upper": _percentile(draws, 0.975), "n": len(rows),
        "method": "within-game-domain stratified root bootstrap percentile 95%",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def _cluster_bootstrap_binary(
    rows: Sequence[tuple[str, bool]], *, seed: int
) -> dict[str, Any]:
    clusters: dict[str, list[int]] = defaultdict(list)
    for cluster_id, outcome in rows:
        clusters[cluster_id].append(int(outcome))
    if not clusters:
        raise FinalResultsError("Cannot bootstrap an empty clustered rate.")
    ids = sorted(clusters)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = [rng.choice(ids) for _ in ids]
        numerator = sum(sum(clusters[cluster_id]) for cluster_id in sampled)
        denominator = sum(len(clusters[cluster_id]) for cluster_id in sampled)
        draws.append(numerator / denominator)
    estimate = sum(sum(values) for values in clusters.values()) / sum(
        len(values) for values in clusters.values()
    )
    return {
        "estimate": estimate, "lower": _percentile(draws, 0.025),
        "upper": _percentile(draws, 0.975), "n_rows": len(rows),
        "n_clusters": len(clusters),
        "method": "prompt-root cluster bootstrap percentile 95% across response-language conditions",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def _part0(
    path: Path, *, target_ids: set[str] | None = None,
    require_complete: bool = True, require_summary: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str, str]], dict[str, Any]]:
    manifest = _manifest(
        path, "inference_hub_part0_accelerated_private_panel",
        require_complete=require_complete,
    )
    if manifest.get("stimulus_reconstruction") != (
        "archived_english_source_prompt_crossed_with_explicit_response_language_instruction;"
        "legacy_translated_input_not_retained"
    ):
        raise FinalResultsError("Part 0 reconstructed-condition provenance changed.")
    if require_summary:
        summary = manifest.get("summary")
        if not isinstance(summary, Mapping) or any(
            summary.get(key) != 0 for key in (
                "subject_transport_failures", "judge_failed_units",
                "subject_model_identity_mismatches", "judge_model_identity_mismatches",
            )
        ):
            raise FinalResultsError("Part 0 contains an operational or identity failure.")
    schedule = manifest.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        raise FinalResultsError("Part 0 schedule is absent.")
    schedule_by_trial = {
        row.get("trial_id"): row for row in schedule if isinstance(row, Mapping)
    }
    if len(schedule_by_trial) != len(schedule):
        raise FinalResultsError("Part 0 schedule ids are invalid or duplicated.")
    subjects = _selected_subject_routes(manifest, target_ids)
    judge = manifest.get("judge")
    if not isinstance(judge, Mapping) or not isinstance(judge.get("route"), str):
        raise FinalResultsError("Part 0 judge identity is absent.")
    output: list[dict[str, Any]] = []
    identities: dict[str, tuple[str, str, str]] = {}
    for target_id, subject in subjects.items():
        rows = _journal_rows(manifest, target_id)
        terminal: dict[str, Mapping[str, Any]] = {}
        retained_subjects: dict[str, Mapping[str, Any]] = {}
        retained_batches: dict[str, Mapping[str, Any]] = {}
        subject_identity_count = judge_identity_count = 0
        for row in rows:
            event = row.get("event")
            if event == "subject_response_retained":
                subject_identity_count += row.get("model_identity_valid") is not True
                raw = row.get("raw_response")
                trial_id = row.get("trial_id")
                if (
                    not isinstance(trial_id, str) or trial_id not in schedule_by_trial
                    or trial_id in retained_subjects
                    or row.get("requested_route") != subject["route"]
                    or row.get("response_model") != subject["route"]
                    or not isinstance(raw, Mapping) or raw.get("model") != subject["route"]
                    or row.get("raw_response_sha256") != _sha256_json(raw)
                ):
                    raise FinalResultsError("Part 0 subject raw-response hash failed.")
                retained_subjects[trial_id] = row
            elif event == "judge_batch_retained":
                judge_identity_count += row.get("model_identity_valid") is not True
                raw = row.get("raw_response")
                batch_id = row.get("batch_id")
                if (
                    not isinstance(batch_id, str) or batch_id in retained_batches
                    or row.get("requested_route") != judge["route"]
                    or row.get("response_model") != judge["route"]
                    or not isinstance(raw, Mapping) or raw.get("model") != judge["route"]
                    or row.get("raw_response_sha256") != _sha256_json(raw)
                ):
                    raise FinalResultsError("Part 0 judge raw-response hash failed.")
                retained_batches[batch_id] = row
            elif event == "unit_completed":
                trial_id = row.get("trial_id")
                trial = schedule_by_trial.get(trial_id)
                if (
                    not isinstance(trial_id, str) or trial is None or trial_id in terminal
                    or row.get("target_id") != target_id
                    or row.get("root_id") != trial.get("root_id")
                    or row.get("language") != trial.get("language")
                    or row.get("outcome") not in {"REFUSAL", "COMPLIANCE", "UNCLEAR", "INVALID"}
                ):
                    raise FinalResultsError("Part 0 terminal row binding is invalid.")
                terminal[trial_id] = row
        if (
            subject_identity_count or judge_identity_count
            or set(terminal) != set(schedule_by_trial)
            or set(retained_subjects) != set(schedule_by_trial)
            or any(
                row.get("judge_dispatched") is not True
                or row.get("judge_batch_id") not in retained_batches
                for row in terminal.values()
            )
        ):
            raise FinalResultsError("Part 0 journal coverage or identity validation failed.")
        by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for trial_id, row in terminal.items():
            by_condition[str(schedule_by_trial[trial_id]["language"])].append(row)
        condition_rows = []
        for condition in manifest.get("languages", []):
            values = by_condition.get(str(condition), [])
            refusal = sum(row["outcome"] == "REFUSAL" for row in values)
            unclear = sum(row["outcome"] == "UNCLEAR" for row in values)
            invalid = sum(row["outcome"] == "INVALID" for row in values)
            condition_rows.append({
                "response_language_condition": condition,
                "condition_definition": "English source request plus explicit requested response language; not a translated-prompt condition",
                "refusal": _wilson(refusal, len(values)),
                "unclear_count": unclear, "invalid_count": invalid,
            })
        overall_refusal = _cluster_bootstrap_binary(
            [
                (
                    str(schedule_by_trial[trial_id]["root_id"]),
                    row["outcome"] == "REFUSAL",
                )
                for trial_id, row in terminal.items()
            ],
            seed=_derive_public_seed(manifest.get("base_seed", DEFAULT_BOOTSTRAP_SEED), "part0", target_id),
        )
        part0_paper_eligible = (
            manifest.get("human_validation_complete") is True
            and isinstance(manifest.get("selected_roots_per_language"), int)
            and manifest["selected_roots_per_language"] >= 48
        )
        output.append({
            "target_id": target_id, "upstream_provider": subject["upstream_provider"],
            "model": subject["model"], "scope": "reconstructed_response_language_conditions",
            "root_count_per_condition": manifest["selected_roots_per_language"],
            "overall_refusal": overall_refusal,
            "conditions": condition_rows,
            "human_validation_complete": manifest.get("human_validation_complete") is True,
            "paper_eligible": part0_paper_eligible,
        })
        identities[target_id] = (str(subject["upstream_provider"]), str(subject["model"]), str(subject["route"]))
    return output, identities, {
        **_manifest_binding(path, manifest),
        "human_validation_complete": manifest.get("human_validation_complete") is True,
    }


def _part0_unavailable_failure(
    manifest: Mapping[str, Any], target_id: str,
) -> dict[str, Any]:
    subjects = _subject_routes(manifest)
    subject = subjects[target_id]
    schedule = manifest.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        raise FinalResultsError("Part 0 schedule is absent.")
    schedule_by_trial = {
        row.get("trial_id"): row for row in schedule if isinstance(row, Mapping)
    }
    if len(schedule_by_trial) != len(schedule):
        raise FinalResultsError("Part 0 schedule ids are invalid or duplicated.")
    judge = manifest.get("judge")
    if not isinstance(judge, Mapping) or not isinstance(judge.get("route"), str):
        raise FinalResultsError("Part 0 judge identity is absent.")
    counts: dict[str, int] = defaultdict(int)
    subject_trials: set[str] = set()
    judge_batches: set[str] = set()
    terminal_trials: set[str] = set()
    for row in _journal_rows(manifest, target_id):
        if (
            row.get("schema_version") != 1
            or row.get("artifact_type") != "inference_hub_part0_private_record"
            or row.get("target_id") != target_id
        ):
            raise FinalResultsError("Part 0 unavailable-target journal binding failed.")
        event = row.get("event")
        if event == "subject_response_retained":
            raw = row.get("raw_response")
            trial_id = row.get("trial_id")
            trial = schedule_by_trial.get(trial_id)
            if (
                not isinstance(trial_id, str) or trial is None
                or trial_id in subject_trials
                or row.get("root_id") != trial.get("root_id")
                or row.get("language") != trial.get("language")
                or row.get("requested_route") != subject["route"]
                or not isinstance(raw, Mapping)
                or row.get("raw_response_sha256") != _sha256_json(raw)
                or row.get("response_model") != raw.get("model")
            ):
                raise FinalResultsError("Part 0 unavailable subject evidence is invalid.")
            subject_trials.add(trial_id)
            if row.get("model_identity_valid") is not True:
                counts["subject_model_identity_mismatch"] += 1
        elif event == "judge_batch_retained":
            raw = row.get("raw_response")
            batch_id = row.get("batch_id")
            if (
                not isinstance(batch_id, str) or not batch_id
                or batch_id in judge_batches
                or row.get("requested_route") != judge["route"]
                or not isinstance(raw, Mapping)
                or row.get("raw_response_sha256") != _sha256_json(raw)
                or row.get("response_model") != raw.get("model")
            ):
                raise FinalResultsError("Part 0 unavailable judge evidence is invalid.")
            judge_batches.add(batch_id)
            if row.get("model_identity_valid") is not True:
                counts["judge_model_identity_mismatch"] += 1
        elif event == "unit_completed":
            trial = schedule_by_trial.get(row.get("trial_id"))
            if (
                trial is None or row.get("trial_id") in terminal_trials
                or row.get("root_id") != trial.get("root_id")
                or row.get("language") != trial.get("language")
                or row.get("outcome") not in {"REFUSAL", "COMPLIANCE", "UNCLEAR", "INVALID"}
            ):
                raise FinalResultsError("Part 0 unavailable terminal evidence is invalid.")
            terminal_trials.add(str(row["trial_id"]))
            operational = row.get("operational_failure")
            judge_failure = row.get("judge_failure")
            if operational is not None:
                if (
                    row.get("outcome") != "INVALID"
                    or not isinstance(operational, Mapping)
                    or not isinstance(operational.get("failure_code"), str)
                ):
                    raise FinalResultsError("Part 0 subject failure evidence is invalid.")
                counts["subject_transport_failure"] += 1
            if judge_failure is not None:
                if (
                    row.get("outcome") != "UNCLEAR"
                    or not isinstance(judge_failure, Mapping)
                    or not isinstance(judge_failure.get("failure_code"), str)
                ):
                    raise FinalResultsError("Part 0 judge failure evidence is invalid.")
                counts["judge_transport_failure"] += 1
        else:
            raise FinalResultsError("Part 0 unavailable-target journal event is unknown.")
    _validate_failure_summary_bounds(
        manifest, counts,
        {
            "subject_model_identity_mismatch": "subject_model_identity_mismatches",
            "judge_model_identity_mismatch": "judge_model_identity_mismatches",
            "subject_transport_failure": "subject_transport_failures",
            "judge_transport_failure": "judge_failed_units",
        },
        part="part0",
    )
    return _failure_provenance(target_id, counts, part="part0")


def _part0_overlay(
    primary_path: Path, replacement_paths: Sequence[Path],
    unavailable_target_ids: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str, str]], dict[str, Any]]:
    unavailable_ids = _explicit_target_ids(
        unavailable_target_ids, label="Part 0 unavailable",
    )
    if not replacement_paths and not unavailable_ids:
        return _part0(primary_path)
    artifact_type = "inference_hub_part0_accelerated_private_panel"
    primary = _manifest(primary_path, artifact_type, require_complete=False)
    if primary.get("complete") is True:
        raise FinalResultsError(
            "Part 0 replacements or unavailable targets require an incomplete primary manifest."
        )
    primary_subjects = _subject_routes(primary)
    unknown_unavailable = unavailable_ids - set(primary_subjects)
    if unknown_unavailable:
        raise FinalResultsError(
            "Part 0 unavailable target is absent from the primary manifest: "
            + ",".join(sorted(unknown_unavailable))
        )
    unavailable_failures = [
        _part0_unavailable_failure(primary, target_id)
        for target_id in primary_subjects if target_id in unavailable_ids
    ]
    seen_replacements: set[str] = set()
    replacements: list[tuple[Path, dict[str, Any], set[str]]] = []
    for path in replacement_paths:
        replacement = _manifest(path, artifact_type)
        replacement_ids = _validate_replacement_manifest(
            primary=primary, replacement=replacement, part="part0",
            seen_target_ids=seen_replacements,
        )
        replacements.append((path, replacement, replacement_ids))
    overlap = seen_replacements & unavailable_ids
    if overlap:
        raise FinalResultsError(
            "Part 0 target cannot be both replaced and unavailable: "
            + ",".join(sorted(overlap))
        )

    retained_ids = set(primary_subjects) - seen_replacements - unavailable_ids
    if retained_ids:
        rows, identities, primary_binding = _part0(
            primary_path, target_ids=retained_ids,
            require_complete=False, require_summary=False,
        )
    else:
        rows, identities = [], {}
        primary_binding = _manifest_binding(primary_path, primary)
    primary_binding.update({
        "complete": primary.get("complete") is True,
        "retained_target_ids": sorted(retained_ids),
    })
    replacement_bindings: list[dict[str, Any]] = []
    for path, _replacement, replacement_ids in replacements:
        new_rows, new_identities, binding = _part0(
            path, target_ids=replacement_ids,
        )
        rows.extend(new_rows)
        identities.update(new_identities)
        replacement_bindings.append({
            **binding, "replacement_target_ids": sorted(replacement_ids),
        })
    row_by_target = {str(row["target_id"]): row for row in rows}
    expected_output_ids = set(primary_subjects) - unavailable_ids
    if set(row_by_target) != expected_output_ids:
        raise FinalResultsError("Part 0 overlay does not cover every primary target exactly once.")
    rows = [
        row_by_target[target_id] for target_id in primary_subjects
        if target_id in expected_output_ids
    ]
    return rows, identities, {
        "overlay_schema_version": 1,
        "primary": primary_binding,
        "replacements": replacement_bindings,
        "replaced_target_ids": sorted(seen_replacements),
        "unavailable_target_ids": sorted(unavailable_ids),
        "unavailable_target_failures": unavailable_failures,
    }


def _part1_manifest(
    path: Path, *, scope: str, bootstrap_seed: int,
    target_ids: set[str] | None = None, require_complete: bool = True,
    require_summary: bool = True, subject_indices: Mapping[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str, str]], dict[str, Any]]:
    if scope not in {"full_384", "balanced_partial"}:
        raise FinalResultsError("Unknown Part 1 scope.")
    manifest = _manifest(
        path, "inference_hub_part1_large_n_exploratory_panel",
        require_complete=require_complete,
    )
    observed_roots = manifest.get("executed_trial_count_per_subject")
    expected_roots = 384 if scope == "full_384" else observed_roots
    if (
        isinstance(expected_roots, bool)
        or not isinstance(expected_roots, int)
        or expected_roots < 12
        or expected_roots > 384
        or expected_roots % 12
        or manifest.get("executed_trial_count_per_subject") != expected_roots
        or (scope == "full_384" and manifest.get("trial_limit") is not None)
        or (scope == "balanced_partial" and expected_roots == 384)
        or (scope == "balanced_partial" and manifest.get("trial_limit") != expected_roots)
    ):
        raise FinalResultsError(f"Part 1 {scope} manifest has the wrong root count/scope.")
    if require_summary:
        summary = manifest.get("summary")
        if not isinstance(summary, Mapping) or summary.get("failed_without_response") != 0 or summary.get("response_model_identity_mismatches") != 0:
            raise FinalResultsError("Part 1 contains an operational or identity failure.")
    all_subjects = _subject_routes(manifest)
    subjects = _selected_subject_routes(manifest, target_ids)
    manifest_indices = {
        target_id: index for index, target_id in enumerate(all_subjects)
    }
    if subject_indices is not None and not set(subjects) <= set(subject_indices):
        raise FinalResultsError("Part 1 overlay subject-index binding is incomplete.")
    output: list[dict[str, Any]] = []
    identities: dict[str, tuple[str, str, str]] = {}
    for target_id, subject in subjects.items():
        subject_index = (
            int(subject_indices[target_id])
            if subject_indices is not None else manifest_indices[target_id]
        )
        journal_rows = _journal_rows(manifest, target_id)
        retained: dict[str, Mapping[str, Any]] = {}
        derived: list[dict[str, Any]] = []
        for row in journal_rows:
            if (
                row.get("schema_version") != 1
                or row.get("artifact_type") != "inference_hub_part1_raw_response"
                or row.get("target_id") != target_id
                or row.get("upstream_provider") != subject["upstream_provider"]
                or row.get("model") != subject["model"]
                or row.get("requested_route") != subject["route"]
            ):
                raise FinalResultsError("Part 1 raw row identity binding failed.")
            trial_id = row.get("trial_id")
            raw = row.get("raw_response")
            if (
                not isinstance(trial_id, str) or not trial_id or trial_id in retained
                or not isinstance(raw, Mapping)
                or row.get("raw_response_sha256") != _sha256_json(raw)
                or row.get("response_model") != subject["route"]
                or raw.get("model") != subject["route"]
                or row.get("model_identity_valid") is not True
                or row.get("parsed_action") not in {"X", "Y", None}
                or row.get("format_valid") is not (row.get("parsed_action") in {"X", "Y"})
            ):
                raise FinalResultsError("Part 1 retained response hash/parser/model identity failed.")
            counterbalance = COUNTERBALANCE_BY_ID.get(str(row.get("counterbalance_id")))
            if counterbalance is None:
                raise FinalResultsError("Part 1 counterbalance identity is unknown.")
            cooperative_label = counterbalance.label_for(WELFARE_PRESERVING)
            retained[trial_id] = row
            derived.append({
                "game": row.get("game"), "domain": row.get("domain"),
                "cooperation": row.get("parsed_action") == cooperative_label,
                "format_valid": row.get("format_valid") is True,
            })
        if len(retained) != expected_roots or len({row.get("root_id") for row in journal_rows}) != expected_roots:
            raise FinalResultsError(f"Part 1 {target_id} lacks exact {expected_roots}-root coverage.")
        cell_counts: dict[tuple[object, object], int] = defaultdict(int)
        for row in derived:
            cell_counts[(row["game"], row["domain"])] += 1
        expected_per_cell = expected_roots // 12
        expected_cells = {(game, domain) for game in GAMES for domain in DOMAINS}
        if set(cell_counts) != expected_cells or set(cell_counts.values()) != {expected_per_cell}:
            raise FinalResultsError(
                f"Part 1 {scope} must cover all 12 game-domain strata equally."
            )
        interval = _stratified_bootstrap(
            derived,
            seed=_derive_public_seed(bootstrap_seed, scope, target_id, subject_index),
        )
        output.append({
            "target_id": target_id, "upstream_provider": subject["upstream_provider"],
            "model": subject["model"], "scope": scope, "root_count": expected_roots,
            "format_valid_count": sum(row["format_valid"] for row in derived),
            "format_invalid_count": sum(not row["format_valid"] for row in derived),
            "cooperation": interval,
            "paper_eligible": scope == "full_384",
            "descriptive_reportable": True,
            "stratum_count": 12,
            "roots_per_stratum": expected_per_cell,
        })
        identities[target_id] = (str(subject["upstream_provider"]), str(subject["model"]), str(subject["route"]))
    return output, identities, {
        **_manifest_binding(path, manifest), "scope": scope,
        "root_count": expected_roots,
    }


def _derive_public_seed(base: int, *parts: object) -> int:
    digest = hashlib.sha256(
        json.dumps([base, *parts], separators=(",", ":"), ensure_ascii=False).encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _combine_part1(
    full_paths: Sequence[Path], n96_paths: Sequence[Path], *, bootstrap_seed: int
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str, str]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    identities: dict[str, tuple[str, str, str]] = {}
    bindings: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for scope, paths in (("full_384", full_paths), ("balanced_partial", n96_paths)):
        for path in paths:
            new_rows, new_identities, binding = _part1_manifest(
                path, scope=scope, bootstrap_seed=bootstrap_seed
            )
            for row in new_rows:
                key = (scope, str(row["target_id"]), int(row["root_count"]))
                if key in seen:
                    raise FinalResultsError(
                        f"Part 1 target/root count is duplicated within {scope}: "
                        f"{key[1]} at n={key[2]}."
                    )
                seen.add(key)
            for target_id, identity in new_identities.items():
                prior = identities.get(target_id)
                if prior is not None and prior != identity:
                    raise FinalResultsError("Part 1 identity changed across full/n96 inputs.")
                identities[target_id] = identity
            rows.extend(new_rows)
            bindings.append(binding)
    if not rows:
        raise FinalResultsError("At least one Part 1 manifest is required.")
    preferred = _preferred_part1(rows)
    for row in rows:
        row["preferred_for_descriptive_outputs"] = (
            preferred[str(row["target_id"])] is row
        )
    return rows, identities, bindings


def _part1_scope(manifest: Mapping[str, Any]) -> str:
    observed = manifest.get("executed_trial_count_per_subject")
    trial_limit = manifest.get("trial_limit")
    if observed == 384 and trial_limit is None:
        return "full_384"
    if (
        isinstance(observed, int) and not isinstance(observed, bool)
        and 12 <= observed < 384 and observed % 12 == 0
        and trial_limit == observed
    ):
        return "balanced_partial"
    raise FinalResultsError("Part 1 replacement has an invalid root count/scope.")


def _part1_unavailable_failure(
    manifest: Mapping[str, Any], target_id: str,
) -> dict[str, Any]:
    subject = _subject_routes(manifest)[target_id]
    counts: dict[str, int] = defaultdict(int)
    seen_trials: set[str] = set()
    for row in _journal_rows(manifest, target_id):
        if (
            row.get("schema_version") != 1
            or row.get("artifact_type") != "inference_hub_part1_raw_response"
            or row.get("target_id") != target_id
            or row.get("upstream_provider") != subject["upstream_provider"]
            or row.get("model") != subject["model"]
            or row.get("requested_route") != subject["route"]
        ):
            raise FinalResultsError("Part 1 unavailable-target row binding failed.")
        trial_id = row.get("trial_id")
        if not isinstance(trial_id, str) or not trial_id or trial_id in seen_trials:
            raise FinalResultsError("Part 1 unavailable-target trial binding failed.")
        seen_trials.add(trial_id)
        raw = row.get("raw_response")
        if raw is None:
            failure = row.get("failure")
            if (
                row.get("raw_response_sha256") is not None
                or row.get("response_model") is not None
                or not isinstance(failure, Mapping)
                or not isinstance(failure.get("failure_code"), str)
            ):
                raise FinalResultsError("Part 1 unavailable transport evidence is invalid.")
            counts["transport_failure_without_response"] += 1
            continue
        if (
            not isinstance(raw, Mapping)
            or row.get("raw_response_sha256") != _sha256_json(raw)
            or row.get("response_model") != raw.get("model")
        ):
            raise FinalResultsError("Part 1 unavailable retained evidence is invalid.")
        if row.get("model_identity_valid") is not True:
            counts["response_model_identity_mismatch"] += 1
    _validate_failure_summary_bounds(
        manifest, counts,
        {
            "transport_failure_without_response": "failed_without_response",
            "response_model_identity_mismatch": "response_model_identity_mismatches",
        },
        part="part1",
    )
    return _failure_provenance(target_id, counts, part="part1")


def _combine_part1_overlay(
    full_paths: Sequence[Path], n96_paths: Sequence[Path],
    replacement_paths: Sequence[Path], *, bootstrap_seed: int,
    unavailable_target_ids: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str, str]], list[dict[str, Any]]]:
    unavailable_ids = _explicit_target_ids(
        unavailable_target_ids, label="Part 1 unavailable",
    )
    if not replacement_paths and not unavailable_ids:
        return _combine_part1(full_paths, n96_paths, bootstrap_seed=bootstrap_seed)
    artifact_type = "inference_hub_part1_large_n_exploratory_panel"
    primaries: list[dict[str, Any]] = []
    for scope, paths in (("full_384", full_paths), ("balanced_partial", n96_paths)):
        for path in paths:
            manifest = _manifest(path, artifact_type, require_complete=False)
            if _part1_scope(manifest) != scope:
                raise FinalResultsError(f"Part 1 {scope} manifest has the wrong root count/scope.")
            subjects = _subject_routes(manifest)
            primaries.append({
                "path": path, "scope": scope, "manifest": manifest,
                "subjects": subjects,
                "subject_indices": {
                    target_id: index for index, target_id in enumerate(subjects)
                },
                "replacements": [], "seen_replacements": set(),
                "unavailable_failures": {},
            })
    if not primaries:
        raise FinalResultsError("At least one Part 1 manifest is required.")

    for target_id in unavailable_ids:
        candidates = [
            primary for primary in primaries if target_id in primary["subjects"]
        ]
        if len(candidates) != 1:
            raise FinalResultsError(
                f"Part 1 unavailable target {target_id} must occur in exactly one "
                "primary manifest."
            )
        primary = candidates[0]
        if primary["manifest"].get("complete") is True:
            raise FinalResultsError(
                "Part 1 unavailable targets require an incomplete primary manifest."
            )
        primary["unavailable_failures"][target_id] = _part1_unavailable_failure(
            primary["manifest"], target_id,
        )

    for path in replacement_paths:
        replacement = _manifest(path, artifact_type)
        replacement_scope = _part1_scope(replacement)
        replacement_ids = set(_subject_routes(replacement))
        candidates = [
            primary for primary in primaries
            if primary["scope"] == replacement_scope
            and replacement_ids <= set(primary["subjects"])
            and _overlay_contract(primary["manifest"], part="part1")
            == _overlay_contract(replacement, part="part1")
        ]
        if len(candidates) != 1:
            raise FinalResultsError(
                "Part 1 replacement must match exactly one primary manifest by "
                "scope, targets, schedule, and scientific contract."
            )
        primary = candidates[0]
        if primary["manifest"].get("complete") is True:
            raise FinalResultsError(
                "Part 1 replacements may only repair an incomplete primary manifest."
            )
        validated_ids = _validate_replacement_manifest(
            primary=primary["manifest"], replacement=replacement, part="part1",
            seen_target_ids=primary["seen_replacements"],
        )
        overlap = validated_ids & unavailable_ids
        if overlap:
            raise FinalResultsError(
                "Part 1 target cannot be both replaced and unavailable: "
                + ",".join(sorted(overlap))
            )
        primary["replacements"].append((path, replacement, validated_ids))

    rows: list[dict[str, Any]] = []
    identities: dict[str, tuple[str, str, str]] = {}
    bindings: list[dict[str, Any]] = []
    seen_rows: set[tuple[str, str, int]] = set()
    for primary in primaries:
        path = primary["path"]
        scope = primary["scope"]
        replacements = primary["replacements"]
        unavailable_for_primary = set(primary["unavailable_failures"])
        if not replacements and not unavailable_for_primary:
            new_rows, new_identities, binding = _part1_manifest(
                path, scope=scope, bootstrap_seed=bootstrap_seed,
            )
        else:
            replaced_ids = set(primary["seen_replacements"])
            retained_ids = (
                set(primary["subjects"]) - replaced_ids - unavailable_for_primary
            )
            new_rows, new_identities, primary_binding = _part1_manifest(
                path, scope=scope, bootstrap_seed=bootstrap_seed,
                target_ids=retained_ids, require_complete=False,
                require_summary=False, subject_indices=primary["subject_indices"],
            )
            primary_binding.update({
                "complete": primary["manifest"].get("complete") is True,
                "retained_target_ids": sorted(retained_ids),
            })
            replacement_bindings: list[dict[str, Any]] = []
            for replacement_path, _replacement, replacement_ids in replacements:
                replacement_rows, replacement_identities, replacement_binding = (
                    _part1_manifest(
                        replacement_path, scope=scope, bootstrap_seed=bootstrap_seed,
                        target_ids=replacement_ids,
                        subject_indices=primary["subject_indices"],
                    )
                )
                new_rows.extend(replacement_rows)
                new_identities.update(replacement_identities)
                replacement_bindings.append({
                    **replacement_binding,
                    "replacement_target_ids": sorted(replacement_ids),
                })
            row_by_target = {str(row["target_id"]): row for row in new_rows}
            expected_output_ids = set(primary["subjects"]) - unavailable_for_primary
            if set(row_by_target) != expected_output_ids:
                raise FinalResultsError(
                    "Part 1 overlay does not cover every primary target exactly once."
                )
            new_rows = [
                row_by_target[target_id] for target_id in primary["subjects"]
                if target_id in expected_output_ids
            ]
            binding = {
                "overlay_schema_version": 1,
                "primary": primary_binding,
                "replacements": replacement_bindings,
                "replaced_target_ids": sorted(replaced_ids),
                "unavailable_target_ids": sorted(unavailable_for_primary),
                "unavailable_target_failures": [
                    primary["unavailable_failures"][target_id]
                    for target_id in primary["subjects"]
                    if target_id in unavailable_for_primary
                ],
                "scope": scope,
            }

        for row in new_rows:
            key = (scope, str(row["target_id"]), int(row["root_count"]))
            if key in seen_rows:
                raise FinalResultsError(
                    f"Part 1 target/root count is duplicated within {scope}: "
                    f"{key[1]} at n={key[2]}."
                )
            seen_rows.add(key)
        for target_id, identity in new_identities.items():
            prior = identities.get(target_id)
            if prior is not None and prior != identity:
                raise FinalResultsError("Part 1 identity changed across full/n96 inputs.")
            identities[target_id] = identity
        rows.extend(new_rows)
        bindings.append(binding)

    preferred = _preferred_part1(rows)
    for row in rows:
        row["preferred_for_descriptive_outputs"] = (
            preferred[str(row["target_id"])] is row
        )
    return rows, identities, bindings


def _reject_text_keys(value: Any, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if not normalized.startswith("contains_") and (normalized in _FORBIDDEN_PUBLIC_KEYS or any(
                token in normalized for token in ("prompt_text", "raw_response", "response_text", "visible_response", "reasoning")
            )):
                raise FinalResultsError(f"Sanitized input contains forbidden field at {path}.{key}.")
            _reject_text_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_text_keys(child, path=f"{path}[{index}]")


def _part2_bound_rows(
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], list[Any], list[Any]]:
    refs = manifest.get("sanitized_artifacts")
    model_ref = refs.get("model_metrics") if isinstance(refs, Mapping) else None
    trajectory_ref = refs.get("trajectory_metrics") if isinstance(refs, Mapping) else None
    if not isinstance(model_ref, Mapping) or not isinstance(trajectory_ref, Mapping):
        raise FinalResultsError("Part 2 sanitized metric bindings are absent.")

    def load_bound(reference: Mapping[str, Any], artifact_type: str) -> dict[str, Any]:
        path = Path(str(reference.get("path", "")))
        if _sha256_file(path) != reference.get("file_sha256"):
            raise FinalResultsError("Part 2 sanitized metric file hash changed.")
        value = _read_json(path, artifact_type)
        if value.get("artifact_type") != artifact_type or value.get("evidence_sha256") != _self_hash(value) or value.get("evidence_sha256") != reference.get("evidence_sha256"):
            raise FinalResultsError("Part 2 sanitized metric self-hash/type binding failed.")
        _reject_text_keys(value)
        return value

    model_artifact = load_bound(
        model_ref, "inference_hub_part2_sanitized_model_metrics",
    )
    trajectory_artifact = load_bound(
        trajectory_ref, "inference_hub_part2_sanitized_trajectory_metrics",
    )
    model_rows = model_artifact.get("rows")
    trajectory_rows = trajectory_artifact.get("rows")
    if not isinstance(model_rows, list) or not isinstance(trajectory_rows, list):
        raise FinalResultsError("Part 2 sanitized metrics lack rows.")
    return model_ref, trajectory_ref, model_rows, trajectory_rows


def _part2(
    manifest_path: Path, *, target_ids: set[str] | None = None,
    require_complete: bool = True, require_summary: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str, str]], dict[str, Any]]:
    manifest = _manifest(
        manifest_path, "inference_hub_part2_corrected_matched_panel",
        require_complete=require_complete,
    )
    if require_summary:
        summary = manifest.get("summary")
        if not isinstance(summary, Mapping) or summary.get("identity_mismatch_count") != 0 or summary.get("transport_failure_count") != 0:
            raise FinalResultsError("Part 2 contains an operational or identity failure.")
    subjects = _selected_subject_routes(manifest, target_ids)
    model_ref, trajectory_ref, model_rows, trajectory_rows = _part2_bound_rows(manifest)
    by_target: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in trajectory_rows:
        if not isinstance(row, Mapping):
            raise FinalResultsError("Part 2 trajectory metric is not an object.")
        if target_ids is not None and row.get("target_id") not in subjects:
            continue
        if row.get("operationally_eligible") is not True:
            raise FinalResultsError("Part 2 contains an ineligible trajectory.")
        by_target[str(row.get("target_id"))].append(row)
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    contract = manifest.get("part2_contract")
    if not isinstance(contract, Mapping) or any(
        contract.get(key) != value
        for key, value in {
            "society_size": 5, "days": 12, "resource_capacity": 50,
            "option_a_private_gain": 1, "option_b_private_gain": 2,
            "option_b_reserve_cost": 2,
        }.items()
    ):
        raise FinalResultsError("Part 2 corrected incentive/dynamics contract changed.")
    expected = int(contract["independent_trajectories"])
    for row in model_rows:
        if not isinstance(row, Mapping):
            raise FinalResultsError("Part 2 model metric is not an object.")
        target_id = row.get("target_id")
        if target_ids is not None and target_id not in subjects:
            continue
        if (
            not isinstance(target_id, str) or target_id in seen or target_id not in subjects
            or row.get("complete_matched_panel") is not True
            or row.get("trajectory_count") != expected
            or row.get("eligible_trajectory_count") != expected
            or len(by_target[target_id]) != expected
        ):
            raise FinalResultsError("Part 2 model/trajectory coverage is incomplete or duplicated.")
        subject = subjects[target_id]
        if row.get("upstream_provider") != subject["upstream_provider"] or row.get("model") != subject["model"]:
            raise FinalResultsError("Part 2 sanitized identity differs from its manifest.")
        intervals = row.get("trajectory_level_95_percent_t_intervals")
        required = {"aurc", "aupc", "restraint_rate", "reserve_nondepletion", "final_reserve", "population_retention", "cumulative_private_payoff", "cumulative_group_payoff"}
        if not isinstance(intervals, Mapping) or not required <= set(intervals):
            raise FinalResultsError("Part 2 trajectory interval set is incomplete.")
        for metric in required:
            interval = intervals[metric]
            if not isinstance(interval, Mapping):
                raise FinalResultsError(f"Part 2 {metric} interval is not an object.")
            values = (interval.get("mean"), interval.get("lower"), interval.get("upper"))
            if (
                interval.get("n") != expected
                or not isinstance(interval.get("method"), str)
                or not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in values
                )
                or not float(values[1]) <= float(values[0]) <= float(values[2])
            ):
                raise FinalResultsError(f"Part 2 {metric} interval/count is invalid.")
        nondepletion = intervals["reserve_nondepletion"]
        successes = nondepletion.get("successes")
        if (
            not isinstance(successes, int) or isinstance(successes, bool)
            or not 0 <= successes <= expected
            or not math.isclose(
                float(nondepletion["mean"]), successes / expected,
                rel_tol=0.0, abs_tol=1e-12,
            )
        ):
            raise FinalResultsError("Part 2 nondepletion Wilson count is invalid.")
        invalid_count = row.get("total_invalid_count")
        scheduled_agent_days = row.get("total_scheduled_agent_days")
        for label, value in (
            ("total_invalid_count", invalid_count),
            ("total_scheduled_agent_days", scheduled_agent_days),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise FinalResultsError(f"Part 2 {label} is invalid.")
        if (
            invalid_count is not None and scheduled_agent_days is not None
            and invalid_count > scheduled_agent_days
        ):
            raise FinalResultsError("Part 2 invalid count exceeds scheduled agent-days.")
        output.append({
            "target_id": target_id, "upstream_provider": subject["upstream_provider"],
            "model": subject["model"], "trajectory_count": expected,
            "trajectory_level_95_percent_t_intervals": dict(intervals),
            "total_invalid_count": invalid_count,
            "total_scheduled_agent_days": scheduled_agent_days,
            "paper_eligible": expected >= 12,
        })
        seen.add(target_id)
    if seen != set(subjects):
        raise FinalResultsError("Part 2 sanitized metrics do not cover every manifest subject.")
    identities = {
        target_id: (str(row["upstream_provider"]), str(row["model"]), str(row["route"]))
        for target_id, row in subjects.items()
    }
    return output, identities, {
        **_manifest_binding(manifest_path, manifest),
        "model_metrics_file_sha256": model_ref["file_sha256"],
        "trajectory_metrics_file_sha256": trajectory_ref["file_sha256"],
    }


def _part2_unavailable_failure(
    manifest: Mapping[str, Any], target_id: str,
) -> dict[str, Any]:
    subject = _subject_routes(manifest)[target_id]
    _model_ref, _trajectory_ref, model_rows, trajectory_rows = _part2_bound_rows(manifest)
    selected_trajectories = [
        row for row in trajectory_rows
        if isinstance(row, Mapping) and row.get("target_id") == target_id
    ]
    if len(selected_trajectories) != sum(
        isinstance(row, Mapping) and row.get("target_id") == target_id
        for row in trajectory_rows
    ):
        raise FinalResultsError("Part 2 unavailable trajectory evidence is invalid.")
    trajectory_ids: set[int] = set()
    identity_count = transport_count = 0
    for row in selected_trajectories:
        trajectory_index = row.get("trajectory_index")
        identity = row.get("identity_mismatch_count")
        transport = row.get("transport_failure_count")
        if (
            not isinstance(trajectory_index, int) or isinstance(trajectory_index, bool)
            or trajectory_index in trajectory_ids
            or not isinstance(identity, int) or isinstance(identity, bool) or identity < 0
            or not isinstance(transport, int) or isinstance(transport, bool) or transport < 0
            or row.get("operationally_eligible") is not (identity == transport == 0)
        ):
            raise FinalResultsError("Part 2 unavailable trajectory evidence is invalid.")
        trajectory_ids.add(trajectory_index)
        identity_count += identity
        transport_count += transport
    selected_models = [
        row for row in model_rows
        if isinstance(row, Mapping) and row.get("target_id") == target_id
    ]
    if len(selected_models) != 1:
        raise FinalResultsError("Part 2 unavailable model evidence is absent or duplicated.")
    model_row = selected_models[0]
    eligible_count = sum(
        row.get("operationally_eligible") is True for row in selected_trajectories
    )
    if (
        model_row.get("upstream_provider") != subject["upstream_provider"]
        or model_row.get("model") != subject["model"]
        or model_row.get("trajectory_count") != len(selected_trajectories)
        or model_row.get("eligible_trajectory_count") != eligible_count
        or model_row.get("total_identity_mismatch_count") != identity_count
        or model_row.get("total_transport_failure_count") != transport_count
        or (
            identity_count + transport_count > 0
            and model_row.get("complete_matched_panel") is not False
        )
    ):
        raise FinalResultsError("Part 2 unavailable model failure binding is invalid.")
    counts = {
        "response_model_identity_mismatch": identity_count,
        "transport_failure": transport_count,
    }
    _validate_failure_summary_bounds(
        manifest, counts,
        {
            "response_model_identity_mismatch": "identity_mismatch_count",
            "transport_failure": "transport_failure_count",
        },
        part="part2",
    )
    return _failure_provenance(target_id, counts, part="part2")


def _part2_overlay(
    primary_path: Path, replacement_paths: Sequence[Path],
    unavailable_target_ids: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str, str]], dict[str, Any]]:
    unavailable_ids = _explicit_target_ids(
        unavailable_target_ids, label="Part 2 unavailable",
    )
    if not replacement_paths and not unavailable_ids:
        return _part2(primary_path)
    artifact_type = "inference_hub_part2_corrected_matched_panel"
    primary = _manifest(primary_path, artifact_type, require_complete=False)
    if primary.get("complete") is True:
        raise FinalResultsError(
            "Part 2 replacements or unavailable targets require an incomplete primary manifest."
        )
    primary_subjects = _subject_routes(primary)
    unknown_unavailable = unavailable_ids - set(primary_subjects)
    if unknown_unavailable:
        raise FinalResultsError(
            "Part 2 unavailable target is absent from the primary manifest: "
            + ",".join(sorted(unknown_unavailable))
        )
    unavailable_failures = [
        _part2_unavailable_failure(primary, target_id)
        for target_id in primary_subjects if target_id in unavailable_ids
    ]
    seen_replacements: set[str] = set()
    replacements: list[tuple[Path, dict[str, Any], set[str]]] = []
    for path in replacement_paths:
        replacement = _manifest(path, artifact_type)
        replacement_ids = _validate_replacement_manifest(
            primary=primary, replacement=replacement, part="part2",
            seen_target_ids=seen_replacements,
        )
        replacements.append((path, replacement, replacement_ids))
    overlap = seen_replacements & unavailable_ids
    if overlap:
        raise FinalResultsError(
            "Part 2 target cannot be both replaced and unavailable: "
            + ",".join(sorted(overlap))
        )

    retained_ids = set(primary_subjects) - seen_replacements - unavailable_ids
    if retained_ids:
        rows, identities, primary_binding = _part2(
            primary_path, target_ids=retained_ids,
            require_complete=False, require_summary=False,
        )
    else:
        rows, identities = [], {}
        primary_binding = _manifest_binding(primary_path, primary)
    primary_binding.update({
        "complete": primary.get("complete") is True,
        "retained_target_ids": sorted(retained_ids),
    })
    replacement_bindings: list[dict[str, Any]] = []
    for path, _replacement, replacement_ids in replacements:
        new_rows, new_identities, binding = _part2(
            path, target_ids=replacement_ids,
        )
        rows.extend(new_rows)
        identities.update(new_identities)
        replacement_bindings.append({
            **binding, "replacement_target_ids": sorted(replacement_ids),
        })
    row_by_target = {str(row["target_id"]): row for row in rows}
    expected_output_ids = set(primary_subjects) - unavailable_ids
    if set(row_by_target) != expected_output_ids:
        raise FinalResultsError("Part 2 overlay does not cover every primary target exactly once.")
    rows = [
        row_by_target[target_id] for target_id in primary_subjects
        if target_id in expected_output_ids
    ]
    return rows, identities, {
        "overlay_schema_version": 1,
        "primary": primary_binding,
        "replacements": replacement_bindings,
        "replaced_target_ids": sorted(seen_replacements),
        "unavailable_target_ids": sorted(unavailable_ids),
        "unavailable_target_failures": unavailable_failures,
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        raise FinalResultsError("Spearman inputs must have the same length of at least three.")
    x, y = _average_ranks(left), _average_ranks(right)
    x_mean, y_mean = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x) * sum((b - y_mean) ** 2 for b in y)
    )
    return numerator / denominator if denominator else None


def _cross_axis(
    *, panel_path: Path, part0: Sequence[Mapping[str, Any]],
    part1: Sequence[Mapping[str, Any]], part2: Sequence[Mapping[str, Any]],
    identities: Sequence[Mapping[str, tuple[str, str, str]]],
) -> dict[str, Any]:
    panel = _read_json(panel_path, "cross-axis panel")
    target_ids = panel.get("subject_target_ids")
    reasons: list[str] = []
    if (
        panel.get("schema_version") != 1 or not isinstance(target_ids, list)
        or len(target_ids) != 24 or len(target_ids) != len(set(target_ids))
    ):
        raise FinalResultsError("Cross-axis panel must contain exactly 24 unique systems.")
    expected = set(target_ids)
    p0 = {str(row["target_id"]): row for row in part0 if row.get("paper_eligible") is True}
    p1 = {
        str(row["target_id"]): row for row in part1
        if row.get("scope") == "full_384" and row.get("paper_eligible") is True
    }
    p2 = {str(row["target_id"]): row for row in part2 if row.get("paper_eligible") is True}
    ignored_outside_panel: dict[str, list[str]] = {}
    for label, values in (("part0_human_validated", p0), ("part1_full_384", p1), ("part2_complete_trajectories", p2)):
        missing, extra = sorted(expected - set(values)), sorted(set(values) - expected)
        if missing:
            reasons.append(f"{label}_missing:" + ",".join(missing))
        if extra:
            ignored_outside_panel[label] = extra
    for target_id in target_ids:
        observed = [mapping.get(target_id) for mapping in identities]
        present = [value for value in observed if value is not None]
        if present and any(value != present[0] for value in present[1:]):
            reasons.append(f"identity_mismatch_across_parts:{target_id}")
    if reasons:
        return {
            "status": "not_emitted_fail_closed", "expected_system_count": 24,
            "reasons": reasons, "ignored_outside_panel": ignored_outside_panel,
            "rows": [], "spearman_pairs": [],
        }

    rows: list[dict[str, Any]] = []
    for target_id in target_ids:
        p2_intervals = p2[target_id]["trajectory_level_95_percent_t_intervals"]
        rows.append({
            "target_id": target_id,
            "part0_refusal_rate": p0[target_id]["overall_refusal"]["estimate"],
            "part1_cooperation_rate": p1[target_id]["cooperation"]["estimate"],
            "part2_restraint_rate": p2_intervals["restraint_rate"]["mean"],
            "part2_aurc": p2_intervals["aurc"]["mean"],
        })
    metrics = (
        "part0_refusal_rate", "part1_cooperation_rate",
        "part2_restraint_rate", "part2_aurc",
    )
    pairs = []
    for left_index, left in enumerate(metrics):
        for right in metrics[left_index + 1 :]:
            rho = _spearman(
                [float(row[left]) for row in rows], [float(row[right]) for row in rows]
            )
            pairs.append({
                "left": left, "right": right, "rho": rho, "n_systems": 24,
                "method": "Spearman rank correlation with average ranks for ties",
                "p_value": None,
                "p_value_status": "not_reported_no_independent_system_superpopulation_claim",
            })
    return {
        "status": "emitted_exact_matched_24", "expected_system_count": 24,
        "reasons": [], "ignored_outside_panel": ignored_outside_panel,
        "rows": rows, "spearman_pairs": pairs,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _flatten_part0(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        for condition in row["conditions"]:
            interval = condition["refusal"]
            output.append({
                "target_id": row["target_id"], "upstream_provider": row["upstream_provider"],
                "model": row["model"],
                "condition_type": "reconstructed_response_language_condition_not_translated_prompt",
                "response_language_condition": condition["response_language_condition"],
                "refusal_rate": interval["estimate"], "ci95_lower": interval["lower"],
                "ci95_upper": interval["upper"], "n_prompt_roots": interval["n"],
                "unclear_count": condition["unclear_count"], "invalid_count": condition["invalid_count"],
                "human_validation_complete": row["human_validation_complete"],
            })
    return output


def _flatten_part1(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "target_id": row["target_id"], "upstream_provider": row["upstream_provider"],
        "model": row["model"], "scope": row["scope"], "root_count": row["root_count"],
        "cooperation_rate": row["cooperation"]["estimate"],
        "ci95_lower": row["cooperation"]["lower"], "ci95_upper": row["cooperation"]["upper"],
        "format_valid_count": row["format_valid_count"],
        "format_invalid_count": row["format_invalid_count"],
        "preferred_for_descriptive_outputs": row[
            "preferred_for_descriptive_outputs"
        ],
    } for row in rows]


def _flatten_part2(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        flat = {
            "target_id": row["target_id"], "upstream_provider": row["upstream_provider"],
            "model": row["model"], "trajectory_count": row["trajectory_count"],
        }
        for metric, interval in row["trajectory_level_95_percent_t_intervals"].items():
            flat[f"{metric}_mean"] = interval["mean"]
            flat[f"{metric}_ci95_lower"] = interval["lower"]
            flat[f"{metric}_ci95_upper"] = interval["upper"]
        output.append(flat)
    return output


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
        "_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#",
        "$": r"\$", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _latex_percent_interval(interval: Mapping[str, Any]) -> str:
    """Format a sanitized rate and 95% interval for a paper table."""
    return (
        f"{100 * float(interval['estimate']):.1f}\\% "
        f"[{100 * float(interval['lower']):.1f}, "
        f"{100 * float(interval['upper']):.1f}]"
    )


def _latex_mean_interval(interval: Mapping[str, Any], *, digits: int = 3) -> str:
    """Format a sanitized trajectory mean and 95% interval."""
    return (
        f"{float(interval['mean']):.{digits}f} "
        f"[{float(interval['lower']):.{digits}f}, "
        f"{float(interval['upper']):.{digits}f}]"
    )


def _latex_table_blocks(
    *, caption: str, label: str, column_spec: str, header: str,
    rows: Sequence[str], max_rows: int,
) -> str:
    """Return deterministic, page-sized table* blocks suitable for ``\\input``."""
    if max_rows < 1 or not rows:
        raise FinalResultsError("A publication table cannot be empty.")
    blocks: list[str] = ["% Generated by analysis.build_final_results; do not edit."]
    chunks = [rows[index : index + max_rows] for index in range(0, len(rows), max_rows)]
    for index, chunk in enumerate(chunks):
        continued = " (continued)" if index else ""
        block_label = label if index == 0 else f"{label}-{index + 1}"
        blocks.extend([
            r"\begin{table*}[t]",
            r"\centering",
            f"\\caption{{{caption}{continued}}}",
            f"\\label{{{block_label}}}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3.2pt}",
            r"\resizebox{\textwidth}{!}{%",
            f"\\begin{{tabular}}{{{column_spec}}}",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            *chunk,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ])
    return "\n".join(blocks)


def _write_publication_tables(
    output_dir: Path, part0: Sequence[Mapping[str, Any]],
    part1: Sequence[Mapping[str, Any]], part2: Sequence[Mapping[str, Any]],
) -> None:
    """Emit scope-explicit, sanitized tables matching the locked result schema."""
    sort_key = lambda row: (
        str(row["upstream_provider"]).casefold(), str(row["model"]).casefold(),
        str(row["target_id"]),
    )

    p0_lines: list[str] = []
    for row in sorted(part0, key=sort_key):
        by_condition = {
            str(condition["response_language_condition"]): condition
            for condition in row["conditions"]
        }
        if set(by_condition) != {"english", "chinese", "russian"}:
            raise FinalResultsError(
                "Part 0 paper table requires exactly English, Chinese, and Russian response conditions."
            )
        unclear = sum(int(condition["unclear_count"]) for condition in by_condition.values())
        invalid = sum(int(condition["invalid_count"]) for condition in by_condition.values())
        p0_lines.append(
            " & ".join((
                _latex_escape(str(row["model"])),
                _latex_escape(str(row["upstream_provider"])),
                str(int(row["root_count_per_condition"])),
                _latex_percent_interval(row["overall_refusal"]),
                _latex_percent_interval(by_condition["english"]["refusal"]),
                _latex_percent_interval(by_condition["chinese"]["refusal"]),
                _latex_percent_interval(by_condition["russian"]["refusal"]),
                f"{unclear}/{invalid}",
                "Exploratory",
            )) + r" \\"
        )
    (output_dir / "part0_results_table.tex").write_text(
        _latex_table_blocks(
            caption=(
                "Part 0 exploratory refusal results over reconstructed response-language "
                "conditions. Every request is the same English source request crossed with "
                "an EN, ZH, or RU response instruction; these are not translated-prompt "
                "conditions, and no benign controls are included. Overall intervals use a "
                "prompt-root cluster bootstrap; condition intervals are Wilson 95\\% intervals."
            ),
            label="tab:final-part0", column_spec="llrllllrl",
            header=(
                "System & Developer & Roots/cond. & Overall refusal [95\\% CI] & "
                "EN [95\\% CI] & ZH [95\\% CI] & RU [95\\% CI] & Unclear/invalid & Status"
            ),
            rows=p0_lines, max_rows=26,
        ),
        encoding="utf-8",
    )

    p1_lines: list[str] = []
    for row in sorted(
        part1,
        key=lambda value: (*sort_key(value), int(value["root_count"]), str(value["scope"])),
    ):
        root_count = int(row["root_count"])
        scope = "Full" if row["scope"] == "full_384" else "Balanced partial"
        cooperation = (
            _latex_percent_interval(row["cooperation"])
            if root_count >= 96 else
            f"{100 * float(row['cooperation']['estimate']):.1f}\\% [--]"
        )
        valid_count = int(row["format_valid_count"])
        invalid_count = int(row["format_invalid_count"])
        p1_lines.append(
            " & ".join((
                _latex_escape(str(row["model"])),
                _latex_escape(str(row["upstream_provider"])),
                scope,
                str(root_count),
                f"{int(row['stratum_count'])}/12 ({int(row['roots_per_stratum'])}/stratum)",
                cooperation,
                f"{valid_count}/{root_count} ({100 * valid_count / root_count:.1f}\\%)",
                str(invalid_count),
                "Exploratory",
            )) + r" \\"
        )
    (output_dir / "part1_results_table.tex").write_text(
        _latex_table_blocks(
            caption=(
                "Part 1 exploratory welfare-preserving choices. Every row is one independently "
                "reported execution scope; balanced 12-, balanced 96-, and full 384-root scopes "
                "are never pooled. Bracketed 95\\% stratified root-bootstrap intervals are "
                "reported only for scopes with at least 96 roots."
            ),
            label="tab:final-part1", column_spec="lllrrllll",
            header=(
                "System & Developer & Scope & Roots & Strata (roots each) & "
                "Welfare-preserving [95\\% CI] & Format valid & Invalid & Status"
            ),
            rows=p1_lines, max_rows=28,
        ),
        encoding="utf-8",
    )

    trajectory_counts = {int(row["trajectory_count"]) for row in part2}
    if len(trajectory_counts) != 1:
        raise FinalResultsError("Part 2 paper table requires one matched trajectory count.")
    trajectory_count = next(iter(trajectory_counts))
    p2_lines: list[str] = []
    for row in sorted(part2, key=sort_key):
        intervals = row["trajectory_level_95_percent_t_intervals"]
        nondepletion = intervals["reserve_nondepletion"]
        invalid_count = row.get("total_invalid_count")
        scheduled = row.get("total_scheduled_agent_days")
        invalid_cell = "--" if invalid_count is None else str(int(invalid_count))
        if invalid_count is not None and scheduled is not None:
            invalid_cell = f"{int(invalid_count)}/{int(scheduled)}"
        p2_lines.append(
            " & ".join((
                _latex_escape(str(row["model"])),
                _latex_escape(str(row["upstream_provider"])),
                str(trajectory_count),
                _latex_mean_interval(intervals["aurc"]),
                _latex_mean_interval(intervals["restraint_rate"]),
                (
                    f"{int(nondepletion['successes'])}/{trajectory_count} "
                    f"[{100 * float(nondepletion['lower']):.1f}, "
                    f"{100 * float(nondepletion['upper']):.1f}]\\%"
                ),
                _latex_mean_interval(intervals["aupc"]),
                invalid_cell,
            )) + r" \\"
        )
    degrees_freedom = trajectory_count - 1
    (output_dir / "part2_results_table.tex").write_text(
        _latex_table_blocks(
            caption=(
                f"Part 2 corrected matched-trajectory results. AURC, restraint, and AUPC are "
                f"trajectory means with 95\\% $t_{{{degrees_freedom}}}$ intervals; reserve "
                "nondepletion is a trajectory count with a Wilson 95\\% interval. Invalid "
                "agent-days are shown over scheduled agent-days when the sanitized denominator "
                "is available."
            ),
            label="tab:final-part2", column_spec="llrlllll",
            header=(
                "System & Developer & Traj. & AURC [95\\% CI] & Restraint [95\\% CI] & "
                "Nondepletion [95\\% CI] & AUPC [95\\% CI] & Invalid agent-days"
            ),
            rows=p2_lines, max_rows=26,
        ),
        encoding="utf-8",
    )


def _preferred_part1(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Prefer full coverage, then the largest balanced partial per target."""
    selected: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        target_id = str(row["target_id"])
        prior = selected.get(target_id)
        if prior is None:
            selected[target_id] = row
            continue
        prior_full = prior.get("scope") == "full_384"
        row_full = row.get("scope") == "full_384"
        if row_full and not prior_full:
            selected[target_id] = row
        elif not row_full and not prior_full and int(row["root_count"]) > int(prior["root_count"]):
            selected[target_id] = row
    return selected


def _write_latex(output_dir: Path, part0: Sequence[Mapping[str, Any]], part1: Sequence[Mapping[str, Any]], part2: Sequence[Mapping[str, Any]], cross: Mapping[str, Any]) -> None:
    rows_path = output_dir / "paper_rows.tex"
    macros_path = output_dir / "paper_macros.tex"
    lines = ["% Generated by analysis.build_final_results; do not edit."]
    p0_by_id = {str(row["target_id"]): row for row in part0}
    p1_by_id = _preferred_part1(part1)
    p2_by_id = {str(row["target_id"]): row for row in part2}
    for target_id in sorted(set(p0_by_id) | set(p1_by_id) | set(p2_by_id)):
        name = _latex_escape(target_id)
        p0_value = p0_by_id.get(target_id, {}).get("overall_refusal", {}).get("estimate")
        p1_value = p1_by_id.get(target_id, {}).get("cooperation", {}).get("estimate")
        p2_value = p2_by_id.get(target_id, {}).get("trajectory_level_95_percent_t_intervals", {}).get("restraint_rate", {}).get("mean")
        format_value = lambda value: "--" if value is None else f"{100 * float(value):.1f}"
        lines.append(f"{name} & {format_value(p0_value)} & {format_value(p1_value)} & {format_value(p2_value)} \\\\")
    rows_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    macros = [
        "% Generated by analysis.build_final_results; do not edit.",
        f"\\newcommand{{\\FinalPartZeroSystems}}{{{len(part0)}}}",
        f"\\newcommand{{\\FinalPartOneReportedSystems}}{{{len(p1_by_id)}}}",
        f"\\newcommand{{\\FinalPartOneFullSystems}}{{{sum(row.get('scope') == 'full_384' for row in p1_by_id.values())}}}",
        f"\\newcommand{{\\FinalPartOneBalancedPartialSystems}}{{{sum(row.get('scope') == 'balanced_partial' for row in p1_by_id.values())}}}",
        f"\\newcommand{{\\FinalPartTwoSystems}}{{{len(part2)}}}",
        f"\\newcommand{{\\FinalMatchedCrossAxisSystems}}{{{24 if cross['status'] == 'emitted_exact_matched_24' else 0}}}",
    ]
    macros_path.write_text("\n".join(macros) + "\n", encoding="utf-8")


def _select_figure_rows(
    part0: Sequence[Mapping[str, Any]], part1: Sequence[Mapping[str, Any]],
    part2: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[Mapping[str, Any]]]:
    """Select matched-axis and broad-Part-1 rows without pooling scopes."""
    p0_by_id = {str(row["target_id"]): row for row in part0}
    p2_by_id = {str(row["target_id"]): row for row in part2}
    p1_by_id = _preferred_part1(part1)
    matched_ids = set(p0_by_id) & set(p2_by_id)
    if not matched_ids:
        raise FinalResultsError("The matched-axis figure has no common Part 0/Part 2 systems.")

    def identity_key(target_id: str) -> tuple[str, str, str]:
        row = p0_by_id[target_id]
        return (
            str(row["upstream_provider"]).casefold(),
            str(row["model"]).casefold(), target_id,
        )

    matched = [
        {
            "target_id": target_id,
            "part0": p0_by_id[target_id],
            "part1": p1_by_id.get(target_id),
            "part2": p2_by_id[target_id],
        }
        for target_id in sorted(matched_ids, key=identity_key)
    ]
    broad_part1 = sorted(
        part1,
        key=lambda row: (
            str(row["scope"]), int(row["root_count"]),
            str(row["upstream_provider"]).casefold(),
            str(row["model"]).casefold(), str(row["target_id"]),
        ),
    )
    return matched, broad_part1


def _plot_label(row: Mapping[str, Any]) -> str:
    """Return a model/provider label safe for Matplotlib's mathtext parser."""
    return f"{row['model']} — {row['upstream_provider']}".replace("$", r"\$")


def _draw_rate_point(
    axis: Any, *, estimate: float, lower: float | None, upper: float | None,
    position: int, color: str, marker: str,
) -> None:
    """Draw one rate and an optional independent-unit 95% interval."""
    if lower is None or upper is None:
        axis.plot(
            estimate, position, marker=marker, linestyle="none", color=color,
            markersize=4.4, markeredgewidth=0.8,
        )
        return
    axis.errorbar(
        estimate, position,
        xerr=[[max(0.0, estimate - lower)], [max(0.0, upper - estimate)]],
        fmt=marker, linestyle="none", color=color, ecolor=color,
        markersize=4.4, markeredgewidth=0.8, elinewidth=0.9, capsize=1.8,
    )


def _figures(output_dir: Path, part0: Sequence[Mapping[str, Any]], part1: Sequence[Mapping[str, Any]], part2: Sequence[Mapping[str, Any]], cross: Mapping[str, Any]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    matched, broad_part1 = _select_figure_rows(part0, part1, part2)
    colors = {
        "refusal": "#0072B2", "cooperation": "#E69F00", "restraint": "#009E73",
    }
    style = {
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8.0, "axes.titlesize": 8.5, "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0, "figure.titlesize": 10.0,
        "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    files: list[str] = []
    with plt.rc_context(style):
        height = max(5.4, 0.285 * len(matched) + 1.3)
        fig, axes = plt.subplots(
            1, 3, figsize=(11.2, height), sharey=True, constrained_layout=True,
        )
        positions = list(range(len(matched)))
        labels = [_plot_label(row["part0"]) for row in matched]
        for position, selected in enumerate(matched):
            refusal = selected["part0"]["overall_refusal"]
            _draw_rate_point(
                axes[0], estimate=float(refusal["estimate"]),
                lower=float(refusal["lower"]), upper=float(refusal["upper"]),
                position=position, color=colors["refusal"], marker="o",
            )
            cooperation_row = selected["part1"]
            if cooperation_row is not None:
                cooperation = cooperation_row["cooperation"]
                root_count = int(cooperation_row["root_count"])
                marker = "o" if cooperation_row["scope"] == "full_384" else (
                    "s" if root_count >= 96 else "^"
                )
                _draw_rate_point(
                    axes[1], estimate=float(cooperation["estimate"]),
                    lower=float(cooperation["lower"]) if root_count >= 96 else None,
                    upper=float(cooperation["upper"]) if root_count >= 96 else None,
                    position=position, color=colors["cooperation"], marker=marker,
                )
            restraint = selected["part2"][
                "trajectory_level_95_percent_t_intervals"
            ]["restraint_rate"]
            _draw_rate_point(
                axes[2], estimate=float(restraint["mean"]),
                lower=float(restraint["lower"]), upper=float(restraint["upper"]),
                position=position, color=colors["restraint"], marker="o",
            )
        panel_specs = (
            (axes[0], "Refusal (exploratory)", colors["refusal"]),
            (axes[1], "Cooperation (exploratory; scope-marked)", colors["cooperation"]),
            (axes[2], "Commons restraint (corrected trajectories)", colors["restraint"]),
        )
        for axis, title, color in panel_specs:
            axis.set_xlim(-0.03, 1.03)
            axis.set_xlabel("Rate")
            axis.set_title(title, color=color, fontweight="semibold")
            axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            axis.grid(axis="x", color="#D0D0D0", linewidth=0.55, alpha=0.8)
            axis.set_axisbelow(True)
        axes[0].set_yticks(positions, labels)
        axes[0].invert_yaxis()
        axes[1].text(
            0.5, -0.12,
            "Part 1 markers: full 384 = circle; balanced 96 = square; "
            "balanced 12 = triangle; blank = unavailable",
            transform=axes[1].transAxes, ha="center", va="top", fontsize=6.5,
        )
        fig.suptitle(
            f"Matched {len(matched)}-system axis summary "
            "(alphabetical; no composite ranking)",
            fontweight="semibold",
        )
        for stem in ("matched_panel_axis_summary", "model_axis_summary"):
            for suffix in ("pdf", "png"):
                path = figure_dir / f"{stem}.{suffix}"
                fig.savefig(path, dpi=240, bbox_inches="tight")
                files.append(str(path.resolve()))
        plt.close(fig)

        groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
        for row in broad_part1:
            groups[(str(row["scope"]), int(row["root_count"]))].append(row)
        facet_chunks: list[
            tuple[tuple[str, int], int, int, list[Mapping[str, Any]]]
        ] = []
        max_facet_rows = 26
        for group_key in sorted(
            groups,
            key=lambda key: (key[0] != "full_384", -key[1], key[0]),
        ):
            rows = groups[group_key]
            chunks = [
                rows[index : index + max_facet_rows]
                for index in range(0, len(rows), max_facet_rows)
            ]
            facet_chunks.extend(
                (group_key, index + 1, len(chunks), chunk)
                for index, chunk in enumerate(chunks)
            )
        column_count = min(3, len(facet_chunks))
        row_count = math.ceil(len(facet_chunks) / column_count)
        fig, axes_grid = plt.subplots(
            row_count, column_count,
            figsize=(4.15 * column_count, 7.0 * row_count),
            squeeze=False, constrained_layout=True,
        )
        for axis, (group_key, chunk_index, chunk_count, rows) in zip(
            axes_grid.flat, facet_chunks
        ):
            scope, root_count = group_key
            marker = "o" if scope == "full_384" else (
                "s" if root_count >= 96 else "^"
            )
            for position, row in enumerate(rows):
                interval = row["cooperation"]
                _draw_rate_point(
                    axis, estimate=float(interval["estimate"]),
                    lower=float(interval["lower"]) if root_count >= 96 else None,
                    upper=float(interval["upper"]) if root_count >= 96 else None,
                    position=position, color=colors["cooperation"], marker=marker,
                )
            scope_label = "Full" if scope == "full_384" else "Balanced partial"
            continuation = (
                f" ({chunk_index}/{chunk_count})" if chunk_count > 1 else ""
            )
            axis.set_title(
                f"{scope_label}, n={root_count}{continuation}",
                color=colors["cooperation"], fontweight="semibold",
            )
            axis.set_xlim(-0.03, 1.03)
            axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            axis.set_xlabel("Welfare-preserving choice rate")
            axis.set_yticks(
                range(len(rows)), [_plot_label(row) for row in rows]
            )
            axis.invert_yaxis()
            axis.grid(axis="x", color="#D0D0D0", linewidth=0.55, alpha=0.8)
            axis.set_axisbelow(True)
        for axis in list(axes_grid.flat)[len(facet_chunks):]:
            axis.set_visible(False)
        fig.suptitle(
            "Broad Part 1 exploratory panel by execution scope "
            "(alphabetical within facets; scopes are not pooled or ranked)",
            fontweight="semibold",
        )
        fig.text(
            0.5, -0.025,
            "Horizontal bars are 95% stratified root-bootstrap intervals; "
            "intervals are omitted for n=12.",
            ha="center", va="bottom", fontsize=7.0,
        )
        for suffix in ("pdf", "png"):
            path = figure_dir / f"part1_scope_facets.{suffix}"
            fig.savefig(path, dpi=240, bbox_inches="tight")
            files.append(str(path.resolve()))
        plt.close(fig)

    if cross["status"] == "emitted_exact_matched_24":
        metrics = ["part0_refusal_rate", "part1_cooperation_rate", "part2_restraint_rate", "part2_aurc"]
        matrix = [[1.0 if left == right else _spearman([float(row[left]) for row in cross["rows"]], [float(row[right]) for row in cross["rows"]]) for right in metrics] for left in metrics]
        with plt.rc_context(style):
            fig, axis = plt.subplots(
                figsize=(5.8, 5.0), constrained_layout=True
            )
            image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
            labels = ["Refusal", "Cooperation", "Restraint", "AURC"]
            axis.set_xticks(range(4), labels, rotation=30, ha="right")
            axis.set_yticks(range(4), labels)
            axis.set_title("Gated exact matched-24 Spearman description")
            for i in range(4):
                for j in range(4):
                    value = matrix[i][j]
                    axis.text(
                        j, i, "NA" if value is None else f"{value:.2f}",
                        ha="center", va="center",
                        color=(
                            "white" if value is not None and abs(value) > 0.55
                            else "black"
                        ),
                    )
            fig.colorbar(image, ax=axis, label="Spearman rho")
            for suffix in ("pdf", "png"):
                path = figure_dir / f"cross_axis_spearman.{suffix}"
                fig.savefig(path, dpi=240, bbox_inches="tight")
                files.append(str(path.resolve()))
            plt.close(fig)
    return files


def build_final_results(
    *, part0_manifest: Path, part1_full_manifests: Sequence[Path],
    part1_n96_manifests: Sequence[Path], part2_manifest: Path,
    panel_path: Path, output_dir: Path, bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    part0_replacement_manifests: Sequence[Path] = (),
    part1_replacement_manifests: Sequence[Path] = (),
    part2_replacement_manifests: Sequence[Path] = (),
    part0_unavailable_targets: Sequence[str] = (),
    part1_unavailable_targets: Sequence[str] = (),
    part2_unavailable_targets: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate private inputs and emit only sanitized final-result derivatives."""
    if output_dir.exists():
        raise FinalResultsError("Output directory already exists; final results are immutable.")
    if not part1_full_manifests and not part1_n96_manifests:
        raise FinalResultsError("At least one Part 1 manifest is required.")
    part0_rows, p0_identities, p0_binding = _part0_overlay(
        part0_manifest, part0_replacement_manifests, part0_unavailable_targets,
    )
    part1_rows, p1_identities, p1_bindings = _combine_part1_overlay(
        part1_full_manifests, part1_n96_manifests,
        part1_replacement_manifests, bootstrap_seed=bootstrap_seed,
        unavailable_target_ids=part1_unavailable_targets,
    )
    part2_rows, p2_identities, p2_binding = _part2_overlay(
        part2_manifest, part2_replacement_manifests, part2_unavailable_targets,
    )
    cross = _cross_axis(
        panel_path=panel_path, part0=part0_rows, part1=part1_rows,
        part2=part2_rows, identities=(p0_identities, p1_identities, p2_identities),
    )
    output_dir.mkdir(parents=True)

    p0_csv = _flatten_part0(part0_rows)
    p1_csv = _flatten_part1(part1_rows)
    p2_csv = _flatten_part2(part2_rows)
    _write_csv(
        output_dir / "part0_model_rates.csv", p0_csv,
        (
            "target_id", "upstream_provider", "model", "condition_type",
            "response_language_condition", "refusal_rate", "ci95_lower", "ci95_upper",
            "n_prompt_roots", "unclear_count", "invalid_count", "human_validation_complete",
        ),
    )
    _write_csv(
        output_dir / "part1_model_rates.csv", p1_csv,
        (
            "target_id", "upstream_provider", "model", "scope", "root_count",
            "cooperation_rate", "ci95_lower", "ci95_upper", "format_valid_count",
            "format_invalid_count", "preferred_for_descriptive_outputs",
        ),
    )
    part2_fields = ["target_id", "upstream_provider", "model", "trajectory_count"]
    for metric in (
        "aurc", "aupc", "restraint_rate", "reserve_nondepletion", "final_reserve",
        "population_retention", "cumulative_private_payoff", "cumulative_group_payoff",
    ):
        part2_fields.extend((f"{metric}_mean", f"{metric}_ci95_lower", f"{metric}_ci95_upper"))
    _write_csv(output_dir / "part2_model_metrics.csv", p2_csv, part2_fields)
    if cross["status"] == "emitted_exact_matched_24":
        _write_csv(
            output_dir / "cross_axis_spearman.csv", cross["spearman_pairs"],
            ("left", "right", "rho", "n_systems", "method", "p_value", "p_value_status"),
        )
    _write_latex(output_dir, part0_rows, part1_rows, part2_rows, cross)
    _write_publication_tables(output_dir, part0_rows, part1_rows, part2_rows)
    figures = _figures(output_dir, part0_rows, part1_rows, part2_rows, cross)

    def output_ref(relative: str) -> dict[str, Any]:
        path = output_dir / relative
        return {"path": relative, "file_sha256": _sha256_file(path)}

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "prosocial_readiness_final_sanitized_results",
        "privacy_contract": {
            "contains_prompt_text": False, "contains_response_text": False,
            "contains_reasoning": False, "contains_raw_responses": False,
            "contains_routes": False,
        },
        "parameters": {
            "part0_condition_interpretation": "reconstructed response-language conditions over English source requests; not translated-prompt conditions",
            "part1_scopes_pooled": False,
            "part1_full_root_count": 384,
            "part1_balanced_partial_root_counts": sorted({int(row["root_count"]) for row in part1_rows if row["scope"] == "balanced_partial"}),
            "part2_uncertainty_unit": "independent_trajectory",
            "bootstrap_seed": bootstrap_seed, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "bindings": {
            "part0": p0_binding, "part1": p1_bindings, "part2": p2_binding,
            "cross_axis_panel": {
                "path": panel_path.name,
                "path_scope": "input_panel_basename_only",
                "file_sha256": _sha256_file(panel_path),
                "canonical_sha256": _sha256_json(_read_json(panel_path, "cross-axis panel")),
            },
        },
        "part0": part0_rows, "part1": part1_rows, "part2": part2_rows,
        "cross_axis": cross,
        "outputs": {
            "part0_csv": output_ref("part0_model_rates.csv"),
            "part1_csv": output_ref("part1_model_rates.csv"),
            "part2_csv": output_ref("part2_model_metrics.csv"),
            "cross_axis_csv": output_ref("cross_axis_spearman.csv") if cross["status"] == "emitted_exact_matched_24" else None,
            "latex_rows": output_ref("paper_rows.tex"),
            "latex_macros": output_ref("paper_macros.tex"),
            "latex_part0_table": output_ref("part0_results_table.tex"),
            "latex_part1_table": output_ref("part1_results_table.tex"),
            "latex_part2_table": output_ref("part2_results_table.tex"),
            "figures": [
                output_ref(str(Path(path).relative_to(output_dir.resolve()))) for path in figures
            ],
        },
    }
    _reject_text_keys(artifact)
    _seal(artifact)
    _atomic_json(output_dir / "final_results.json", artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build fail-closed sanitized final results for Safety Beyond Refusal."
    )
    parser.add_argument("--part0-manifest", type=Path, required=True)
    parser.add_argument(
        "--part0-replacement-manifest", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--part0-unavailable-target", action="append", default=[]
    )
    parser.add_argument("--part1-full-manifest", type=Path, action="append", default=[])
    parser.add_argument(
        "--part1-partial-manifest", "--part1-n96-manifest",
        dest="part1_n96_manifest", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--part1-replacement-manifest", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--part1-unavailable-target", action="append", default=[]
    )
    parser.add_argument("--part2-manifest", type=Path, required=True)
    parser.add_argument(
        "--part2-replacement-manifest", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--part2-unavailable-target", action="append", default=[]
    )
    parser.add_argument("--panel-config", type=Path, default=Path("experiments/sota_cross_axis_panel.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = build_final_results(
        part0_manifest=args.part0_manifest,
        part1_full_manifests=args.part1_full_manifest,
        part1_n96_manifests=args.part1_n96_manifest,
        part2_manifest=args.part2_manifest, panel_path=args.panel_config,
        output_dir=args.output_dir, bootstrap_seed=args.bootstrap_seed,
        part0_replacement_manifests=args.part0_replacement_manifest,
        part1_replacement_manifests=args.part1_replacement_manifest,
        part2_replacement_manifests=args.part2_replacement_manifest,
        part0_unavailable_targets=args.part0_unavailable_target,
        part1_unavailable_targets=args.part1_unavailable_target,
        part2_unavailable_targets=args.part2_unavailable_target,
    )
    print(f"Sanitized final results: {args.output_dir / 'final_results.json'}")
    print(f"Matched cross-axis status: {artifact['cross_axis']['status']}")
    return 0


def cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except (FinalResultsError, OSError, ValueError) as error:
        print(f"Final-results build failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
