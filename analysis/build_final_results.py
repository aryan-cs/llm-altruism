"""Build fail-closed, text-free paper artifacts from completed hosted panels.

This module never emits prompts, visible responses, reasoning, raw responses, or
routes.  Part 0's reconstructed conditions are labelled response-language
conditions, Part 1's 384-root and n=96 scopes are never pooled, and Part 2 uses
independent trajectories as its uncertainty unit.  Cross-axis correlations are
created only for the exact frozen 24-system panel when every gate passes.
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
from experiments.part1.confirmatory_design import COUNTERBALANCE_BY_ID, WELFARE_PRESERVING


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


def _manifest(path: Path, artifact_type: str) -> dict[str, Any]:
    value = _read_json(path.resolve(), f"{artifact_type} manifest")
    if (
        value.get("schema_version") != 1
        or value.get("artifact_type") != artifact_type
        or value.get("evidence_sha256") != _self_hash(value)
    ):
        raise FinalResultsError(f"{artifact_type} manifest schema/type/self-hash failed.")
    if value.get("complete") is not True:
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


def _part0(path: Path) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str, str]], dict[str, Any]]:
    manifest = _manifest(path, "inference_hub_part0_accelerated_private_panel")
    if manifest.get("stimulus_reconstruction") != (
        "archived_english_source_prompt_crossed_with_explicit_response_language_instruction;"
        "legacy_translated_input_not_retained"
    ):
        raise FinalResultsError("Part 0 reconstructed-condition provenance changed.")
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
    subjects = _subject_routes(manifest)
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
        all_rows = list(terminal.values())
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
        "manifest_path": str(path.resolve()), "file_sha256": _sha256_file(path),
        "evidence_sha256": manifest["evidence_sha256"],
        "human_validation_complete": manifest.get("human_validation_complete") is True,
    }


def _part1_manifest(
    path: Path, *, scope: str, bootstrap_seed: int
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str, str]], dict[str, Any]]:
    if scope not in {"full_384", "n96_shard"}:
        raise FinalResultsError("Unknown Part 1 scope.")
    manifest = _manifest(path, "inference_hub_part1_large_n_exploratory_panel")
    expected_roots = 384 if scope == "full_384" else 96
    if (
        manifest.get("executed_trial_count_per_subject") != expected_roots
        or (scope == "full_384" and manifest.get("trial_limit") is not None)
        or (scope == "n96_shard" and manifest.get("trial_limit") != 96)
    ):
        raise FinalResultsError(f"Part 1 {scope} manifest has the wrong root count/scope.")
    summary = manifest.get("summary")
    if not isinstance(summary, Mapping) or summary.get("failed_without_response") != 0 or summary.get("response_model_identity_mismatches") != 0:
        raise FinalResultsError("Part 1 contains an operational or identity failure.")
    subjects = _subject_routes(manifest)
    output: list[dict[str, Any]] = []
    identities: dict[str, tuple[str, str, str]] = {}
    for subject_index, (target_id, subject) in enumerate(subjects.items()):
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
        expected_per_cell = 32 if scope == "full_384" else 8
        if len(cell_counts) != 12 or set(cell_counts.values()) != {expected_per_cell}:
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
        "manifest_path": str(path.resolve()), "file_sha256": _sha256_file(path),
        "evidence_sha256": manifest["evidence_sha256"], "scope": scope,
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
    seen: set[tuple[str, str]] = set()
    for scope, paths in (("full_384", full_paths), ("n96_shard", n96_paths)):
        for path in paths:
            new_rows, new_identities, binding = _part1_manifest(
                path, scope=scope, bootstrap_seed=bootstrap_seed
            )
            for row in new_rows:
                key = (scope, str(row["target_id"]))
                if key in seen:
                    raise FinalResultsError(f"Part 1 target is duplicated within {scope}: {key[1]}.")
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


def _part2(manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str, str]], dict[str, Any]]:
    manifest = _manifest(manifest_path, "inference_hub_part2_corrected_matched_panel")
    summary = manifest.get("summary")
    if not isinstance(summary, Mapping) or summary.get("identity_mismatch_count") != 0 or summary.get("transport_failure_count") != 0:
        raise FinalResultsError("Part 2 contains an operational or identity failure.")
    subjects = _subject_routes(manifest)
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

    model_artifact = load_bound(model_ref, "inference_hub_part2_sanitized_model_metrics")
    trajectory_artifact = load_bound(trajectory_ref, "inference_hub_part2_sanitized_trajectory_metrics")
    model_rows = model_artifact.get("rows")
    trajectory_rows = trajectory_artifact.get("rows")
    if not isinstance(model_rows, list) or not isinstance(trajectory_rows, list):
        raise FinalResultsError("Part 2 sanitized metrics lack rows.")
    by_target: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in trajectory_rows:
        if not isinstance(row, Mapping) or row.get("operationally_eligible") is not True:
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
        output.append({
            "target_id": target_id, "upstream_provider": subject["upstream_provider"],
            "model": subject["model"], "trajectory_count": expected,
            "trajectory_level_95_percent_t_intervals": dict(intervals),
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
        "manifest_path": str(manifest_path.resolve()), "file_sha256": _sha256_file(manifest_path),
        "evidence_sha256": manifest["evidence_sha256"],
        "model_metrics_file_sha256": model_ref["file_sha256"],
        "trajectory_metrics_file_sha256": trajectory_ref["file_sha256"],
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
    replacements = {"\\": r"\textbackslash{}", "_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#"}
    return "".join(replacements.get(char, char) for char in value)


def _preferred_part1(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Prefer full coverage, otherwise retain balanced n=96 descriptive rows."""
    selected: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        target_id = str(row["target_id"])
        prior = selected.get(target_id)
        if prior is None or (
            prior.get("scope") != "full_384" and row.get("scope") == "full_384"
        ):
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
        f"\\newcommand{{\\FinalPartOneBalancedNinetySixSystems}}{{{sum(row.get('scope') == 'n96_shard' for row in p1_by_id.values())}}}",
        f"\\newcommand{{\\FinalPartTwoSystems}}{{{len(part2)}}}",
        f"\\newcommand{{\\FinalMatchedCrossAxisSystems}}{{{24 if cross['status'] == 'emitted_exact_matched_24' else 0}}}",
    ]
    macros_path.write_text("\n".join(macros) + "\n", encoding="utf-8")


def _figures(output_dir: Path, part0: Sequence[Mapping[str, Any]], part1: Sequence[Mapping[str, Any]], part2: Sequence[Mapping[str, Any]], cross: Mapping[str, Any]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    preferred_part1 = _preferred_part1(part1)
    ids = sorted(set(str(row["target_id"]) for row in part0) | set(preferred_part1) | set(str(row["target_id"]) for row in part2))
    maps = (
        ("Refusal", {str(row["target_id"]): row["overall_refusal"]["estimate"] for row in part0}),
        ("Cooperation", {target_id: row["cooperation"]["estimate"] for target_id, row in preferred_part1.items()}),
        ("Restraint", {str(row["target_id"]): row["trajectory_level_95_percent_t_intervals"]["restraint_rate"]["mean"] for row in part2}),
    )
    height = max(4.0, 0.28 * len(ids))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, height), sharey=True)
    positions = list(range(len(ids)))
    for axis, (label, values) in zip(axes, maps):
        xs = [values.get(target_id, math.nan) for target_id in ids]
        axis.scatter(xs, positions, s=22)
        axis.set_xlim(-0.03, 1.03)
        axis.set_xlabel(f"{label} rate")
        axis.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(positions, ids, fontsize=7)
    fig.tight_layout()
    files = []
    for suffix in ("pdf", "png"):
        path = figure_dir / f"model_axis_summary.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        files.append(str(path.resolve()))
    plt.close(fig)
    if cross["status"] == "emitted_exact_matched_24":
        metrics = ["part0_refusal_rate", "part1_cooperation_rate", "part2_restraint_rate", "part2_aurc"]
        matrix = [[1.0 if left == right else _spearman([float(row[left]) for row in cross["rows"]], [float(row[right]) for row in cross["rows"]]) for right in metrics] for left in metrics]
        fig, axis = plt.subplots(figsize=(6.4, 5.4))
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        labels = ["Refusal", "Cooperation", "Restraint", "AURC"]
        axis.set_xticks(range(4), labels, rotation=30, ha="right")
        axis.set_yticks(range(4), labels)
        for i in range(4):
            for j in range(4):
                value = matrix[i][j]
                axis.text(j, i, "NA" if value is None else f"{value:.2f}", ha="center", va="center", color="white" if value is not None and abs(value) > 0.55 else "black")
        fig.colorbar(image, ax=axis, label="Spearman rho")
        fig.tight_layout()
        for suffix in ("pdf", "png"):
            path = figure_dir / f"cross_axis_spearman.{suffix}"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            files.append(str(path.resolve()))
        plt.close(fig)
    return files


def build_final_results(
    *, part0_manifest: Path, part1_full_manifests: Sequence[Path],
    part1_n96_manifests: Sequence[Path], part2_manifest: Path,
    panel_path: Path, output_dir: Path, bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate private inputs and emit only sanitized final-result derivatives."""
    if output_dir.exists():
        raise FinalResultsError("Output directory already exists; final results are immutable.")
    if not part1_full_manifests and not part1_n96_manifests:
        raise FinalResultsError("At least one Part 1 manifest is required.")
    part0_rows, p0_identities, p0_binding = _part0(part0_manifest)
    part1_rows, p1_identities, p1_bindings = _combine_part1(
        part1_full_manifests, part1_n96_manifests, bootstrap_seed=bootstrap_seed
    )
    part2_rows, p2_identities, p2_binding = _part2(part2_manifest)
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
            "format_invalid_count",
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
            "part1_full_root_count": 384, "part1_shard_root_count": 96,
            "part2_uncertainty_unit": "independent_trajectory",
            "bootstrap_seed": bootstrap_seed, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "bindings": {
            "part0": p0_binding, "part1": p1_bindings, "part2": p2_binding,
            "cross_axis_panel": {
                "path": str(panel_path.resolve()), "file_sha256": _sha256_file(panel_path),
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
    parser.add_argument("--part1-full-manifest", type=Path, action="append", default=[])
    parser.add_argument("--part1-n96-manifest", type=Path, action="append", default=[])
    parser.add_argument("--part2-manifest", type=Path, required=True)
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
