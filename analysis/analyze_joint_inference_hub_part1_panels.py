"""Fail-closed joint analysis of private hosted exploratory Part 1 panels.

Every input is first validated as a complete, independently bound panel by the
single-panel analyzer.  A one-subject replacement shard may fill only a target
that is operationally quarantined in, or absent from, the base manifests.  The
choice is deliberately made without consulting the target's X/Y observations.

The emitted artifact contains identifiers, hashes, and derived aggregates only.
It is exploratory evidence and can never be promoted to confirmatory or paper
evidence.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from analysis.analyze_inference_hub_part1_panel import (
    BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    EXPECTED_CELLS,
    EXPECTED_ROOTS,
    HostedPart1AnalysisError,
    _atomic_json,
    _counts,
    _descriptive_groups,
    _family,
    _interval,
    _read_chained_journal,
    _read_json,
    _safe_file_stem,
    _seal,
    _sha256_file,
    _sha256_json,
    _subject_routes,
    analyze_panel,
)
from experiments.misc.inference_hub_part1_panel import build_draft_trials
from experiments.part1.confirmatory_design import (
    COUNTERBALANCE_BY_ID,
    DOMAINS,
    GAMES,
    WELFARE_PRESERVING,
)


SCHEMA_VERSION = 1
_IDENTITY_KEYS = ("upstream_provider", "model", "route")


class JointHostedPart1AnalysisError(HostedPart1AnalysisError):
    """The supplied panel collection cannot support a safe joint analysis."""


@dataclass(frozen=True)
class _ValidatedInput:
    role: str
    index: int
    path: Path
    manifest: dict[str, Any]
    analysis: dict[str, Any]
    subjects: dict[str, dict[str, Any]]
    eligible: frozenset[str]
    quarantined: dict[str, dict[str, Any]]
    registry_identities: dict[str, tuple[str, str, str]]

    @property
    def label(self) -> str:
        return f"{self.role}[{self.index}]"


def _identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    values = tuple(row.get(key) for key in _IDENTITY_KEYS)
    if not all(isinstance(value, str) and value for value in values):
        raise JointHostedPart1AnalysisError("A route/provider/model identity is incomplete.")
    return values  # type: ignore[return-value]


def _folded_identity(identity: Sequence[str]) -> tuple[str, ...]:
    return tuple(value.casefold() for value in identity)


def _registry_identities(manifest: Mapping[str, Any]) -> dict[str, tuple[str, str, str]]:
    """Return all compatibility-selected identities in a bound registry."""

    inputs = manifest.get("input_artifacts")
    if not isinstance(inputs, Mapping):
        raise JointHostedPart1AnalysisError("Manifest input bindings disappeared.")
    registry_ref = inputs.get("registry")
    compatibility_ref = inputs.get("compatibility")
    if not isinstance(registry_ref, Mapping) or not isinstance(compatibility_ref, Mapping):
        raise JointHostedPart1AnalysisError("Manifest registry bindings are malformed.")
    registry = _read_json(Path(str(registry_ref.get("path", ""))), "bound registry")
    compatibility = _read_json(
        Path(str(compatibility_ref.get("path", ""))), "bound compatibility"
    )
    registry_rows = registry.get("targets")
    compatibility_rows = compatibility.get("targets")
    if not isinstance(registry_rows, list) or not isinstance(compatibility_rows, list):
        raise JointHostedPart1AnalysisError("Bound registry target lists are malformed.")

    registered: dict[str, Mapping[str, Any]] = {}
    folded_ids: dict[str, str] = {}
    for row in registry_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), str):
            raise JointHostedPart1AnalysisError("Bound registry contains an invalid target.")
        target_id = str(row["id"])
        folded = target_id.casefold()
        if target_id in registered or (folded in folded_ids and folded_ids[folded] != target_id):
            raise JointHostedPart1AnalysisError("Registry ambiguity: target id is duplicated by case.")
        registered[target_id] = row
        folded_ids[folded] = target_id

    selected_routes: dict[str, str] = {}
    for row in compatibility_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("target_id"), str):
            raise JointHostedPart1AnalysisError("Compatibility contains an invalid target.")
        target_id = str(row["target_id"])
        route = row.get("selected_execution_candidate")
        if route is None:
            continue
        if not isinstance(route, str) or not route or target_id in selected_routes:
            raise JointHostedPart1AnalysisError(
                "Registry ambiguity: compatibility-selected route is duplicated or invalid."
            )
        selected_routes[target_id] = route

    result: dict[str, tuple[str, str, str]] = {}
    for target_id, route in selected_routes.items():
        row = registered.get(target_id)
        if row is None:
            raise JointHostedPart1AnalysisError(
                "Registry ambiguity: compatibility target is absent from its registry."
            )
        provider = row.get("upstream_provider")
        model = row.get("model")
        if not isinstance(provider, str) or not provider or not isinstance(model, str) or not model:
            raise JointHostedPart1AnalysisError("Registry identity is incomplete.")
        result[target_id] = (provider, model, route)
    return result


def _binding(item: _ValidatedInput) -> dict[str, Any]:
    bindings = item.analysis["bindings"]
    return {
        "role": item.role,
        "ordered_index": item.index,
        "manifest_path": str(item.path),
        "manifest_file_sha256": bindings["panel_manifest_file_sha256"],
        "manifest_evidence_sha256": bindings["panel_manifest_evidence_sha256"],
        "independent_analysis_evidence_sha256": item.analysis["evidence_sha256"],
        "independent_analysis_canonical_sha256": _sha256_json(item.analysis),
        "source_artifacts": bindings["source_artifacts"],
        "input_artifacts": bindings["input_artifacts"],
    }


def _validate_input(path: Path, *, role: str, index: int, bootstrap_seed: int) -> _ValidatedInput:
    resolved = path.resolve()
    try:
        analysis = analyze_panel(
            resolved,
            bootstrap_seed=bootstrap_seed,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
        )
    except HostedPart1AnalysisError as error:
        raise JointHostedPart1AnalysisError(
            f"Independent validation failed for {role}[{index}]: {error}"
        ) from error
    # Bind the independently produced analysis to the exact bytes re-read here.
    if analysis.get("evidence_sha256") != _sha256_json(
        {key: value for key, value in analysis.items() if key != "evidence_sha256"}
    ):
        raise JointHostedPart1AnalysisError(f"Independent analysis self-hash failed for {role}[{index}].")
    manifest = _read_json(resolved, f"{role} manifest {index}")
    bindings = analysis.get("bindings")
    if (
        not isinstance(bindings, Mapping)
        or bindings.get("panel_manifest_file_sha256") != _sha256_file(resolved)
        or bindings.get("panel_manifest_evidence_sha256") != manifest.get("evidence_sha256")
    ):
        raise JointHostedPart1AnalysisError(f"Manifest changed while validating {role}[{index}].")
    subjects = _subject_routes(manifest)
    eligible = frozenset(
        str(row["target_id"])
        for row in analysis.get("subjects", [])
        if isinstance(row, Mapping) and isinstance(row.get("target_id"), str)
    )
    quarantined_rows = analysis.get("quarantined_targets")
    if not isinstance(quarantined_rows, list):
        raise JointHostedPart1AnalysisError("Independent analysis quarantine report is malformed.")
    quarantined = {
        str(row["target_id"]): dict(row)
        for row in quarantined_rows
        if isinstance(row, Mapping) and isinstance(row.get("target_id"), str)
    }
    if eligible | set(quarantined) != set(subjects) or eligible & set(quarantined):
        raise JointHostedPart1AnalysisError("Independent analysis did not classify every subject once.")
    return _ValidatedInput(
        role=role,
        index=index,
        path=resolved,
        manifest=manifest,
        analysis=analysis,
        subjects=subjects,
        eligible=eligible,
        quarantined=quarantined,
        registry_identities=_registry_identities(manifest),
    )


def _check_schedule(inputs: Sequence[_ValidatedInput]) -> tuple[int, str, str]:
    first = inputs[0].manifest
    expected = (
        first.get("base_seed"),
        first.get("full_primary_schedule_sha256"),
        first.get("executed_schedule_sha256"),
    )
    if isinstance(expected[0], bool) or not isinstance(expected[0], int):
        raise JointHostedPart1AnalysisError("Base seed is invalid.")
    for item in inputs[1:]:
        observed = (
            item.manifest.get("base_seed"),
            item.manifest.get("full_primary_schedule_sha256"),
            item.manifest.get("executed_schedule_sha256"),
        )
        if observed != expected:
            raise JointHostedPart1AnalysisError(
                "Schedule/prompt drift: every manifest must bind the identical 384-root schedule and base seed."
            )
    return expected  # type: ignore[return-value]


def _check_registry_ambiguity(inputs: Sequence[_ValidatedInput]) -> dict[str, tuple[str, str, str]]:
    canonical: dict[str, tuple[str, str, str]] = {}
    folded_targets: dict[str, str] = {}
    for item in inputs:
        for target_id, identity in item.registry_identities.items():
            folded = target_id.casefold()
            prior_target = folded_targets.get(folded)
            if prior_target is not None and prior_target != target_id:
                raise JointHostedPart1AnalysisError(
                    f"Registry ambiguity for case-folded target id {target_id!r}."
                )
            folded_targets[folded] = target_id
            prior = canonical.get(target_id)
            if prior is not None and prior != identity:
                raise JointHostedPart1AnalysisError(
                    f"Registry ambiguity for target {target_id}: route/provider/model identity differs."
                )
            canonical[target_id] = prior or identity

    # Also reject different target ids assigned the same execution route or model identity.
    route_owner: dict[str, str] = {}
    model_owner: dict[tuple[str, str], str] = {}
    relevant = {target_id for item in inputs for target_id in item.subjects}
    for target_id in relevant:
        identity = canonical.get(target_id)
        if identity is None:
            raise JointHostedPart1AnalysisError(
                f"Registry ambiguity: selected target {target_id} lacks a canonical identity."
            )
        provider, model, route = identity
        prior_route = route_owner.setdefault(route.casefold(), target_id)
        prior_model = model_owner.setdefault((provider.casefold(), model.casefold()), target_id)
        if prior_route != target_id or prior_model != target_id:
            raise JointHostedPart1AnalysisError(
                "Registry ambiguity: distinct targets share a route or provider/model identity."
            )
    return canonical


def _check_judge_overlap(inputs: Sequence[_ValidatedInput]) -> None:
    subject_rows = [row for item in inputs for row in item.subjects.values()]
    for item in inputs:
        judge = item.manifest.get("judge_reservation")
        if not isinstance(judge, Mapping):
            raise JointHostedPart1AnalysisError("Judge reservation disappeared.")
        judge_target = str(judge.get("target_id", "")).casefold()
        judge_identity = _folded_identity(_identity(judge))
        for subject in subject_rows:
            subject_target = str(subject.get("target_id", "")).casefold()
            subject_identity = _folded_identity(_identity(subject))
            if (
                subject_target == judge_target
                or subject_identity[2] == judge_identity[2]
                or subject_identity[:2] == judge_identity[:2]
            ):
                raise JointHostedPart1AnalysisError(
                    "Judge overlap: a reserved judge identity appears as a subject across manifests."
                )


def _source_ref(item: _ValidatedInput) -> dict[str, Any]:
    binding = _binding(item)
    return {
        "role": item.role,
        "ordered_index": item.index,
        "manifest_file_sha256": binding["manifest_file_sha256"],
        "manifest_evidence_sha256": binding["manifest_evidence_sha256"],
        "independent_analysis_evidence_sha256": binding[
            "independent_analysis_evidence_sha256"
        ],
        "source_artifacts": binding["source_artifacts"],
    }


def _rows_for(item: _ValidatedInput, target_id: str, trials: Mapping[str, Any]) -> list[dict[str, Any]]:
    journals = item.manifest.get("journals")
    if not isinstance(journals, Mapping) or not isinstance(journals.get("raw_responses"), Mapping):
        raise JointHostedPart1AnalysisError("Validated journal bindings disappeared.")
    raw_ref = journals["raw_responses"].get(target_id)
    path = item.path.parent / "raw_responses" / f"{_safe_file_stem(target_id)}.jsonl"
    rows = _read_chained_journal(path, raw_ref, label=f"joint raw responses for {target_id}")
    derived: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        trial_id = row.get("trial_id")
        if not isinstance(trial_id, str) or trial_id in seen or trial_id not in trials:
            raise JointHostedPart1AnalysisError("Validated subject coverage changed during joint analysis.")
        seen.add(trial_id)
        trial = trials[trial_id]
        counterbalance = COUNTERBALANCE_BY_ID[trial.counterbalance_id]
        action = row.get("parsed_action")
        derived.append(
            {
                "target_id": target_id,
                "game": trial.game,
                "domain": trial.domain,
                "root_id": trial.root_id,
                "action": action,
                "format_valid": action in {"X", "Y"},
                "provider_failure": False,
                "model_identity_mismatch": False,
                "action_x": action == "X",
                "cooperation": action == counterbalance.label_for(WELFARE_PRESERVING),
            }
        )
    if seen != set(trials) or len({row["root_id"] for row in derived}) != EXPECTED_ROOTS:
        raise JointHostedPart1AnalysisError("Validated subject no longer has exact 384-root coverage.")
    return derived


def analyze_joint_panels(
    base_manifest_paths: Sequence[Path],
    replacement_manifest_paths: Sequence[Path] = (),
    *,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Validate, select, and jointly bootstrap an ordered finite panel."""

    if not base_manifest_paths:
        raise JointHostedPart1AnalysisError("At least one base manifest is required.")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise JointHostedPart1AnalysisError("Bootstrap seed must be an integer.")
    if bootstrap_replicates != BOOTSTRAP_REPLICATES:
        raise JointHostedPart1AnalysisError("Joint analysis requires exactly 5,000 bootstrap replicates.")
    paths = [Path(path).resolve() for path in (*base_manifest_paths, *replacement_manifest_paths)]
    if len(paths) != len(set(paths)):
        raise JointHostedPart1AnalysisError("An input manifest was supplied more than once.")

    bases = [
        _validate_input(Path(path), role="base", index=index, bootstrap_seed=bootstrap_seed)
        for index, path in enumerate(base_manifest_paths)
    ]
    replacements = [
        _validate_input(Path(path), role="replacement", index=index, bootstrap_seed=bootstrap_seed)
        for index, path in enumerate(replacement_manifest_paths)
    ]
    inputs = [*bases, *replacements]
    base_seed, compact_schedule_hash, detailed_schedule_hash = _check_schedule(inputs)
    canonical_identities = _check_registry_ambiguity(inputs)
    _check_judge_overlap(inputs)

    replacement_by_target: dict[str, _ValidatedInput] = {}
    for replacement in replacements:
        if len(replacement.subjects) != 1:
            raise JointHostedPart1AnalysisError(
                f"Replacement shard {replacement.index} must contain exactly one subject."
            )
        target_id = next(iter(replacement.subjects))
        if target_id in replacement_by_target:
            raise JointHostedPart1AnalysisError(f"Duplicate replacements for target {target_id}.")
        replacement_by_target[target_id] = replacement

    eligible_base: dict[str, _ValidatedInput] = {}
    base_occurrences: dict[str, list[_ValidatedInput]] = defaultdict(list)
    planned_order: list[str] = []
    for base in bases:
        for target_id in base.subjects:
            if target_id not in planned_order:
                planned_order.append(target_id)
            base_occurrences[target_id].append(base)
            if target_id in base.eligible:
                if target_id in eligible_base:
                    raise JointHostedPart1AnalysisError(
                        f"Overlapping eligible base subject: {target_id}."
                    )
                eligible_base[target_id] = base
    for target_id in replacement_by_target:
        if target_id not in planned_order:
            planned_order.append(target_id)

    selected: dict[str, _ValidatedInput] = {}
    replaced_targets: list[dict[str, Any]] = []
    unavailable_targets: list[dict[str, Any]] = []
    quarantined_targets: list[dict[str, Any]] = []
    for base in bases:
        for target_id, report in base.quarantined.items():
            quarantined_targets.append({**report, "source": _source_ref(base)})

    for target_id in planned_order:
        replacement = replacement_by_target.get(target_id)
        base_source = eligible_base.get(target_id)
        if base_source is not None:
            if replacement is not None:
                raise JointHostedPart1AnalysisError(
                    f"Outcome-selection guard: replacement for aggregate-eligible base target {target_id} is forbidden."
                )
            selected[target_id] = base_source
            continue

        base_status = "operationally_quarantined" if base_occurrences.get(target_id) else "missing"
        if replacement is None:
            unavailable_targets.append(
                {
                    "target_id": target_id,
                    "reason": "no_eligible_base_and_no_replacement",
                    "base_status": base_status,
                    "base_sources": [_source_ref(item) for item in base_occurrences.get(target_id, [])],
                }
            )
            continue

        expected_identity = canonical_identities.get(target_id)
        observed_identity = _identity(replacement.subjects[target_id])
        base_registry_identities = {
            item.registry_identities[target_id]
            for item in bases
            if target_id in item.registry_identities
        }
        if len(base_registry_identities) != 1:
            raise JointHostedPart1AnalysisError(
                f"Replacement target {target_id} has no unambiguous base-registry identity."
            )
        expected_from_base = next(iter(base_registry_identities))
        if (
            expected_identity is None
            or observed_identity != expected_from_base
            or expected_identity != expected_from_base
        ):
            raise JointHostedPart1AnalysisError(
                f"Replacement identity mismatch for {target_id}: exact route/provider/model identity is required."
            )
        if target_id not in replacement.eligible:
            report = replacement.quarantined[target_id]
            quarantined_targets.append({**report, "source": _source_ref(replacement)})
            unavailable_targets.append(
                {
                    "target_id": target_id,
                    "reason": "replacement_operationally_quarantined",
                    "base_status": base_status,
                    "replacement_source": _source_ref(replacement),
                }
            )
            continue
        selected[target_id] = replacement
        replaced_targets.append(
            {
                "target_id": target_id,
                "base_status": base_status,
                "selection_basis": "exact_target_identity_after_base_missing_or_operational_quarantine_only",
                "identity": dict(zip(_IDENTITY_KEYS, observed_identity)),
                "base_sources": [_source_ref(item) for item in base_occurrences.get(target_id, [])],
                "replacement_source": _source_ref(replacement),
            }
        )

    trial_list = list(build_draft_trials(base_seed=base_seed))
    trials = {trial.trial_id: trial for trial in trial_list}
    ordered_subject_ids = [target_id for target_id in planned_order if target_id in selected]
    observations = {
        target_id: _rows_for(selected[target_id], target_id, trials)
        for target_id in ordered_subject_ids
    }
    ordered_cells = [(game, domain) for game in GAMES for domain in DOMAINS]
    roots_by_cell = {
        cell: [trial.root_id for trial in trial_list if (trial.game, trial.domain) == cell]
        for cell in ordered_cells
    }
    rows_by_subject_root = {
        target_id: {str(row["root_id"]): row for row in rows}
        for target_id, rows in observations.items()
    }
    action_draws = {target_id: [] for target_id in ordered_subject_ids}
    cooperation_draws = {target_id: [] for target_id in ordered_subject_ids}
    panel_action_draws: list[float] = []
    panel_cooperation_draws: list[float] = []
    rng = random.Random(bootstrap_seed)
    for _ in range(bootstrap_replicates if ordered_subject_ids else 0):
        action_totals = dict.fromkeys(ordered_subject_ids, 0)
        cooperation_totals = dict.fromkeys(ordered_subject_ids, 0)
        for cell in ordered_cells:
            roots = roots_by_cell[cell]
            sampled_counts = Counter(rng.choices(roots, k=len(roots)))
            for target_id in ordered_subject_ids:
                lookup = rows_by_subject_root[target_id]
                action_totals[target_id] += sum(
                    count * int(lookup[root]["action_x"])
                    for root, count in sampled_counts.items()
                )
                cooperation_totals[target_id] += sum(
                    count * int(lookup[root]["cooperation"])
                    for root, count in sampled_counts.items()
                )
        for target_id in ordered_subject_ids:
            action_draws[target_id].append(action_totals[target_id] / EXPECTED_ROOTS)
            cooperation_draws[target_id].append(cooperation_totals[target_id] / EXPECTED_ROOTS)
        panel_action_draws.append(
            sum(action_totals.values()) / (EXPECTED_ROOTS * len(ordered_subject_ids))
        )
        panel_cooperation_draws.append(
            sum(cooperation_totals.values()) / (EXPECTED_ROOTS * len(ordered_subject_ids))
        )

    subject_results: list[dict[str, Any]] = []
    for target_id in ordered_subject_ids:
        item = selected[target_id]
        subject = item.subjects[target_id]
        rows = observations[target_id]
        counts = _counts(rows)
        per_cell = [
            {
                "game": game,
                "domain": domain,
                **_counts(
                    [row for row in rows if row["game"] == game and row["domain"] == domain]
                ),
            }
            for game, domain in ordered_cells
        ]
        subject_results.append(
            {
                "target_id": target_id,
                "upstream_provider": subject["upstream_provider"],
                "model": subject["model"],
                "route": subject["route"],
                "family": _family(str(subject["model"])),
                "selected_source": _source_ref(item),
                "counts": counts,
                "primary_action_x_rate_95_ci": _interval(
                    counts["primary_action_x_rate_format_invalid_retained_as_non_x"],
                    action_draws[target_id],
                ),
                "primary_cooperation_rate_95_ci": _interval(
                    counts[
                        "primary_cooperation_rate_format_invalid_retained_as_noncooperation"
                    ],
                    cooperation_draws[target_id],
                ),
                "per_game_domain": per_cell,
            }
        )

    panel_action = (
        sum(
            row["counts"]["primary_action_x_rate_format_invalid_retained_as_non_x"]
            for row in subject_results
        )
        / len(subject_results)
        if subject_results
        else None
    )
    panel_cooperation = (
        sum(
            row["counts"][
                "primary_cooperation_rate_format_invalid_retained_as_noncooperation"
            ]
            for row in subject_results
        )
        / len(subject_results)
        if subject_results
        else None
    )
    panel_cells = (
        [
            {
                "game": game,
                "domain": domain,
                "equal_subject_mean_format_valid_rate": sum(
                    subject["per_game_domain"][index]["format_valid_rate"]
                    for subject in subject_results
                )
                / len(subject_results),
                "equal_subject_mean_action_x_rate_format_invalid_retained_as_non_x": sum(
                    subject["per_game_domain"][index][
                        "primary_action_x_rate_format_invalid_retained_as_non_x"
                    ]
                    for subject in subject_results
                )
                / len(subject_results),
                "equal_subject_mean_cooperation_rate_format_invalid_retained_as_noncooperation": sum(
                    subject["per_game_domain"][index][
                        "primary_cooperation_rate_format_invalid_retained_as_noncooperation"
                    ]
                    for subject in subject_results
                )
                / len(subject_results),
            }
            for index, (game, domain) in enumerate(ordered_cells)
        ]
        if subject_results
        else []
    )

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "joint_inference_hub_part1_exploratory_private_panel_analysis",
        "analysis_role": "exploratory_only_no_confirmatory_or_paper_promotion",
        "confirmatory_or_paper_promotion_permitted": False,
        "draft_bank_human_approved": False,
        "privacy_contract": {
            "contains_prompt_text": False,
            "contains_raw_response_or_response_text": False,
            "contains_reasoning": False,
            "contains_only_identifiers_hash_bindings_and_derived_aggregates": True,
        },
        "parameters": {
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_method": "one_joint_shared_root_resampling_with_replacement_within_each_of_12_game_by_domain_strata",
            "confidence_interval": "deterministic_nonparametric_percentile_95",
            "panel_interval_combination": "single_joint_bootstrap_distribution_not_averaged_input_intervals",
            "selection_inputs": "target_identity_and_operational_eligibility_only_never_x_y_outcomes",
            "invalid_handling": "format_invalid_responses_retained_in_primary_denominator_as_non_x_and_noncooperation",
        },
        "schedule_binding": {
            "base_seed": base_seed,
            "root_count": EXPECTED_ROOTS,
            "game_domain_strata": EXPECTED_CELLS,
            "full_primary_schedule_sha256": compact_schedule_hash,
            "executed_schedule_sha256": detailed_schedule_hash,
        },
        "bindings": {
            "ordered_base_inputs": [_binding(item) for item in bases],
            "ordered_replacement_inputs": [_binding(item) for item in replacements],
        },
        "coverage": {
            "base_manifest_count": len(bases),
            "replacement_manifest_count": len(replacements),
            "unique_planned_target_count": len(planned_order),
            "aggregate_eligible_subject_count": len(subject_results),
            "unavailable_target_count": len(unavailable_targets),
            "replaced_target_count": len(replaced_targets),
            "quarantined_source_target_count": len(quarantined_targets),
            "roots_per_subject": EXPECTED_ROOTS,
            "retained_eligible_subject_trial_count": len(subject_results) * EXPECTED_ROOTS,
            "game_domain_strata": EXPECTED_CELLS,
            "roots_per_stratum_per_subject": 32,
            "judge_rows": 0,
        },
        "quarantined_targets": quarantined_targets,
        "unavailable_targets": unavailable_targets,
        "replaced_targets": replaced_targets,
        "subjects": subject_results,
        "overall_equal_subject": (
            {
                "weighting": "joint_finite_panel_equal_subject_weight",
                "subject_count": len(subject_results),
                "primary_action_x_rate_95_ci": _interval(panel_action, panel_action_draws),
                "primary_cooperation_rate_95_ci": _interval(
                    panel_cooperation, panel_cooperation_draws
                ),
                "per_game_domain": panel_cells,
                "inference_status": "exploratory_joint_finite_panel_summary_over_uniquely_selected_operationally_eligible_targets_only",
            }
            if panel_action is not None and panel_cooperation is not None
            else None
        ),
        "provider_descriptive": _descriptive_groups(subject_results, "upstream_provider"),
        "family_descriptive": _descriptive_groups(subject_results, "family"),
        "family_definition": "deterministic_casefolded_model_identifier_marker_v1; unmatched models are other",
        "limitations": [
            "Exploratory draft scenarios were not human-approved.",
            "Replacement eligibility used only exact identity and operational availability, never observed X/Y outcomes.",
            "Provider and family summaries are descriptive and do not establish independent replication.",
            "Operationally unavailable targets contribute no rows to any aggregate or bootstrap.",
            "This artifact cannot be promoted to confirmatory or paper evidence.",
        ],
    }
    _seal(artifact)
    return artifact


# A concise alias for callers that mirror the single-panel analyzer API.
analyze_panels = analyze_joint_panels


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and jointly aggregate ordered private hosted Part 1 panels."
    )
    parser.add_argument("--base-manifest", action="append", type=Path, required=True)
    parser.add_argument("--replacement-manifest", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = analyze_joint_panels(
        args.base_manifest,
        args.replacement_manifest,
        bootstrap_seed=args.bootstrap_seed,
    )
    _atomic_json(args.output, artifact)
    print(
        f"Validated {artifact['coverage']['base_manifest_count']} base manifests and "
        f"{artifact['coverage']['replacement_manifest_count']} replacement shards; "
        f"{artifact['coverage']['aggregate_eligible_subject_count']} unique subjects are aggregate-eligible."
    )
    print(f"Exploratory joint aggregate artifact: {args.output}")
    return 0


def cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except (HostedPart1AnalysisError, OSError, ValueError) as error:
        print(f"Joint hosted Part 1 analysis failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
