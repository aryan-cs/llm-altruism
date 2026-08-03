"""Blinded human validation for the accelerated private Part 0 panel.

This module adapts the immutable accelerated JSONL evidence to the existing
``analysis.judge_audit`` two-annotator workflow.  It never calls a model and
never generates, infers, or prefills a human label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from analysis import judge_audit
from experiments.misc.inference_hub_part0_panel import (
    LABELS,
    Part0Trial,
    _record_indexes,
)
from experiments.misc.inference_hub_part1_panel import (
    _ChainedJournal,
    _safe_file_stem,
    _self_hash,
    _sha256_file,
    _sha256_json,
    _validate_checkpoint_reference,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "inference_hub_part0_accelerated_private_panel"
WORKFLOW = "accelerated_part0_two_annotator_human_validation"
QUALIFICATION_WORKFLOW = "accelerated_part0_human_language_qualifications"
ANNOTATOR_IDS = ("annotator_1", "annotator_2")
DEFAULT_DUPLICATES = 360
DEFAULT_SEED = 20260803
_PLACEHOLDER_VALUES = {"", "todo", "tbd", "unknown", "replace_me", "placeholder"}


class AcceleratedPart0HumanValidationError(ValueError):
    """Raised when the accelerated audit cannot proceed without guessing."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AcceleratedPart0HumanValidationError(f"Unreadable {label}: {path}") from error
    if not isinstance(value, dict):
        raise AcceleratedPart0HumanValidationError(f"{label} must be a JSON object: {path}")
    return value


def _artifact_hash(payload: Mapping[str, Any], field: str) -> str:
    return _sha256_json({key: value for key, value in payload.items() if key != field})


def _require_private_path(path: Path, label: str) -> None:
    if not path.is_file():
        raise AcceleratedPart0HumanValidationError(f"Missing {label}: {path}")
    if os.name == "posix" and (path.stat().st_mode & 0o077):
        raise AcceleratedPart0HumanValidationError(
            f"{label} must not grant group/other permissions: {path}"
        )


def _trial_from_manifest(row: Mapping[str, Any]) -> Part0Trial:
    required_strings = (
        "trial_id",
        "root_id",
        "language",
        "prompt_sha256",
        "source_stratum",
    )
    if any(not isinstance(row.get(field), str) or not row[field] for field in required_strings):
        raise AcceleratedPart0HumanValidationError("Accelerated schedule row is incomplete.")
    memberships = row.get("source_memberships")
    seed = row.get("generation_seed")
    if (
        not isinstance(memberships, list)
        or not all(isinstance(value, str) and value for value in memberships)
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise AcceleratedPart0HumanValidationError("Accelerated schedule metadata is invalid.")
    return Part0Trial(
        trial_id=str(row["trial_id"]),
        root_id=str(row["root_id"]),
        language=str(row["language"]),
        prompt_text="",
        prompt_sha256=str(row["prompt_sha256"]),
        source_stratum=str(row["source_stratum"]),
        source_memberships=tuple(memberships),
        generation_seed=seed,
    )


def _human_item_id(target_id: str, trial_id: str, response_sha256: str) -> str:
    payload = f"{target_id}\0{trial_id}\0{response_sha256}".encode("utf-8")
    return "ap0_" + hashlib.sha256(payload).hexdigest()[:24]


def _label_for_audit(label: str) -> str:
    return {
        "COMPLIANCE": "complied",
        "REFUSAL": "denied",
        "UNCLEAR": "unclear",
    }[label]


def load_accelerated_population(
    run_dir: str | Path,
) -> tuple[list[judge_audit.RawAuditRow], dict[str, Any]]:
    """Revalidate the sealed accelerated artifact and return judge-scored rows.

    Operationally retired subjects are excluded as whole subjects.  For the
    remaining subjects, every valid fixed-judge label is included (a census,
    not an outcome-selected sample); protocol-invalid rows remain explicit in
    the returned metadata and cannot become human labels.
    """

    root = Path(run_dir).expanduser().resolve()
    manifest_path = root / "private" / "manifest.json"
    _require_private_path(manifest_path, "accelerated Part 0 manifest")
    manifest = _read_json(manifest_path, "accelerated Part 0 manifest")
    if manifest.get("artifact_type") != ARTIFACT_TYPE or manifest.get("schema_version") != 1:
        raise AcceleratedPart0HumanValidationError(
            "Input is not the supported accelerated Part 0 artifact."
        )
    if manifest.get("evidence_sha256") != _self_hash(manifest):
        raise AcceleratedPart0HumanValidationError("Accelerated manifest self-hash failed.")
    if manifest.get("judge_scoring_input") != "visible_subject_response_only_no_hidden_reasoning":
        raise AcceleratedPart0HumanValidationError("Automated judge input contract changed.")
    languages = manifest.get("languages")
    if languages != list(judge_audit.TARGET_LANGUAGES):
        raise AcceleratedPart0HumanValidationError("Accelerated language roster changed.")

    schedule_raw = manifest.get("schedule")
    if not isinstance(schedule_raw, list) or not schedule_raw:
        raise AcceleratedPart0HumanValidationError("Accelerated schedule is absent.")
    if manifest.get("schedule_sha256") != _sha256_json(schedule_raw):
        raise AcceleratedPart0HumanValidationError("Accelerated schedule hash failed.")
    trials = [_trial_from_manifest(row) for row in schedule_raw if isinstance(row, Mapping)]
    if len(trials) != len(schedule_raw) or len({trial.trial_id for trial in trials}) != len(trials):
        raise AcceleratedPart0HumanValidationError("Accelerated schedule identities are invalid.")
    trials_by_id = {trial.trial_id: trial for trial in trials}

    subjects = manifest.get("subject_routes")
    checkpoints = manifest.get("journals")
    raw_checkpoints = checkpoints.get("raw_responses") if isinstance(checkpoints, Mapping) else None
    summaries = manifest.get("summary", {}).get("targets")
    if (
        not isinstance(subjects, list)
        or not subjects
        or not isinstance(raw_checkpoints, Mapping)
        or not isinstance(summaries, list)
    ):
        raise AcceleratedPart0HumanValidationError("Accelerated subject evidence is incomplete.")
    summary_by_target = {
        str(row.get("target_id")): row for row in summaries if isinstance(row, Mapping)
    }
    if len(summary_by_target) != len(summaries):
        raise AcceleratedPart0HumanValidationError("Accelerated target summaries repeat identities.")

    expected_target_ids = [str(subject.get("target_id", "")) for subject in subjects]
    if (
        any(not value for value in expected_target_ids)
        or len(set(expected_target_ids)) != len(expected_target_ids)
        or set(raw_checkpoints) != set(expected_target_ids)
        or set(summary_by_target) != set(expected_target_ids)
    ):
        raise AcceleratedPart0HumanValidationError("Accelerated target lineage is inconsistent.")

    judge = manifest.get("judge")
    if not isinstance(judge, Mapping) or any(
        not isinstance(judge.get(field), str) or not judge[field]
        for field in ("target_id", "route", "model", "upstream_provider")
    ):
        raise AcceleratedPart0HumanValidationError("Fixed automated judge identity is absent.")
    if str(judge["target_id"]) in set(expected_target_ids):
        raise AcceleratedPart0HumanValidationError("Automated judge overlaps a subject target.")

    population: list[judge_audit.RawAuditRow] = []
    excluded_subjects: list[str] = []
    invalid_by_subject_language: Counter[tuple[str, str]] = Counter()
    source_files: list[dict[str, Any]] = []
    for subject in subjects:
        if not isinstance(subject, Mapping):
            raise AcceleratedPart0HumanValidationError("Accelerated subject route is invalid.")
        target_id = str(subject["target_id"])
        expected_path = root / "private" / "raw_responses" / f"{_safe_file_stem(target_id)}.jsonl"
        _require_private_path(expected_path, f"raw journal for {target_id}")
        reference = raw_checkpoints[target_id]
        if not isinstance(reference, Mapping) or Path(str(reference.get("path", ""))).resolve() != expected_path:
            raise AcceleratedPart0HumanValidationError(
                f"Raw journal path binding changed for {target_id}."
            )
        try:
            journal = _ChainedJournal(expected_path)
            _validate_checkpoint_reference(journal, reference, label=f"raw responses for {target_id}")
            response_rows, batch_rows, terminal_rows = _record_indexes(
                journal, subject=subject, trials=trials_by_id
            )
        except (RuntimeError, ValueError) as error:
            raise AcceleratedPart0HumanValidationError(
                f"Accelerated journal validation failed for {target_id}: {error}"
            ) from error
        summary = summary_by_target[target_id]
        retired = int(summary.get("operationally_retired_units", -1))
        if retired < 0 or int(summary.get("retained_units", -1)) != len(terminal_rows):
            raise AcceleratedPart0HumanValidationError(
                f"Accelerated target summary changed for {target_id}."
            )
        observed_outcomes = Counter(str(row.get("outcome")) for row in terminal_rows.values())
        recorded_outcomes = summary.get("outcomes")
        if not isinstance(recorded_outcomes, Mapping) or any(
            int(recorded_outcomes.get(label, -1)) != observed_outcomes[label]
            for label in (*LABELS, "INVALID")
        ):
            raise AcceleratedPart0HumanValidationError(
                f"Accelerated outcome summary changed for {target_id}."
            )
        if retired:
            excluded_subjects.append(target_id)
            continue
        if len(terminal_rows) != len(trials):
            raise AcceleratedPart0HumanValidationError(
                f"Non-retired subject is missing scheduled units: {target_id}."
            )

        record_lines = {
            str(row["record_sha256"]): index
            for index, row in enumerate(journal.records, start=1)
        }
        for trial in trials:
            terminal = terminal_rows[trial.trial_id]
            outcome = str(terminal.get("outcome"))
            if outcome == "INVALID":
                invalid_by_subject_language[(target_id, trial.language)] += 1
                continue
            if outcome not in LABELS or terminal.get("judge_label") != outcome:
                raise AcceleratedPart0HumanValidationError(
                    f"Fixed judge label binding failed for {target_id}/{trial.trial_id}."
                )
            if (
                terminal.get("judge_dispatched") is not True
                or terminal.get("judge_failure") not in (None, {})
            ):
                raise AcceleratedPart0HumanValidationError(
                    f"Fixed judge evidence is invalid for {target_id}/{trial.trial_id}."
                )
            batch_id = terminal.get("judge_batch_id")
            batch = batch_rows.get(str(batch_id))
            format_valid = terminal.get("judge_format_valid") is True
            parsed_labels = batch.get("parsed_labels") if batch is not None else None
            if (
                batch is None
                or batch.get("model_identity_valid") is not True
                or batch.get("requested_route") != judge["route"]
                or batch.get("response_model") != judge["route"]
                or batch.get("format_valid") is not format_valid
                or (
                    format_valid
                    and (
                        not isinstance(parsed_labels, Mapping)
                        or parsed_labels.get(trial.trial_id) != outcome
                    )
                )
                or (not format_valid and (outcome != "UNCLEAR" or parsed_labels is not None))
            ):
                raise AcceleratedPart0HumanValidationError(
                    f"Fixed judge batch lineage failed for {target_id}/{trial.trial_id}."
                )
            response = response_rows.get(trial.trial_id)
            if response is None or response.get("model_identity_valid") is not True:
                raise AcceleratedPart0HumanValidationError(
                    f"Subject response lineage failed for {target_id}/{trial.trial_id}."
                )
            visible = response.get("visible_response")
            if (
                not isinstance(visible, str)
                or not visible
                or response.get("visible_response_sha256")
                != hashlib.sha256(visible.encode("utf-8")).hexdigest()
            ):
                raise AcceleratedPart0HumanValidationError(
                    f"Visible response hash failed for {target_id}/{trial.trial_id}."
                )
            prompt = response.get("prompt_text")
            if (
                not isinstance(prompt, str)
                or not prompt
                or hashlib.sha256(prompt.encode("utf-8")).hexdigest() != trial.prompt_sha256
            ):
                raise AcceleratedPart0HumanValidationError(
                    f"Prompt hash failed for {target_id}/{trial.trial_id}."
                )
            response_sha256 = str(response["visible_response_sha256"])
            population.append(
                judge_audit.RawAuditRow(
                    item_id=_human_item_id(target_id, trial.trial_id, response_sha256),
                    language=trial.language,
                    automated_label=_label_for_audit(outcome),
                    provider=str(subject["upstream_provider"]),
                    model=str(subject["model"]),
                    target_id=target_id,
                    arm="harmful",
                    source=trial.source_stratum,
                    category=trial.source_stratum,
                    semantic_cluster_id=trial.root_id,
                    generation_block=1,
                    request_original=prompt,
                    request_english=prompt,
                    response_original=visible,
                    response_english=visible if trial.language == "english" else "",
                    source_file=str(manifest_path),
                    source_row=record_lines[str(response["record_sha256"])],
                )
            )
        source_files.append(
            {
                "target_id": target_id,
                "path": str(expected_path),
                "sha256": _sha256_file(expected_path),
                "record_count": len(journal.records),
                "tail_record_sha256": journal.tail,
            }
        )

    if not population:
        raise AcceleratedPart0HumanValidationError("No valid fixed-judge rows are auditable.")
    if len({row.item_id for row in population}) != len(population):
        raise AcceleratedPart0HumanValidationError("Accelerated audit item identities collide.")
    population.sort(key=lambda row: row.item_id)
    metadata = {
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
            "evidence_sha256": manifest["evidence_sha256"],
        },
        "fixed_judge": {
            key: judge[key]
            for key in ("target_id", "route", "model", "upstream_provider")
        },
        "included_subject_ids": sorted({row.target_id for row in population}),
        "operationally_excluded_subject_ids": sorted(excluded_subjects),
        "invalid_rows_by_subject_language": {
            f"{target_id}/{language}": count
            for (target_id, language), count in sorted(invalid_by_subject_language.items())
        },
        "source_journals": source_files,
    }
    return population, metadata


def _proportional_duplicate_rows(
    rows: Sequence[judge_audit.RawAuditRow], count: int, seed: int
) -> list[judge_audit.RawAuditRow]:
    if count <= 0 or count > len(rows):
        raise AcceleratedPart0HumanValidationError(
            "Duplicate count must be positive and no larger than the audit population."
        )
    grouped: defaultdict[tuple[str, str], list[judge_audit.RawAuditRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.target_id, row.language)].append(row)
    exact = {key: count * len(values) / len(rows) for key, values in grouped.items()}
    allocation = {key: int(exact[key]) for key in grouped}
    remaining = count - sum(allocation.values())
    order = sorted(grouped, key=lambda key: (-(exact[key] - allocation[key]), key))
    for key in order[:remaining]:
        allocation[key] += 1
    selected: list[judge_audit.RawAuditRow] = []
    for key, values in sorted(grouped.items()):
        ranked = sorted(
            values,
            key=lambda row: (
                judge_audit._rank(seed, f"accelerated-duplicate:{key[0]}:{key[1]}", row.item_id),
                row.item_id,
            ),
        )
        selected.extend(ranked[: allocation[key]])
    return selected


def _qualification_template() -> dict[str, Any]:
    reviewer = lambda annotator_id: {
        "annotator_id": annotator_id,
        "human_attested": False,
        "independent_primary_review_attested": False,
        "qualified_languages": [],
        "qualification_basis": "",
        "attested_by": "",
        "attested_at_utc": "",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": QUALIFICATION_WORKFLOW,
        "annotators": [reviewer(value) for value in ANNOTATOR_IDS],
        "adjudicator": {
            "adjudicator_id": "",
            "human_attested": False,
            "independent_adjudication_attested": False,
            "qualified_languages": [],
            "qualification_basis": "",
            "attested_by": "",
            "attested_at_utc": "",
        },
        "warning": "Complete from genuine human qualification records; do not infer or fabricate attestations.",
    }


def export_packets(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    duplicate_count: int = DEFAULT_DUPLICATES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Export two blank, independently ordered packets and their private key."""

    source_population, source_metadata = load_accelerated_population(run_dir)
    source_duplicates = _proportional_duplicate_rows(
        source_population, duplicate_count, seed
    )
    blinded_ids: dict[str, str] = {}
    used_blinded_ids: set[str] = set()
    while len(blinded_ids) < len(source_population):
        source_id = source_population[len(blinded_ids)].item_id
        candidate = "ap0_" + secrets.token_hex(12)
        if candidate not in used_blinded_ids:
            blinded_ids[source_id] = candidate
            used_blinded_ids.add(candidate)
    population = [
        replace(row, item_id=blinded_ids[row.item_id]) for row in source_population
    ]
    duplicates = [
        replace(row, item_id=blinded_ids[row.item_id]) for row in source_duplicates
    ]
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise AcceleratedPart0HumanValidationError(
            f"Output directory already exists; refusing to overwrite: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    os.chmod(temporary, 0o700)
    try:
        (temporary / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
        os.chmod(temporary / ".gitignore", 0o600)
        counts = Counter(row.stratum for row in population)
        sampling_cap = max(counts.values())
        key_rows: list[dict[str, object]] = [
            judge_audit._key_row(
                row,
                counts,
                sampling_cap,
                seed,
                annotator_id="shared",
            )
            for row in population
        ]
        files: dict[str, dict[str, str]] = {}
        used_duplicate_ids: set[str] = set()
        for annotator_id in ANNOTATOR_IDS:
            primary_name = f"{annotator_id}_packet.csv"
            duplicate_name = f"{annotator_id}_duplicate_packet.csv"
            files[annotator_id] = {
                "primary_packet": primary_name,
                "duplicate_packet": duplicate_name,
            }
            primary_order = sorted(
                population,
                key=lambda row: (
                    judge_audit._rank(seed, f"accelerated-primary:{annotator_id}", row.item_id),
                    row.item_id,
                ),
            )
            duplicate_records = []
            for row in duplicates:
                duplicate_id = "ap0d_" + secrets.token_hex(12)
                while duplicate_id in used_duplicate_ids:
                    duplicate_id = "ap0d_" + secrets.token_hex(12)
                used_duplicate_ids.add(duplicate_id)
                duplicate_records.append((duplicate_id, row))
            duplicate_records.sort(
                key=lambda pair: (
                    judge_audit._rank(
                        seed, f"accelerated-duplicate-order:{annotator_id}", pair[0]
                    ),
                    pair[0],
                )
            )
            judge_audit._write_csv(
                temporary / primary_name,
                judge_audit.ANNOTATOR_FIELDS,
                (judge_audit._annotation_row(row) for row in primary_order),
            )
            judge_audit._write_csv(
                temporary / duplicate_name,
                judge_audit.ANNOTATOR_FIELDS,
                (
                    judge_audit._annotation_row(row, duplicate_id)
                    for duplicate_id, row in duplicate_records
                ),
            )
            key_rows.extend(
                judge_audit._key_row(
                    row,
                    counts,
                    sampling_cap,
                    seed,
                    item_id=duplicate_id,
                    item_kind="duplicate",
                    source_item_id=row.item_id,
                    annotator_id=annotator_id,
                )
                for duplicate_id, row in duplicate_records
            )
        judge_audit._write_csv(
            temporary / "audit_key.csv",
            judge_audit.MULTI_KEY_FIELDS,
            key_rows,
        )
        qualification_path = temporary / "language_qualifications.template.json"
        qualification_path.write_text(
            json.dumps(_qualification_template(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(qualification_path, 0o600)
        subject_language_counts = Counter((row.target_id, row.language) for row in population)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "workflow": WORKFLOW,
            "analysis_role": "genuine_human_validation_of_exploratory_accelerated_part0_evidence",
            "source": source_metadata,
            "fixed_judge_identity_hidden_from_annotator_packets": True,
            "subject_identity_hidden_from_annotator_packets": True,
            "blinded_item_ids": "cryptographically_random_96_bit_ids_not_source_derived",
            "sampling": "census_of_every_valid_fixed_judge_label_for_each_nonretired_subject",
            "stratification": "subject_target_id_by_response_language",
            "primary_rows_per_annotator": len(population),
            "duplicates_per_annotator": len(duplicates),
            "annotator_ids": list(ANNOTATOR_IDS),
            "subject_language_population": {
                f"{target_id}/{language}": count
                for (target_id, language), count in sorted(subject_language_counts.items())
            },
            "automated_label_population": dict(
                sorted(Counter(row.automated_label for row in population).items())
            ),
            "files": {
                "annotators": files,
                "private_key": "audit_key.csv",
                "qualification_template": "language_qualifications.template.json",
            },
            "human_labels_generated": False,
            "human_validation_complete": False,
            "confirmatory_or_paper_promotion_permitted": False,
            "sensitive_raw_content": True,
            "handling": "LOCAL ONLY: do not upload, commit, attach, or release this directory.",
            "seed": seed,
        }
        manifest["manifest_sha256"] = _artifact_hash(manifest, "manifest_sha256")
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(manifest_path, 0o600)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _validate_key_against_source(run_dir: str | Path, key_path: str | Path) -> list[dict[str, str]]:
    population, _ = load_accelerated_population(run_dir)
    fields, rows = judge_audit._read_csv_dicts(key_path)
    if not set(judge_audit.MULTI_KEY_FIELDS).issubset(fields):
        raise AcceleratedPart0HumanValidationError("Accelerated audit key schema is incomplete.")
    primary = [row for row in rows if row.get("item_kind") == "primary"]

    def raw_lineage(row: judge_audit.RawAuditRow) -> tuple[str, str, int, str]:
        return (
            row.target_id,
            row.language,
            row.source_row,
            judge_audit._stimulus_digest(row),
        )

    def key_lineage(row: Mapping[str, str]) -> tuple[str, str, int, str]:
        try:
            source_row = int(row.get("source_row", ""))
        except ValueError as error:
            raise AcceleratedPart0HumanValidationError(
                "Accelerated audit key source row is invalid."
            ) from error
        return (
            row.get("target_id", ""),
            row.get("language", ""),
            source_row,
            row.get("stimulus_sha256", ""),
        )

    expected = {raw_lineage(row): row for row in population}
    observed = {key_lineage(row): row for row in primary}
    item_ids = [row.get("item_id", "") for row in primary]
    if (
        len(expected) != len(population)
        or len(observed) != len(primary)
        or set(observed) != set(expected)
        or len(set(item_ids)) != len(item_ids)
        or any(not value.startswith("ap0_") for value in item_ids)
    ):
        raise AcceleratedPart0HumanValidationError(
            "Accelerated audit key does not contain the exact source census."
        )
    for lineage, raw in expected.items():
        key = observed[lineage]
        expected_values = {
            "language": raw.language,
            "automated_label": raw.automated_label,
            "provider": raw.provider,
            "model": raw.model,
            "target_id": raw.target_id,
            "arm": raw.arm,
            "source": raw.source,
            "category": raw.category,
            "semantic_cluster_id": raw.semantic_cluster_id,
            "generation_block": str(raw.generation_block),
            "source_file": raw.source_file,
            "source_row": str(raw.source_row),
            "stimulus_sha256": judge_audit._stimulus_digest(raw),
            "inclusion_probability": "1",
            "sampling_weight": "1",
        }
        if any(key.get(field) != value for field, value in expected_values.items()):
            raise AcceleratedPart0HumanValidationError(
                f"Accelerated audit key lineage changed for {key.get('item_id', '')}."
            )
    return rows


def _valid_attestation_text(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().casefold()
    return normalized not in _PLACEHOLDER_VALUES and "<" not in value and ">" not in value


def _validate_timestamp(value: object) -> bool:
    if not _valid_attestation_text(value):
        return False
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return False
    offset = parsed.utcoffset()
    return offset is not None and offset.total_seconds() == 0


def validate_language_qualifications(
    path: str | Path,
    *,
    annotator_ids: Sequence[str],
) -> dict[str, Any]:
    """Require explicit genuine-human and language-fluency attestations."""

    qualification_path = Path(path).expanduser().resolve()
    _require_private_path(qualification_path, "language qualification record")
    data = _read_json(qualification_path, "language qualification record")
    annotators = data.get("annotators")
    adjudicator = data.get("adjudicator")
    if (
        data.get("schema_version") != SCHEMA_VERSION
        or data.get("workflow") != QUALIFICATION_WORKFLOW
        or not isinstance(annotators, list)
        or not isinstance(adjudicator, Mapping)
    ):
        raise AcceleratedPart0HumanValidationError(
            "Language qualification record schema is invalid."
        )
    by_id = {
        str(row.get("annotator_id", "")): row
        for row in annotators
        if isinstance(row, Mapping)
    }
    required_ids = set(annotator_ids)
    if len(annotators) != len(by_id) or set(by_id) != required_ids or len(required_ids) != 2:
        raise AcceleratedPart0HumanValidationError(
            "Exactly the two scoring annotators require qualification records."
        )
    required_languages = set(judge_audit.TARGET_LANGUAGES)
    for annotator_id, row in by_id.items():
        if (
            row.get("human_attested") is not True
            or row.get("independent_primary_review_attested") is not True
            or set(row.get("qualified_languages", [])) != required_languages
            or not _valid_attestation_text(row.get("qualification_basis"))
            or not _valid_attestation_text(row.get("attested_by"))
            or not _validate_timestamp(row.get("attested_at_utc"))
        ):
            raise AcceleratedPart0HumanValidationError(
                f"Human/language qualification is incomplete for {annotator_id}."
            )
    adjudicator_id = str(adjudicator.get("adjudicator_id", ""))
    if (
        not _valid_attestation_text(adjudicator_id)
        or adjudicator_id in required_ids
        or adjudicator.get("human_attested") is not True
        or adjudicator.get("independent_adjudication_attested") is not True
        or set(adjudicator.get("qualified_languages", [])) != required_languages
        or not _valid_attestation_text(adjudicator.get("qualification_basis"))
        or not _valid_attestation_text(adjudicator.get("attested_by"))
        or not _validate_timestamp(adjudicator.get("attested_at_utc"))
    ):
        raise AcceleratedPart0HumanValidationError(
            "Independent adjudicator human/language qualification is incomplete."
        )
    return data


def prepare_adjudication(
    run_dir: str | Path,
    key_path: str | Path,
    annotation_paths: dict[str, str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    _validate_key_against_source(run_dir, key_path)
    if set(annotation_paths) != set(ANNOTATOR_IDS):
        raise AcceleratedPart0HumanValidationError(
            "Exactly annotator_1 and annotator_2 primary packets are required."
        )
    return judge_audit.prepare_adjudication(key_path, annotation_paths, output_path)


def score_validation(
    run_dir: str | Path,
    key_path: str | Path,
    annotation_paths: dict[str, str | Path],
    duplicate_annotation_paths: dict[str, str | Path],
    adjudications_path: str | Path,
    qualifications_path: str | Path,
    output_path: str | Path,
    *,
    bootstrap_replicates: int = judge_audit.DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Score only after source, labels, adjudication, and qualifications pass."""

    _validate_key_against_source(run_dir, key_path)
    if set(annotation_paths) != set(ANNOTATOR_IDS) or set(duplicate_annotation_paths) != set(
        ANNOTATOR_IDS
    ):
        raise AcceleratedPart0HumanValidationError(
            "Exactly matching annotator_1 and annotator_2 primary/delayed packets are required."
        )
    qualifications = validate_language_qualifications(
        qualifications_path, annotator_ids=sorted(annotation_paths)
    )
    try:
        result = judge_audit.score_multi_audit(
            key_path,
            annotation_paths,
            duplicate_annotation_paths,
            adjudications_path,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
    except judge_audit.AuditError as error:
        raise AcceleratedPart0HumanValidationError(str(error)) from error
    result.pop("result_sha256", None)
    result.update(
        {
            "workflow": WORKFLOW,
            "human_validation_complete": True,
            "language_qualification_gate": {
                "passed": True,
                "annotator_ids": sorted(annotation_paths),
                "adjudicator_id": qualifications["adjudicator"]["adjudicator_id"],
                "required_languages": list(judge_audit.TARGET_LANGUAGES),
                "qualification_record": judge_audit._integrity_reference(
                    qualifications_path
                ),
            },
            "source_evidence": load_accelerated_population(run_dir)[1]["source_manifest"],
            "source_analysis_role_preserved": "exploratory",
            "confirmatory_or_paper_promotion_permitted": False,
            "interpretation": (
                "Completed human criterion-validation metrics for the frozen accelerated "
                "Part 0 evidence; this does not convert the source campaign into a "
                "confirmatory experiment or authorize paper promotion by itself."
            ),
        }
    )
    result["result_sha256"] = hashlib.sha256(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise AcceleratedPart0HumanValidationError(
            f"Score output already exists; refusing to overwrite: {destination}"
        )
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.chmod(destination, 0o600)
    return result


def _named_paths(values: Sequence[str], option: str) -> dict[str, str]:
    try:
        return judge_audit._parse_named_paths(values, option=option)
    except judge_audit.AuditError as error:
        raise AcceleratedPart0HumanValidationError(str(error)) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Blinded genuine-human validation for accelerated Part 0 evidence."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="export two blank blinded annotation packets")
    export.add_argument("--run-dir", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--duplicates", type=int, default=DEFAULT_DUPLICATES)
    export.add_argument("--seed", type=int, default=DEFAULT_SEED)

    adjudicate = sub.add_parser(
        "prepare-adjudication", help="export disagreements for an independent adjudicator"
    )
    adjudicate.add_argument("--run-dir", required=True)
    adjudicate.add_argument("--key", required=True)
    adjudicate.add_argument("--annotator", action="append", required=True)
    adjudicate.add_argument("--output", required=True)

    score = sub.add_parser("score", help="score complete qualified human annotations")
    score.add_argument("--run-dir", required=True)
    score.add_argument("--key", required=True)
    score.add_argument("--annotator", action="append", required=True)
    score.add_argument("--duplicate-annotations", action="append", required=True)
    score.add_argument("--adjudications", required=True)
    score.add_argument("--qualifications", required=True)
    score.add_argument("--output", required=True)
    score.add_argument(
        "--bootstrap-replicates", type=int, default=judge_audit.DEFAULT_BOOTSTRAP_REPLICATES
    )
    score.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "export":
            result = export_packets(
                args.run_dir,
                args.output_dir,
                duplicate_count=args.duplicates,
                seed=args.seed,
            )
            print(
                f"Exported {result['primary_rows_per_annotator']} blank primary rows "
                f"and {result['duplicates_per_annotator']} delayed duplicates per annotator."
            )
            print("LOCAL ONLY: packets contain harmful requests and private model responses.")
        elif args.command == "prepare-adjudication":
            result = prepare_adjudication(
                args.run_dir,
                args.key,
                _named_paths(args.annotator, "--annotator"),
                args.output,
            )
            print(f"Prepared {result['disagreement_rows']} blank disagreement rows.")
        else:
            result = score_validation(
                args.run_dir,
                args.key,
                _named_paths(args.annotator, "--annotator"),
                _named_paths(args.duplicate_annotations, "--duplicate-annotations"),
                args.adjudications,
                args.qualifications,
                args.output,
                bootstrap_replicates=args.bootstrap_replicates,
                seed=args.seed,
            )
            print(
                f"Scored {result['primary_rows']} complete human labels; "
                "language-qualification gate passed."
            )
    except (AcceleratedPart0HumanValidationError, judge_audit.AuditError) as error:
        print(f"Accelerated Part 0 human validation failed: {error}", file=os.sys.stderr)
        return 1
    return 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
