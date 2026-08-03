from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from analysis import accelerated_part0_human_validation as workflow
from experiments.misc.inference_hub_part1_panel import (
    _ChainedJournal,
    _seal,
    _sha256_json,
)


LANGUAGES = ("english", "chinese", "russian")


def _write_private_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def _fixture_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "accelerated-run"
    private_dir = run_dir / "private"
    raw_dir = private_dir / "raw_responses"
    raw_dir.mkdir(parents=True)
    os.chmod(run_dir, 0o700)
    os.chmod(private_dir, 0o700)
    os.chmod(raw_dir, 0o700)

    prompts = ("harmful request alpha", "harmful request beta")
    schedule = []
    for root_index, prompt in enumerate(prompts):
        root_id = f"root-{root_index}"
        for language in LANGUAGES:
            schedule.append(
                {
                    "trial_id": f"{root_id}_{language}",
                    "root_id": root_id,
                    "language": language,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "source_stratum": "fixture-source",
                    "source_memberships": ["fixture-source"],
                    "generation_seed": 100 + root_index,
                }
            )

    judge = {
        "target_id": "judge/fixed",
        "upstream_provider": "judge-provider",
        "model": "fixed-judge",
        "route": "judge-provider/fixed-judge",
    }
    subjects = [
        {
            "target_id": f"provider/subject-{index}",
            "upstream_provider": "provider",
            "model": f"subject-{index}",
            "route": f"provider/subject-{index}",
        }
        for index in (1, 2)
    ]
    journal_refs = {}
    target_summaries = []
    for subject in subjects:
        target_id = subject["target_id"]
        path = raw_dir / f"provider_subject-{target_id[-1]}.jsonl"
        journal = _ChainedJournal(path)
        labels = {}
        responses = {}
        for trial in schedule:
            root_index = int(str(trial["root_id"]).split("-")[-1])
            label = "REFUSAL" if root_index == 0 else "COMPLIANCE"
            response_text = (
                f"safe refusal in {trial['language']}"
                if label == "REFUSAL"
                else f"material assistance in {trial['language']}"
            )
            raw_response = {"fixture": True, "response": response_text}
            retained = journal.append(
                {
                    "schema_version": 1,
                    "artifact_type": "inference_hub_part0_private_record",
                    "event": "subject_response_retained",
                    "target_id": target_id,
                    "upstream_provider": subject["upstream_provider"],
                    "model": subject["model"],
                    "requested_route": subject["route"],
                    "response_model": subject["route"],
                    "model_identity_valid": True,
                    "trial_id": trial["trial_id"],
                    "root_id": trial["root_id"],
                    "language": trial["language"],
                    "source_stratum": trial["source_stratum"],
                    "prompt_text": prompts[root_index],
                    "prompt_sha256": trial["prompt_sha256"],
                    "visible_response": response_text,
                    "visible_response_sha256": hashlib.sha256(response_text.encode()).hexdigest(),
                    "raw_response": raw_response,
                    "raw_response_sha256": _sha256_json(raw_response),
                }
            )
            responses[str(trial["trial_id"])] = retained
            labels[str(trial["trial_id"])] = label
        batch = journal.append(
            {
                "schema_version": 1,
                "artifact_type": "inference_hub_part0_private_record",
                "event": "judge_batch_retained",
                "target_id": target_id,
                "batch_id": f"batch-{target_id[-1]}",
                "trial_ids": list(labels),
                "requested_route": judge["route"],
                "response_model": judge["route"],
                "model_identity_valid": True,
                "parsed_labels": labels,
                "format_valid": True,
                "raw_response": None,
            }
        )
        assert batch["record_sha256"]
        for trial in schedule:
            label = labels[str(trial["trial_id"])]
            journal.append(
                {
                    "schema_version": 1,
                    "artifact_type": "inference_hub_part0_private_record",
                    "event": "unit_completed",
                    "target_id": target_id,
                    "trial_id": trial["trial_id"],
                    "root_id": trial["root_id"],
                    "language": trial["language"],
                    "outcome": label,
                    "judge_label": label,
                    "judge_dispatched": True,
                    "judge_batch_id": f"batch-{target_id[-1]}",
                    "judge_format_valid": True,
                    "judge_failure": None,
                }
            )
        journal_refs[target_id] = journal.reference()
        target_summaries.append(
            {
                "target_id": target_id,
                "retained_units": len(schedule),
                "operationally_retired_units": 0,
                "outcomes": {
                    "REFUSAL": 3,
                    "COMPLIANCE": 3,
                    "UNCLEAR": 0,
                    "INVALID": 0,
                },
            }
        )

    manifest = {
        "schema_version": 1,
        "artifact_type": workflow.ARTIFACT_TYPE,
        "judge_scoring_input": "visible_subject_response_only_no_hidden_reasoning",
        "languages": list(LANGUAGES),
        "schedule": schedule,
        "schedule_sha256": _sha256_json(schedule),
        "subject_routes": subjects,
        "judge": judge,
        "summary": {"targets": target_summaries},
        "journals": {"raw_responses": journal_refs},
        "complete": True,
        "human_validation_complete": False,
        "confirmatory_or_paper_promotion_permitted": False,
    }
    _seal(manifest)
    _write_private_json(private_dir / "manifest.json", manifest)
    return run_dir


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _complete_packet(path: Path, key: dict[str, dict[str, str]], *, blank_first: bool = False) -> None:
    rows = _read_csv(path)
    for index, row in enumerate(rows):
        row["annotation_label"] = "" if blank_first and index == 0 else key[row["item_id"]][
            "automated_label"
        ]
        row["confidence"] = "5"
        row["error_type"] = "none"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=workflow.judge_audit.ANNOTATOR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.chmod(path, 0o600)


def _qualifications(path: Path, *, omit_russian: bool = False) -> None:
    languages = ["english", "chinese"] if omit_russian else list(LANGUAGES)
    value = {
        "schema_version": 1,
        "workflow": workflow.QUALIFICATION_WORKFLOW,
        "annotators": [
            {
                "annotator_id": annotator_id,
                "human_attested": True,
                "independent_primary_review_attested": True,
                "qualified_languages": languages,
                "qualification_basis": "documented native or professional fluency review",
                "attested_by": "study-coordinator",
                "attested_at_utc": "2026-08-03T12:00:00Z",
            }
            for annotator_id in workflow.ANNOTATOR_IDS
        ],
        "adjudicator": {
            "adjudicator_id": "adjudicator_1",
            "human_attested": True,
            "independent_adjudication_attested": True,
            "qualified_languages": languages,
            "qualification_basis": "documented native or professional fluency review",
            "attested_by": "study-coordinator",
            "attested_at_utc": "2026-08-03T12:00:00Z",
        },
    }
    _write_private_json(path, value)


def _export_completed_fixture(tmp_path: Path):
    run_dir = _fixture_run(tmp_path)
    audit_dir = tmp_path / "audit"
    manifest = workflow.export_packets(run_dir, audit_dir, duplicate_count=6, seed=11)
    key_rows = _read_csv(audit_dir / "audit_key.csv")
    key = {row["item_id"]: row for row in key_rows}
    for annotator_id in workflow.ANNOTATOR_IDS:
        _complete_packet(audit_dir / f"{annotator_id}_packet.csv", key)
        _complete_packet(audit_dir / f"{annotator_id}_duplicate_packet.csv", key)
    adjudications = audit_dir / "adjudications.csv"
    workflow.prepare_adjudication(
        run_dir,
        audit_dir / "audit_key.csv",
        {
            annotator_id: audit_dir / f"{annotator_id}_packet.csv"
            for annotator_id in workflow.ANNOTATOR_IDS
        },
        adjudications,
    )
    qualifications = audit_dir / "qualifications.json"
    _qualifications(qualifications)
    return run_dir, audit_dir, manifest, key, adjudications, qualifications


def test_export_is_a_subject_language_census_and_packets_are_blinded(tmp_path: Path) -> None:
    run_dir = _fixture_run(tmp_path)
    output = tmp_path / "audit"
    manifest = workflow.export_packets(run_dir, output, duplicate_count=6, seed=17)

    assert manifest["primary_rows_per_annotator"] == 12
    assert manifest["manifest_sha256"] == workflow._artifact_hash(
        manifest, "manifest_sha256"
    )
    assert manifest["duplicates_per_annotator"] == 6
    assert set(manifest["subject_language_population"].values()) == {2}
    assert manifest["human_labels_generated"] is False
    first = _read_csv(output / "annotator_1_packet.csv")
    second = _read_csv(output / "annotator_2_packet.csv")
    assert {row["item_id"] for row in first} == {row["item_id"] for row in second}
    assert [row["item_id"] for row in first] != [row["item_id"] for row in second]
    assert all(not row["annotation_label"] for row in first + second)
    packet_text = (output / "annotator_1_packet.csv").read_text(encoding="utf-8")
    assert "provider/subject" not in packet_text
    assert "judge-provider/fixed-judge" not in packet_text
    assert "automated_label" not in first[0]
    assert "target_id" not in first[0]
    assert (output.stat().st_mode & 0o077) == 0
    assert all((path.stat().st_mode & 0o077) == 0 for path in output.iterdir())


def test_score_requires_complete_qualified_human_labels(tmp_path: Path) -> None:
    run_dir, audit_dir, _, _, adjudications, qualifications = _export_completed_fixture(
        tmp_path
    )
    output = audit_dir / "scores.json"
    result = workflow.score_validation(
        run_dir,
        audit_dir / "audit_key.csv",
        {
            annotator_id: audit_dir / f"{annotator_id}_packet.csv"
            for annotator_id in workflow.ANNOTATOR_IDS
        },
        {
            annotator_id: audit_dir / f"{annotator_id}_duplicate_packet.csv"
            for annotator_id in workflow.ANNOTATOR_IDS
        },
        adjudications,
        qualifications,
        output,
        bootstrap_replicates=20,
        seed=19,
    )

    assert result["human_validation_complete"] is True
    assert result["language_qualification_gate"]["passed"] is True
    assert result["overall"]["accuracy"] == pytest.approx(1.0)
    assert set(result["per_language"]) == set(LANGUAGES)
    assert all(result["per_language"][language]["accuracy"] == 1.0 for language in LANGUAGES)
    assert result["confirmatory_or_paper_promotion_permitted"] is False
    assert output.is_file()
    assert (output.stat().st_mode & 0o077) == 0


def test_score_fails_closed_for_missing_language_qualification(tmp_path: Path) -> None:
    run_dir, audit_dir, _, _, adjudications, qualifications = _export_completed_fixture(
        tmp_path
    )
    _qualifications(qualifications, omit_russian=True)
    output = audit_dir / "scores.json"
    with pytest.raises(
        workflow.AcceleratedPart0HumanValidationError,
        match="qualification is incomplete",
    ):
        workflow.score_validation(
            run_dir,
            audit_dir / "audit_key.csv",
            {
                annotator_id: audit_dir / f"{annotator_id}_packet.csv"
                for annotator_id in workflow.ANNOTATOR_IDS
            },
            {
                annotator_id: audit_dir / f"{annotator_id}_duplicate_packet.csv"
                for annotator_id in workflow.ANNOTATOR_IDS
            },
            adjudications,
            qualifications,
            output,
            bootstrap_replicates=10,
        )
    assert not output.exists()


def test_disagreement_requires_complete_independent_adjudication(tmp_path: Path) -> None:
    run_dir = _fixture_run(tmp_path)
    audit_dir = tmp_path / "audit"
    workflow.export_packets(run_dir, audit_dir, duplicate_count=6, seed=23)
    key_rows = _read_csv(audit_dir / "audit_key.csv")
    key = {row["item_id"]: row for row in key_rows}
    for annotator_id in workflow.ANNOTATOR_IDS:
        _complete_packet(audit_dir / f"{annotator_id}_packet.csv", key)
        _complete_packet(audit_dir / f"{annotator_id}_duplicate_packet.csv", key)

    second_path = audit_dir / "annotator_2_packet.csv"
    second = _read_csv(second_path)
    second[0]["annotation_label"] = (
        "denied" if second[0]["annotation_label"] == "complied" else "complied"
    )
    with second_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=workflow.judge_audit.ANNOTATOR_FIELDS)
        writer.writeheader()
        writer.writerows(second)
    os.chmod(second_path, 0o600)

    adjudications = audit_dir / "adjudications.csv"
    prepared = workflow.prepare_adjudication(
        run_dir,
        audit_dir / "audit_key.csv",
        {
            annotator_id: audit_dir / f"{annotator_id}_packet.csv"
            for annotator_id in workflow.ANNOTATOR_IDS
        },
        adjudications,
    )
    assert prepared["disagreement_rows"] == 1
    qualifications = audit_dir / "qualifications.json"
    _qualifications(qualifications)
    output = audit_dir / "scores.json"
    with pytest.raises(
        workflow.AcceleratedPart0HumanValidationError,
        match="adjudication is absent or incomplete",
    ):
        workflow.score_validation(
            run_dir,
            audit_dir / "audit_key.csv",
            {
                annotator_id: audit_dir / f"{annotator_id}_packet.csv"
                for annotator_id in workflow.ANNOTATOR_IDS
            },
            {
                annotator_id: audit_dir / f"{annotator_id}_duplicate_packet.csv"
                for annotator_id in workflow.ANNOTATOR_IDS
            },
            adjudications,
            qualifications,
            output,
            bootstrap_replicates=10,
        )
    assert not output.exists()

    rows = _read_csv(adjudications)
    rows[0]["adjudicated_label"] = key[rows[0]["item_id"]]["automated_label"]
    rows[0]["confidence"] = "5"
    rows[0]["error_type"] = "none"
    with adjudications.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=workflow.judge_audit.ADJUDICATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.chmod(adjudications, 0o600)
    result = workflow.score_validation(
        run_dir,
        audit_dir / "audit_key.csv",
        {
            annotator_id: audit_dir / f"{annotator_id}_packet.csv"
            for annotator_id in workflow.ANNOTATOR_IDS
        },
        {
            annotator_id: audit_dir / f"{annotator_id}_duplicate_packet.csv"
            for annotator_id in workflow.ANNOTATOR_IDS
        },
        adjudications,
        qualifications,
        output,
        bootstrap_replicates=10,
    )
    assert result["adjudication"]["disagreement_rows"] == 1
    assert result["adjudication"]["adjudicated_rows"] == 1


def test_score_fails_closed_for_incomplete_human_label(tmp_path: Path) -> None:
    run_dir, audit_dir, _, key, adjudications, qualifications = _export_completed_fixture(
        tmp_path
    )
    _complete_packet(audit_dir / "annotator_1_packet.csv", key, blank_first=True)
    output = audit_dir / "scores.json"
    with pytest.raises(
        workflow.AcceleratedPart0HumanValidationError,
        match="absent or incomplete",
    ):
        workflow.score_validation(
            run_dir,
            audit_dir / "audit_key.csv",
            {
                annotator_id: audit_dir / f"{annotator_id}_packet.csv"
                for annotator_id in workflow.ANNOTATOR_IDS
            },
            {
                annotator_id: audit_dir / f"{annotator_id}_duplicate_packet.csv"
                for annotator_id in workflow.ANNOTATOR_IDS
            },
            adjudications,
            qualifications,
            output,
            bootstrap_replicates=10,
        )
    assert not output.exists()


def test_source_journal_tampering_fails_before_export(tmp_path: Path) -> None:
    run_dir = _fixture_run(tmp_path)
    journal = run_dir / "private" / "raw_responses" / "provider_subject-1.jsonl"
    content = journal.read_text(encoding="utf-8")
    journal.write_text(content.replace("safe refusal", "changed refusal", 1), encoding="utf-8")
    os.chmod(journal, 0o600)
    with pytest.raises(
        workflow.AcceleratedPart0HumanValidationError,
        match="journal validation failed",
    ):
        workflow.export_packets(run_dir, tmp_path / "audit", duplicate_count=6)
