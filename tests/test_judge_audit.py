from __future__ import annotations

import csv
import json
import stat
from collections import Counter
from pathlib import Path

import pytest

from analysis.judge_audit import (
    ADJUDICATION_FIELDS,
    ANNOTATOR_FIELDS,
    DEFAULT_DUPLICATES,
    DEFAULT_PER_STRATUM,
    ERROR_TYPES,
    AuditError,
    generate_audit,
    main,
    prepare_adjudication,
    score_audit,
    score_multi_audit,
    _fleiss_kappa,
    _normalize_automated_label,
)


RAW_FIELDS = (
    "provider",
    "model",
    "language",
    "prompt",
    "reasoning",
    "response",
    "response_en",
    "complied?",
)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def _raw_fixture_rows(english_complied_count: int = 2) -> list[dict[str, object]]:
    """Return structural unit-test records, not research observations."""

    rows: list[dict[str, object]] = []
    counts = {
        (language, label): 2
        for language in ("english", "chinese", "russian")
        for label in ("complied", "denied")
    }
    counts[("english", "complied")] = english_complied_count
    for (language, label), count in counts.items():
        for index in range(count):
            denied = label == "denied"
            rows.append(
                {
                    "provider": "unit-test-provider",
                    "model": f"unit-test-model-{index % 2}",
                    "language": language,
                    "prompt": (
                        "A benchmark request whose operational content is intentionally omitted "
                        f"({language}-{label}-{index})."
                    ),
                    "reasoning": "A brief fixture rationale with no operational content.",
                    "response": (
                        "I cannot provide the omitted operational content."
                        if denied
                        else "I would provide the requested content in this structural fixture."
                    ),
                    "response_en": "",
                    "complied?": "false" if denied else "true",
                }
            )
    return rows


def test_response_only_label_takes_precedence_over_retained_legacy_label() -> None:
    assert _normalize_automated_label(
        {"new_complied": "false", "complied?": "true"}
    ) == "denied"
    with pytest.raises(AuditError, match="unjudged"):
        _normalize_automated_label({"new_complied": "", "complied?": "true"})


def test_fleiss_kappa_supports_more_than_two_annotators() -> None:
    reliability = _fleiss_kappa(
        [
            ["complied", "complied", "denied", "denied"],
            ["complied", "complied", "denied", "denied"],
            ["complied", "denied", "denied", "denied"],
        ]
    )

    assert reliability["n_items"] == 4
    assert reliability["n_raters"] == 3
    assert reliability["observed_agreement"] == pytest.approx(5 / 6)
    assert reliability["fleiss_kappa"] == pytest.approx(23 / 35)


def _generated_audit(
    tmp_path: Path, *, english_complied_count: int = 2, name: str = "private-audit"
) -> Path:
    raw_path = tmp_path / f"{name}-raw.csv"
    _write_csv(raw_path, RAW_FIELDS, _raw_fixture_rows(english_complied_count))
    output = tmp_path / name
    generate_audit([raw_path], output, seed=17, per_stratum=2, duplicate_count=6)
    return output


def _complete_annotations(output: Path, *, add_unclear: bool = False) -> None:
    _, key_rows = _read_csv(output / "audit_key.csv")
    primary_key = {row["item_id"]: row for row in key_rows if row["item_kind"] == "primary"}
    duplicate_key = {row["item_id"]: row for row in key_rows if row["item_kind"] == "duplicate"}
    _, primary_rows = _read_csv(output / "annotator_packet.csv")
    _, duplicate_rows = _read_csv(output / "duplicate_packet.csv")

    english_complied = sorted(
        item_id
        for item_id, row in primary_key.items()
        if row["language"] == "english" and row["automated_label"] == "complied"
    )
    changed_label = english_complied[0]
    unclear_label = next(
        item_id
        for item_id, row in sorted(primary_key.items())
        if row["language"] == "chinese" and row["automated_label"] == "complied"
    )
    labels: dict[str, str] = {}
    for row in primary_rows:
        item_id = row["item_id"]
        label = primary_key[item_id]["automated_label"]
        if item_id == changed_label:
            label = "denied"
        if add_unclear and item_id == unclear_label:
            label = "unclear"
        labels[item_id] = label
        row.update(annotation_label=label, confidence="4", error_type="none")
    _write_csv(output / "annotator_packet.completed.csv", ANNOTATOR_FIELDS, primary_rows)

    for row in duplicate_rows:
        source_id = duplicate_key[row["item_id"]]["source_item_id"]
        row.update(annotation_label=labels[source_id], confidence="4", error_type="none")
    _write_csv(output / "duplicate_packet.completed.csv", ANNOTATOR_FIELDS, duplicate_rows)


def _complete_multi_annotations(
    output: Path,
) -> tuple[dict[str, Path], dict[str, Path], str]:
    _, key_rows = _read_csv(output / "audit_key.csv")
    primary_key = {
        row["item_id"]: row for row in key_rows if row["item_kind"] == "primary"
    }
    disagreement_id = next(
        item_id
        for item_id, row in sorted(primary_key.items())
        if row["language"] == "english"
    )
    primary_paths: dict[str, Path] = {}
    duplicate_paths: dict[str, Path] = {}
    for annotator_id in ("annotator_1", "annotator_2"):
        _, primary_rows = _read_csv(output / f"{annotator_id}_packet.csv")
        labels: dict[str, str] = {}
        for row in primary_rows:
            label = primary_key[row["item_id"]]["automated_label"]
            if annotator_id == "annotator_2" and row["item_id"] == disagreement_id:
                label = "denied" if label == "complied" else "complied"
            labels[row["item_id"]] = label
            row.update(annotation_label=label, confidence="4", error_type="none")
        primary_path = output / f"{annotator_id}_packet.completed.csv"
        _write_csv(primary_path, ANNOTATOR_FIELDS, primary_rows)
        primary_paths[annotator_id] = primary_path
        duplicate_key = {
            row["item_id"]: row
            for row in key_rows
            if row["item_kind"] == "duplicate"
            and row["annotator_id"] == annotator_id
        }
        _, duplicate_rows = _read_csv(
            output / f"{annotator_id}_duplicate_packet.csv"
        )
        for row in duplicate_rows:
            source_id = duplicate_key[row["item_id"]]["source_item_id"]
            row.update(
                annotation_label=labels[source_id],
                confidence="4",
                error_type="none",
            )
        duplicate_path = output / f"{annotator_id}_duplicate_packet.completed.csv"
        _write_csv(duplicate_path, ANNOTATOR_FIELDS, duplicate_rows)
        duplicate_paths[annotator_id] = duplicate_path
    return primary_paths, duplicate_paths, disagreement_id


def test_generate_is_deterministic_stratified_blinded_and_blank(tmp_path: Path) -> None:
    raw_path = tmp_path / "part0.csv"
    _write_csv(raw_path, RAW_FIELDS, _raw_fixture_rows())
    first = tmp_path / "audit-one"
    second = tmp_path / "audit-two"

    first_manifest = generate_audit(
        [raw_path], first, seed=321, per_stratum=2, duplicate_count=6
    )
    second_manifest = generate_audit(
        [raw_path], second, seed=321, per_stratum=2, duplicate_count=6
    )

    assert first_manifest["primary_rows"] == 12
    assert first_manifest["duplicate_rows"] == 6
    assert first_manifest["human_labels_generated"] is False
    for filename in ("annotator_packet.csv", "duplicate_packet.csv", "audit_key.csv"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()

    fields, primary = _read_csv(first / "annotator_packet.csv")
    duplicate_fields, duplicates = _read_csv(first / "duplicate_packet.csv")
    assert tuple(fields) == ANNOTATOR_FIELDS
    assert tuple(duplicate_fields) == ANNOTATOR_FIELDS
    assert {"provider", "model", "automated_label", "source_item_id"}.isdisjoint(fields)
    assert len(primary) == 12
    assert len(duplicates) == 6
    assert all(row["item_id"].startswith("ja_") for row in primary)
    assert all(row["item_id"].startswith("jd_") for row in duplicates)
    assert all(
        not row["annotation_label"] and not row["confidence"] and not row["error_type"]
        for row in primary + duplicates
    )

    _, key_rows = _read_csv(first / "audit_key.csv")
    primary_key_rows = [row for row in key_rows if row["item_kind"] == "primary"]
    duplicate_key_rows = [row for row in key_rows if row["item_kind"] == "duplicate"]
    assert len(primary_key_rows) == 12
    assert len(duplicate_key_rows) == 6
    assert {
        (row["language"], row["automated_label"]) for row in primary_key_rows
    } == {
        (language, label)
        for language in ("english", "chinese", "russian")
        for label in ("complied", "denied")
    }
    assert Counter((row["language"], row["automated_label"]) for row in primary_key_rows) == {
        (language, label): 2
        for language in ("english", "chinese", "russian")
        for label in ("complied", "denied")
    }
    assert stat.S_IMODE(first.stat().st_mode) == 0o700
    assert (first / ".gitignore").read_text(encoding="utf-8") == "*\n!.gitignore\n"
    assert DEFAULT_PER_STRATUM == 100
    assert DEFAULT_DUPLICATES == 120


def test_generate_two_independent_complete_packets_with_per_rater_duplicates(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "multi-raw.csv"
    _write_csv(raw_path, RAW_FIELDS, _raw_fixture_rows())
    output = tmp_path / "multi-audit"

    manifest = generate_audit(
        [raw_path],
        output,
        seed=51,
        per_stratum=2,
        duplicate_count=6,
        annotator_count=2,
    )

    assert manifest["schema_version"] == 2
    assert manifest["annotator_count"] == 2
    assert manifest["independent_primary_packets"] == 2
    assert manifest["duplicate_rows_per_annotator"] == 6
    assert "fluent" in manifest["language_qualification"]
    primary_packets = []
    duplicate_ids: set[str] = set()
    for annotator_id in ("annotator_1", "annotator_2"):
        fields, primary = _read_csv(output / f"{annotator_id}_packet.csv")
        duplicate_fields, duplicates = _read_csv(
            output / f"{annotator_id}_duplicate_packet.csv"
        )
        assert tuple(fields) == ANNOTATOR_FIELDS
        assert tuple(duplicate_fields) == ANNOTATOR_FIELDS
        assert len(primary) == 12
        assert len(duplicates) == 6
        assert all(
            not row["annotation_label"]
            and not row["confidence"]
            and not row["error_type"]
            for row in primary + duplicates
        )
        primary_packets.append(primary)
        current_duplicate_ids = {row["item_id"] for row in duplicates}
        assert duplicate_ids.isdisjoint(current_duplicate_ids)
        duplicate_ids.update(current_duplicate_ids)
    assert {row["item_id"] for row in primary_packets[0]} == {
        row["item_id"] for row in primary_packets[1]
    }
    assert [row["item_id"] for row in primary_packets[0]] != [
        row["item_id"] for row in primary_packets[1]
    ]

    key_fields, key_rows = _read_csv(output / "audit_key.csv")
    assert "annotator_id" in key_fields
    assert Counter(
        row["annotator_id"] for row in key_rows if row["item_kind"] == "duplicate"
    ) == {"annotator_1": 6, "annotator_2": 6}
    assert {
        row["annotator_id"] for row in key_rows if row["item_kind"] == "primary"
    } == {"shared"}


def test_multi_annotator_reliability_adjudication_and_intra_rater_checks(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "multi-workflow-raw.csv"
    _write_csv(raw_path, RAW_FIELDS, _raw_fixture_rows())
    output = tmp_path / "multi-workflow"
    generate_audit(
        [raw_path],
        output,
        seed=73,
        per_stratum=2,
        duplicate_count=6,
        annotator_count=2,
    )
    primary_paths, duplicate_paths, disagreement_id = _complete_multi_annotations(
        output
    )
    adjudication_path = output / "adjudication_packet.csv"

    prepared = prepare_adjudication(
        output / "audit_key.csv", primary_paths, adjudication_path
    )

    assert prepared["primary_rows"] == 12
    assert prepared["agreement_rows"] == 11
    assert prepared["disagreement_rows"] == 1
    assert prepared["human_labels_generated"] is False
    fields, adjudications = _read_csv(adjudication_path)
    assert tuple(fields) == ADJUDICATION_FIELDS
    assert [row["item_id"] for row in adjudications] == [disagreement_id]
    assert adjudications[0]["adjudicated_label"] == ""
    assert adjudications[0]["confidence"] == ""
    assert set(json.loads(adjudications[0]["annotator_labels"])) == {
        "annotator_1",
        "annotator_2",
    }

    with pytest.raises(AuditError, match="adjudication is absent or incomplete"):
        score_multi_audit(
            output / "audit_key.csv",
            primary_paths,
            duplicate_paths,
            adjudication_path,
            bootstrap_replicates=10,
        )

    _, first_primary = _read_csv(primary_paths["annotator_1"])
    first_label = next(
        row["annotation_label"]
        for row in first_primary
        if row["item_id"] == disagreement_id
    )
    adjudications[0].update(
        adjudicated_label=first_label,
        confidence="5",
        error_type="partial_or_mixed_response",
        notes="Resolved by an independent adjudicator in this structural test.",
    )
    completed_adjudication = output / "adjudication_packet.completed.csv"
    _write_csv(completed_adjudication, ADJUDICATION_FIELDS, adjudications)
    score_path = output / "multi_scores.json"

    result = score_multi_audit(
        output / "audit_key.csv",
        primary_paths,
        duplicate_paths,
        completed_adjudication,
        output_path=score_path,
        bootstrap_replicates=40,
        seed=19,
    )

    assert json.loads(score_path.read_text(encoding="utf-8")) == result
    assert result["adjudication"] == {
        "agreement_rows": 11,
        "disagreement_rows": 1,
        "adjudicated_rows": 1,
        "final_label_rule": (
            "unanimous primary label when all annotators agree; otherwise the "
            "completed independent adjudication label"
        ),
    }
    inter = result["inter_rater_reliability"]
    assert inter["overall"]["n_items"] == 12
    assert inter["overall"]["cohen"]["observed_agreement"] == pytest.approx(11 / 12)
    assert inter["overall"]["cohen"]["cohen_kappa"] < 1.0
    assert inter["overall"]["fleiss"]["n_raters"] == 2
    assert set(inter["per_language"]) == {"english", "chinese", "russian"}
    for annotator_id in ("annotator_1", "annotator_2"):
        intra = result["intra_rater_reliability"][annotator_id]
        assert intra["n_pairs"] == 6
        assert intra["observed_agreement"] == 1.0
        assert intra["cohen_kappa"] == 1.0
    assert result["bootstrap_confidence_intervals"]["replicates"] == 40

    cli_score_path = output / "multi_scores_cli.json"
    assert main(
        [
            "score-multi",
            "--key",
            str(output / "audit_key.csv"),
            "--annotator",
            f"annotator_1={primary_paths['annotator_1']}",
            "--annotator",
            f"annotator_2={primary_paths['annotator_2']}",
            "--duplicate-annotations",
            f"annotator_1={duplicate_paths['annotator_1']}",
            "--duplicate-annotations",
            f"annotator_2={duplicate_paths['annotator_2']}",
            "--adjudications",
            str(completed_adjudication),
            "--output",
            str(cli_score_path),
            "--bootstrap-replicates",
            "10",
        ]
    ) == 0
    assert json.loads(cli_score_path.read_text(encoding="utf-8"))["schema_version"] == 2


def test_generate_cli_supports_verdict_schema_and_writes_no_labels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = _raw_fixture_rows()
    for row in rows:
        row["verdict"] = "denied" if row.pop("complied?") == "false" else "complied"
    fields = tuple(field for field in RAW_FIELDS if field != "complied?") + ("verdict",)
    raw_path = tmp_path / "part0-verdict.csv"
    _write_csv(raw_path, fields, rows)
    output = tmp_path / "cli-audit"

    exit_code = main(
        [
            "generate",
            "--input",
            str(raw_path),
            "--output-dir",
            str(output),
            "--seed",
            "9",
            "--per-stratum",
            "2",
            "--duplicates",
            "6",
        ]
    )

    assert exit_code == 0
    assert "LOCAL ONLY" in capsys.readouterr().out
    _, rows = _read_csv(output / "annotator_packet.csv")
    assert {row["annotation_label"] for row in rows} == {""}


def test_generate_fails_atomically_when_any_stratum_is_insufficient(tmp_path: Path) -> None:
    rows = _raw_fixture_rows()
    rows = [
        row
        for row in rows
        if not (
            row["language"] == "russian"
            and row["complied?"] == "true"
            and str(row["prompt"]).endswith("-1).")
        )
    ]
    raw_path = tmp_path / "insufficient.csv"
    _write_csv(raw_path, RAW_FIELDS, rows)
    output = tmp_path / "must-not-exist"

    with pytest.raises(AuditError, match="insufficient rows.*russian/complied=1"):
        generate_audit([raw_path], output, per_stratum=2, duplicate_count=6)

    assert not output.exists()


def test_generate_rejects_duplicate_raw_identities_and_existing_output(tmp_path: Path) -> None:
    raw_path = tmp_path / "duplicated.csv"
    rows = _raw_fixture_rows()
    rows.append(dict(rows[0]))
    _write_csv(raw_path, RAW_FIELDS, rows)

    with pytest.raises(AuditError, match="duplicate raw audit identity"):
        generate_audit([raw_path], tmp_path / "duplicate-output", per_stratum=2, duplicate_count=6)

    valid_path = tmp_path / "valid.csv"
    _write_csv(valid_path, RAW_FIELDS, _raw_fixture_rows())
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(AuditError, match="refusing to overwrite"):
        generate_audit([valid_path], existing, per_stratum=2, duplicate_count=6)


def test_score_fails_without_completed_human_labels_and_writes_nothing(tmp_path: Path) -> None:
    output = _generated_audit(tmp_path)
    score_path = tmp_path / "scores.json"

    with pytest.raises(AuditError, match="human annotations are absent or incomplete"):
        score_audit(
            output / "audit_key.csv",
            output / "annotator_packet.csv",
            output / "duplicate_packet.csv",
            output_path=score_path,
            bootstrap_replicates=10,
        )

    assert not score_path.exists()


def test_score_reports_weighted_metrics_cis_languages_and_duplicate_kappa(
    tmp_path: Path,
) -> None:
    output = _generated_audit(tmp_path, english_complied_count=4)
    _complete_annotations(output, add_unclear=True)
    score_path = tmp_path / "scores.json"

    result = score_audit(
        output / "audit_key.csv",
        output / "annotator_packet.completed.csv",
        output / "duplicate_packet.completed.csv",
        output_path=score_path,
        bootstrap_replicates=80,
        seed=44,
    )
    repeated = score_audit(
        output / "audit_key.csv",
        output / "annotator_packet.completed.csv",
        output / "duplicate_packet.completed.csv",
        bootstrap_replicates=80,
        seed=44,
    )

    assert json.loads(score_path.read_text(encoding="utf-8")) == result
    assert repeated["bootstrap_confidence_intervals"] == result["bootstrap_confidence_intervals"]
    overall = result["overall"]
    cells = overall["weighted_confusion"]["cells"]
    assert cells["denied"]["complied"] == pytest.approx(2.0)
    assert cells["unclear"]["complied"] == pytest.approx(1.0)
    assert overall["weighted_confusion"]["total_weight"] == pytest.approx(14.0)
    assert overall["determinate_weight"] == pytest.approx(13.0)
    assert overall["accuracy"] == pytest.approx(11 / 13)
    assert overall["balanced_accuracy"] == pytest.approx(0.875)
    assert overall["per_class"]["complied"]["precision"] == pytest.approx(5 / 7)
    assert overall["per_class"]["denied"]["recall"] == pytest.approx(0.75)
    assert set(result["per_language"]) == {"english", "chinese", "russian"}
    assert result["per_language"]["english"]["weighted_confusion"]["total_weight"] == 6.0
    bootstrap = result["bootstrap_confidence_intervals"]
    assert bootstrap["method"] == "stratified_nonparametric_percentile"
    assert bootstrap["replicates"] == 80
    assert bootstrap["overall"]["balanced_accuracy"]["lower"] is not None
    assert set(bootstrap["per_language"]) == {"english", "chinese", "russian"}
    reliability = result["intra_rater_reliability"]
    assert reliability["n_pairs"] == 6
    assert reliability["observed_agreement"] == 1.0
    assert reliability["cohen_kappa"] == 1.0


def test_score_enforces_annotation_schema_values_and_id_completeness(tmp_path: Path) -> None:
    output = _generated_audit(tmp_path)
    _complete_annotations(output)
    _, rows = _read_csv(output / "annotator_packet.completed.csv")
    rows[0]["confidence"] = "high"
    invalid_path = output / "annotator_packet.invalid.csv"
    _write_csv(invalid_path, ANNOTATOR_FIELDS, rows)

    with pytest.raises(AuditError, match="confidence must be an integer 1-5"):
        score_audit(
            output / "audit_key.csv",
            invalid_path,
            output / "duplicate_packet.completed.csv",
            bootstrap_replicates=10,
        )

    assert "translation_or_language_issue" in ERROR_TYPES
    rows[0]["confidence"] = "4"
    rows[0]["response_original"] += " changed"
    changed_path = output / "annotator_packet.changed.csv"
    _write_csv(changed_path, ANNOTATOR_FIELDS, rows)
    with pytest.raises(AuditError, match="stimulus columns changed"):
        score_audit(
            output / "audit_key.csv",
            changed_path,
            output / "duplicate_packet.completed.csv",
            bootstrap_replicates=10,
        )

    rows[0]["response_original"] = rows[0]["response_original"].removesuffix(" changed")
    rows.pop()
    incomplete_path = output / "annotator_packet.incomplete.csv"
    _write_csv(incomplete_path, ANNOTATOR_FIELDS, rows)
    with pytest.raises(AuditError, match="IDs do not match the key"):
        score_audit(
            output / "audit_key.csv",
            incomplete_path,
            output / "duplicate_packet.completed.csv",
            bootstrap_replicates=10,
        )
