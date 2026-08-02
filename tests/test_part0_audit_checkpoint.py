from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "data" / "analysis" / "part0_rejudge_audit_checkpoint.json"


def test_sanitized_part0_checkpoint_is_internally_consistent_and_source_bound() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    adjudication = payload["adjudication"]
    transitions = (
        adjudication["legacy_denied_to_response_only_complied"],
        adjudication["legacy_complied_to_response_only_denied"],
        adjudication["unchanged_denied"],
        adjudication["unchanged_complied"],
    )
    assert sum(transitions) == adjudication["adjudicated_rows"] == 1243
    assert sum(transitions[:2]) == adjudication["changed_rows"] == 300
    assert adjudication["changed_fraction"] == 300 / 1243
    assert sum(payload["coverage"]["by_language"].values()) == 1243
    assert sum(payload["coverage"]["by_model"].values()) == 1243

    source = ROOT / payload["source"]["relative_path"]
    if not source.is_file():
        return
    source_bytes = source.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == payload["source"]["sha256"]
    assert len(source_bytes) == payload["source"]["size_bytes"]
    rows = list(csv.DictReader(source_bytes.decode("utf-8").splitlines()))
    judged = [row for row in rows if row["new_complied"].lower() in {"true", "false"}]
    counts = Counter(
        (row["legacy_complied"].lower(), row["new_complied"].lower())
        for row in judged
    )
    assert len(rows) == payload["source"]["total_rows"]
    assert counts == {
        ("false", "false"): 650,
        ("false", "true"): 25,
        ("true", "false"): 275,
        ("true", "true"): 293,
    }
