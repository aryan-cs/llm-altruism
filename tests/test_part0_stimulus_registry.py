import csv
import hashlib
import io

import pytest

from experiments.part0 import stimulus_registry
from experiments.part0.stimulus_registry import (
    StimulusRegistryError,
    build_stimulus_registry,
    normalized_prompt_key,
    validate_stimulus_registry,
)


def _csv_bytes(headers: list[str], rows: list[list[str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _payloads() -> dict[str, bytes]:
    return {
        "harmbench_harmful": _csv_bytes(
            [
                "Behavior",
                "FunctionalCategory",
                "SemanticCategory",
                "Tags",
                "ContextString",
                "BehaviorID",
            ],
            [
                ["Harmful   prompt one", "standard", "cyber", "tag-a", "", "hb-1"],
                ["Harmful prompt two", "standard", "fraud", "", "", "hb-2"],
            ],
        ),
        "jbb_harmful": _csv_bytes(
            ["Index", "Goal", "Target", "Behavior", "Category", "Source"],
            [
                ["0", "Harmful prompt one", "target", "Malware", "Cyber", "TDC"],
                ["1", "Harmful prompt three", "target", "Fraud", "Economic", "Original"],
            ],
        ),
        "jbb_benign": _csv_bytes(
            ["Index", "Goal", "Target", "Behavior", "Category", "Source"],
            [
                ["0", "Benign prompt one", "target", "Malware", "Cyber", "Original"],
                ["1", "Benign prompt two", "target", "Fraud", "Economic", "Original"],
            ],
        ),
    }


def _install_payload_hashes(
    monkeypatch: pytest.MonkeyPatch,
    payloads: dict[str, bytes],
) -> None:
    specs = {
        name: {
            "url": f"https://example.test/{name}.csv",
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in payloads.items()
    }
    monkeypatch.setattr(stimulus_registry, "SOURCE_SPECS", specs)


def test_normalized_prompt_key_collapses_unicode_and_whitespace() -> None:
    assert normalized_prompt_key("  Ｈarmful\n prompt  ") == "harmful prompt"


def test_registry_deduplicates_and_preserves_every_source_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _payloads()
    _install_payload_hashes(monkeypatch, payloads)

    registry = build_stimulus_registry(payloads)

    assert registry["counts"] == {
        "harmful_source_rows": 4,
        "harmful_unique_prompts": 3,
        "harmful_duplicates_collapsed": 1,
        "paired_control_prompts": 2,
        "jbb_pairs": 2,
    }
    harmful = [item for item in registry["stimuli"] if item["arm"] == "harmful"]
    controls = [item for item in registry["stimuli"] if item["arm"] == "control"]
    overlapped = next(
        item
        for item in harmful
        if normalized_prompt_key(item["prompt_english"]) == "harmful prompt one"
    )
    assert {row["source_name"] for row in overlapped["provenance"]} == {
        "harmbench",
        "jbb",
    }
    assert overlapped["pair_id"] == "jbb_pair_000"
    assert [item["pair_id"] for item in controls] == [
        "jbb_pair_000",
        "jbb_pair_001",
    ]
    assert len({item["base_prompt_id"] for item in registry["stimuli"]}) == 5
    with pytest.raises(StimulusRegistryError, match="response instruction"):
        validate_stimulus_registry(registry, require_production_approval=True)

    # Synthetic approvals exercise the production gate only; they are not
    # research annotations or artifacts retained by the repository.
    registry["response_instructions"] = {}
    for language, text in (
        ("english", "Unit-test English response instruction."),
        ("chinese", "单元测试中文回复说明。"),
        ("russian", "Тестовая инструкция ответа."),
    ):
        registry["response_instructions"][language] = {
            "text": text,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "review_status": "approved",
            "reviewer_ids": ["unit-test-reviewer"],
        }
    for item in registry["stimuli"]:
        item["semantic_cluster_id"] = item["base_prompt_id"]
        item["semantic_cluster_review"] = {
            "status": "approved",
            "reviewer_ids": ["unit-test-reviewer"],
        }
        item["localized_stimuli"]["english"]["review_status"] = "approved"
        item["localized_stimuli"]["english"]["reviewer_ids"] = [
            "unit-test-reviewer"
        ]
        for language, text in (
            ("chinese", f"测试译文 {item['base_prompt_id']}"),
            ("russian", f"Тестовый перевод {item['base_prompt_id']}"),
        ):
            item["localized_stimuli"][language] = {
                "text": text,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "translation_method": "unit_test_fixture",
                "review_status": "approved",
                "reviewer_ids": ["unit-test-reviewer"],
            }
    validate_stimulus_registry(registry, require_production_approval=True)


def test_registry_rejects_mismatched_pair_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _payloads()
    decoded = payloads["jbb_benign"].decode("utf-8").replace(
        "Malware,Cyber", "Unrelated,Cyber"
    )
    payloads["jbb_benign"] = decoded.encode("utf-8")
    _install_payload_hashes(monkeypatch, payloads)

    with pytest.raises(StimulusRegistryError, match="mismatched behaviors"):
        build_stimulus_registry(payloads)


def test_registry_rejects_unexpected_or_missing_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _payloads()
    _install_payload_hashes(monkeypatch, payloads)
    payloads.pop("jbb_benign")

    with pytest.raises(StimulusRegistryError, match="Source payload keys differ"):
        build_stimulus_registry(payloads)
