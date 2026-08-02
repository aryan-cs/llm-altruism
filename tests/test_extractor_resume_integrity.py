from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import campaign
from experiments.misc.run_metadata import metadata_payload_sha256
from experiments.part0 import part_0
from experiments.part1 import part_1
from experiments.part2 import part_2


def _generation_record(extracted: dict[str, str]) -> SimpleNamespace:
    value = {
        "protocol": "independent-final-answer-extraction-v3",
        "status": "success",
        "kind": "test",
        "subject": {
            "provider": "openai",
            "model": "gpt-4.1-mini",
            "truncated": False,
            "model_identity_match": True,
        },
        "extractor": {
            "provider": "inference_hub",
            "model": "google/gemma-3-27b-it",
            "response": {
                "truncated": False,
                "model_identity_match": True,
            },
        },
        "extracted_final": extracted,
    }
    return SimpleNamespace(to_dict=lambda: value)


def _part1_partial_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(part_1, "run_experiment_preflight", lambda *args, **kwargs: None)
    calls = 0

    def interrupt_after_one(self, query, **kwargs):
        nonlocal calls
        del self, query, kwargs
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        extracted = {"action": "DEFECT", "justification": "Visible answer."}
        return json.dumps(extracted), _generation_record(extracted)

    monkeypatch.setattr(part_1.Agent1, "query_for_grading", interrupt_after_one)
    csv_path = part_1.run_part_1(
        provider="openai",
        model="gpt-4.1-mini",
        games=["prisoners_dilemma"],
        frames=["advice"],
        domains=["workplace"],
        presentations=["structured"],
        limit=2,
        output_token_cap=512,
    )
    return Path(csv_path)


def _part2_partial_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(part_2, "run_experiment_preflight", lambda *args, **kwargs: None)
    calls = 0

    def interrupt_on_day_two(self, query, **kwargs):
        nonlocal calls
        del self, query, kwargs
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt()
        extracted = {"action": "RESTRAIN", "reasoning": "Visible answer."}
        return json.dumps(extracted), _generation_record(extracted)

    monkeypatch.setattr(part_2.Agent2, "query_for_grading", interrupt_on_day_two)
    csv_path = part_2.run_part_2(
        provider="openai",
        model="gpt-4.1-mini",
        society_size=2,
        days=2,
        resource="water",
        selfish_gain=2,
        depletion_units=2,
        community_benefit=5,
        output_token_cap=512,
    )
    return Path(csv_path)


def test_strict_part1_resume_accepts_untampered_contract_and_hash_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = _part1_partial_run(monkeypatch, tmp_path)

    def finish(self, query, **kwargs):
        del self, query, kwargs
        extracted = {"action": "DEFECT", "justification": "Visible answer."}
        return json.dumps(extracted), _generation_record(extracted)

    monkeypatch.setattr(part_1.Agent1, "query_for_grading", finish)
    assert Path(part_1.run_part_1(resume=True)) == csv_path
    metadata = json.loads(
        csv_path.with_name(f"{csv_path.stem}_meta.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "complete"
    assert metadata["attempt_log"]["hash_chain_status"] == "verified"
    assert metadata["artifact_integrity"]["results"]["sha256"]


def test_strict_part1_resume_rejects_attempt_metadata_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = _part1_partial_run(monkeypatch, tmp_path)
    metadata_path = csv_path.with_name(f"{csv_path.stem}_meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["attempt_log"]["total_attempts"] += 1
    metadata["metadata_sha256"] = metadata_payload_sha256(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="Attempt log metadata mismatch"):
        part_1.run_part_1(resume=True)


def test_strict_part2_resume_rejects_result_or_contract_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = _part2_partial_run(monkeypatch, tmp_path)
    csv_path.write_text(
        csv_path.read_text(encoding="utf-8").replace("RESTRAIN", "OVERUSE", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="integrity mismatch"):
        part_2.run_part_2(resume=True)


def test_strict_part2_resume_rejects_changed_run_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    csv_path = _part2_partial_run(monkeypatch, tmp_path)
    metadata_path = csv_path.with_name(f"{csv_path.stem}_meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["society_config"]["resource"] = "oil"
    metadata["metadata_sha256"] = metadata_payload_sha256(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="resume contract mismatch"):
        part_2.run_part_2(resume=True)


def test_strict_part0_resume_rejects_changed_extractor_prompt_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(part_0, "run_experiment_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(part_0.Agent0, "query_for_grading", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(part_0, "unload_ollama_model", lambda *args, **kwargs: None)
    csv_path = Path(
        part_0.run_alignment_test(
            models={"openai": ["gpt-4.1-mini"]},
            prompts=["prompt"],
            languages=["english"],
            output_token_cap=512,
        )
    )
    metadata_path = csv_path.with_name(f"{csv_path.stem}_meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["grading_protocol"]["extractor"]["prompt_template_sha256"] = "0" * 64
    metadata["metadata_sha256"] = metadata_payload_sha256(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="does not exactly match"):
        part_0.run_alignment_test(resume=True)


def test_campaign_verifies_schema_hashes_units_attempts_and_resume_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(part_1, "run_experiment_preflight", lambda *args, **kwargs: None)

    def answer(self, query, **kwargs):
        del self, query, kwargs
        extracted = {"action": "DEFECT", "justification": "Visible answer."}
        return json.dumps(extracted), _generation_record(extracted)

    monkeypatch.setattr(part_1.Agent1, "query_for_grading", answer)
    csv_path = Path(
        part_1.run_part_1(
            provider="openai",
            model="gpt-4.1-mini",
            games=["prisoners_dilemma"],
            frames=["advice"],
            domains=["workplace"],
            presentations=["structured"],
            limit=1,
            output_token_cap=512,
        )
    )
    metadata_path = csv_path.with_name(f"{csv_path.stem}_meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["git_commit"] = "a" * 40
    metadata["git_dirty"] = False
    metadata["metadata_sha256"] = metadata_payload_sha256(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(campaign, "REPO_ROOT", tmp_path)
    job = {
        "id": "strict-part1",
        "experiment": "part_1",
        "expected": {
            "provider": metadata["provider"],
            "model": metadata["model"],
            "games": metadata["games"],
            "frames": metadata["frames"],
            "domains": metadata["domains"],
            "presentations": metadata["presentations"],
            "limit": metadata["limit"],
            "ordering": {
                key: metadata["ordering"][key]
                for key in ("seed", "strategy", "counterbalance_index")
            },
            "row_count": 1,
            "grading_protocol": metadata["grading_protocol"],
            "campaign_git_commit": metadata["git_commit"],
        },
    }

    verified = campaign.verify_artifact(job, metadata_path)
    assert verified["rows"] == 1

    csv_path.write_text(
        csv_path.read_text(encoding="utf-8").replace("DEFECT", "COOPERATE", 1),
        encoding="utf-8",
    )
    with pytest.raises(campaign.CampaignError, match="integrity"):
        campaign.verify_artifact(job, metadata_path)
