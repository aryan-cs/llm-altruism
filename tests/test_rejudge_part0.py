from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from analysis import rejudge_part0


FIELDS = ["provider", "model", "language", "prompt", "reasoning", "response", "complied?"]


class SequenceJudge:
    def __init__(self, results: list[object]) -> None:
        self.results = iter(results)
        self.prompts: list[str] = []
        self.timeouts: list[float] = []

    def judge(self, prompt: str, *, timeout: float) -> object:
        self.prompts.append(prompt)
        self.timeouts.append(timeout)
        result = next(self.results)
        if isinstance(result, BaseException):
            raise result
        return result


class AlwaysFailJudge:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def judge(self, prompt: str, *, timeout: float) -> str:
        del timeout
        self.prompts.append(prompt)
        raise TimeoutError("judge deadline expired")


class InterruptingJudge:
    def __init__(self) -> None:
        self.calls = 0

    def judge(self, prompt: str, *, timeout: float) -> str:
        del prompt, timeout
        self.calls += 1
        if self.calls == 2:
            raise KeyboardInterrupt
        return '{"complied": false}'


def _source_rows() -> list[dict[str, str]]:
    return [
        {
            "provider": "ollama",
            "model": "model-a",
            "language": "russian",
            "prompt": "base request one",
            "reasoning": "PRIVATE_RATIONALE_CANARY_ONE",
            "response": "original response one",
            "complied?": "true",
        },
        {
            "provider": "ollama",
            "model": "model-b",
            "language": "english",
            "prompt": "base request two",
            "reasoning": "PRIVATE_RATIONALE_CANARY_TWO",
            "response": "original response two",
            "complied?": "false",
        },
    ]


def _write_input(path: Path, rows: list[dict[str, str]] | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows or _source_rows())


def _read_output(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_response_only_prompts_never_contain_rationale_canaries(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "rejudged.csv"
    _write_input(source)
    judge = SequenceJudge(['{"complied": false}', {"complied": True}])

    summary = rejudge_part0.rejudge_csv(
        source,
        output,
        judge=judge,
        max_attempts=1,
        timeout=7.5,
        retry_delay=0,
    )

    assert len(judge.prompts) == 2
    assert judge.timeouts == [7.5, 7.5]
    assert all("PRIVATE_RATIONALE_CANARY" not in prompt for prompt in judge.prompts)
    assert "base request one" in judge.prompts[0]
    assert "original response one" in judge.prompts[0]
    assert "localized request" in judge.prompts[0]
    assert "original response two" in judge.prompts[1]
    rows = _read_output(output)
    assert rows[0]["reasoning"] == "PRIVATE_RATIONALE_CANARY_ONE"
    assert rows[0]["new_verdict"] == "denied"
    assert rows[1]["new_verdict"] == "complied"
    assert summary["overall"]["disagreements"] == 2
    assert summary["by_language"]["russian"]["total"] == 1
    assert summary["by_model"]["ollama/model-b"]["judged"] == 1


def test_interrupted_checkpoint_resumes_without_rejudging_completed_row(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "rejudged.csv"
    _write_input(source)

    with pytest.raises(KeyboardInterrupt):
        rejudge_part0.rejudge_csv(
            source,
            output,
            judge=InterruptingJudge(),
            max_attempts=1,
            retry_delay=0,
        )

    partial = _read_output(output)
    assert len(partial) == 1
    first_id = partial[0]["row_id"]
    resume_judge = SequenceJudge([{"complied": True}])
    summary = rejudge_part0.rejudge_csv(
        source,
        output,
        judge=resume_judge,
        max_attempts=1,
        retry_delay=0,
        resume=True,
    )

    rows = _read_output(output)
    assert len(rows) == 2
    assert len({row["row_id"] for row in rows}) == 2
    assert rows[0]["row_id"] == first_id
    assert len(resume_judge.prompts) == 1
    assert "base request two" in resume_judge.prompts[0]
    assert summary["resumed_rows"] == 1
    assert summary["processed_this_run"] == 1
    assert summary["complete"] is True


def test_resume_rejects_checkpoint_whose_original_fields_changed(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "rejudged.csv"
    _write_input(source)
    rejudge_part0.rejudge_csv(
        source,
        output,
        judge=SequenceJudge([{"complied": False}, {"complied": False}]),
        max_attempts=1,
        retry_delay=0,
    )
    rows = _read_output(output)
    rows[0]["response"] = "tampered checkpoint response"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(rejudge_part0.ResumeIntegrityError, match="differs"):
        rejudge_part0.rejudge_csv(
            source,
            output,
            judge=SequenceJudge([]),
            max_attempts=1,
            retry_delay=0,
            resume=True,
        )


def test_resume_rejects_mixed_judge_protocol(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "rejudged.csv"
    _write_input(source)
    rejudge_part0.rejudge_csv(
        source,
        output,
        judge=SequenceJudge([{"complied": False}, {"complied": False}]),
        max_attempts=1,
        retry_delay=0,
    )

    with pytest.raises(rejudge_part0.ResumeIntegrityError, match="judge_model differs"):
        rejudge_part0.rejudge_csv(
            source,
            output,
            judge=SequenceJudge([]),
            model="a-different-judge",
            max_attempts=1,
            retry_delay=0,
            resume=True,
        )


def test_exhausted_or_invalid_judge_is_explicitly_unjudged(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "rejudged.csv"
    _write_input(source, [_source_rows()[0], _source_rows()[1]])
    failure = AlwaysFailJudge()

    summary = rejudge_part0.rejudge_csv(
        source,
        output,
        judge=failure,
        max_attempts=2,
        timeout=0.25,
        retry_delay=0,
    )

    rows = _read_output(output)
    assert len(failure.prompts) == 4
    assert [row["new_verdict"] for row in rows] == ["unjudged", "unjudged"]
    assert [row["new_complied"] for row in rows] == ["", ""]
    assert [row["attempts"] for row in rows] == ["2", "2"]
    assert all("TimeoutError" in row["error"] for row in rows)
    assert summary["overall"]["judged"] == 0
    assert summary["overall"]["unjudged"] == 2
    assert summary["overall"]["disagreements"] == 0
    assert summary["complete"] is False


def test_retry_unjudged_replaces_only_failed_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "rejudged.csv"
    _write_input(source)
    rejudge_part0.rejudge_csv(
        source,
        output,
        judge=SequenceJudge([TimeoutError("first failed"), {"complied": False}]),
        max_attempts=1,
        retry_delay=0,
    )
    before = _read_output(output)
    assert [row["new_verdict"] for row in before] == ["unjudged", "denied"]
    retained_id = before[1]["row_id"]

    retry_judge = SequenceJudge([{"complied": True}])
    summary = rejudge_part0.rejudge_csv(
        source,
        output,
        judge=retry_judge,
        max_attempts=1,
        retry_delay=0,
        resume=True,
        retry_unjudged=True,
    )

    after = _read_output(output)
    assert len(after) == 2
    assert len({row["row_id"] for row in after}) == 2
    assert retained_id in {row["row_id"] for row in after}
    assert summary["overall"]["unjudged"] == 0
    assert summary["complete"] is True


def test_invalid_payload_is_not_coerced_to_a_label(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "rejudged.csv"
    _write_input(source, [_source_rows()[0]])

    rejudge_part0.rejudge_csv(
        source,
        output,
        judge=SequenceJudge([{"complied": "false", "explanation": "not schema-valid"}]),
        max_attempts=1,
        retry_delay=0,
    )

    row = _read_output(output)[0]
    assert row["new_verdict"] == "unjudged"
    assert row["new_complied"] == ""
    assert "JudgeResponseError" in row["error"]


def test_input_is_never_overwritten_and_existing_output_requires_resume(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "existing.csv"
    _write_input(source)
    original_bytes = source.read_bytes()

    with pytest.raises(ValueError, match="never overwritten"):
        rejudge_part0.rejudge_csv(
            source,
            source,
            judge=SequenceJudge([]),
            max_attempts=1,
            retry_delay=0,
        )
    assert source.read_bytes() == original_bytes

    output.write_text("do not replace me\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--resume"):
        rejudge_part0.rejudge_csv(
            source,
            output,
            judge=SequenceJudge([]),
            max_attempts=1,
            retry_delay=0,
        )
    assert output.read_text(encoding="utf-8") == "do not replace me\n"


def test_retry_timeout_and_concurrency_bounds_are_enforced(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _write_input(source)

    for suffix, kwargs in (
        ("attempts", {"max_attempts": rejudge_part0.MAX_ATTEMPTS_LIMIT + 1}),
        ("timeout", {"timeout": float("inf")}),
        ("workers", {"workers": rejudge_part0.MAX_WORKERS + 1}),
    ):
        with pytest.raises(ValueError):
            rejudge_part0.rejudge_csv(
                source,
                tmp_path / f"{suffix}.csv",
                judge=SequenceJudge([]),
                retry_delay=0,
                **kwargs,
            )
        assert not (tmp_path / f"{suffix}.csv").exists()


def test_row_ids_and_provenance_are_stable_and_document_protocol(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    output_one = tmp_path / "one.csv"
    output_two = tmp_path / "two.csv"
    _write_input(source)
    judgments = [{"complied": False}, {"complied": True}]

    rejudge_part0.rejudge_csv(
        source,
        output_one,
        judge=SequenceJudge(list(judgments)),
        max_attempts=1,
        retry_delay=0,
    )
    rejudge_part0.rejudge_csv(
        source,
        output_two,
        judge=SequenceJudge(list(judgments)),
        max_attempts=1,
        retry_delay=0,
    )

    first = _read_output(output_one)
    second = _read_output(output_two)
    assert [row["row_id"] for row in first] == [row["row_id"] for row in second]
    assert all(row["row_id"].startswith("p0_") and len(row["row_id"]) == 67 for row in first)
    provenance = json.loads(first[0]["judge_provenance"])
    assert first[0]["judge_provider"] == "ollama"
    assert first[0]["judge_model"] == "gpt-oss:20b"
    assert first[0]["judge_protocol"] == rejudge_part0.PROTOCOL
    assert provenance["adjudication_basis"] == "model_response_only"
    assert provenance["localized_prompt_available"] is False
    assert provenance["original_language_response_used"] is True
    assert len(provenance["judge_rubric_sha256"]) == 64
    assert len(provenance["judge_system_prompt_sha256"]) == 64


def test_rejudge_metadata_freezes_artifact_and_judge_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "rejudged.csv"
    _write_input(source)
    summary = rejudge_part0.rejudge_csv(
        source,
        output,
        judge=SequenceJudge([{"complied": False}, {"complied": True}]),
        max_attempts=1,
        retry_delay=0,
    )

    metadata = rejudge_part0.build_rejudge_metadata(
        input_path=source,
        output_path=output,
        summary=summary,
        provider="ollama",
        model="gpt-oss:20b",
        ollama_host="http://127.0.0.1:11435",
        judge_model_digest="digest-abc",
        judge_server_version="0.test",
        judge_server_config="parallel=4,context=32768",
    )

    assert metadata["status"] == "complete"
    assert metadata["row_count"] == 2
    assert len(metadata["source_csv_sha256"]) == 64
    assert len(metadata["csv_sha256"]) == 64
    assert metadata["judge"]["model_digest"] == "digest-abc"
    assert metadata["judge"]["decoding"]["think"] == "low"
    assert metadata["human_validation_status"] == "not_completed"
