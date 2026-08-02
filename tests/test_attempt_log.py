from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.misc.attempt_log import DurableAttemptLogger, summarize_attempt_log


def test_durable_attempt_log_preserves_each_outcome_across_resume(tmp_path: Path) -> None:
    path = tmp_path / "run_attempts.jsonl"
    logger = DurableAttemptLogger(path, experiment="part_1")
    common = {
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "unit_id": "prompt-1",
        "unit": {"prompt_id": "prompt-1"},
        "max_attempts": 3,
        "prompt_text": "Choose A or B.",
    }
    logger.append(
        **common,
        attempt=1,
        outcome="invalid_response",
        raw_response="not-json",
        error={"exception_type": "ResponseParseError", "message": "Invalid JSON"},
        will_retry=True,
    )
    logger.append(
        **common,
        attempt=2,
        outcome="success",
        raw_response='{"action":"A"}',
        parsed_response={"action": "A"},
    )

    resumed = DurableAttemptLogger(path, experiment="part_1")
    resumed.append(
        provider="openai",
        model="gpt-4.1-mini",
        unit_id="prompt-2",
        unit={"prompt_id": "prompt-2"},
        attempt=1,
        max_attempts=3,
        prompt_text="Choose C or D.",
        outcome="provider_error",
        error={"exception_type": "TimeoutError", "message": "timed out"},
    )

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["sequence"] for record in records] == [1, 2, 3]
    assert [record["outcome"] for record in records] == [
        "invalid_response",
        "success",
        "provider_error",
    ]
    assert records[0]["raw_response"] == "not-json"
    assert records[1]["parsed_response"] == {"action": "A"}
    assert records[1]["prompt_sha256"] == hashlib.sha256(
        b"Choose A or B."
    ).hexdigest()

    summary = summarize_attempt_log(path)
    assert summary.total_attempts == 3
    assert summary.successful_attempts == 1
    assert summary.invalid_response_attempts == 1
    assert summary.provider_error_attempts == 1
    assert summary.retry_attempts == 1
    assert summary.retried_units == 1
    assert summary.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_attempt_log_refuses_to_append_after_truncated_json(tmp_path: Path) -> None:
    path = tmp_path / "truncated.jsonl"
    path.write_text('{"outcome":"success"', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        DurableAttemptLogger(path, experiment="part_2")
