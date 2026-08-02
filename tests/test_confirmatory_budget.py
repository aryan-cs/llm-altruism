from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.confirmatory_budget import (
    ConfirmatoryBudgetError,
    build_frozen_budget,
    create_ledger,
    record_attempt,
    reserve_environment_attempt,
    successful_posts_by_role,
    validate_frozen_budget,
    validate_ledger,
)


def test_exact_thirty_route_budget_arithmetic() -> None:
    base = successful_posts_by_role(30, include_rejudge_allowance=False)
    assert base == {
        "discovery": 30,
        "part0_subject": 52_740,
        "part0_judge": 52_740,
        "part1_subject": 11_880,
        "part2_subject": 216_360,
    }
    budget = build_frozen_budget(30)
    assert budget["base_successful_posts"] == 333_750
    assert budget["rejudge_inclusive_posts"] == 386_328
    assert budget["planned_physical_attempt_bound"] == 424_961
    assert budget["physical_attempt_ceiling"] == 430_000
    assert budget["base_maximum_output_tokens"] == 35_994_720
    assert validate_frozen_budget(budget) == budget


def test_budget_rejects_attempt_ceiling_below_frozen_bound() -> None:
    with pytest.raises(ConfirmatoryBudgetError, match="exceeding the ceiling"):
        build_frozen_budget(30, physical_attempt_ceiling=424_960)


def test_budget_self_hash_and_design_are_fail_closed() -> None:
    budget = build_frozen_budget()
    tampered = deepcopy(budget)
    tampered["design"]["part2_runs"] = 15
    with pytest.raises(ConfirmatoryBudgetError, match="self-hash"):
        validate_frozen_budget(tampered)
    tampered["budget_sha256"] = budget["budget_sha256"]
    with pytest.raises(ConfirmatoryBudgetError):
        validate_frozen_budget(tampered)


def test_ledger_records_usage_and_replays() -> None:
    budget = build_frozen_budget()
    ledger = create_ledger(budget)
    updated = record_attempt(
        ledger,
        budget,
        role="part1_subject",
        attempt_id="part1-route-a-root-001-attempt-1",
        request_sha256="a" * 64,
        outcome="success",
        input_tokens=123,
        output_tokens=4,
    )
    assert updated["physical_attempts"] == 1
    assert updated["attempts_by_role"]["part1_subject"] == 1
    assert updated["input_tokens"] == 123
    assert updated["output_tokens"] == 4
    assert validate_ledger(updated, budget) == updated


def test_ledger_rejects_duplicate_and_tampering() -> None:
    budget = build_frozen_budget()
    ledger = record_attempt(
        create_ledger(budget),
        budget,
        role="discovery",
        attempt_id="route-a",
        request_sha256="b" * 64,
        outcome="success",
        input_tokens=1,
        output_tokens=1,
    )
    with pytest.raises(ConfirmatoryBudgetError, match="already present"):
        record_attempt(
            ledger,
            budget,
            role="discovery",
            attempt_id="route-a",
            request_sha256="b" * 64,
            outcome="success",
            input_tokens=1,
            output_tokens=1,
        )
    tampered = deepcopy(ledger)
    tampered["physical_attempts"] = 0
    with pytest.raises(ConfirmatoryBudgetError, match="self-hash"):
        validate_ledger(tampered, budget)


def test_role_cap_and_global_token_cap_are_checked_before_append() -> None:
    budget = build_frozen_budget(1)
    ledger = create_ledger(budget)
    saturated = deepcopy(ledger)
    saturated.pop("ledger_sha256")
    cap = budget["role_attempt_caps"]["discovery"]
    saturated["records"] = [
        {
            "attempt_id": f"d-{index}",
            "role": "discovery",
            "request_sha256": "c" * 64,
            "outcome": "provider_error",
            "input_tokens": 1,
            "output_tokens": 0,
        }
        for index in range(cap)
    ]
    saturated["attempts_by_role"]["discovery"] = cap
    saturated["physical_attempts"] = cap
    saturated["input_tokens"] = cap
    hash_payload = deepcopy(saturated)
    hash_payload.pop("ledger_sha256", None)
    saturated["ledger_sha256"] = hashlib.sha256(
        json.dumps(
            hash_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ConfirmatoryBudgetError, match="Role cap"):
        record_attempt(
            saturated,
            budget,
            role="discovery",
            attempt_id="too-many",
            request_sha256="d" * 64,
            outcome="provider_error",
            input_tokens=1,
            output_tokens=0,
        )


def test_environment_reservation_is_durable_and_role_specific(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = build_frozen_budget(1)
    budget_path = tmp_path / "budget.json"
    ledger_path = tmp_path / "ledger.json"
    budget_path.write_text(json.dumps(budget), encoding="utf-8")
    ledger_path.write_text(json.dumps(create_ledger(budget)), encoding="utf-8")
    monkeypatch.setenv("CONFIRMATORY_BUDGET_PATH", str(budget_path))
    monkeypatch.setenv("CONFIRMATORY_LEDGER_PATH", str(ledger_path))
    monkeypatch.setenv("CONFIRMATORY_EXPERIMENT", "part0")
    reserve_environment_attempt(
        provider="inference_hub",
        model="judge-route",
        system_prompt="judge",
        query="classify",
        max_tokens=32,
    )
    reserve_environment_attempt(
        provider="inference_hub",
        model="subject-route",
        system_prompt="subject",
        query="respond",
        max_tokens=512,
    )
    ledger = validate_ledger(
        json.loads(ledger_path.read_text(encoding="utf-8")), budget
    )
    assert ledger["physical_attempts"] == 2
    assert ledger["attempts_by_role"]["part0_judge"] == 1
    assert ledger["attempts_by_role"]["part0_subject"] == 1
    assert all(
        record["outcome"] == "reserved_before_dispatch"
        for record in ledger["records"]
    )

    tiny = build_frozen_budget(1, token_ceiling=1_200_000)
    current = create_ledger(tiny)
    with pytest.raises(ConfirmatoryBudgetError, match="Token ceiling"):
        record_attempt(
            current,
            tiny,
            role="part0_subject",
            attempt_id="too-large",
            request_sha256="e" * 64,
            outcome="success",
            input_tokens=1_200_000,
            output_tokens=1,
        )
