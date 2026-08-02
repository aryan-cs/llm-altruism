from __future__ import annotations

import hashlib
import json
import stat
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from experiments.misc.attempt_log import DurableAttemptLogger, load_attempt_records
from experiments.part1 import confirmatory_runner
from experiments.part1.confirmatory_design import (
    APPROVED_REVIEW,
    INDEPENDENT_AUTHORSHIP,
    HumanContentApproval,
    MappingReview,
    build_draft_bank,
    rehash_root,
)
from experiments.part1.confirmatory_runner import (
    EXPECTED_PRIMARY_TRIALS,
    EXPECTED_SECONDARY_TRIALS,
    EXPECTED_TRIALS_PER_MODEL,
    ConfirmatoryPart1Error,
    FrozenRoute,
    RouteIdentityError,
    build_confirmatory_schedule,
    execute_trial,
    freeze_execution_plan,
    load_production_bank,
    run_frozen_plan,
    validate_execution_plan,
)
from providers.api_call import ProviderResponse


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _approved_roots():
    roots = []
    for root in build_draft_bank():
        production_root = rehash_root(
            replace(
                root,
                authorship_method=INDEPENDENT_AUTHORSHIP,
                mapping_review=MappingReview(
                    moral_confounds_status=APPROVED_REVIEW,
                    welfare_mapping_status=APPROVED_REVIEW,
                ),
            )
        )
        approvals = tuple(
            HumanContentApproval(
                reviewer_slot=slot,
                reviewer_id=f"real-test-reviewer-{slot}",
                decision=APPROVED_REVIEW,
                approves_moral_neutrality=True,
                approves_welfare_mapping=True,
                approves_payoff_ordering=True,
                approves_material_distinctness=True,
                reviewed_content_hash=production_root.content_hash,
                reviewed_at_utc=f"2026-08-0{slot}T00:00:00+00:00",
            )
            for slot in (1, 2, 3)
        )
        roots.append(
            replace(
                production_root,
                human_content_approvals=approvals,
            )
        )
    return tuple(roots)


def _write_bank(tmp_path: Path, roots=None) -> tuple[Path, str]:
    payload = {
        "schema_version": 1,
        "roots": [asdict(root) for root in (roots or _approved_roots())],
    }
    path = tmp_path / "approved-part1-bank.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _route(role: str) -> FrozenRoute:
    route = f"unit-test/{role}"
    registry_hash = _hash("unit-test-model-registry")
    identity = {
        "provider": "openai_compatible",
        "route": route,
        "model": route,
        "registry_version": "unit-test-v1",
        "registry_hash": registry_hash,
        "verification_status": "verified",
        "verification_evidence": {
            "request_id": f"verified-{role}",
            "route": route,
        },
    }
    return FrozenRoute(
        provider="openai_compatible",
        route=route,
        registry_version="unit-test-v1",
        registry_hash=registry_hash,
        identity=identity,
    )


def _response(
    route: FrozenRoute,
    content: str,
    request_id: str,
    *,
    truncated: bool | None = False,
) -> ProviderResponse:
    return ProviderResponse(
        provider=route.provider,
        model=route.route,
        requested_model=route.route,
        response_model=route.route,
        model_identity_match=True,
        content=content,
        reasoning="PRIVATE HIDDEN REASONING",
        raw_response={"id": request_id, "content": content, "secret": "raw"},
        finish_reason="stop",
        truncated=truncated,
        usage={"input_tokens": 10, "output_tokens": 2},
        request_id=request_id,
    )


@pytest.fixture(scope="module")
def approved_roots():
    return _approved_roots()


@pytest.fixture()
def frozen_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    approved_roots,
):
    path, bank_hash = _write_bank(tmp_path, approved_roots)
    loaded = load_production_bank(path, expected_sha256=bank_hash)
    subject, extractor = _route("subject"), _route("extractor")
    identities = {route.route: route.identity for route in (subject, extractor)}
    monkeypatch.setattr(
        confirmatory_runner,
        "resolve_model_registry_entry",
        lambda provider, model: (
            deepcopy(identities.get(model))
            if provider == "openai_compatible" and model in identities
            else None
        ),
    )
    monkeypatch.setattr(confirmatory_runner, "git_commit", lambda: "a" * 40)
    monkeypatch.setattr(confirmatory_runner, "git_dirty", lambda: False)
    plan = freeze_execution_plan(
        loaded,
        subject_route=subject,
        extractor_route=extractor,
    )
    return path, loaded, plan, subject, extractor


def test_loader_requires_exact_hash_and_three_real_unanimous_approvals(
    tmp_path: Path,
    approved_roots,
) -> None:
    path, bank_hash = _write_bank(tmp_path, approved_roots)
    loaded = load_production_bank(path, expected_sha256=bank_hash)
    assert len(loaded.roots) == 384

    with pytest.raises(ConfirmatoryPart1Error, match="hash mismatch"):
        load_production_bank(path, expected_sha256="0" * 64)

    stale = list(approved_roots)
    approvals = list(stale[0].human_content_approvals)
    approvals[2] = replace(approvals[2], decision="NOT_REVIEWED")
    stale[0] = replace(stale[0], human_content_approvals=tuple(approvals))
    bad_path, bad_hash = _write_bank(tmp_path, stale)
    with pytest.raises(ConfirmatoryPart1Error, match="human-approval"):
        load_production_bank(bad_path, expected_sha256=bad_hash)


def test_bank_schema_rejects_unknown_fields(tmp_path: Path, approved_roots) -> None:
    path, _ = _write_bank(tmp_path, approved_roots)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["synthetic_approval_override"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ConfirmatoryPart1Error, match="fields are not exact"):
        load_production_bank(path, expected_sha256=digest)


def test_complete_schedule_is_exact_hash_bound_and_independently_seeded(
    frozen_fixture,
) -> None:
    _, loaded, plan, subject, _ = frozen_fixture
    schedule = plan["schedule"]
    assert len(schedule) == EXPECTED_TRIALS_PER_MODEL == 4_224
    assert sum(row["phase"] == "primary" for row in schedule) == EXPECTED_PRIMARY_TRIALS
    assert sum(row["phase"] == "secondary" for row in schedule) == EXPECTED_SECONDARY_TRIALS
    assert [row["execution_index"] for row in schedule] == list(range(4_224))
    assert len({row["trial_id"] for row in schedule}) == 4_224
    assert len(
        {row["generation_settings"]["generation_seed"] for row in schedule}
    ) == 4_224
    assert len({row["extractor_seed"] for row in schedule}) == 4_224
    assert all(row["subject_route"] == subject.to_dict() for row in schedule)
    assert schedule == build_confirmatory_schedule(loaded, subject_route=subject)


def test_plan_rejects_prompt_seed_route_and_source_freeze_tampering(
    frozen_fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, loaded, plan, _, _ = frozen_fixture
    for mutate in (
        lambda value: value["schedule"][0].__setitem__("prompt_text", "tampered"),
        lambda value: value["schedule"][0]["generation_settings"].__setitem__(
            "generation_seed", 1
        ),
        lambda value: value["subject_route"].__setitem__("route", "other/model"),
    ):
        tampered = deepcopy(plan)
        mutate(tampered)
        tampered["plan_sha256"] = confirmatory_runner.stable_json_hash(
            {key: item for key, item in tampered.items() if key != "plan_sha256"}
        )
        with pytest.raises((ConfirmatoryPart1Error, RouteIdentityError)):
            validate_execution_plan(tampered, loaded)

    monkeypatch.setattr(confirmatory_runner, "git_dirty", lambda: True)
    with pytest.raises(ConfirmatoryPart1Error, match="clean Git"):
        validate_execution_plan(plan, loaded)


def test_trial_runs_subject_then_visible_only_extractor_and_scores_exact_token(
    frozen_fixture,
    tmp_path: Path,
) -> None:
    _, _, plan, subject, extractor = frozen_fixture
    trial = plan["schedule"][0]
    attempts = tmp_path / "attempts.jsonl"
    logger = DurableAttemptLogger(attempts, experiment="part_1_confirmatory")
    calls: list[dict[str, Any]] = []

    def fake_call(provider, model, system_prompt, query, **kwargs):
        calls.append(
            {
                "provider": provider,
                "model": model,
                "system_prompt": system_prompt,
                "query": query,
                **kwargs,
            }
        )
        if model == subject.route:
            return _response(subject, "Reasoning that remains visible.\nX", "subject-1")
        return _response(
            extractor,
            json.dumps({"response": "Reasoning that remains visible.\nX"}),
            "extractor-1",
        )

    result = execute_trial(
        trial,
        extractor_route=extractor,
        attempt_logger=logger,
        detailed_call=fake_call,
    )

    assert result["status"] == "SCORED"
    assert result["choice"] == "X"
    assert result["parse_reason"] == "exact_final_token"
    assert result["welfare_preserving"] is (
        trial["counterbalance_id"] in {"CB_X_FIRST", "CB_X_SECOND"}
    )
    assert len(calls) == 2
    assert calls[0]["seed"] == trial["generation_settings"]["generation_seed"]
    assert calls[1]["seed"] == trial["extractor_seed"]
    assert "PRIVATE HIDDEN REASONING" not in calls[1]["query"]
    encoded = base64_from_extractor_prompt(calls[1]["query"])
    assert encoded == "Reasoning that remains visible.\nX"
    records = load_attempt_records(attempts)
    assert [record["outcome"] for record in records] == ["success", "success"]
    assert all(record["raw_response"] is None for record in records)
    assert all("Reasoning that remains visible" not in record["prompt_text"] for record in records)


def base64_from_extractor_prompt(prompt: str) -> str:
    import base64

    encoded = prompt.split("<assistant_response_base64>", 1)[1].split(
        "</assistant_response_base64>", 1
    )[0]
    return base64.b64decode(encoded).decode("utf-8")


def test_malformed_terminal_choice_is_retained_invalid_without_semantic_retry(
    frozen_fixture,
    tmp_path: Path,
) -> None:
    _, _, plan, subject, extractor = frozen_fixture
    trial = plan["schedule"][1]
    attempts = tmp_path / "attempts.jsonl"
    logger = DurableAttemptLogger(attempts, experiment="part_1_confirmatory")
    call_count = 0

    def fake_call(provider, model, system_prompt, query, **kwargs):
        nonlocal call_count
        call_count += 1
        if model == subject.route:
            return _response(subject, "I choose X.", "subject-malformed")
        return _response(
            extractor,
            json.dumps({"response": "I choose X."}),
            "extractor-malformed",
        )

    result = execute_trial(
        trial,
        extractor_route=extractor,
        attempt_logger=logger,
        detailed_call=fake_call,
    )
    assert result["status"] == "INVALID"
    assert result["choice"] == "INVALID"
    assert result["unscorable_reason"] == "parser:malformed_final_token"
    assert call_count == 2


def test_truncation_and_extractor_hallucination_fail_closed_without_retry(
    frozen_fixture,
    tmp_path: Path,
) -> None:
    _, _, plan, subject, extractor = frozen_fixture

    truncated_logger = DurableAttemptLogger(
        tmp_path / "truncated.jsonl", experiment="part_1_confirmatory"
    )
    calls = 0

    def truncated_call(provider, model, system_prompt, query, **kwargs):
        nonlocal calls
        calls += 1
        return _response(subject, "X", "truncated", truncated=True)

    result = execute_trial(
        plan["schedule"][2],
        extractor_route=extractor,
        attempt_logger=truncated_logger,
        detailed_call=truncated_call,
    )
    assert result["status"] == "INVALID"
    assert result["unscorable_reason"] == "subject:truncation_status_True"
    assert calls == 1

    hallucination_logger = DurableAttemptLogger(
        tmp_path / "hallucination.jsonl", experiment="part_1_confirmatory"
    )

    def hallucination_call(provider, model, system_prompt, query, **kwargs):
        if model == subject.route:
            return _response(subject, "X", "subject-grounded")
        return _response(
            extractor,
            json.dumps({"response": "Y"}),
            "extractor-ungrounded",
        )

    result = execute_trial(
        plan["schedule"][3],
        extractor_route=extractor,
        attempt_logger=hallucination_logger,
        detailed_call=hallucination_call,
    )
    assert result["status"] == "INVALID"
    assert result["unscorable_reason"] == "extractor:semantic_invalid:ValueError"
    assert [item["outcome"] for item in load_attempt_records(hallucination_logger.path)] == [
        "success",
        "invalid_response",
    ]


def test_transport_retries_only_same_route_with_identical_seed(
    frozen_fixture,
    tmp_path: Path,
) -> None:
    _, _, plan, subject, extractor = frozen_fixture
    logger = DurableAttemptLogger(
        tmp_path / "transport.jsonl", experiment="part_1_confirmatory"
    )
    subject_seeds: list[int] = []

    def flaky_call(provider, model, system_prompt, query, **kwargs):
        if model == subject.route:
            subject_seeds.append(kwargs["seed"])
            if len(subject_seeds) == 1:
                raise ConnectionError("temporary network loss")
            return _response(subject, "Y", "subject-recovered")
        return _response(extractor, json.dumps({"response": "Y"}), "extractor-ok")

    result = execute_trial(
        plan["schedule"][4],
        extractor_route=extractor,
        attempt_logger=logger,
        detailed_call=flaky_call,
    )
    assert result["status"] == "SCORED"
    assert subject_seeds == [
        plan["schedule"][4]["generation_settings"]["generation_seed"]
    ] * 2
    records = load_attempt_records(logger.path)
    assert [item["outcome"] for item in records] == [
        "provider_error",
        "success",
        "success",
    ]
    assert {item["model"] for item in records[:2]} == {subject.route}


def test_returned_alias_or_missing_identity_is_fatal(
    frozen_fixture,
    tmp_path: Path,
) -> None:
    _, _, plan, subject, extractor = frozen_fixture
    logger = DurableAttemptLogger(
        tmp_path / "identity.jsonl", experiment="part_1_confirmatory"
    )

    def alias_call(provider, model, system_prompt, query, **kwargs):
        response = _response(subject, "X", "aliased")
        return replace(response, response_model="alias/latest", model_identity_match=False)

    with pytest.raises(RouteIdentityError):
        execute_trial(
            plan["schedule"][5],
            extractor_route=extractor,
            attempt_logger=logger,
            detailed_call=alias_call,
        )
    assert load_attempt_records(logger.path)[0]["outcome"] == "invalid_response"


def test_private_full_lifecycle_and_exact_completed_resume(
    frozen_fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank_path, _, plan, subject, extractor = frozen_fixture
    private_root = tmp_path / "private"
    monkeypatch.setattr(confirmatory_runner, "PRIVATE_RESULTS_ROOT", private_root)
    # The production plan cardinality is already tested above.  Limit this I/O
    # lifecycle test to two exact plan rows so it exercises real artifact and
    # resume behavior without issuing 8,448 fixture provider calls.
    small_plan = deepcopy(plan)
    small_plan["schedule"] = small_plan["schedule"][:2]
    small_plan["plan_sha256"] = confirmatory_runner.stable_json_hash(
        {key: item for key, item in small_plan.items() if key != "plan_sha256"}
    )
    monkeypatch.setattr(
        confirmatory_runner,
        "validate_execution_plan",
        lambda supplied, loaded: None,
    )
    calls = 0

    def fake_call(provider, model, system_prompt, query, **kwargs):
        nonlocal calls
        calls += 1
        if model == subject.route:
            return _response(subject, "X", f"subject-{calls}")
        return _response(extractor, json.dumps({"response": "X"}), f"extractor-{calls}")

    output = private_root / "model-run"
    results_path = run_frozen_plan(
        plan=small_plan,
        bank_path=bank_path,
        output_directory=output,
        detailed_call=fake_call,
    )
    assert len(results_path.read_text(encoding="utf-8").splitlines()) == 2
    assert calls == 4
    for artifact in output.iterdir():
        assert stat.S_IMODE(artifact.stat().st_mode) == 0o600
    assert stat.S_IMODE(output.stat().st_mode) == 0o700

    resumed = run_frozen_plan(
        plan=small_plan,
        bank_path=bank_path,
        output_directory=output,
        resume=True,
        detailed_call=lambda *args, **kwargs: pytest.fail("completed run replayed"),
    )
    assert resumed == results_path


def test_resume_refuses_result_attempt_or_permission_tampering(
    frozen_fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank_path, _, plan, subject, extractor = frozen_fixture
    private_root = tmp_path / "private"
    monkeypatch.setattr(confirmatory_runner, "PRIVATE_RESULTS_ROOT", private_root)
    small_plan = deepcopy(plan)
    small_plan["schedule"] = small_plan["schedule"][:1]
    small_plan["plan_sha256"] = confirmatory_runner.stable_json_hash(
        {key: item for key, item in small_plan.items() if key != "plan_sha256"}
    )
    monkeypatch.setattr(confirmatory_runner, "validate_execution_plan", lambda *args: None)

    def fake_call(provider, model, system_prompt, query, **kwargs):
        if model == subject.route:
            return _response(subject, "Y", "subject")
        return _response(extractor, json.dumps({"response": "Y"}), "extractor")

    output = private_root / "tamper-run"
    results = run_frozen_plan(
        plan=small_plan,
        bank_path=bank_path,
        output_directory=output,
        detailed_call=fake_call,
    )
    results.chmod(0o644)
    with pytest.raises(ConfirmatoryPart1Error, match="permissive"):
        run_frozen_plan(
            plan=small_plan,
            bank_path=bank_path,
            output_directory=output,
            resume=True,
            detailed_call=fake_call,
        )


def test_outputs_outside_private_root_are_refused(
    frozen_fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank_path, _, plan, _, _ = frozen_fixture
    monkeypatch.setattr(
        confirmatory_runner, "PRIVATE_RESULTS_ROOT", tmp_path / "approved-private"
    )
    with pytest.raises(ConfirmatoryPart1Error, match="must stay under"):
        run_frozen_plan(
            plan=plan,
            bank_path=bank_path,
            output_directory=tmp_path / "public-output",
        )
