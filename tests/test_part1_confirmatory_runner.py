from __future__ import annotations

import hashlib
import json
import stat
from collections import Counter, defaultdict
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
    EXPECTED_SMOKE_TRIALS,
    EXPECTED_TRIALS_PER_MODEL,
    ConfirmatoryPart1Error,
    FrozenRoute,
    RouteIdentityError,
    build_confirmatory_schedule,
    build_sacrificial_smoke_schedule,
    execute_trial,
    freeze_execution_plan,
    freeze_sacrificial_smoke_plan,
    load_production_bank,
    run_frozen_plan,
    validate_execution_plan,
)
from providers.api_call import ProviderResponse


_REAL_VALIDATE_COMPLETED_SMOKE_DIRECTORY = (
    confirmatory_runner.validate_completed_smoke_directory
)


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
    monkeypatch.setattr(
        confirmatory_runner,
        "require_fresh_route_verification",
        lambda _entry: None,
    )
    smoke_directory = confirmatory_runner.PRIVATE_RESULTS_ROOT / "unit-test-smoke"
    smoke_gate = {
        "schema_version": 1,
        "status": "unit-test-completed-smoke",
        "smoke_directory": str(smoke_directory.resolve()),
        "smoke_gate_sha256": "f" * 64,
    }
    monkeypatch.setattr(
        confirmatory_runner,
        "validate_completed_smoke_directory",
        lambda *_args, **_kwargs: deepcopy(smoke_gate),
    )
    plan = freeze_execution_plan(
        loaded,
        subject_route=subject,
        extractor_route=extractor,
        completed_smoke_directory=smoke_directory,
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


def test_strict_resume_accepts_stale_frozen_identity_but_new_freeze_does_not(
    frozen_fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank_path, loaded, plan, subject, extractor = frozen_fixture
    private_root = tmp_path / "private"
    output = private_root / "resume-stale-evidence"
    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)
    plan_path = output / "part1_confirmatory_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    plan_path.chmod(0o600)
    monkeypatch.setattr(confirmatory_runner, "PRIVATE_RESULTS_ROOT", private_root)

    def stale(_entry):
        raise ValueError("route evidence is older than 168 hours")

    monkeypatch.setattr(confirmatory_runner, "require_fresh_route_verification", stale)
    with pytest.raises(RouteIdentityError, match="production-current"):
        confirmatory_runner.freeze_verified_route(subject.provider, subject.route)

    captured: dict[str, Any] = {}

    def fake_run_frozen_plan(**kwargs):
        captured.update(kwargs)
        return output / "part1_confirmatory_results.jsonl"

    monkeypatch.setattr(confirmatory_runner, "run_frozen_plan", fake_run_frozen_plan)
    smoke_directory = plan["completed_smoke_gate"]["smoke_directory"]
    assert (
        confirmatory_runner.main(
            [
                "--mode",
                "production",
                "--bank",
                str(bank_path),
                "--bank-sha256",
                loaded.file_sha256,
                "--subject-provider",
                subject.provider,
                "--subject-model",
                subject.route,
                "--extractor-provider",
                extractor.provider,
                "--extractor-model",
                extractor.route,
                "--completed-smoke-directory",
                smoke_directory,
                "--output-directory",
                str(output),
                "--resume",
            ]
        )
        == 0
    )
    assert captured["plan"] == plan
    assert captured["resume"] is True


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


def test_sacrificial_smoke_schedule_is_small_deterministic_and_fully_balanced(
    frozen_fixture,
) -> None:
    _, loaded, production_plan, subject, extractor = frozen_fixture
    smoke_plan = freeze_sacrificial_smoke_plan(
        loaded,
        subject_route=subject,
        extractor_route=extractor,
    )
    schedule = smoke_plan["schedule"]

    assert smoke_plan["execution_mode"] == "sacrificial_smoke"
    assert smoke_plan["analysis_eligibility"] == {
        "eligible": False,
        "purpose": "sacrificial_part1_full_path_smoke",
        "exclusion_required": True,
    }
    assert len(schedule) == EXPECTED_SMOKE_TRIALS == 48
    assert sum(item["phase"] == "primary" for item in schedule) == 12
    assert sum(item["phase"] == "secondary" for item in schedule) == 36
    assert Counter(item["frame_id"] for item in schedule) == Counter(
        {
            "self_direct": 12,
            "advice": 12,
            "observer_evaluation": 12,
            "prediction": 12,
        }
    )
    assert set(Counter(item["game"] for item in schedule).values()) == {24}
    assert set(Counter(item["domain"] for item in schedule).values()) == {8}
    assert set(Counter(item["counterbalance_id"] for item in schedule).values()) == {
        12
    }
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in schedule:
        assert item["execution_mode"] == "sacrificial_smoke"
        assert item["analysis_eligible"] is False
        by_cell[(item["game"], item["domain"])].append(item)
    assert len(by_cell) == 12
    for rows in by_cell.values():
        assert {row["frame_id"] for row in rows} == {
            "self_direct",
            "advice",
            "observer_evaluation",
            "prediction",
        }
        assert {row["counterbalance_id"] for row in rows} == {
            "CB_X_FIRST",
            "CB_Y_FIRST",
            "CB_X_SECOND",
            "CB_Y_SECOND",
        }
    assert schedule == build_sacrificial_smoke_schedule(production_plan["schedule"])
    with pytest.raises(ConfirmatoryPart1Error, match="exact unique production"):
        build_sacrificial_smoke_schedule(production_plan["schedule"][:-1])


def test_cli_exposes_only_exact_production_and_sacrificial_smoke_modes() -> None:
    required = [
        "--bank",
        "/private/bank.json",
        "--bank-sha256",
        "0" * 64,
        "--subject-provider",
        "inference_hub",
        "--subject-model",
        "exact/subject",
        "--extractor-provider",
        "inference_hub",
        "--extractor-model",
        "exact/extractor",
        "--output-directory",
        "data/private/part1_confirmatory/smoke",
    ]
    smoke = confirmatory_runner._parser().parse_args(
        ["--mode", "sacrificial-smoke", *required]
    )
    assert smoke.mode == "sacrificial-smoke"
    production = confirmatory_runner._parser().parse_args(required)
    assert production.mode == "production"
    with pytest.raises(SystemExit):
        confirmatory_runner._parser().parse_args(["--mode", "smoke", *required])


def test_sacrificial_smoke_runs_all_48_subject_extractor_parser_paths_and_marks_data(
    frozen_fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank_path, loaded, _, subject, extractor = frozen_fixture
    private_root = tmp_path / "private"
    monkeypatch.setattr(confirmatory_runner, "PRIVATE_RESULTS_ROOT", private_root)
    smoke_plan = freeze_sacrificial_smoke_plan(
        loaded,
        subject_route=subject,
        extractor_route=extractor,
    )
    calls: list[tuple[str, int]] = []
    latest_visible = ""
    subject_count = 0

    def fake_call(provider, model, system_prompt, query, **kwargs):
        nonlocal latest_visible, subject_count
        calls.append((model, kwargs["seed"]))
        if model == subject.route:
            subject_count += 1
            token = "X" if subject_count % 2 else "Y"
            latest_visible = f"Visible smoke reasoning {subject_count}.\n{token}"
            return _response(subject, latest_visible, f"smoke-subject-{subject_count}")
        return _response(
            extractor,
            json.dumps({"response": latest_visible}),
            f"smoke-extractor-{subject_count}",
        )

    output = private_root / "full-smoke"
    results_path = run_frozen_plan(
        plan=smoke_plan,
        bank_path=bank_path,
        output_directory=output,
        detailed_call=fake_call,
    )
    records = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == EXPECTED_SMOKE_TRIALS
    assert len(calls) == EXPECTED_SMOKE_TRIALS * 2 == 96
    assert Counter(model for model, _ in calls) == Counter(
        {subject.route: 48, extractor.route: 48}
    )
    assert Counter(record["choice"] for record in records) == Counter({"X": 24, "Y": 24})
    assert all(record["status"] == "SCORED" for record in records)
    assert all(record["execution_mode"] == "sacrificial_smoke" for record in records)
    assert all(record["analysis_eligible"] is False for record in records)

    marker_path = output / "part1_confirmatory_analysis_exclude.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    metadata = json.loads(
        (output / "part1_confirmatory_meta.json").read_text(encoding="utf-8")
    )
    assert marker["reason"] == "sacrificial_part1_full_path_smoke"
    assert marker["analysis_eligible"] is False
    assert marker["execution_mode"] == "sacrificial_smoke"
    assert marker["total_results"] == 48
    assert marker["plan_sha256"] == smoke_plan["plan_sha256"]
    assert marker["schedule_sha256"] == confirmatory_runner.stable_json_hash(
        smoke_plan["schedule"]
    )
    assert marker["results_sha256"] == hashlib.sha256(
        results_path.read_bytes()
    ).hexdigest()
    assert marker["metadata_sha256"] == metadata["metadata_sha256"]
    assert marker["marker_payload_sha256"] == confirmatory_runner.stable_json_hash(
        {
            key: value
            for key, value in marker.items()
            if key != "marker_payload_sha256"
        }
    )
    assert stat.S_IMODE(marker_path.stat().st_mode) == 0o600
    gate = _REAL_VALIDATE_COMPLETED_SMOKE_DIRECTORY(
        loaded,
        subject_route=subject,
        extractor_route=extractor,
        primary_seed=smoke_plan["primary_seed_base"],
        secondary_seed=smoke_plan["secondary_seed_base"],
        extractor_seed=smoke_plan["extractor_seed_base"],
        smoke_directory=output,
    )
    assert gate["status"] == "validated_complete_full_path_smoke"
    assert gate["result_count"] == 48
    assert len(gate["smoke_gate_sha256"]) == 64

    before_resume_calls = len(calls)
    assert run_frozen_plan(
        plan=smoke_plan,
        bank_path=bank_path,
        output_directory=output,
        resume=True,
        detailed_call=fake_call,
    ) == results_path
    assert len(calls) == before_resume_calls


def test_smoke_plan_mode_design_and_schedule_tampering_fail_closed(
    frozen_fixture,
) -> None:
    _, loaded, _, subject, extractor = frozen_fixture
    smoke_plan = freeze_sacrificial_smoke_plan(
        loaded,
        subject_route=subject,
        extractor_route=extractor,
    )

    mutations = (
        lambda value: value["smoke_design"].__setitem__("expected_trials", 47),
        lambda value: value["schedule"].pop(),
        lambda value: value["schedule"][0].__setitem__("analysis_eligible", True),
        lambda value: value.__setitem__("execution_mode", "production"),
    )
    for mutate in mutations:
        tampered = deepcopy(smoke_plan)
        mutate(tampered)
        tampered["plan_sha256"] = confirmatory_runner.stable_json_hash(
            {key: item for key, item in tampered.items() if key != "plan_sha256"}
        )
        with pytest.raises(ConfirmatoryPart1Error):
            validate_execution_plan(tampered, loaded)

    disguised = deepcopy(smoke_plan)
    disguised["execution_mode"] = "production"
    disguised["analysis_eligibility"] = {
        "eligible": True,
        "purpose": "confirmatory_part1_inference",
        "exclusion_required": False,
    }
    disguised["planned_trial_count"] = EXPECTED_TRIALS_PER_MODEL
    disguised.pop("smoke_design")
    for row in disguised["schedule"]:
        row["execution_mode"] = "production"
        row["analysis_eligible"] = True
    disguised["plan_sha256"] = confirmatory_runner.stable_json_hash(
        {key: item for key, item in disguised.items() if key != "plan_sha256"}
    )
    with pytest.raises(ConfirmatoryPart1Error, match="smoke gate|schedule"):
        validate_execution_plan(disguised, loaded)


def test_completed_smoke_refuses_missing_or_tampered_exclusion_before_calls(
    frozen_fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank_path, loaded, _, subject, extractor = frozen_fixture
    private_root = tmp_path / "private"
    monkeypatch.setattr(confirmatory_runner, "PRIVATE_RESULTS_ROOT", private_root)
    smoke_plan = freeze_sacrificial_smoke_plan(
        loaded,
        subject_route=subject,
        extractor_route=extractor,
    )
    latest_visible = "X"

    def fake_call(provider, model, system_prompt, query, **kwargs):
        nonlocal latest_visible
        if model == subject.route:
            latest_visible = "X"
            return _response(subject, latest_visible, "subject-smoke")
        return _response(extractor, json.dumps({"response": latest_visible}), "extractor-smoke")

    output = private_root / "marker-tamper"
    run_frozen_plan(
        plan=smoke_plan,
        bank_path=bank_path,
        output_directory=output,
        detailed_call=fake_call,
    )
    marker_path = output / "part1_confirmatory_analysis_exclude.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["results_sha256"] = "0" * 64
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    marker_path.chmod(0o600)
    with pytest.raises(ConfirmatoryPart1Error, match="exclusion marker"):
        run_frozen_plan(
            plan=smoke_plan,
            bank_path=bank_path,
            output_directory=output,
            resume=True,
            detailed_call=lambda *args, **kwargs: pytest.fail(
                "marker tampering must fail before provider calls"
            ),
        )

    marker_path.unlink()
    with pytest.raises(ConfirmatoryPart1Error, match="missing its exclusion marker"):
        run_frozen_plan(
            plan=smoke_plan,
            bank_path=bank_path,
            output_directory=output,
            resume=True,
            detailed_call=lambda *args, **kwargs: pytest.fail(
                "missing marker must fail before provider calls"
            ),
        )


def test_production_resume_rejects_any_smoke_exclusion_marker(
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
    latest_visible = "X"

    def fake_call(provider, model, system_prompt, query, **kwargs):
        if model == subject.route:
            return _response(subject, latest_visible, "subject-production")
        return _response(extractor, json.dumps({"response": latest_visible}), "extractor-production")

    output = private_root / "production-marker"
    run_frozen_plan(
        plan=small_plan,
        bank_path=bank_path,
        output_directory=output,
        detailed_call=fake_call,
    )
    marker_path = output / "part1_confirmatory_analysis_exclude.json"
    marker_path.write_text("{}\n", encoding="utf-8")
    marker_path.chmod(0o600)
    with pytest.raises(ConfirmatoryPart1Error, match="Production Part 1 output"):
        run_frozen_plan(
            plan=small_plan,
            bank_path=bank_path,
            output_directory=output,
            resume=True,
            detailed_call=fake_call,
        )
