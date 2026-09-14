from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import hmac
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

import analysis.validate_inference_hub_part2_operational_overlays as validator
from experiments.misc import inference_hub_part2_cascading_operational_repair as repair
from experiments.misc import inference_hub_part2_panel as runner
from experiments.misc.inference_hub_discovery import InferenceHubDiscoveryError


ROUTE = "gcp/google/gemini-3.5-flash"
CURSOR_EPOCH = "cursor-epoch-0123456789abcdef0123456789abcdef"
ROOT = Path(__file__).resolve().parents[1]
IS_ANONYMOUS_SUPPLEMENT = (ROOT / "SUPPLEMENT_MANIFEST.json").is_file()


def _bind_validator_runtime_hashes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use distributed source hashes only inside the anonymized test archive."""

    cascading_path = Path(repair.__file__).resolve()
    paths = {
        "EXPECTED_CASCADING_REPAIR_IMPLEMENTATION_SHA256": cascading_path,
        "EXPECTED_RUNNER_IMPLEMENTATION_SHA256": Path(runner.__file__).resolve(),
        "EXPECTED_REPAIR_IMPLEMENTATION_SHA256": Path(
            validator.repair.__file__
        ).resolve(),
        "EXPECTED_DISCOVERY_IMPLEMENTATION_SHA256": cascading_path.with_name(
            "inference_hub_discovery.py"
        ),
        "EXPECTED_RATE_LIMIT_IMPLEMENTATION_SHA256": cascading_path.with_name(
            "inference_hub_rate_limit.py"
        ),
    }
    for attribute, path in paths.items():
        if IS_ANONYMOUS_SUPPLEMENT or attribute == (
            "EXPECTED_CASCADING_REPAIR_IMPLEMENTATION_SHA256"
        ):
            monkeypatch.setattr(validator, attribute, runner._sha256_file(path))


class _Limiter:
    def __init__(self, path: Path) -> None:
        self.state_path = path


class _Client:
    def __init__(
        self,
        slot: int,
        *,
        base_url: str = runner.DEFAULT_BASE_URL,
        limiter_path: Path | None = None,
        catalog_error: int | None = None,
        catalog_routes: Sequence[str] = (ROUTE,),
        preflight_response_route: str = ROUTE,
        experiment_response_route: str = ROUTE,
        experiment_http_statuses: Sequence[int] = (),
    ) -> None:
        self.base_url = base_url
        self.rate_limit_contract = deepcopy(
            validator.EXPECTED_SHARED_RATE_LIMIT_CONTRACT
        )
        self.rate_limiter = _Limiter(
            limiter_path or Path(f"/tmp/v4-limiter-{slot}.json")
        )
        self.catalog_error = catalog_error
        self.catalog_routes = tuple(catalog_routes)
        self.preflight_response_route = preflight_response_route
        self.experiment_response_route = experiment_response_route
        self.experiment_http_statuses = list(experiment_http_statuses)
        self.calls: list[dict[str, Any]] = []

    @property
    def get_calls(self) -> list[dict[str, Any]]:
        return [row for row in self.calls if row["method"] == "GET"]

    @property
    def post_calls(self) -> list[dict[str, Any]]:
        return [row for row in self.calls if row["method"] == "POST"]

    @property
    def experiment_calls(self) -> list[dict[str, Any]]:
        return [row for row in self.post_calls if row["preflight"] is False]

    def get(self, path: str) -> dict[str, Any]:
        assert path == "/models"
        self.calls.append({"method": "GET", "path": path})
        if self.catalog_error is not None:
            raise InferenceHubDiscoveryError(
                "redacted", failure_code="http_error",
                http_status=self.catalog_error,
            )
        return {"data": [{"id": route} for route in self.catalog_routes]}

    def post(
        self, path: str, body: dict[str, Any], *, upstream_provider: str | None = None,
    ) -> dict[str, Any]:
        assert path == "/chat/completions"
        preflight = body == repair._preflight_request(ROUTE)
        self.calls.append(
            {
                "method": "POST",
                "path": path,
                "body": deepcopy(body),
                "upstream_provider": upstream_provider,
                "preflight": preflight,
            }
        )
        if not preflight and self.experiment_http_statuses:
            status = self.experiment_http_statuses.pop(0)
            raise InferenceHubDiscoveryError(
                "redacted", failure_code="http_error", http_status=status,
            )
        return {
            "id": "fixture",
            "model": (
                self.preflight_response_route
                if preflight
                else self.experiment_response_route
            ),
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "action": "OPTION_A",
                        "reasoning": "Preserve the shared reserve.",
                    }),
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


def _accounts(
    *clients: _Client,
    cursor_epoch: str = CURSOR_EPOCH,
    credentials: Sequence[str] | None = None,
) -> tuple[repair.AccountClient, ...]:
    secrets = tuple(credentials or (
        "fixture-secret-one", "fixture-secret-two", "fixture-secret-three",
    ))
    assert len(secrets) == len(clients)
    return tuple(
        repair.AccountClient(
            f"account-slot-{index:02d}",
            client,
            repair._credential_commitment(
                secrets[index - 1],
                cursor_epoch=cursor_epoch,
                base_url=client.base_url,
            ),
        )
        for index, client in enumerate(clients, 1)
    )


def _pool(
    accounts: Sequence[repair.AccountClient],
    *,
    qualified_slots: Sequence[str] = ("account-slot-01", "account-slot-03"),
    cursor_epoch: str = CURSOR_EPOCH,
    preflight_session_index: int = 0,
    preflight_session_sha256: str = "a" * 64,
    start_ordinal: int = 0,
) -> repair.QualifiedAccountPool:
    return repair.QualifiedAccountPool(
        accounts,
        qualified_slots=qualified_slots,
        cursor_epoch=cursor_epoch,
        preflight_session_index=preflight_session_index,
        preflight_session_sha256=preflight_session_sha256,
        start_ordinal=start_ordinal,
    )


def _credential_file(
    path: Path,
    *,
    mode: int = 0o600,
    credentials: Sequence[str] = (
        "fixture-secret-one",
        "fixture-secret-two",
        "fixture-secret-three",
    ),
) -> Path:
    assert len(credentials) == 3
    path.write_text(
        "".join(
            f"BETTER_GOS_NVIDIA_API_KEY{'' if index == 1 else f'_{index}'}={value}\n"
            for index, value in enumerate(credentials, 1)
        ),
        encoding="utf-8",
    )
    path.chmod(mode)
    return path


def _clock(day: str = "2026-09-12") -> Any:
    counter = 0

    def now() -> str:
        nonlocal counter
        counter += 1
        return f"{day}T00:00:{counter:02d}Z"

    return now


def test_blind_preflight_is_generic_and_persists_no_probe_secrets() -> None:
    accounts = _accounts(
        _Client(1),
        _Client(2, catalog_error=401),
        _Client(3, preflight_response_route="wrong/route"),
    )

    ledger = repair.blind_preflight_accounts(
        accounts, exact_route=ROUTE, cursor_epoch=CURSOR_EPOCH,
        now=_clock(),
    )

    assert ledger["configured_account_slots"] == [
        "account-slot-01", "account-slot-02", "account-slot-03",
    ]
    assert ledger["qualified_account_slots"] == ["account-slot-01"]
    assert ledger["results"][1]["catalog_http_status"] == 401
    assert ledger["results"][2]["chat_response_identity_matched"] is False
    assert [len(account.client.get_calls) for account in accounts] == [1, 1, 1]
    assert [len(account.client.post_calls) for account in accounts] == [1, 0, 1]
    assert ledger["request_contract_sha256"] == runner._sha256_json(
        repair._preflight_request(ROUTE)
    )
    serialized = json.dumps(ledger).casefold()
    for forbidden in (
        "authorization", "bearer ", "api_key", "raw_response", "request_body",
        repair.PREFLIGHT_MESSAGE.casefold(),
    ):
        assert forbidden not in serialized
    assert repair._validate_preflight_ledger(
        ledger, exact_route=ROUTE,
    ) == (CURSOR_EPOCH, ("account-slot-01",))


def test_preflight_rejects_resealed_qualification_and_secret_field_tamper() -> None:
    ledger = repair.blind_preflight_accounts(
        _accounts(_Client(1), _Client(2), _Client(3)), exact_route=ROUTE,
        cursor_epoch=CURSOR_EPOCH, now=_clock(),
    )
    changed = deepcopy(ledger)
    changed["qualified_account_slots"] = ["account-slot-01"]
    changed["qualified_set_sha256"] = repair._qualified_set_sha256(
        changed["qualified_account_slots"],
        {
            result["account_slot"]: result["account_commitment"]
            for result in changed["results"]
        },
    )
    runner._seal(changed)
    with pytest.raises(repair.Part2CascadingRepairError, match="qualified-set"):
        repair._validate_preflight_ledger(changed, exact_route=ROUTE)

    exposed = deepcopy(ledger)
    exposed["api_key"] = "must-not-persist"
    runner._seal(exposed)
    with pytest.raises(
        repair.Part2CascadingRepairError, match="secret-like field",
    ):
        repair._assert_no_secret_like_fields(exposed, label="fixture")


@pytest.mark.parametrize(
    "cursor_epoch",
    [
        "cursor-epoch-fixture",
        "cursor-epoch-0123456789abcdef",
        "cursor-epoch-0123456789ABCDEF0123456789ABCDEF",
        "cursor-epoch-0123456789abcdef0123456789abcdef0",
    ],
)
def test_preflight_ledger_rejects_noncanonical_cursor_epoch(
    cursor_epoch: str,
) -> None:
    ledger = repair.blind_preflight_accounts(
        _accounts(_Client(1), _Client(2), _Client(3)),
        exact_route=ROUTE,
        cursor_epoch=CURSOR_EPOCH,
        now=_clock(),
    )
    ledger["cursor_epoch"] = cursor_epoch
    runner._seal(ledger)

    with pytest.raises(
        repair.Part2CascadingRepairError, match="cursor epoch"
    ):
        repair._validate_preflight_ledger(ledger, exact_route=ROUTE)


def test_resume_preflight_session_binds_next_round_robin_reservation() -> None:
    initial_clients = (_Client(1), _Client(2, catalog_error=401), _Client(3))
    initial_accounts = _accounts(*initial_clients)
    initial = repair.blind_preflight_accounts(
        initial_accounts,
        exact_route=ROUTE,
        cursor_epoch=CURSOR_EPOCH,
        now=_clock("2026-09-12"),
    )
    fresh_clients = (_Client(11), _Client(12, catalog_error=401), _Client(13))
    fresh_accounts = _accounts(*fresh_clients)
    fresh = repair.blind_preflight_accounts(
        fresh_accounts,
        exact_route=ROUTE,
        cursor_epoch=CURSOR_EPOCH,
        now=_clock("2026-09-13"),
    )

    updated = repair._append_resume_preflight(
        initial,
        fresh,
        exact_route=ROUTE,
        starting_global_dispatch_ordinal=7,
    )
    assert repair._validate_preflight_ledger(
        updated, exact_route=ROUTE,
    ) == (
        CURSOR_EPOCH,
        ("account-slot-01", "account-slot-03"),
    )
    session = updated["resume_preflight_sessions"][0]
    pool = _pool(
        fresh_accounts,
        preflight_session_index=session["session_index"],
        preflight_session_sha256=session["session_sha256"],
        start_ordinal=7,
    )
    row, client = pool.reserve(_Journal(), {"event": "fixture-reservation"})
    assert row["global_dispatch_ordinal"] == 8
    assert row["account_slot"] == "account-slot-03"
    assert row["preflight_session_index"] == 1
    assert row["preflight_session_sha256"] == session["session_sha256"]
    assert client is fresh_clients[2]


def test_preflight_rejects_route_absence_and_returned_model_mismatch() -> None:
    clients = (
        _Client(1),
        _Client(2, catalog_routes=("gcp/google/another-model",)),
        _Client(3, preflight_response_route="gcp/google/another-model"),
    )
    ledger = repair.blind_preflight_accounts(
        _accounts(*clients),
        exact_route=ROUTE,
        cursor_epoch=CURSOR_EPOCH,
        now=_clock(),
    )

    assert ledger["qualified_account_slots"] == ["account-slot-01"]
    assert ledger["results"][1]["authentication_succeeded"] is True
    assert ledger["results"][1]["exact_route_catalog_listed"] is False
    assert ledger["results"][1]["chat_attempted"] is False
    assert ledger["results"][2]["exact_route_catalog_listed"] is True
    assert ledger["results"][2]["chat_response_identity_matched"] is False
    assert [len(client.get_calls) for client in clients] == [1, 1, 1]
    assert [len(client.post_calls) for client in clients] == [1, 0, 1]


def test_preflight_rejects_endpoint_override_before_any_network_call() -> None:
    clients = tuple(
        _Client(index, base_url="https://example.invalid/v1")
        for index in range(1, 4)
    )

    with pytest.raises(
        repair.Part2CascadingRepairError, match="frozen InferenceHub endpoint"
    ):
        repair.blind_preflight_accounts(
            _accounts(*clients),
            exact_route=ROUTE,
            cursor_epoch=CURSOR_EPOCH,
            now=_clock(),
        )

    assert all(client.calls == [] for client in clients)


@pytest.mark.parametrize("mode", [0o644, 0o640])
def test_runtime_accounts_rejects_insecure_credential_file_before_client_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: int,
) -> None:
    credential_path = _credential_file(tmp_path / ".env", mode=mode)
    created: list[None] = []

    def must_not_create(*_args: Any, **_kwargs: Any) -> Any:
        created.append(None)
        raise AssertionError("client construction preceded credential-file validation")

    monkeypatch.setattr(repair, "InferenceHubClient", must_not_create)
    with pytest.raises(
        repair.Part2CascadingRepairError,
        match="(?i)credential.*(?:0600|owner-held)",
    ):
        repair._runtime_accounts(
            credential_path,
            expected_count=3,
            timeout_seconds=repair.DEFAULT_TIMEOUT_SECONDS,
            rate_profile=repair.DEFAULT_RATE_PROFILE,
            cursor_epoch=CURSOR_EPOCH,
        )
    assert created == []


def test_runtime_accounts_rejects_symlink_credential_file_before_client_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_path = _credential_file(tmp_path / "credentials.env")
    credential_path = tmp_path / ".env"
    credential_path.symlink_to(real_path)
    created: list[None] = []

    def must_not_create(*_args: Any, **_kwargs: Any) -> Any:
        created.append(None)
        raise AssertionError("client construction followed a credential symlink")

    monkeypatch.setattr(repair, "InferenceHubClient", must_not_create)
    with pytest.raises(
        repair.Part2CascadingRepairError,
        match="(?i)credential.*(?:symlink|regular)",
    ):
        repair._runtime_accounts(
            credential_path,
            expected_count=3,
            timeout_seconds=repair.DEFAULT_TIMEOUT_SECONDS,
            rate_profile=repair.DEFAULT_RATE_PROFILE,
            cursor_epoch=CURSOR_EPOCH,
        )
    assert created == []


def test_runtime_accounts_rejects_ambient_endpoint_override_before_reading_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential_path = _credential_file(tmp_path / ".env")
    monkeypatch.setenv("INFERENCE_HUB_BASE_URL", "https://example.invalid/v1")
    created: list[None] = []

    def must_not_create(*_args: Any, **_kwargs: Any) -> Any:
        created.append(None)
        raise AssertionError("client construction preceded endpoint validation")

    monkeypatch.setattr(repair, "InferenceHubClient", must_not_create)
    with pytest.raises(
        repair.Part2CascadingRepairError, match="endpoint override"
    ):
        repair._runtime_accounts(
            credential_path,
            expected_count=3,
            timeout_seconds=repair.DEFAULT_TIMEOUT_SECONDS,
            rate_profile=repair.DEFAULT_RATE_PROFILE,
            cursor_epoch=CURSOR_EPOCH,
        )
    assert created == []


def test_concurrent_runtime_accounts_bind_each_hidden_key_to_its_own_commitment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = (
        _credential_file(
            tmp_path / "pool-a.env",
            credentials=(
                "opaque-pool-a-slot-1",
                "opaque-pool-a-slot-2",
                "opaque-pool-a-slot-3",
            ),
        ),
        _credential_file(
            tmp_path / "pool-b.env",
            credentials=(
                "opaque-pool-b-slot-1",
                "opaque-pool-b-slot-2",
                "opaque-pool-b-slot-3",
            ),
        ),
    )
    cursor_epochs = (
        CURSOR_EPOCH,
        "cursor-epoch-fedcba9876543210fedcba9876543210",
    )
    original_client = repair.InferenceHubClient
    constructor_barrier = threading.Barrier(2)
    start_barrier = threading.Barrier(2)

    def synchronized_client(**kwargs: Any) -> Any:
        constructor_barrier.wait(timeout=5)
        return original_client(**kwargs)

    def obsolete_ambient_factory(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("runtime accounts used the ambient-env client factory")

    monkeypatch.setattr(repair, "InferenceHubClient", synchronized_client)
    monkeypatch.setattr(runner, "_runtime_client", obsolete_ambient_factory)
    monkeypatch.setenv("NVIDIA_API_KEY", "opaque-ambient-sentinel")
    monkeypatch.setenv("INFERENCE_HUB_BASE_URL", runner.DEFAULT_BASE_URL)

    def construct(index: int) -> tuple[repair.AccountClient, ...]:
        start_barrier.wait(timeout=5)
        return repair._runtime_accounts(
            paths[index],
            expected_count=3,
            timeout_seconds=repair.DEFAULT_TIMEOUT_SECONDS,
            rate_profile=repair.DEFAULT_RATE_PROFILE,
            cursor_epoch=cursor_epochs[index],
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(construct, index) for index in range(2)]
        pools = tuple(future.result(timeout=10) for future in futures)

    for accounts, cursor_epoch in zip(pools, cursor_epochs):
        assert [account.account_slot for account in accounts] == [
            "account-slot-01", "account-slot-02", "account-slot-03",
        ]
        for account in accounts:
            observed_commitment = repair._credential_commitment(
                account.client._api_key,
                cursor_epoch=cursor_epoch,
                base_url=account.client.base_url,
            )
            assert hmac.compare_digest(
                observed_commitment, account.account_commitment,
            )
    assert os.environ["NVIDIA_API_KEY"] == "opaque-ambient-sentinel"
    assert os.environ["INFERENCE_HUB_BASE_URL"] == runner.DEFAULT_BASE_URL
    assert {
        path.resolve() for path in tmp_path.rglob("*") if path.is_file()
    } == {path.resolve() for path in paths}


class _Journal:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def append(self, payload: dict[str, Any]) -> dict[str, Any]:
        row = deepcopy(payload)
        self.records.append(row)
        return row


def _subject() -> dict[str, Any]:
    return {
        "target_id": repair.EXPECTED_TARGET_ID,
        "upstream_provider": "google",
        "model": "gemini-3.5-flash",
        "route": ROUTE,
        "supported_controls": ["seed", "temperature", "top_p", "structured_response"],
    }


def test_pool_uses_only_qualified_slots_with_independent_exact_round_robin() -> None:
    clients = (_Client(1), _Client(2), _Client(3))
    accounts = _accounts(*clients)
    pool = _pool(accounts)
    journal = _Journal()

    allocations = [
        pool.reserve(journal, {"event": "fixture-reservation"})
        for _ in range(6)
    ]

    assert [row[0]["account_slot"] for row in allocations] == [
        "account-slot-01", "account-slot-03",
        "account-slot-01", "account-slot-03",
        "account-slot-01", "account-slot-03",
    ]
    assert [row[0]["global_dispatch_ordinal"] for row in allocations] == list(
        range(1, 7)
    )
    assert [row[1] for row in allocations] == [
        clients[0], clients[2], clients[0], clients[2], clients[0], clients[2],
    ]
    limiter_paths = {
        account.client.rate_limiter.state_path.resolve()
        for account in accounts
        if account.account_slot in pool.qualified_slots
    }
    assert len(limiter_paths) == 2
    binding = repair._credential_pool_binding(
        accounts=accounts,
        qualified_slots=pool.qualified_slots,
        cursor_epoch=CURSOR_EPOCH,
        exact_route=ROUTE,
    )
    assert set(binding["qualified_account_rate_limiters"]) == {
        "account-slot-01", "account-slot-03",
    }
    assert len({
        row["state_path_sha256"]
        for row in binding["qualified_account_rate_limiters"].values()
    }) == 2


def test_pool_rejects_shared_limiter_path() -> None:
    shared = Path("/tmp/v4-shared-limiter.json")
    accounts = _accounts(
        _Client(1, limiter_path=shared),
        _Client(2),
        _Client(3, limiter_path=shared),
    )

    with pytest.raises(
        repair.Part2CascadingRepairError, match="independent limiter scopes"
    ):
        _pool(accounts)


@pytest.mark.parametrize("status", [401, 403])
def test_post_preflight_auth_failure_quarantines_all_later_reservations(
    status: int,
) -> None:
    clients = (
        _Client(1, experiment_http_statuses=(status,)),
        _Client(2),
        _Client(3),
    )
    pool = _pool(_accounts(*clients))
    journal = _Journal()

    with pytest.raises(
        repair.CredentialQualificationInvalidatedError, match="quarantined"
    ):
        repair._dispatch_unit(
            journal=journal,
            subject=_subject(),
            trajectory_index=repair.EXPECTED_TRAJECTORY_INDEX,
            environment_seed=repair.EXPECTED_ENVIRONMENT_SEED,
            day=1,
            slot=0,
            prompt="scientific prompt",
            system_prompt="system",
            prior_attempt=0,
            max_attempts=repair.DEFAULT_MAX_ATTEMPTS,
            initial_backoff_seconds=repair.DEFAULT_BACKOFF_SECONDS,
            pool=pool,
            sleep_fn=lambda _seconds: None,
        )

    reservations = [
        row for row in journal.records if row["event"] == "reserved_before_dispatch"
    ]
    terminal = [row for row in journal.records if row["event"] == "semantic_result"]
    assert len(reservations) == len(terminal) == 1
    assert terminal[0]["failure"]["http_status"] == status
    assert len(clients[0].experiment_calls) == 1
    before = deepcopy(journal.records)
    with pytest.raises(repair.CredentialQualificationInvalidatedError):
        pool.reserve(journal, {"event": "must-not-be-reserved"})
    assert journal.records == before
    assert clients[2].experiment_calls == []


def test_http_400_retries_identical_request_with_true_status_and_next_slot() -> None:
    first, second, third = (
        _Client(1, experiment_http_statuses=(400,)),
        _Client(2),
        _Client(3),
    )
    pool = _pool(
        _accounts(first, second, third),
        qualified_slots=("account-slot-01", "account-slot-02"),
    )
    journal = _Journal()
    sleeps: list[float] = []

    result = repair._dispatch_unit(
        journal=journal, subject=_subject(), trajectory_index=1,
        environment_seed=repair.EXPECTED_ENVIRONMENT_SEED,
        day=1, slot=0, prompt="scientific prompt", system_prompt="system",
        prior_attempt=0, max_attempts=8, initial_backoff_seconds=1.0,
        pool=pool, sleep_fn=sleeps.append,
    )

    reservations = [
        row for row in journal.records if row["event"] == "reserved_before_dispatch"
    ]
    failure = next(row for row in journal.records if row["event"] == "attempt_failed")
    assert [row["account_slot"] for row in reservations] == [
        "account-slot-01", "account-slot-02",
    ]
    assert [row["global_dispatch_ordinal"] for row in reservations] == [1, 2]
    assert len({row["request_sha256"] for row in reservations}) == 1
    assert first.experiment_calls[-1]["body"] == second.experiment_calls[-1]["body"]
    assert failure["failure"] == {
        "failure_code": "http_error",
        "transient": True,
        "http_status": 400,
        "error_type": "InferenceHubDiscoveryError",
    }
    assert sleeps == [1.0]
    assert result["action"] == "OPTION_A"


def _day_one_reservation_payload(
    *, attempt_id: str, attempt_number: int, slot: int = 0,
    contract: runner.Part2Contract | None = None,
) -> dict[str, Any]:
    subject = _subject()
    contract = contract or runner.Part2Contract(1, 1, 12, 10, 2, 2, 5, 0.2)
    agent = repair.Agent2(f"slot_{slot:02d}", "inference_hub", ROUTE)
    prompt = agent.build_commons_prompt(
        selfish_gain=contract.private_gain,
        depletion_units=contract.reserve_cost,
        community_benefit=contract.community_benefit,
        day=1,
        living_agents=contract.society_size,
        resource_units=contract.capacity,
        resource_capacity=contract.capacity,
        previous_overuse_count=None,
        cumulative_private_payoff=0,
        cumulative_group_payoff=0,
    )
    generation_seed = repair._derive_seed(
        "inference_hub_part2_generation_v1",
        subject["target_id"],
        repair.EXPECTED_ENVIRONMENT_SEED,
        1,
        slot,
    )
    body, controls = runner._request_contract(
        subject,
        prompt=prompt,
        system_prompt=agent.system_prompt,
        generation_seed=generation_seed,
    )
    request_bytes = runner._canonical_bytes(body)
    return repair._reservation_payload(
        attempt_id=attempt_id,
        subject=subject,
        trajectory_index=repair.EXPECTED_TRAJECTORY_INDEX,
        environment_seed=repair.EXPECTED_ENVIRONMENT_SEED,
        day=1,
        slot=slot,
        attempt_number=attempt_number,
        generation_seed=generation_seed,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
        request_body_bytes=len(request_bytes),
        controls=controls,
    )


def test_stale_eighth_physical_attempt_resumes_with_skipped_ninth_and_zero_calls(
    tmp_path: Path,
) -> None:
    clients = (_Client(1), _Client(2), _Client(3))
    pool = _pool(_accounts(*clients))
    journal_path = tmp_path / "trajectory.jsonl"
    journal = runner._ChainedJournal(journal_path)
    failure = {
        "failure_code": "http_error",
        "transient": True,
        "http_status": 500,
        "error_type": "InferenceHubDiscoveryError",
    }
    for attempt_number in range(1, repair.DEFAULT_MAX_ATTEMPTS + 1):
        attempt_id = f"fixture-attempt-{attempt_number}"
        reservation, _client = pool.reserve(
            journal,
            _day_one_reservation_payload(
                attempt_id=attempt_id, attempt_number=attempt_number,
            ),
        )
        if attempt_number < repair.DEFAULT_MAX_ATTEMPTS:
            journal.append(
                repair._terminal_payload(
                    event="attempt_failed",
                    attempt_id=attempt_id,
                    subject=_subject(),
                    trajectory_index=repair.EXPECTED_TRAJECTORY_INDEX,
                    day=1,
                    slot=0,
                    request_sha256=reservation["request_sha256"],
                    failure=failure,
                )
            )

    result = repair._run_trajectory(
        subject=_subject(),
        trajectory_index=repair.EXPECTED_TRAJECTORY_INDEX,
        environment_seed=repair.EXPECTED_ENVIRONMENT_SEED,
        contract=runner.Part2Contract(1, 1, 12, 10, 2, 2, 5, 0.2),
        journal=journal,
        pool=pool,
        participant_workers=1,
        max_attempts=repair.DEFAULT_MAX_ATTEMPTS,
        initial_backoff_seconds=repair.DEFAULT_BACKOFF_SECONDS,
        sleep_fn=lambda _seconds: None,
    )

    assert result["operationally_eligible"] is False
    assert all(client.experiment_calls == [] for client in clients)
    reservations = [
        row for row in journal.records if row["event"] == "reserved_before_dispatch"
    ]
    assert len(reservations) == 9
    assert reservations[-1]["dispatch_skipped"] is True
    assert reservations[-1]["attempt_number"] == 9
    assert reservations[-1]["account_slot"] is None
    assert reservations[-1]["global_dispatch_ordinal"] is None
    assert any(
        row["event"] == "attempt_failed"
        and row["attempt_id"] == "fixture-attempt-8"
        and row["failure"]["failure_code"] == "stale_reserved_attempt"
        for row in journal.records
    )
    assert journal.records[-1]["failure"]["failure_code"] == (
        "resume_attempt_budget_exhausted"
    )


@pytest.mark.parametrize(
    ("attempt_number", "dispatch_skipped", "expected"),
    [
        (2, False, "not monotonic"),
        (1, True, "exact ceiling"),
    ],
)
def test_malformed_crash_suffix_is_rejected_without_network(
    tmp_path: Path,
    attempt_number: int,
    dispatch_skipped: bool,
    expected: str,
) -> None:
    clients = (_Client(1), _Client(2), _Client(3))
    pool = _pool(_accounts(*clients))
    journal = runner._ChainedJournal(
        tmp_path / f"malformed-{attempt_number}-{dispatch_skipped}.jsonl"
    )
    pool.reserve(
        journal,
        _day_one_reservation_payload(
            attempt_id="malformed-crash-suffix",
            attempt_number=attempt_number,
        ),
        dispatch_skipped=dispatch_skipped,
    )

    with pytest.raises(repair.Part2CascadingRepairError, match=expected):
        repair._v4_result_index(
            journal,
            _subject(),
            repair.EXPECTED_TRAJECTORY_INDEX,
            environment_seed=repair.EXPECTED_ENVIRONMENT_SEED,
            max_attempts=repair.DEFAULT_MAX_ATTEMPTS,
        )
    assert all(client.calls == [] for client in clients)


def _metric_row(
    subject: dict[str, Any], trajectory_index: int, *, eligible: bool = True,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "target_id": subject["target_id"],
        "upstream_provider": subject["upstream_provider"],
        "model": subject["model"],
        "trajectory_index": trajectory_index,
        "environment_seed_index": trajectory_index,
        "environment_seed": validator.EXPECTED_COMMON_ENVIRONMENT_SEEDS[trajectory_index],
        "operationally_eligible": eligible,
        "scheduled_agent_days": 4,
        "responses_received": 4 if eligible else 3,
        "invalid_count": 0 if eligible else 1,
        "identity_mismatch_count": 0,
        "transport_failure_count": 0 if eligible else 1,
        "restraint_count": 4 if eligible else 3,
        "overuse_count": 0,
        "restraint_rate": 1.0 if eligible else 0.75,
        "aurc": 1.0,
        "aupc": 1.0,
        "reserve_nondepletion": True,
        "final_reserve": 10,
        "final_population": 2,
        "population_retention": 1.0,
        "cumulative_private_payoff": 4 if eligible else 3,
        "cumulative_group_payoff": 10 if eligible else 0,
        "operational_repair_round": None,
        "source_replaced_for_operational_failure": False,
    }


def _parent_chain(tmp_path: Path) -> Any:
    subjects = []
    for index in range(21):
        target_id = repair.EXPECTED_TARGET_ID if index == 0 else f"fixture/target-{index:02d}"
        route = ROUTE if index == 0 else f"region/model-{index:02d}"
        subjects.append({
            "target_id": target_id,
            "upstream_provider": "google" if index == 0 else "fixture",
            "model": route.rsplit("/", 1)[-1],
            "route": route,
            "candidate_index": 0,
            "supported_controls": ["seed", "temperature", "top_p", "structured_response"],
            "selected_profile_id": f"profile-{index}",
            "selected_profile_request_sha256": f"{index:064x}",
        })
    hydrated_subjects = [
        {
            **subject,
            "id": subject["target_id"],
            "provider": subject["upstream_provider"],
            "target_model": subject["model"],
            "endpoint_profile": "nvidia_inference_hub",
            "verification_status": "fixture_verified",
            "route_source": "fixture_registry",
            "compatibility_max_tokens": runner.PART2_MAX_TOKENS_FLOOR,
        }
        for subject in subjects
    ]
    rows = []
    for subject in subjects:
        for trajectory_index in range(12):
            eligible = not (
                subject["target_id"] == repair.EXPECTED_TARGET_ID
                and trajectory_index == repair.EXPECTED_TRAJECTORY_INDEX
            )
            rows.append(_metric_row(subject, trajectory_index, eligible=eligible))
    source_path = tmp_path / "source/private/manifest.json"
    parent_path = tmp_path / "parent/private/manifest.json"
    for path in (source_path, parent_path):
        path.parent.mkdir(parents=True, mode=0o700)
        path.parent.chmod(0o700)
        path.write_text("{}\n", encoding="utf-8")
        path.chmod(0o600)
        lock = path.parent / ".run.lock"
        lock.touch(mode=0o600)
        lock.chmod(0o600)
    part2_contract = {
        "society_size": 2,
        "days": 2,
        "independent_trajectories": 12,
        "resource_capacity": 10,
        "option_a_private_gain": 1,
        "option_b_private_gain": 2,
        "option_b_reserve_cost": 2,
        "unanimous_a_group_payoff": 5,
        "unanimous_b_group_payoff": -5,
        "invalid_policy": "retain_as_INVALID_zero_effect_no_semantic_retry",
        "collapse_death_rate": 0.2,
        "attrition_policy": "matched_seed_day_random_sample_v1",
    }
    source = SimpleNamespace(
        manifest={
            "evidence_sha256": "a" * 64,
            "panel_id": validator.EXPECTED_PANEL_ID,
            "base_seed": validator.EXPECTED_BASE_SEED,
            "part2_contract": part2_contract,
            "common_environment_seeds": list(
                validator.EXPECTED_COMMON_ENVIRONMENT_SEEDS
            ),
            "subject_routes": subjects,
            "source_artifacts": {
                str(Path(runner.__file__).resolve()): (
                    validator.EXPECTED_RUNNER_IMPLEMENTATION_SHA256
                ),
            },
        },
        subjects=tuple(hydrated_subjects),
        environment_seeds=validator.EXPECTED_COMMON_ENVIRONMENT_SEEDS,
        contract=runner.Part2Contract(2, 2, 12, 10, 2, 2, 5, 0.2),
    )
    selected_key = (repair.EXPECTED_TARGET_ID, repair.EXPECTED_TRAJECTORY_INDEX)
    failure_keys = [
        (row["target_id"], row["trajectory_index"]) for row in rows
    ][:38]
    if selected_key not in failure_keys:
        failure_keys[-1] = selected_key
    source_failures = {key: {} for key in failure_keys}
    successful = {key: (1, {}) for key in failure_keys if key != selected_key}
    selected_row = next(
        row for row in rows
        if (row["target_id"], row["trajectory_index"]) == selected_key
    )
    return SimpleNamespace(
        source=source,
        parent_manifest={
            "evidence_sha256": "b" * 64,
            "summary": {
                "source_operational_failure_trajectories": 38,
                "operational_repairs_succeeded": 37,
                "operational_repairs_unresolved": 1,
            },
        },
        source_failures=source_failures,
        successful=successful,
        unresolved={selected_key: selected_row},
        effective_trajectories=tuple(rows),
        source_path=source_path,
        parent_path=parent_path,
    )


def _crashed_initial_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Path, Path, tuple[_Client, ...]]:
    chain = _parent_chain(tmp_path)
    clients = (_Client(1), _Client(2, catalog_error=401), _Client(3))
    monkeypatch.setattr(repair, "_load_parent_chain", lambda *_args: chain)
    monkeypatch.setattr(
        repair,
        "_runtime_accounts",
        lambda *_args, **kwargs: _accounts(
            *clients, cursor_epoch=kwargs["cursor_epoch"],
        ),
    )
    original_run_trajectory = repair._run_trajectory

    def simulated_crash(**_kwargs: Any) -> Any:
        raise RuntimeError("simulated crash after initial checkpoint")

    monkeypatch.setattr(repair, "_run_trajectory", simulated_crash)
    output_dir = tmp_path / "v4-crash-audit"
    credential_path = _credential_file(tmp_path / "crash-audit.env")
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            repair.run_cascading_repair(
                source_manifest_path=chain.source_path,
                parent_overlay_manifest_path=chain.parent_path,
                output_dir=output_dir,
                credential_env_file=credential_path,
                sleep_fn=lambda _seconds: None,
            )
    finally:
        monkeypatch.setattr(repair, "_run_trajectory", original_run_trajectory)
    return chain, output_dir, credential_path, clients


def _checkpoint_journal(
    output_dir: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], runner._ChainedJournal]:
    manifest_path = output_dir / "private/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ledger = json.loads(
        Path(manifest["preflight_ledger"]["path"]).read_text(encoding="utf-8")
    )
    journal_key = (
        f"{repair.EXPECTED_TARGET_ID}::"
        f"{repair.EXPECTED_TRAJECTORY_INDEX}::1"
    )
    journal_path = Path(manifest["journals"][journal_key]["path"])
    return manifest_path, manifest, ledger, runner._ChainedJournal(journal_path)


def _append_bound_reservation(
    journal: runner._ChainedJournal,
    *,
    ledger: Mapping[str, Any],
    attempt_id: str,
    attempt_number: int,
    participant_slot: int,
    dispatch_ordinal: int | None,
    account_slot: str | None,
    reserved_at_utc: str,
    preflight_session_index: int = 0,
    dispatch_skipped: bool = False,
    contract: runner.Part2Contract | None = None,
) -> dict[str, Any]:
    payload = _day_one_reservation_payload(
        attempt_id=attempt_id,
        attempt_number=attempt_number,
        slot=participant_slot,
        contract=contract,
    )
    session_sha256 = (
        ledger["initial_session_sha256"]
        if preflight_session_index == 0
        else ledger["resume_preflight_sessions"][
            preflight_session_index - 1
        ]["session_sha256"]
    )
    payload.update(
        {
            "account_slot": account_slot,
            "global_dispatch_ordinal": dispatch_ordinal,
            "cursor_epoch": ledger["cursor_epoch"],
            "preflight_session_index": preflight_session_index,
            "preflight_session_sha256": session_sha256,
            "reserved_at_utc": reserved_at_utc,
        }
    )
    if dispatch_skipped:
        payload["dispatch_skipped"] = True
    return journal.append(payload)


def _reseal_checkpoint_journal(
    manifest_path: Path,
    manifest: dict[str, Any],
    journal: runner._ChainedJournal,
    *,
    updated_at_utc: str | None = None,
) -> None:
    journal_key = (
        f"{repair.EXPECTED_TARGET_ID}::"
        f"{repair.EXPECTED_TRAJECTORY_INDEX}::1"
    )
    manifest["journals"][journal_key] = journal.reference()
    manifest["last_updated_at_utc"] = updated_at_utc or runner._utc_now()
    runner._seal(manifest)
    runner._atomic_json(manifest_path, manifest)


def _crash_existing_resume(
    *,
    chain: Any,
    output_dir: Path,
    credential_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_run_trajectory = repair._run_trajectory

    def simulated_resume_crash(**kwargs: Any) -> Any:
        if kwargs.get("pool") is None:
            return original_run_trajectory(**kwargs)
        raise RuntimeError("simulated crash after resume checkpoint")

    monkeypatch.setattr(repair, "_run_trajectory", simulated_resume_crash)
    try:
        with pytest.raises(RuntimeError, match="simulated crash after resume"):
            repair.run_cascading_repair(
                source_manifest_path=chain.source_path,
                parent_overlay_manifest_path=chain.parent_path,
                output_dir=output_dir,
                credential_env_file=credential_path,
                resume=True,
                sleep_fn=lambda _seconds: None,
            )
    finally:
        monkeypatch.setattr(
            repair, "_run_trajectory", original_run_trajectory,
        )


def _crashed_resume_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Path, Path, tuple[_Client, ...]]:
    chain, output_dir, credential_path, clients = _crashed_initial_checkpoint(
        tmp_path, monkeypatch,
    )
    _crash_existing_resume(
        chain=chain,
        output_dir=output_dir,
        credential_path=credential_path,
        monkeypatch=monkeypatch,
    )
    return chain, output_dir, credential_path, clients


def _install_no_live_call_tripwire(
    monkeypatch: pytest.MonkeyPatch,
) -> list[None]:
    credential_loads: list[None] = []

    def must_not_load_credentials(*_args: Any, **_kwargs: Any) -> Any:
        credential_loads.append(None)
        raise AssertionError("invalid retained evidence reached credential loading")

    monkeypatch.setattr(repair, "_runtime_accounts", must_not_load_credentials)
    return credential_loads


@pytest.mark.parametrize(
    "tamper",
    ["extra_key", "wrong_exact_path", "wrong_suffix_case"],
)
def test_resume_rejects_nonexact_journal_reference_before_live_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    chain, output_dir, credential_path, clients = _crashed_initial_checkpoint(
        tmp_path, monkeypatch,
    )
    manifest_path = output_dir / "private/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = (
        f"{repair.EXPECTED_TARGET_ID}::"
        f"{repair.EXPECTED_TRAJECTORY_INDEX}::1"
    )
    reference = manifest["journals"][key]
    if tamper == "extra_key":
        reference["unexpected"] = "must-be-rejected"
    elif tamper == "wrong_exact_path":
        reference["path"] = manifest["journals"][
            f"{repair.EXPECTED_TARGET_ID}::"
            f"{repair.EXPECTED_TRAJECTORY_INDEX}::2"
        ]["path"]
    else:
        reference["path"] = str(
            Path(reference["path"]).with_suffix(".JSONL")
        )
    runner._seal(manifest)
    runner._atomic_json(manifest_path, manifest)
    baseline_calls = [len(client.calls) for client in clients]
    credential_loads = _install_no_live_call_tripwire(monkeypatch)

    with pytest.raises(
        repair.Part2CascadingRepairError,
        match="(?i)journal|trajectory.*(?:reference|path)",
    ):
        repair.run_cascading_repair(
            source_manifest_path=chain.source_path,
            parent_overlay_manifest_path=chain.parent_path,
            output_dir=output_dir,
            credential_env_file=credential_path,
            resume=True,
            sleep_fn=lambda _seconds: None,
        )

    assert credential_loads == []
    assert [len(client.calls) for client in clients] == baseline_calls


def test_resume_rejects_child_journal_leaf_symlink_without_read_or_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain, output_dir, credential_path, clients = _crashed_initial_checkpoint(
        tmp_path, monkeypatch,
    )
    manifest_path = output_dir / "private/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = (
        f"{repair.EXPECTED_TARGET_ID}::"
        f"{repair.EXPECTED_TRAJECTORY_INDEX}::1"
    )
    journal_path = Path(manifest["journals"][key]["path"])
    assert not journal_path.exists()
    outside = tmp_path / "outside-child-journal.jsonl"
    outside.write_bytes(b'{"sentinel":"must-remain-byte-identical"}\n')
    outside.chmod(0o600)
    before = outside.read_bytes()
    journal_path.symlink_to(outside)
    baseline_calls = [len(client.calls) for client in clients]
    credential_loads = _install_no_live_call_tripwire(monkeypatch)

    with pytest.raises(
        repair.Part2CascadingRepairError,
        match="(?i)journal.*(?:symlink|regular)",
    ):
        repair.run_cascading_repair(
            source_manifest_path=chain.source_path,
            parent_overlay_manifest_path=chain.parent_path,
            output_dir=output_dir,
            credential_env_file=credential_path,
            resume=True,
            sleep_fn=lambda _seconds: None,
        )

    assert journal_path.is_symlink()
    assert outside.read_bytes() == before
    assert credential_loads == []
    assert [len(client.calls) for client in clients] == baseline_calls


def test_resume_rejects_initial_reservation_predating_child_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain, output_dir, credential_path, clients = _crashed_initial_checkpoint(
        tmp_path, monkeypatch,
    )
    manifest_path, manifest, ledger, journal = _checkpoint_journal(output_dir)
    _append_bound_reservation(
        journal,
        ledger=ledger,
        attempt_id="before-child-creation",
        attempt_number=1,
        participant_slot=0,
        dispatch_ordinal=1,
        account_slot="account-slot-01",
        reserved_at_utc="2099-01-01T00:00:01Z",
    )
    manifest["created_at_utc"] = "2099-01-01T00:00:02Z"
    _reseal_checkpoint_journal(
        manifest_path,
        manifest,
        journal,
        updated_at_utc="2099-01-01T00:00:03Z",
    )
    baseline_calls = [len(client.calls) for client in clients]
    credential_loads = _install_no_live_call_tripwire(monkeypatch)

    with pytest.raises(
        repair.Part2CascadingRepairError,
        match="(?i)dispatch.*predates.*child.*creation",
    ):
        repair.run_cascading_repair(
            source_manifest_path=chain.source_path,
            parent_overlay_manifest_path=chain.parent_path,
            output_dir=output_dir,
            credential_env_file=credential_path,
            resume=True,
            sleep_fn=lambda _seconds: None,
        )

    assert credential_loads == []
    assert [len(client.calls) for client in clients] == baseline_calls


def test_resume_rejects_latest_session_reservation_predating_last_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain, output_dir, credential_path, clients = _crashed_resume_checkpoint(
        tmp_path, monkeypatch,
    )
    manifest_path, manifest, ledger, journal = _checkpoint_journal(output_dir)
    assert manifest["resume_count"] == 1
    _append_bound_reservation(
        journal,
        ledger=ledger,
        attempt_id="before-latest-resume",
        attempt_number=1,
        participant_slot=0,
        dispatch_ordinal=1,
        account_slot="account-slot-01",
        reserved_at_utc="2099-01-01T00:00:01Z",
        preflight_session_index=1,
    )
    manifest["last_resumed_at_utc"] = "2099-01-01T00:00:02Z"
    _reseal_checkpoint_journal(
        manifest_path,
        manifest,
        journal,
        updated_at_utc="2099-01-01T00:00:03Z",
    )
    baseline_calls = [len(client.calls) for client in clients]
    credential_loads = _install_no_live_call_tripwire(monkeypatch)

    with pytest.raises(
        repair.Part2CascadingRepairError,
        match="(?i)dispatch.*predates.*resume.*checkpoint",
    ):
        repair.run_cascading_repair(
            source_manifest_path=chain.source_path,
            parent_overlay_manifest_path=chain.parent_path,
            output_dir=output_dir,
            credential_env_file=credential_path,
            resume=True,
            sleep_fn=lambda _seconds: None,
        )

    assert credential_loads == []
    assert [len(client.calls) for client in clients] == baseline_calls


def test_resume_rejects_stale_recovery_terminal_predating_last_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain, output_dir, credential_path, clients = _crashed_initial_checkpoint(
        tmp_path, monkeypatch,
    )
    manifest_path, manifest, ledger, journal = _checkpoint_journal(output_dir)
    reservation = _append_bound_reservation(
        journal,
        ledger=ledger,
        attempt_id="stale-before-latest-resume",
        attempt_number=1,
        participant_slot=0,
        dispatch_ordinal=1,
        account_slot="account-slot-01",
        reserved_at_utc=manifest["created_at_utc"],
        contract=chain.source.contract,
    )
    _reseal_checkpoint_journal(manifest_path, manifest, journal)
    _crash_existing_resume(
        chain=chain,
        output_dir=output_dir,
        credential_path=credential_path,
        monkeypatch=monkeypatch,
    )
    manifest_path, manifest, ledger, journal = _checkpoint_journal(output_dir)
    assert ledger["resume_preflight_sessions"][-1][
        "starting_global_dispatch_ordinal"
    ] == 1
    stale = repair._terminal_payload(
        event="attempt_failed",
        attempt_id=reservation["attempt_id"],
        subject=_subject(),
        trajectory_index=repair.EXPECTED_TRAJECTORY_INDEX,
        day=1,
        slot=0,
        request_sha256=reservation["request_sha256"],
        failure={
            "failure_code": "stale_reserved_attempt",
            "transient": True,
            "http_status": None,
        },
    )
    stale["completed_at_utc"] = "2099-01-01T00:00:01Z"
    journal.append(stale)
    manifest["last_resumed_at_utc"] = "2099-01-01T00:00:02Z"
    _reseal_checkpoint_journal(
        manifest_path,
        manifest,
        journal,
        updated_at_utc="2099-01-01T00:00:03Z",
    )
    baseline_calls = [len(client.calls) for client in clients]
    credential_loads = _install_no_live_call_tripwire(monkeypatch)

    with pytest.raises(
        repair.Part2CascadingRepairError,
        match="(?i)(?:stale|recovery).*predates.*(?:resume|checkpoint)",
    ):
        repair.run_cascading_repair(
            source_manifest_path=chain.source_path,
            parent_overlay_manifest_path=chain.parent_path,
            output_dir=output_dir,
            credential_env_file=credential_path,
            resume=True,
            sleep_fn=lambda _seconds: None,
        )

    assert credential_loads == []
    assert [len(client.calls) for client in clients] == baseline_calls


def test_resume_rejects_synthetic_recovery_terminal_predating_last_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain, output_dir, credential_path, clients = _crashed_resume_checkpoint(
        tmp_path, monkeypatch,
    )
    manifest_path, manifest, ledger, journal = _checkpoint_journal(output_dir)
    pending = _append_bound_reservation(
        journal,
        ledger=ledger,
        attempt_id="synthetic-before-latest-resume",
        attempt_number=repair.DEFAULT_MAX_ATTEMPTS + 1,
        participant_slot=0,
        dispatch_ordinal=None,
        account_slot=None,
        reserved_at_utc="2097-01-01T00:00:00Z",
        preflight_session_index=1,
        dispatch_skipped=True,
    )

    fresh = repair.blind_preflight_accounts(
        _accounts(
            _Client(21), _Client(22, catalog_error=401), _Client(23),
            cursor_epoch=ledger["cursor_epoch"],
        ),
        exact_route=ROUTE,
        cursor_epoch=ledger["cursor_epoch"],
        now=_clock("2098-01-01"),
    )
    ledger = repair._append_resume_preflight(
        ledger,
        fresh,
        exact_route=ROUTE,
        starting_global_dispatch_ordinal=0,
    )
    ledger_path = output_dir / "private/preflight/preflight-002.json"
    runner._atomic_json(ledger_path, ledger)

    contract = runner.Part2Contract(1, 1, 12, 10, 2, 2, 5, 0.2)
    agent = repair.Agent2("slot_00", "inference_hub", ROUTE)
    prompt = agent.build_commons_prompt(
        selfish_gain=contract.private_gain,
        depletion_units=contract.reserve_cost,
        community_benefit=contract.community_benefit,
        day=1,
        living_agents=1,
        resource_units=contract.capacity,
        resource_capacity=contract.capacity,
        previous_overuse_count=None,
        cumulative_private_payoff=0,
        cumulative_group_payoff=0,
    )
    original_now = runner._utc_now
    monkeypatch.setattr(
        runner, "_utc_now", lambda: "2098-01-01T00:00:10Z",
    )
    try:
        repair._dispatch_unit(
            journal=journal,
            subject=_subject(),
            trajectory_index=repair.EXPECTED_TRAJECTORY_INDEX,
            environment_seed=repair.EXPECTED_ENVIRONMENT_SEED,
            day=1,
            slot=0,
            prompt=prompt,
            system_prompt=agent.system_prompt,
            prior_attempt=repair.DEFAULT_MAX_ATTEMPTS,
            max_attempts=repair.DEFAULT_MAX_ATTEMPTS,
            initial_backoff_seconds=repair.DEFAULT_BACKOFF_SECONDS,
            pool=_pool(_accounts(_Client(31), _Client(32), _Client(33))),
            sleep_fn=lambda _seconds: None,
            pending_synthetic=pending,
        )
    finally:
        monkeypatch.setattr(runner, "_utc_now", original_now)

    manifest["preflight_ledger"] = {
        "path": str(ledger_path.resolve()),
        "file_sha256": runner._sha256_file(ledger_path),
        "evidence_sha256": ledger["evidence_sha256"],
    }
    manifest["resume_count"] = 2
    manifest["last_resumed_at_utc"] = "2099-01-01T00:00:00Z"
    _reseal_checkpoint_journal(
        manifest_path,
        manifest,
        journal,
        updated_at_utc="2099-01-01T00:00:01Z",
    )
    baseline_calls = [len(client.calls) for client in clients]
    credential_loads = _install_no_live_call_tripwire(monkeypatch)

    with pytest.raises(
        repair.Part2CascadingRepairError,
        match="(?i)(?:synthetic|recovery).*predates.*(?:resume|checkpoint)",
    ):
        repair.run_cascading_repair(
            source_manifest_path=chain.source_path,
            parent_overlay_manifest_path=chain.parent_path,
            output_dir=output_dir,
            credential_env_file=credential_path,
            resume=True,
            sleep_fn=lambda _seconds: None,
        )

    assert credential_loads == []
    assert [len(client.calls) for client in clients] == baseline_calls


def test_no_qualified_initial_preflight_allows_only_empty_bootstrap_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _parent_chain(tmp_path)
    rejected = tuple(_Client(index, catalog_error=401) for index in range(1, 4))
    qualified = (_Client(11), _Client(12, catalog_error=401), _Client(13))
    monkeypatch.setattr(repair, "_load_parent_chain", lambda *_args: chain)
    monkeypatch.setattr(
        repair,
        "_runtime_accounts",
        lambda *_args, **kwargs: _accounts(
            *rejected, cursor_epoch=kwargs["cursor_epoch"],
        ),
    )
    output_dir = tmp_path / "v4-empty-bootstrap-retry"
    credential_path = _credential_file(tmp_path / "empty-bootstrap.env")

    with pytest.raises(
        repair.Part2CascadingRepairError, match="No configured account passed"
    ):
        repair.run_cascading_repair(
            source_manifest_path=chain.source_path,
            parent_overlay_manifest_path=chain.parent_path,
            output_dir=output_dir,
            credential_env_file=credential_path,
            sleep_fn=lambda _seconds: None,
        )

    assert sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file()
    ) == ["private/.run.lock"]
    assert not (output_dir / "private/manifest.json").exists()
    assert not any((output_dir / "private/preflight").iterdir())
    assert not any(
        path.suffix == ".jsonl"
        for path in (output_dir / "private/trajectories").rglob("*")
    )

    monkeypatch.setattr(
        repair,
        "_runtime_accounts",
        lambda *_args, **kwargs: _accounts(
            *qualified, cursor_epoch=kwargs["cursor_epoch"],
        ),
    )
    manifest = repair.run_cascading_repair(
        source_manifest_path=chain.source_path,
        parent_overlay_manifest_path=chain.parent_path,
        output_dir=output_dir,
        credential_env_file=credential_path,
        sleep_fn=lambda _seconds: None,
    )

    assert manifest["complete"] is True
    assert [len(client.get_calls) for client in rejected] == [1, 1, 1]
    assert [len(client.post_calls) for client in rejected] == [0, 0, 0]
    assert [len(client.experiment_calls) for client in qualified] == [2, 0, 2]


def test_sealed_preflight_bootstrap_freshly_reprobes_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _parent_chain(tmp_path)
    initial_clients = (_Client(1), _Client(2, catalog_error=401), _Client(3))
    retry_clients = (_Client(11), _Client(12), _Client(13))
    runtime_calls = 0

    def runtime_accounts(*_args: Any, **kwargs: Any) -> Any:
        nonlocal runtime_calls
        runtime_calls += 1
        selected = initial_clients if runtime_calls == 1 else retry_clients
        return _accounts(*selected, cursor_epoch=kwargs["cursor_epoch"])

    monkeypatch.setattr(repair, "_load_parent_chain", lambda *_args: chain)
    monkeypatch.setattr(repair, "_runtime_accounts", runtime_accounts)
    original_atomic_json = runner._atomic_json
    crashed_after_ledger = False

    def crash_after_initial_ledger(path: Path, value: Mapping[str, Any]) -> None:
        nonlocal crashed_after_ledger
        original_atomic_json(path, value)
        if path.name == "preflight-000.json" and not crashed_after_ledger:
            crashed_after_ledger = True
            raise RuntimeError("simulated crash after sealed bootstrap preflight")

    monkeypatch.setattr(runner, "_atomic_json", crash_after_initial_ledger)
    output_dir = tmp_path / "v4-sealed-preflight-bootstrap"
    credential_path = _credential_file(tmp_path / "sealed-bootstrap.env")
    with pytest.raises(RuntimeError, match="sealed bootstrap preflight"):
        repair.run_cascading_repair(
            source_manifest_path=chain.source_path,
            parent_overlay_manifest_path=chain.parent_path,
            output_dir=output_dir,
            credential_env_file=credential_path,
            sleep_fn=lambda _seconds: None,
        )
    monkeypatch.setattr(runner, "_atomic_json", original_atomic_json)

    ledger_path = output_dir / "private/preflight/preflight-000.json"
    first_ledger_bytes = ledger_path.read_bytes()
    first_ledger = json.loads(first_ledger_bytes)
    assert first_ledger["qualified_account_slots"] == [
        "account-slot-01", "account-slot-03",
    ]
    assert not (output_dir / "private/manifest.json").exists()

    manifest = repair.run_cascading_repair(
        source_manifest_path=chain.source_path,
        parent_overlay_manifest_path=chain.parent_path,
        output_dir=output_dir,
        credential_env_file=credential_path,
        sleep_fn=lambda _seconds: None,
    )

    replacement = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert runtime_calls == 2
    assert replacement["cursor_epoch"] == first_ledger["cursor_epoch"]
    assert replacement["qualified_account_slots"] == [
        "account-slot-01", "account-slot-02", "account-slot-03",
    ]
    assert ledger_path.read_bytes() != first_ledger_bytes
    assert [len(client.get_calls) for client in retry_clients] == [1, 1, 1]
    assert [
        sum(call["preflight"] is True for call in client.post_calls)
        for client in retry_clients
    ] == [1, 1, 1]
    assert [len(client.experiment_calls) for client in retry_clients] == [2, 1, 1]
    assert manifest["credential_pool"]["qualified_account_slots"] == [
        "account-slot-01", "account-slot-02", "account-slot-03",
    ]
    assert manifest["complete"] is True


@pytest.mark.parametrize("residue", ["manifest", "journal"])
def test_no_qualified_bootstrap_retry_rejects_any_persisted_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    residue: str,
) -> None:
    chain = _parent_chain(tmp_path)
    rejected = tuple(_Client(index, catalog_error=401) for index in range(1, 4))
    monkeypatch.setattr(repair, "_load_parent_chain", lambda *_args: chain)
    monkeypatch.setattr(
        repair,
        "_runtime_accounts",
        lambda *_args, **kwargs: _accounts(
            *rejected, cursor_epoch=kwargs["cursor_epoch"],
        ),
    )
    output_dir = tmp_path / f"v4-bootstrap-residue-{residue}"
    credential_path = _credential_file(tmp_path / f"bootstrap-{residue}.env")
    with pytest.raises(repair.Part2CascadingRepairError):
        repair.run_cascading_repair(
            source_manifest_path=chain.source_path,
            parent_overlay_manifest_path=chain.parent_path,
            output_dir=output_dir,
            credential_env_file=credential_path,
            sleep_fn=lambda _seconds: None,
        )

    if residue == "manifest":
        evidence_path = output_dir / "private/manifest.json"
        evidence_path.write_text("{}\n", encoding="utf-8")
        evidence_path.chmod(0o600)
    else:
        evidence_path = (
            output_dir / "private/trajectories"
            / runner._safe_file_stem(repair.EXPECTED_TARGET_ID)
            / "seed-001-round-01.jsonl"
        )
        evidence_path.write_text("{}\n", encoding="utf-8")
        evidence_path.chmod(0o600)

    baseline_calls = [len(client.calls) for client in rejected]
    credential_loads = _install_no_live_call_tripwire(monkeypatch)
    with pytest.raises(
        repair.Part2CascadingRepairError,
        match="(?i)(?:bootstrap|committed|journal|preflight)",
    ):
        repair.run_cascading_repair(
            source_manifest_path=chain.source_path,
            parent_overlay_manifest_path=chain.parent_path,
            output_dir=output_dir,
            credential_env_file=credential_path,
            sleep_fn=lambda _seconds: None,
        )

    assert credential_loads == []
    assert [len(client.calls) for client in rejected] == baseline_calls


def test_resume_rejects_retained_401_before_credentials_or_fresh_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _parent_chain(tmp_path)
    clients = (
        _Client(1, experiment_http_statuses=(401,)),
        _Client(2, catalog_error=401),
        _Client(3, experiment_http_statuses=(401,)),
    )
    monkeypatch.setattr(repair, "_load_parent_chain", lambda *_args: chain)
    monkeypatch.setattr(
        repair,
        "_runtime_accounts",
        lambda *_args, **kwargs: _accounts(
            *clients, cursor_epoch=kwargs["cursor_epoch"],
        ),
    )
    output_dir = tmp_path / "v4-retained-401"
    credential_path = _credential_file(tmp_path / "retained-401.env")
    with pytest.raises(repair.CredentialQualificationInvalidatedError):
        repair.run_cascading_repair(
            source_manifest_path=chain.source_path,
            parent_overlay_manifest_path=chain.parent_path,
            output_dir=output_dir,
            credential_env_file=credential_path,
            sleep_fn=lambda _seconds: None,
        )

    manifest_path = output_dir / "private/manifest.json"
    before_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline_calls = [len(client.calls) for client in clients]
    credential_loads = _install_no_live_call_tripwire(monkeypatch)
    with pytest.raises(
        repair.CredentialQualificationInvalidatedError,
        match="(?i)retained.*preflight-qualified",
    ):
        repair.run_cascading_repair(
            source_manifest_path=chain.source_path,
            parent_overlay_manifest_path=chain.parent_path,
            output_dir=output_dir,
            credential_env_file=credential_path,
            resume=True,
            sleep_fn=lambda _seconds: None,
        )

    after_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert credential_loads == []
    assert [len(client.calls) for client in clients] == baseline_calls
    assert after_manifest == before_manifest
    assert not (output_dir / "private/preflight/preflight-001.json").exists()


def test_crash_after_physical_reservation_resumes_and_recursively_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_validator_runtime_hashes(monkeypatch)
    chain = _parent_chain(tmp_path)
    clients = (_Client(1), _Client(2, catalog_error=401), _Client(3))
    runtime_calls = 0

    def runtime_accounts(*_args: Any, **kwargs: Any) -> Any:
        nonlocal runtime_calls
        runtime_calls += 1
        return _accounts(*clients, cursor_epoch=kwargs["cursor_epoch"])

    monkeypatch.setattr(repair, "_load_parent_chain", lambda *_args: chain)
    monkeypatch.setattr(repair, "_runtime_accounts", runtime_accounts)
    original_reserve = repair.QualifiedAccountPool.reserve
    crash_lock = threading.Lock()
    reservation_written = False

    def crash_after_one_reservation(
        pool: repair.QualifiedAccountPool,
        journal: Any,
        payload: Mapping[str, Any],
        *,
        dispatch_skipped: bool = False,
    ) -> Any:
        nonlocal reservation_written
        with crash_lock:
            if reservation_written:
                raise RuntimeError("simulated process loss after reservation")
            appended = original_reserve(
                pool, journal, payload, dispatch_skipped=dispatch_skipped,
            )
            reservation_written = True
            raise RuntimeError("simulated process loss after reservation")

    monkeypatch.setattr(
        repair.QualifiedAccountPool, "reserve", crash_after_one_reservation,
    )
    output_dir = tmp_path / "v4-physical-reservation-crash"
    credential_path = _credential_file(tmp_path / "physical-crash.env")
    with pytest.raises(RuntimeError, match="process loss after reservation"):
        repair.run_cascading_repair(
            source_manifest_path=chain.source_path,
            parent_overlay_manifest_path=chain.parent_path,
            output_dir=output_dir,
            credential_env_file=credential_path,
            sleep_fn=lambda _seconds: None,
        )
    monkeypatch.setattr(
        repair.QualifiedAccountPool, "reserve", original_reserve,
    )

    manifest_path, before_manifest, initial_ledger, journal = (
        _checkpoint_journal(output_dir)
    )
    assert before_manifest.get("resume_count", 0) == 0
    assert len(journal.records) == 1
    first = journal.records[0]
    assert first["event"] == "reserved_before_dispatch"
    assert first["global_dispatch_ordinal"] == 1
    assert first["account_slot"] == "account-slot-01"
    assert first["preflight_session_index"] == 0
    assert first["preflight_session_sha256"] == initial_ledger[
        "initial_session_sha256"
    ]

    manifest = repair.run_cascading_repair(
        source_manifest_path=chain.source_path,
        parent_overlay_manifest_path=chain.parent_path,
        output_dir=output_dir,
        credential_env_file=credential_path,
        resume=True,
        sleep_fn=lambda _seconds: None,
    )

    manifest_path, sealed_manifest, ledger, journal = _checkpoint_journal(
        output_dir
    )
    assert runtime_calls == 2
    assert manifest == sealed_manifest
    assert manifest["complete"] is True
    assert manifest["resume_count"] == 1
    session = ledger["resume_preflight_sessions"][0]
    assert session["session_index"] == 1
    assert session["starting_global_dispatch_ordinal"] == 1
    physical = [
        row for row in journal.records
        if row["event"] == "reserved_before_dispatch"
        and row.get("dispatch_skipped") is not True
    ]
    assert [row["global_dispatch_ordinal"] for row in physical] == [1, 2, 3, 4, 5]
    assert [row["account_slot"] for row in physical] == [
        "account-slot-01", "account-slot-03", "account-slot-01",
        "account-slot-03", "account-slot-01",
    ]
    assert all(
        row["preflight_session_index"] == 1
        and row["preflight_session_sha256"] == session["session_sha256"]
        for row in physical[1:]
    )
    stale = next(
        row for row in journal.records
        if row["event"] == "attempt_failed"
        and row["attempt_id"] == first["attempt_id"]
    )
    assert stale["failure"] == {
        "failure_code": "stale_reserved_attempt",
        "transient": True,
        "http_status": None,
    }
    assert repair._utc_datetime(
        stale["completed_at_utc"], label="fixture stale completion"
    ) >= repair._utc_datetime(
        manifest["last_resumed_at_utc"], label="fixture resume checkpoint"
    )
    assert manifest["evidence_sha256"] == runner._self_hash(manifest)

    source_evidence = validator._SourceEvidence(
        manifest_path=chain.source_path.resolve(),
        manifest_file_sha256=runner._sha256_file(chain.source_path),
        manifest=chain.source.manifest,
        subjects=chain.source.subjects,
        contract=chain.source.contract,
        environment_seeds=chain.source.environment_seeds,
        trajectories=(),
        models=(),
    )
    parent_evidence = validator._ValidatedPartialOverlay(
        source=source_evidence,
        parent_manifest_path=chain.parent_path.resolve(),
        parent_manifest_file_sha256=runner._sha256_file(chain.parent_path),
        parent_manifest=chain.parent_manifest,
        effective_trajectories=chain.effective_trajectories,
        effective_models=(),
        source_failures=chain.source_failures,
        successful=chain.successful,
        unresolved=chain.unresolved,
        repair_round_count=8,
    )
    tracker = validator._FileTracker()
    tracker.add(manifest_path)
    validated = validator._validate_cascading_overlay(
        manifest_path,
        manifest,
        parent=parent_evidence,
        tracker=tracker,
        global_attempt_ids=set(),
    )
    tracker.verify()
    assert validated.overlay_manifest == manifest
    assert validated.repair_round_count == 9


def test_resume_rejects_physical_ordinals_two_then_one_before_live_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain, output_dir, credential_path, clients = _crashed_initial_checkpoint(
        tmp_path, monkeypatch,
    )
    manifest_path, manifest, ledger, journal = _checkpoint_journal(output_dir)
    _append_bound_reservation(
        journal,
        ledger=ledger,
        attempt_id="out-of-order-ordinal-two",
        attempt_number=1,
        participant_slot=0,
        dispatch_ordinal=2,
        account_slot="account-slot-03",
        reserved_at_utc="2099-01-01T00:00:00Z",
    )
    _append_bound_reservation(
        journal,
        ledger=ledger,
        attempt_id="out-of-order-ordinal-one",
        attempt_number=1,
        participant_slot=1,
        dispatch_ordinal=1,
        account_slot="account-slot-01",
        reserved_at_utc="2099-01-01T00:00:01Z",
    )
    _reseal_checkpoint_journal(manifest_path, manifest, journal)
    baseline_calls = [len(client.calls) for client in clients]
    runtime_calls = 0

    def must_not_load_credentials(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal runtime_calls
        runtime_calls += 1
        raise AssertionError("invalid ordinals reached credential loading")

    monkeypatch.setattr(repair, "_runtime_accounts", must_not_load_credentials)
    with pytest.raises(
        repair.Part2CascadingRepairError, match="dispatch provenance"
    ):
        repair.run_cascading_repair(
            source_manifest_path=chain.source_path,
            parent_overlay_manifest_path=chain.parent_path,
            output_dir=output_dir,
            credential_env_file=credential_path,
            resume=True,
            sleep_fn=lambda _seconds: None,
        )

    assert runtime_calls == 0
    assert [len(client.calls) for client in clients] == baseline_calls


def test_resume_rejects_retry_predating_terminal_before_live_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain, output_dir, credential_path, clients = _crashed_initial_checkpoint(
        tmp_path, monkeypatch,
    )
    manifest_path, manifest, ledger, journal = _checkpoint_journal(output_dir)
    first = _append_bound_reservation(
        journal,
        ledger=ledger,
        attempt_id="chronology-attempt-one",
        attempt_number=1,
        participant_slot=0,
        dispatch_ordinal=1,
        account_slot="account-slot-01",
        reserved_at_utc="2099-01-01T00:00:00Z",
    )
    terminal = repair._terminal_payload(
        event="attempt_failed",
        attempt_id="chronology-attempt-one",
        subject=_subject(),
        trajectory_index=repair.EXPECTED_TRAJECTORY_INDEX,
        day=1,
        slot=0,
        request_sha256=first["request_sha256"],
        failure={
            "failure_code": "http_error",
            "transient": True,
            "http_status": 500,
            "error_type": "InferenceHubDiscoveryError",
        },
    )
    terminal["completed_at_utc"] = "2099-01-01T00:00:02Z"
    journal.append(terminal)
    _append_bound_reservation(
        journal,
        ledger=ledger,
        attempt_id="chronology-attempt-two",
        attempt_number=2,
        participant_slot=0,
        dispatch_ordinal=2,
        account_slot="account-slot-03",
        reserved_at_utc="2099-01-01T00:00:01Z",
    )
    _reseal_checkpoint_journal(manifest_path, manifest, journal)
    baseline_calls = [len(client.calls) for client in clients]
    runtime_calls = 0

    def must_not_load_credentials(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal runtime_calls
        runtime_calls += 1
        raise AssertionError("invalid retry chronology reached credential loading")

    monkeypatch.setattr(repair, "_runtime_accounts", must_not_load_credentials)
    with pytest.raises(
        repair.Part2CascadingRepairError,
        match="retry reservation predates its failed predecessor",
    ):
        repair.run_cascading_repair(
            source_manifest_path=chain.source_path,
            parent_overlay_manifest_path=chain.parent_path,
            output_dir=output_dir,
            credential_env_file=credential_path,
            resume=True,
            sleep_fn=lambda _seconds: None,
        )

    assert runtime_calls == 0
    assert [len(client.calls) for client in clients] == baseline_calls


@pytest.mark.parametrize("status", [401, 403])
def test_run_cannot_complete_after_qualified_credential_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    chain = _parent_chain(tmp_path)
    clients = (
        _Client(1, experiment_http_statuses=(status,)),
        _Client(2, catalog_error=401),
        _Client(3, experiment_http_statuses=(status,)),
    )
    monkeypatch.setattr(repair, "_load_parent_chain", lambda *_args: chain)
    monkeypatch.setattr(
        repair,
        "_runtime_accounts",
        lambda *_args, **kwargs: _accounts(
            *clients, cursor_epoch=kwargs["cursor_epoch"],
        ),
    )
    output_dir = tmp_path / f"v4-auth-{status}"

    with pytest.raises(repair.CredentialQualificationInvalidatedError):
        repair.run_cascading_repair(
            source_manifest_path=chain.source_path,
            parent_overlay_manifest_path=chain.parent_path,
            output_dir=output_dir,
            credential_env_file=_credential_file(tmp_path / f"auth-{status}.env"),
            sleep_fn=lambda _seconds: None,
        )

    manifest_path = output_dir / "private/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    assert "completed_at_utc" not in manifest
    assert manifest["sanitized_artifacts"] == {}
    assert not (output_dir / "sanitized/effective_trajectory_metrics.json").exists()
    assert clients[1].experiment_calls == []
    assert 1 <= sum(len(client.experiment_calls) for client in clients) <= 2


def test_resume_rejects_swapped_credential_commitments_before_preflight_or_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _parent_chain(tmp_path)
    monkeypatch.setattr(repair, "_load_parent_chain", lambda *_args: chain)
    initial_clients = (_Client(1), _Client(2), _Client(3))
    resume_clients = (_Client(11), _Client(12), _Client(13))
    runtime_calls = 0

    def runtime_accounts(*_args: Any, **kwargs: Any) -> Any:
        nonlocal runtime_calls
        runtime_calls += 1
        if runtime_calls == 1:
            return _accounts(
                *initial_clients, cursor_epoch=kwargs["cursor_epoch"],
            )
        return _accounts(
            *resume_clients,
            cursor_epoch=kwargs["cursor_epoch"],
            credentials=(
                "fixture-secret-two",
                "fixture-secret-one",
                "fixture-secret-three",
            ),
        )

    monkeypatch.setattr(repair, "_runtime_accounts", runtime_accounts)
    original_run_trajectory = repair._run_trajectory

    def simulated_crash(**_kwargs: Any) -> Any:
        raise RuntimeError("simulated crash after the initial preflight checkpoint")

    monkeypatch.setattr(repair, "_run_trajectory", simulated_crash)
    output_dir = tmp_path / "v4-swapped-resume"
    credential_path = _credential_file(tmp_path / "swapped.env")
    with pytest.raises(RuntimeError, match="simulated crash"):
        repair.run_cascading_repair(
            source_manifest_path=chain.source_path,
            parent_overlay_manifest_path=chain.parent_path,
            output_dir=output_dir,
            credential_env_file=credential_path,
            sleep_fn=lambda _seconds: None,
        )
    monkeypatch.setattr(repair, "_run_trajectory", original_run_trajectory)
    assert runtime_calls == 1
    assert all(client.calls for client in initial_clients)

    with pytest.raises(
        repair.Part2CascadingRepairError,
        match="credentials or endpoint changed before blind preflight",
    ):
        repair.run_cascading_repair(
            source_manifest_path=chain.source_path,
            parent_overlay_manifest_path=chain.parent_path,
            output_dir=output_dir,
            credential_env_file=credential_path,
            resume=True,
            sleep_fn=lambda _seconds: None,
        )

    assert runtime_calls == 2
    assert all(client.calls == [] for client in resume_clients)


def test_terminally_exhausted_resume_rejects_before_credentials_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _parent_chain(tmp_path)
    monkeypatch.setattr(repair, "_load_parent_chain", lambda *_args: chain)
    initial_clients = tuple(
        _Client(index, experiment_response_route="wrong/route")
        for index in range(1, 4)
    )
    runtime_calls = 0

    def runtime_accounts(*_args: Any, **kwargs: Any) -> Any:
        nonlocal runtime_calls
        runtime_calls += 1
        if runtime_calls > 1:
            raise AssertionError(
                "terminally exhausted resume reached credential loading"
            )
        return _accounts(
            *initial_clients, cursor_epoch=kwargs["cursor_epoch"],
        )

    monkeypatch.setattr(repair, "_runtime_accounts", runtime_accounts)
    output_dir = tmp_path / "v4-terminally-exhausted"
    credential_path = _credential_file(tmp_path / "terminal.env")
    manifest = repair.run_cascading_repair(
        source_manifest_path=chain.source_path,
        parent_overlay_manifest_path=chain.parent_path,
        output_dir=output_dir,
        credential_env_file=credential_path,
        sleep_fn=lambda _seconds: None,
    )
    assert manifest["complete"] is False
    assert manifest["summary"]["cascading_repairs_unresolved"] == 1
    initial_call_counts = [len(client.calls) for client in initial_clients]

    with pytest.raises(
        repair.Part2CascadingRepairError, match="terminally exhausted"
    ):
        repair.run_cascading_repair(
            source_manifest_path=chain.source_path,
            parent_overlay_manifest_path=chain.parent_path,
            output_dir=output_dir,
            credential_env_file=credential_path,
            resume=True,
            sleep_fn=lambda _seconds: None,
        )

    assert runtime_calls == 1
    assert [len(client.calls) for client in initial_clients] == initial_call_counts


def test_complete_cascading_run_preserves_parent_rows_and_seals_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_validator_runtime_hashes(monkeypatch)
    chain = _parent_chain(tmp_path)
    clients = (_Client(1), _Client(2, catalog_error=401), _Client(3))
    monkeypatch.setattr(repair, "_load_parent_chain", lambda *_args: chain)
    monkeypatch.setattr(
        repair, "_runtime_accounts",
        lambda *_args, **kwargs: _accounts(
            *clients, cursor_epoch=kwargs["cursor_epoch"],
        ),
    )
    credential_file = _credential_file(tmp_path / ".env")

    manifest = repair.run_cascading_repair(
        source_manifest_path=chain.source_path,
        parent_overlay_manifest_path=chain.parent_path,
        output_dir=tmp_path / "v4",
        credential_env_file=credential_file,
        sleep_fn=lambda _seconds: None,
    )

    assert manifest["complete"] is True
    assert manifest["summary"] == {
        "original_source_operational_failure_trajectories": 38,
        "parent_repairs_succeeded": 37,
        "parent_repairs_unresolved": 1,
        "cascading_repairs_succeeded": 1,
        "cascading_repairs_unresolved": 0,
    }
    assert manifest["credential_pool"]["qualified_account_slots"] == [
        "account-slot-01", "account-slot-03",
    ]
    assert [len(client.get_calls) for client in clients] == [1, 1, 1]
    assert [len(client.post_calls) for client in clients] == [3, 0, 3]
    assert [len(client.experiment_calls) for client in clients] == [2, 0, 2]
    assert manifest["evidence_sha256"] == runner._self_hash(manifest)
    payload = json.loads(Path(
        manifest["sanitized_artifacts"]["effective_trajectory_metrics"]["path"]
    ).read_text(encoding="utf-8"))
    selected_key = (repair.EXPECTED_TARGET_ID, repair.EXPECTED_TRAJECTORY_INDEX)
    before = {
        (row["target_id"], row["trajectory_index"]): row
        for row in chain.effective_trajectories
    }
    after = {(row["target_id"], row["trajectory_index"]): row for row in payload["rows"]}
    assert after[selected_key]["operationally_eligible"] is True
    assert after[selected_key]["source_replaced_for_operational_failure"] is True
    assert all(after[key] == row for key, row in before.items() if key != selected_key)
    refs = list(manifest["journals"].values())
    assert refs[0]["record_count"] > 0
    assert all(ref["record_count"] == 0 for ref in refs[1:])

    source_evidence = validator._SourceEvidence(
        manifest_path=chain.source_path.resolve(),
        manifest_file_sha256=runner._sha256_file(chain.source_path),
        manifest=chain.source.manifest,
        subjects=chain.source.subjects,
        contract=chain.source.contract,
        environment_seeds=chain.source.environment_seeds,
        trajectories=(),
        models=(),
    )
    parent_evidence = validator._ValidatedPartialOverlay(
        source=source_evidence,
        parent_manifest_path=chain.parent_path.resolve(),
        parent_manifest_file_sha256=runner._sha256_file(chain.parent_path),
        parent_manifest=chain.parent_manifest,
        effective_trajectories=chain.effective_trajectories,
        effective_models=(),
        source_failures=chain.source_failures,
        successful=chain.successful,
        unresolved=chain.unresolved,
        repair_round_count=8,
    )
    tracker = validator._FileTracker()
    manifest_path = tmp_path / "v4/private/manifest.json"
    tracker.add(manifest_path)
    validated = validator._validate_cascading_overlay(
        manifest_path, manifest, parent=parent_evidence, tracker=tracker,
        global_attempt_ids=set(),
    )
    tracker.verify()
    assert validated.source_operational_failure_count == 38
    assert validated.repair_round_count == 9
    assert validated.effective_trajectories == tuple(payload["rows"])

    persisted_paths = [
        manifest_path,
        Path(manifest["preflight_ledger"]["path"]),
        Path(manifest["sanitized_artifacts"]["effective_trajectory_metrics"]["path"]),
        Path(manifest["sanitized_artifacts"]["effective_model_metrics"]["path"]),
    ]
    for path in persisted_paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        repair._assert_no_secret_like_fields(value, label=path.name)
        serialized = json.dumps(value, sort_keys=True)
        assert "fixture-secret-one" not in serialized
        assert "fixture-secret-two" not in serialized
        assert "fixture-secret-three" not in serialized
    for path in (tmp_path / "v4").rglob("*"):
        if path.is_file():
            serialized = path.read_text(encoding="utf-8")
            assert "fixture-secret-one" not in serialized
            assert "fixture-secret-two" not in serialized
            assert "fixture-secret-three" not in serialized
