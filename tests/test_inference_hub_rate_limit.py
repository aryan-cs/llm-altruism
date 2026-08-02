import json
import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from experiments.misc.inference_hub_rate_limit import (
    RATE_LIMIT_SCHEMA_VERSION,
    InferenceHubRateLimiter,
    RateLimitPolicy,
    provider_for_route,
)


def _process_limit_worker(state_path: str, provider: str, queue: object) -> None:
    limiter = InferenceHubRateLimiter(
        policy=_policy(), state_path=Path(state_path)
    )
    with limiter.limit(provider):
        queue.put(("enter", provider, time.time(), os.getpid()))
        time.sleep(0.05)
        queue.put(("exit", provider, time.time(), os.getpid()))


def _policy(**overrides: object) -> RateLimitPolicy:
    values = {
        "global_concurrency": 2,
        "provider_concurrency": 1,
        "global_requests_per_second": 1_000.0,
        "provider_requests_per_second": 1_000.0,
        "lease_seconds": 2.0,
        "poll_seconds": 0.002,
        "transient_cooldown_seconds": 0.02,
        "throttle_cooldown_seconds": 0.04,
    }
    values.update(overrides)
    return RateLimitPolicy(**values)


def test_provider_route_normalization() -> None:
    assert provider_for_route("aws/anthropic/claude-opus-4-5") == "anthropic"
    assert provider_for_route("anthropic/claude-sonnet") == "anthropic"
    assert provider_for_route("azure/openai/gpt-5") == "openai"
    assert provider_for_route("gcp/google/gemini-2.5-pro") == "google"
    assert provider_for_route("nvidia/google/gemini-3-pro") == "google"
    assert provider_for_route("zai-org/glm-5") == "zai-org"
    assert provider_for_route(None) == "inference_hub_metadata"


def test_thread_contention_enforces_global_and_provider_caps(tmp_path: Path) -> None:
    limiter = InferenceHubRateLimiter(
        policy=_policy(), state_path=tmp_path / "state.json"
    )
    guard = threading.Lock()
    active_global = 0
    active_by_provider: dict[str, int] = {}
    observed_global = 0
    observed_provider = 0

    def work(provider: str) -> None:
        nonlocal active_global, observed_global, observed_provider
        with limiter.limit(provider):
            with guard:
                active_global += 1
                active_by_provider[provider] = active_by_provider.get(provider, 0) + 1
                observed_global = max(observed_global, active_global)
                observed_provider = max(
                    observed_provider, active_by_provider[provider]
                )
            time.sleep(0.03)
            with guard:
                active_global -= 1
                active_by_provider[provider] -= 1

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(work, ["anthropic", "anthropic", "openai"] * 2))

    assert observed_global == 2
    assert observed_provider == 1


def test_process_contention_uses_the_same_file_backed_caps(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_process_limit_worker,
            args=(str(tmp_path / "state.json"), provider, queue),
        )
        for provider in ("anthropic", "anthropic", "openai")
    ]
    for process in processes:
        process.start()
    events = [queue.get(timeout=10) for _ in range(6)]
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    active_global = 0
    active_by_provider: dict[str, int] = {}
    for event, provider, _timestamp, _pid in sorted(events, key=lambda row: row[2]):
        if event == "enter":
            active_global += 1
            active_by_provider[provider] = active_by_provider.get(provider, 0) + 1
            assert active_global <= 2
            assert active_by_provider[provider] <= 1
        else:
            active_global -= 1
            active_by_provider[provider] -= 1
    assert active_global == 0


def test_provider_start_spacing_is_enforced(tmp_path: Path) -> None:
    limiter = InferenceHubRateLimiter(
        policy=_policy(provider_requests_per_second=10.0),
        state_path=tmp_path / "state.json",
    )
    starts = []
    for _ in range(2):
        with limiter.limit("anthropic"):
            starts.append(time.time())
    assert starts[1] - starts[0] >= 0.09


def test_heartbeat_prevents_overadmission_past_initial_lease(tmp_path: Path) -> None:
    limiter = InferenceHubRateLimiter(
        policy=_policy(lease_seconds=0.09, poll_seconds=0.005),
        state_path=tmp_path / "state.json",
    )
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first() -> None:
        with limiter.limit("anthropic"):
            first_entered.set()
            assert release_first.wait(timeout=2)

    def second() -> None:
        with limiter.limit("anthropic"):
            second_entered.set()

    first_thread = threading.Thread(target=first)
    first_thread.start()
    assert first_entered.wait(timeout=2)
    time.sleep(0.14)
    second_thread = threading.Thread(target=second)
    second_thread.start()
    assert not second_entered.wait(timeout=0.06)
    release_first.set()
    assert second_entered.wait(timeout=2)
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)


def test_expired_or_dead_lease_is_recovered(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    policy = _policy()
    state_path.write_text(
        json.dumps(
            {
                "schema_version": RATE_LIMIT_SCHEMA_VERSION,
                "policy_sha256": policy.evidence()["policy_sha256"],
                "next_global_at": 0.0,
                "next_provider_at": {},
                "leases": [
                    {
                        "token": "dead",
                        "pid": os.getpid(),
                        "provider": "anthropic",
                        "expires_at": time.time() - 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    limiter = InferenceHubRateLimiter(policy=policy, state_path=state_path)
    with limiter.limit("anthropic"):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert [lease["token"] for lease in state["leases"]] != ["dead"]


def test_penalty_publishes_shared_provider_cooldown(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    limiter = InferenceHubRateLimiter(policy=_policy(), state_path=state_path)
    before = time.time()
    delay = limiter.penalize("anthropic", http_status=429, retry_after="0.2")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert delay == pytest.approx(0.2)
    assert state["next_provider_at"]["anthropic"] >= before + 0.19
    assert state["next_global_at"] >= before + 0.03


@pytest.mark.parametrize(
    ("http_status", "expected_delay"),
    [(408, 0.02), (429, 0.04), (500, 0.04), (503, 0.04), (529, 0.04)],
)
def test_all_server_errors_receive_full_throttle_cooldown(
    tmp_path: Path, http_status: int, expected_delay: float
) -> None:
    limiter = InferenceHubRateLimiter(
        policy=_policy(), state_path=tmp_path / f"state-{http_status}.json"
    )
    assert limiter.penalize("provider", http_status=http_status) == pytest.approx(
        expected_delay
    )


def test_policy_contract_is_stable_and_credential_free() -> None:
    evidence = RateLimitPolicy().evidence()
    assert evidence["algorithm"].startswith("cross_process_provider_aware")
    assert len(evidence["policy_sha256"]) == 64
    assert "key" not in json.dumps(evidence).lower()


def test_every_registered_route_falls_back_to_exact_upstream_provider() -> None:
    registry_path = Path("agents/agent_config.registry.json")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = [
        row for row in registry["targets"] if row.get("provider") == "inference_hub"
    ]
    assert rows
    assert {
        row["id"]: (provider_for_route(row["route"]), row["upstream_provider"])
        for row in rows
        if provider_for_route(row["route"]) != row["upstream_provider"]
    } == {}


def test_corrupt_state_fails_closed(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text("{truncated", encoding="utf-8")
    limiter = InferenceHubRateLimiter(policy=_policy(), state_path=state_path)
    with pytest.raises(RuntimeError, match="refusing to fail open"):
        limiter.acquire("anthropic")
