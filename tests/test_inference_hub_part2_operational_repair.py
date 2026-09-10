import json
from pathlib import Path

import pytest

from experiments.misc.inference_hub_part2_operational_repair import (
    Part2OperationalRepairError,
    PooledInferenceHubClient,
    _pooled_runtime_client,
    _read_credential_pool,
)
from experiments.misc.inference_hub_part2_panel import (
    HIGH_LATENCY_ORIGINAL_SCALE_RATE_PROFILE,
)


class _Client:
    base_url = "https://inference-api.nvidia.com/v1"
    rate_limit_contract = {"policy_sha256": "same-policy"}

    def __init__(self, account: int) -> None:
        self.account = account
        self.calls = []

    def post(self, path, body, *, upstream_provider=None):
        self.calls.append((path, body, upstream_provider))
        return {"account": self.account}


def test_pooled_client_round_robins_exact_route_calls() -> None:
    clients = [_Client(index) for index in range(3)]
    pool = PooledInferenceHubClient(clients)

    observed = [
        pool.post("/chat/completions", {"model": "azure/anthropic/claude-haiku-4-5"}, upstream_provider="anthropic")["account"]
        for _ in range(7)
    ]

    assert observed == [0, 1, 2, 0, 1, 2, 0]
    assert pool.account_count == 3
    assert sum(len(client.calls) for client in clients) == 7
    assert all(
        call[2] == "anthropic"
        for client in clients
        for call in client.calls
    )


def test_credential_pool_loads_exact_unique_ordinals_without_serializing(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "BETTER_GOS_NVIDIA_API_KEY=secret-one\n"
        "BETTER_GOS_NVIDIA_API_KEY_2=secret-two\n"
        "BETTER_GOS_NVIDIA_API_KEY_3=secret-three\n"
        "UNRELATED=value\n",
        encoding="utf-8",
    )

    keys = _read_credential_pool(path, 3)

    assert keys == ("secret-one", "secret-two", "secret-three")
    binding = {
        "account_count": len(keys),
        "selection_policy": "thread_safe_round_robin",
        "rate_limit_scope": "independent_per_account",
    }
    serialized = json.dumps(binding)
    assert "secret" not in serialized


def test_pooled_runtime_uses_independent_account_limiters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "BETTER_GOS_NVIDIA_API_KEY=secret-one\n"
        "BETTER_GOS_NVIDIA_API_KEY_2=secret-two\n"
        "BETTER_GOS_NVIDIA_API_KEY_3=secret-three\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("INFERENCE_HUB_BASE_URL", _Client.base_url)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    pool = _pooled_runtime_client(
        path, 3, 60.0, HIGH_LATENCY_ORIGINAL_SCALE_RATE_PROFILE,
    )

    assert pool.account_count == 3
    assert len({client.rate_limiter.state_path for client in pool._clients}) == 3
    assert "NVIDIA_API_KEY" not in __import__("os").environ


@pytest.mark.parametrize(
    "contents",
    [
        "BETTER_GOS_NVIDIA_API_KEY=one\nBETTER_GOS_NVIDIA_API_KEY_3=three\n",
        "BETTER_GOS_NVIDIA_API_KEY=one\nBETTER_GOS_NVIDIA_API_KEY_2=one\nBETTER_GOS_NVIDIA_API_KEY_3=three\n",
    ],
)
def test_credential_pool_fails_closed_on_missing_or_duplicate_accounts(
    tmp_path: Path, contents: str,
) -> None:
    path = tmp_path / ".env"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(Part2OperationalRepairError, match="exactly 3 unique"):
        _read_credential_pool(path, 3)
