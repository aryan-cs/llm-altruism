from __future__ import annotations

from experiments.misc import inference_hub_exploratory_accelerated as accelerated


def test_accelerated_policy_is_bounded_and_rate_limited() -> None:
    contract = accelerated.EXPLORATORY_ACCELERATED_POLICY.evidence()
    assert contract["global_concurrency"] == 12
    assert contract["provider_concurrency"] == 3
    assert contract["global_requests_per_second"] == 8.0
    assert contract["provider_requests_per_second"] == 2.0
    assert contract["throttle_cooldown_seconds"] == 30.0


def test_launcher_exposes_only_exploratory_campaigns() -> None:
    assert set(accelerated._RUNNERS) == {"role", "sensitivity"}
