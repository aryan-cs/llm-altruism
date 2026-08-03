from __future__ import annotations

from experiments.misc import inference_hub_part1_deadline_accelerated as accelerated


def test_part1_deadline_policy_is_bounded_provider_aware_and_rate_limited() -> None:
    contract = accelerated.PART1_DEADLINE_POLICY.evidence()
    assert contract["global_concurrency"] == 24
    assert contract["provider_concurrency"] == 4
    assert contract["global_requests_per_second"] == 12.0
    assert contract["provider_requests_per_second"] == 2.5
    assert contract["throttle_cooldown_seconds"] == 30.0


def test_part1_deadline_launcher_has_a_distinct_policy_scope() -> None:
    assert accelerated.POLICY_VERSION == 1
