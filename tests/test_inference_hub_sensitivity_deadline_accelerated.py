from __future__ import annotations

from experiments.misc import inference_hub_sensitivity_deadline_accelerated as accelerated


def test_sensitivity_deadline_policy_is_bounded_and_provider_aware() -> None:
    contract = accelerated.SENSITIVITY_DEADLINE_POLICY.evidence()
    assert contract["global_concurrency"] == 24
    assert contract["provider_concurrency"] == 4
    assert contract["global_requests_per_second"] == 12.0
    assert contract["provider_requests_per_second"] == 2.5
    assert contract["throttle_cooldown_seconds"] == 30.0


def test_sensitivity_deadline_launcher_has_a_distinct_policy_scope() -> None:
    assert accelerated.POLICY_VERSION == 1
