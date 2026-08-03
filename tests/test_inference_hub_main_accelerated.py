from __future__ import annotations

from experiments.misc import inference_hub_main_accelerated as accelerated


def test_main_accelerated_policy_remains_bounded_and_rate_limited() -> None:
    contract = accelerated.MAIN_ACCELERATED_POLICY.evidence()
    assert contract["global_concurrency"] == 12
    assert contract["provider_concurrency"] == 2
    assert contract["global_requests_per_second"] == 8.0
    assert contract["provider_requests_per_second"] == 1.5
    assert contract["throttle_cooldown_seconds"] == 30.0


def test_main_launcher_exposes_exactly_three_phase_runners() -> None:
    assert set(accelerated._RUNNERS) == {"part0", "part1", "part2"}
