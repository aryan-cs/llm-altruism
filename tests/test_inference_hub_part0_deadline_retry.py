from __future__ import annotations

from experiments.misc import inference_hub_part0_deadline_retry as accelerated


def test_part0_deadline_policy_is_bounded_provider_aware_and_rate_limited() -> None:
    contract = accelerated.PART0_DEADLINE_POLICY.evidence()
    assert contract["global_concurrency"] == 16
    assert contract["provider_concurrency"] == 3
    assert contract["global_requests_per_second"] == 10.0
    assert contract["provider_requests_per_second"] == 2.0
    assert contract["throttle_cooldown_seconds"] == 30.0


def test_part0_deadline_launcher_has_a_distinct_policy_scope() -> None:
    assert accelerated.POLICY_VERSION == 1
    assert accelerated.PERIODIC_HTTP_400_RETRY_CONTRACT == {
        "classification": "same_payload_http_400_operational_retry_v1",
        "identical_request_bytes_required": True,
        "bounded_by_max_attempts_per_request": True,
        "failed_attempts_retained_in_ledger": True,
    }
