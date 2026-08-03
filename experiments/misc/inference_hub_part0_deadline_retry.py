"""Run Part 0 with bounded deadline parallelism and periodic HTTP-400 retry.

The Part 0 request, judge, parser, identity, journal, and denominator contracts
remain unchanged.  Some InferenceHub routes intermittently return HTTP 400 for
otherwise compatibility-validated identical payloads, so this launcher treats
that status as an operationally retryable response within the existing bounded
attempt budget.  Every failed attempt remains in the append-only ledger.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from agents.agent_config import validate_endpoint_base_url
from experiments.misc.inference_hub_discovery import (
    DEFAULT_BASE_URL,
    InferenceHubClient,
    InferenceHubDiscoveryError,
)
from experiments.misc.inference_hub_provider_safe_v2 import _scheduled_selector
from experiments.misc.inference_hub_rate_limit import (
    InferenceHubRateLimiter,
    RateLimitPolicy,
)


POLICY_VERSION = 1
PART0_DEADLINE_POLICY = RateLimitPolicy(
    global_concurrency=16,
    provider_concurrency=3,
    global_requests_per_second=10.0,
    provider_requests_per_second=2.0,
)


def _deadline_client(timeout_seconds: float) -> InferenceHubClient:
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise InferenceHubDiscoveryError(
            "NVIDIA_API_KEY is required for an InferenceHub campaign."
        )
    base_url = validate_endpoint_base_url(
        "inference_hub", os.getenv("INFERENCE_HUB_BASE_URL", DEFAULT_BASE_URL)
    )
    scope_id = hashlib.sha256(
        (f"{base_url}\0{api_key}\0part0-deadline-retry-v{POLICY_VERSION}").encode(
            "utf-8"
        )
    ).hexdigest()
    return InferenceHubClient(
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        rate_limiter=InferenceHubRateLimiter(
            policy=PART0_DEADLINE_POLICY,
            scope_id=scope_id,
        ),
    )


def cli(argv: Sequence[str] | None = None) -> int:
    from experiments.misc import inference_hub_part0_panel as runner
    from experiments.misc import inference_hub_part1_panel as base

    source_paths = getattr(runner, "_SOURCE_PATHS")
    for source in (
        Path(__file__).with_name("inference_hub_provider_safe_v2.py").resolve(),
        Path(__file__).resolve(),
    ):
        if source not in source_paths:
            source_paths = (*source_paths, source)
    setattr(runner, "_SOURCE_PATHS", source_paths)
    runner.select_routes = _scheduled_selector(runner.select_routes)
    runner._client_from_environment = _deadline_client

    ordinary_transient = base._transient

    def periodic_transient(error: Exception) -> tuple[bool, str, int | None]:
        retryable, failure_code, http_status = ordinary_transient(error)
        if isinstance(error, InferenceHubDiscoveryError) and http_status == 400:
            return True, "http_400_periodic_retry", http_status
        return retryable, failure_code, http_status

    runner._transient = periodic_transient
    return runner.cli(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(cli())
