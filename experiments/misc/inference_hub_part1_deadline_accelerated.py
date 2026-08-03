"""Run the 75-route Part 1 panel with bounded deadline parallelism.

This launcher preserves the provider-safe-v2 prompt, parsing, identity, journal,
and denominator contracts.  It changes only operational scheduling: at most 24
requests globally and four per upstream provider, with explicit start-rate
limits and the shared limiter's normal 30-second throttle cooldown.
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
PART1_DEADLINE_POLICY = RateLimitPolicy(
    global_concurrency=24,
    provider_concurrency=4,
    global_requests_per_second=12.0,
    provider_requests_per_second=2.5,
)


def _deadline_client(timeout_seconds: float) -> InferenceHubClient:
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise InferenceHubDiscoveryError(
            "NVIDIA_API_KEY is required for an InferenceHub campaign."
        )
    base_url = validate_endpoint_base_url(
        "inference_hub",
        os.getenv("INFERENCE_HUB_BASE_URL", DEFAULT_BASE_URL),
    )
    scope_id = hashlib.sha256(
        (f"{base_url}\0{api_key}\0part1-deadline-v{POLICY_VERSION}").encode("utf-8")
    ).hexdigest()
    return InferenceHubClient(
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        rate_limiter=InferenceHubRateLimiter(
            policy=PART1_DEADLINE_POLICY,
            scope_id=scope_id,
        ),
    )


def cli(argv: Sequence[str] | None = None) -> int:
    from experiments.misc import inference_hub_part1_panel as base
    from experiments.misc import inference_hub_part1_stratified_panel as runner

    source_paths = getattr(base, "_SOURCE_PATHS")
    for source in (
        Path(__file__).with_name("inference_hub_provider_safe_v2.py").resolve(),
        Path(__file__).resolve(),
    ):
        if source not in source_paths:
            source_paths = (*source_paths, source)
    setattr(base, "_SOURCE_PATHS", source_paths)
    base.select_routes = _scheduled_selector(base.select_routes)
    base._client_from_environment = _deadline_client
    return runner.cli(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(cli())
