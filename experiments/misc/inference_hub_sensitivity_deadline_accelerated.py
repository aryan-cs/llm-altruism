"""Run the two-seed sensitivity matrix with bounded deadline parallelism.

The scientific design, prompts, parsing, identity checks, invalid policy,
journals, and all-scheduled denominators remain those of the v1 sensitivity
runner.  This launcher changes only operational scheduling and keeps explicit
global, provider, start-rate, and throttle-cooldown bounds.
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
from experiments.misc.inference_hub_rate_limit import (
    InferenceHubRateLimiter,
    RateLimitPolicy,
)


POLICY_VERSION = 1
SENSITIVITY_DEADLINE_POLICY = RateLimitPolicy(
    global_concurrency=24,
    provider_concurrency=3,
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
        "inference_hub", os.getenv("INFERENCE_HUB_BASE_URL", DEFAULT_BASE_URL)
    )
    scope_id = hashlib.sha256(
        (f"{base_url}\0{api_key}\0sensitivity-deadline-v{POLICY_VERSION}").encode(
            "utf-8"
        )
    ).hexdigest()
    return InferenceHubClient(
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        rate_limiter=InferenceHubRateLimiter(
            policy=SENSITIVITY_DEADLINE_POLICY,
            scope_id=scope_id,
        ),
    )


def cli(argv: Sequence[str] | None = None) -> int:
    from experiments.misc import inference_hub_part2_sensitivity_v1 as runner

    source_paths = getattr(runner, "_SOURCE_PATHS")
    source = Path(__file__).resolve()
    if source not in source_paths:
        setattr(runner, "_SOURCE_PATHS", (*source_paths, source))
    runner._provider_safe_client = _deadline_client
    return runner.cli(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(cli())
