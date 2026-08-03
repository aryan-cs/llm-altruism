"""Launch deadline exploratory campaigns with bounded provider parallelism.

The main Part 0--2 panels retain the conservative provider-safe-v1 limiter.
This launcher is restricted to the explicitly exploratory Part 1 role
calibration and Part 2 sensitivity campaigns. It shares a separate,
file-backed limiter across those two processes and caps each upstream provider
at three in-flight requests and two starts per second.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Callable, Sequence

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
EXPLORATORY_ACCELERATED_POLICY = RateLimitPolicy(
    global_concurrency=12,
    provider_concurrency=3,
    global_requests_per_second=8.0,
    provider_requests_per_second=2.0,
)


def _accelerated_client(timeout_seconds: float) -> InferenceHubClient:
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
        (
            f"{base_url}\0{api_key}\0"
            f"exploratory-accelerated-v{POLICY_VERSION}"
        ).encode("utf-8")
    ).hexdigest()
    return InferenceHubClient(
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        rate_limiter=InferenceHubRateLimiter(
            policy=EXPLORATORY_ACCELERATED_POLICY,
            scope_id=scope_id,
        ),
    )


def _bind_source(module: object) -> None:
    source_paths = getattr(module, "_SOURCE_PATHS")
    source = Path(__file__).resolve()
    if source not in source_paths:
        setattr(module, "_SOURCE_PATHS", (*source_paths, source))


def _role(arguments: Sequence[str]) -> int:
    from experiments.misc import inference_hub_part1_role_calibration_v1 as runner

    _bind_source(runner)
    runner._provider_safe_client = _accelerated_client
    arguments = list(arguments)
    if "--max-workers-per-provider" not in arguments:
        arguments.extend(["--max-workers-per-provider", "3"])
    return runner.cli(arguments)


def _sensitivity(arguments: Sequence[str]) -> int:
    from experiments.misc import inference_hub_part2_sensitivity_v1 as runner

    _bind_source(runner)
    runner._provider_safe_client = _accelerated_client
    return runner.cli(list(arguments))


_RUNNERS: dict[str, Callable[[Sequence[str]], int]] = {
    "role": _role,
    "sensitivity": _sensitivity,
}


def cli(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in _RUNNERS:
        choices = ", ".join(sorted(_RUNNERS))
        print(
            "Usage: python -m "
            "experiments.misc.inference_hub_exploratory_accelerated "
            f"{{{choices}}} [runner arguments]",
            file=sys.stderr,
        )
        return 2
    return _RUNNERS[arguments[0]](arguments[1:])


if __name__ == "__main__":
    raise SystemExit(cli())
