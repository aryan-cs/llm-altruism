"""Launch main hosted panels with bounded deadline-safe parallelism.

This source-bound launcher preserves each Part 0--2 request and evidence
contract while using a separate shared limiter capped at two in-flight
requests and 1.5 starts per second for any upstream provider. Subject routes
are stably round-robined by provider before submission to avoid worker-pool
head-of-line blocking.
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
from experiments.misc.inference_hub_provider_safe_v2 import _scheduled_selector
from experiments.misc.inference_hub_rate_limit import (
    InferenceHubRateLimiter,
    RateLimitPolicy,
)


POLICY_VERSION = 1
MAIN_ACCELERATED_POLICY = RateLimitPolicy(
    global_concurrency=12,
    provider_concurrency=2,
    global_requests_per_second=8.0,
    provider_requests_per_second=1.5,
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
        (f"{base_url}\0{api_key}\0main-accelerated-v{POLICY_VERSION}").encode(
            "utf-8"
        )
    ).hexdigest()
    return InferenceHubClient(
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        rate_limiter=InferenceHubRateLimiter(
            policy=MAIN_ACCELERATED_POLICY,
            scope_id=scope_id,
        ),
    )


def _bind_sources(module: object) -> None:
    source_paths = getattr(module, "_SOURCE_PATHS")
    for source in (
        Path(__file__).with_name("inference_hub_provider_safe_v2.py").resolve(),
        Path(__file__).resolve(),
    ):
        if source not in source_paths:
            source_paths = (*source_paths, source)
    setattr(module, "_SOURCE_PATHS", source_paths)


def _part0(arguments: Sequence[str]) -> int:
    from experiments.misc import inference_hub_part0_panel as runner

    _bind_sources(runner)
    runner.select_routes = _scheduled_selector(runner.select_routes)
    runner._client_from_environment = _accelerated_client
    return runner.cli(list(arguments))


def _part1(arguments: Sequence[str]) -> int:
    from experiments.misc import inference_hub_part1_panel as base
    from experiments.misc import inference_hub_part1_stratified_panel as runner

    _bind_sources(base)
    base.select_routes = _scheduled_selector(base.select_routes)
    base._client_from_environment = _accelerated_client
    return runner.cli(list(arguments))


def _part2(arguments: Sequence[str]) -> int:
    from experiments.misc import inference_hub_part2_panel as runner

    _bind_sources(runner)
    runner.select_routes = _scheduled_selector(runner.select_routes)
    runner._client_from_environment = _accelerated_client
    return runner.cli(list(arguments))


_RUNNERS: dict[str, Callable[[Sequence[str]], int]] = {
    "part0": _part0,
    "part1": _part1,
    "part2": _part2,
}


def cli(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in _RUNNERS:
        choices = ", ".join(sorted(_RUNNERS))
        print(
            "Usage: python -m experiments.misc.inference_hub_main_accelerated "
            f"{{{choices}}} [runner arguments]",
            file=sys.stderr,
        )
        return 2
    return _RUNNERS[arguments[0]](arguments[1:])


if __name__ == "__main__":
    raise SystemExit(cli())
