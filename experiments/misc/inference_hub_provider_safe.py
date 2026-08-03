"""Launch hosted panels with one shared in-flight request per provider.

This versioned wrapper leaves the source-bound historical runners unchanged.
Fresh Part 0, Part 1, and Part 2 campaigns launched here share a file-backed
limiter scoped to the endpoint and credential, so distinct processes can run
concurrently without ever dispatching two requests to the same upstream
provider at once.
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


PROVIDER_SAFE_POLICY_VERSION = 1
PROVIDER_SAFE_POLICY = RateLimitPolicy(provider_concurrency=1)


def _provider_safe_client(timeout_seconds: float) -> InferenceHubClient:
    """Build a client whose limiter is shared by all provider-safe campaigns."""

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
            f"provider-safe-v{PROVIDER_SAFE_POLICY_VERSION}"
        ).encode("utf-8")
    ).hexdigest()
    limiter = InferenceHubRateLimiter(
        policy=PROVIDER_SAFE_POLICY,
        scope_id=scope_id,
    )
    return InferenceHubClient(
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        rate_limiter=limiter,
    )


def _bind_source(module: object) -> None:
    source_paths = getattr(module, "_SOURCE_PATHS")
    source = Path(__file__).resolve()
    if source not in source_paths:
        setattr(module, "_SOURCE_PATHS", (*source_paths, source))


def _part0(arguments: Sequence[str]) -> int:
    from experiments.misc import inference_hub_part0_panel as runner

    _bind_source(runner)
    runner._client_from_environment = _provider_safe_client
    return runner.cli(list(arguments))


def _part1(arguments: Sequence[str]) -> int:
    from experiments.misc import inference_hub_part1_panel as base
    from experiments.misc import inference_hub_part1_stratified_panel as runner

    _bind_source(base)
    base._client_from_environment = _provider_safe_client
    return runner.cli(list(arguments))


def _part2(arguments: Sequence[str]) -> int:
    from experiments.misc import inference_hub_part2_panel as runner

    _bind_source(runner)
    runner._client_from_environment = _provider_safe_client
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
            f"Usage: python -m experiments.misc.inference_hub_provider_safe "
            f"{{{choices}}} [runner arguments]",
            file=sys.stderr,
        )
        return 2
    return _RUNNERS[arguments[0]](arguments[1:])


if __name__ == "__main__":
    raise SystemExit(cli())
