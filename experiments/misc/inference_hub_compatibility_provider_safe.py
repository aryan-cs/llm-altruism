"""Retry route compatibility through the shared provider-safe limiter.

Each invocation should use a fresh attempt ledger and output path.  That makes
periodic availability retries new evidence instead of silently replaying a
previous terminal failure, while the shared client still guarantees at most
one in-flight request for each upstream provider across concurrent campaigns.
"""

from __future__ import annotations

from collections.abc import Sequence

from experiments.misc import inference_hub_compatibility as compatibility
from experiments.misc.inference_hub_provider_safe import _provider_safe_client


def cli(argv: Sequence[str] | None = None) -> int:
    compatibility._client_from_environment = _provider_safe_client
    return compatibility.cli(None if argv is None else list(argv))


if __name__ == "__main__":
    raise SystemExit(cli())
