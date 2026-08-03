"""Probe InferenceHub routes for profiles that return visible content.

Some reasoning routes pass a structured-response transport probe while placing
the entire answer in a hidden reasoning field and returning ``content=null``.
That profile is unusable for behavioral scoring.  This versioned wrapper keeps
the original compatibility implementation intact but orders profiles without
``structured_response`` first.  The resulting artifact records the effective
profile order and can be used for separately labelled remediation runs.
"""

from __future__ import annotations

from collections.abc import Sequence

from experiments.misc import inference_hub_compatibility as compatibility
from experiments.misc.inference_hub_provider_safe import _provider_safe_client


VISIBLE_CONTENT_PROFILE_ORDER = (
    ("seed", "temperature", "top_p"),
    ("seed", "temperature"),
    ("seed", "top_p"),
    ("temperature", "top_p"),
    ("seed",),
    ("temperature",),
    ("top_p",),
    (),
)


def cli(argv: Sequence[str] | None = None) -> int:
    compatibility.EXECUTION_PROFILE_ORDER = VISIBLE_CONTENT_PROFILE_ORDER
    compatibility._client_from_environment = _provider_safe_client
    return compatibility.cli(None if argv is None else list(argv))


if __name__ == "__main__":
    raise SystemExit(cli())
