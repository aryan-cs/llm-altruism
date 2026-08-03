"""Provider-safe panel launcher with provider-round-robin scheduling.

The base runners interleave trials across subjects, but their frozen subject
order can place many routes from one upstream provider at the front of a
bounded worker pool.  Those tasks then wait on the same provider lease while
other providers remain idle.  This versioned launcher preserves every request
and evidence contract while stably round-robining selected subjects by upstream
provider before work is submitted.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.misc.inference_hub_provider_safe import (
    _provider_safe_client,
)


def provider_round_robin(
    subjects: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return a stable round robin over first-seen upstream providers."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    provider_order: list[str] = []
    for subject in subjects:
        provider = str(subject.get("upstream_provider", ""))
        if not provider:
            raise ValueError("Selected subject lacks an upstream provider.")
        if provider not in groups:
            provider_order.append(provider)
        groups[provider].append(dict(subject))
    ordered: list[dict[str, Any]] = []
    offset = 0
    while len(ordered) < len(subjects):
        for provider in provider_order:
            if offset < len(groups[provider]):
                ordered.append(groups[provider][offset])
        offset += 1
    return ordered


def _scheduled_selector(original: Callable[..., Any]) -> Callable[..., Any]:
    def select_routes(*args: Any, **kwargs: Any) -> Any:
        subjects, judge = original(*args, **kwargs)
        return provider_round_robin(subjects), judge

    return select_routes


def _bind_sources(module: object) -> None:
    source_paths = getattr(module, "_SOURCE_PATHS")
    for source in (
        Path(__file__).with_name("inference_hub_provider_safe.py").resolve(),
        Path(__file__).resolve(),
    ):
        if source not in source_paths:
            source_paths = (*source_paths, source)
    setattr(module, "_SOURCE_PATHS", source_paths)


def _part0(arguments: Sequence[str]) -> int:
    from experiments.misc import inference_hub_part0_panel as runner

    _bind_sources(runner)
    runner.select_routes = _scheduled_selector(runner.select_routes)
    runner._client_from_environment = _provider_safe_client
    return runner.cli(list(arguments))


def _part1(arguments: Sequence[str]) -> int:
    from experiments.misc import inference_hub_part1_panel as base
    from experiments.misc import inference_hub_part1_stratified_panel as runner

    _bind_sources(base)
    base.select_routes = _scheduled_selector(base.select_routes)
    base._client_from_environment = _provider_safe_client
    return runner.cli(list(arguments))


def _part2(arguments: Sequence[str]) -> int:
    from experiments.misc import inference_hub_part2_panel as runner

    _bind_sources(runner)
    runner.select_routes = _scheduled_selector(runner.select_routes)
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
            f"Usage: python -m experiments.misc.inference_hub_provider_safe_v2 "
            f"{{{choices}}} [runner arguments]",
            file=sys.stderr,
        )
        return 2
    return _RUNNERS[arguments[0]](arguments[1:])


if __name__ == "__main__":
    raise SystemExit(cli())
