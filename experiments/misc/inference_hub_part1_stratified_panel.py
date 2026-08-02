"""Run a balanced deadline subset of the hosted Part 1 panel.

The base panel's generic ``--limit`` option is a prefix of its 384-root
schedule.  Because that schedule is grouped by game/domain, a 96-root prefix
contains only three of twelve strata.  This entrypoint replaces only the
subset-selection function: it samples an equal, deterministic number of roots
from every game-by-domain stratum and otherwise delegates to the hardened base
runner unchanged.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.misc import inference_hub_part1_panel as base


STRATUM_COUNT = 12
_BASE_BUILD_DRAFT_TRIALS = base.build_draft_trials


def build_stratified_trials(
    *, base_seed: int, limit: int | None = None
) -> tuple[Any, ...]:
    full = tuple(_BASE_BUILD_DRAFT_TRIALS(base_seed=base_seed, limit=None))
    if limit is None:
        return full
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not STRATUM_COUNT <= limit <= len(full)
        or limit % STRATUM_COUNT
    ):
        raise base.InferenceHubPart1PanelError(
            "A stratified trial limit must be a multiple of 12 from 12 through 384."
        )

    grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for trial in full:
        grouped[(trial.game, trial.domain)].append(trial)
    if len(grouped) != STRATUM_COUNT or any(len(rows) != 32 for rows in grouped.values()):
        raise base.InferenceHubPart1PanelError(
            "The frozen full schedule is not 12 strata by 32 roots."
        )

    per_stratum = limit // STRATUM_COUNT
    chosen: list[Any] = []
    for stratum, rows in sorted(grouped.items()):
        ranked = sorted(
            rows,
            key=lambda trial: hashlib.sha256(
                f"part1-stratified-v1\0{base_seed}\0{stratum[0]}\0{stratum[1]}\0{trial.root_id}".encode(
                    "utf-8"
                )
            ).hexdigest(),
        )
        chosen.extend(ranked[:per_stratum])
    chosen.sort(
        key=lambda trial: hashlib.sha256(
            f"part1-stratified-schedule-v1\0{base_seed}\0{trial.root_id}".encode(
                "utf-8"
            )
        ).hexdigest()
    )
    return tuple(chosen)


def cli(argv: list[str] | None = None) -> int:
    source = Path(__file__).resolve()
    if source not in base._SOURCE_PATHS:
        base._SOURCE_PATHS = (*base._SOURCE_PATHS, source)
    base.build_draft_trials = build_stratified_trials
    return base.cli(argv)


if __name__ == "__main__":
    raise SystemExit(cli())
