from collections import Counter

import pytest

from experiments.misc import inference_hub_part1_panel as base
from experiments.misc.inference_hub_part1_stratified_panel import (
    build_stratified_trials,
)


def test_n96_covers_all_twelve_strata_equally() -> None:
    trials = build_stratified_trials(base_seed=20_260_802, limit=96)
    counts = Counter((trial.game, trial.domain) for trial in trials)

    assert len(trials) == len({trial.root_id for trial in trials}) == 96
    assert len(counts) == 12
    assert set(counts.values()) == {8}


def test_stratified_selection_is_deterministic_and_outcome_blind() -> None:
    first = build_stratified_trials(base_seed=20_260_802, limit=96)
    second = build_stratified_trials(base_seed=20_260_802, limit=96)
    changed = build_stratified_trials(base_seed=20_260_803, limit=96)

    assert [trial.root_id for trial in first] == [trial.root_id for trial in second]
    assert [trial.root_id for trial in first] != [trial.root_id for trial in changed]


@pytest.mark.parametrize("limit", [1, 95, 97, 385])
def test_stratified_limit_rejects_unbalanced_sizes(limit: int) -> None:
    with pytest.raises(base.InferenceHubPart1PanelError):
        build_stratified_trials(base_seed=20_260_802, limit=limit)
