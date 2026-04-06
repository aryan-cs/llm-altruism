"""Heuristics for classifying agent strategies from action histories."""

from __future__ import annotations

from statistics import mean

from .metrics import COOPERATIVE_ACTIONS_BY_GAME, cooperation_rate


def _all_numeric(actions: list[str]) -> bool:
    if not actions:
        return False
    try:
        for action in actions:
            float(action)
    except (TypeError, ValueError):
        return False
    return True


def _is_tit_for_tat(actions: list[str], opponent_actions: list[str], cooperative_actions: set[str]) -> bool:
    if len(actions) < 2 or len(opponent_actions) < 2:
        return False
    if actions[0] not in cooperative_actions:
        return False
    for idx in range(1, min(len(actions), len(opponent_actions))):
        if actions[idx] != opponent_actions[idx - 1]:
            return False
    return True


def _is_grim_trigger(actions: list[str], opponent_actions: list[str], cooperative_actions: set[str]) -> bool:
    if len(actions) < 2 or len(opponent_actions) < 2:
        return False

    defected = False
    for idx, action in enumerate(actions):
        if defected and action in cooperative_actions:
            return False
        if idx > 0 and opponent_actions[idx - 1] not in cooperative_actions:
            defected = True
    return defected


def _is_pavlov(actions: list[str], opponent_actions: list[str]) -> bool:
    if len(actions) < 3 or len(opponent_actions) < 3:
        return False
    matches = 0
    total = 0

    for idx in range(1, min(len(actions), len(opponent_actions))):
        previous_outcome_equal = actions[idx - 1] == opponent_actions[idx - 1]
        stayed = actions[idx] == actions[idx - 1]
        total += 1
        if previous_outcome_equal == stayed:
            matches += 1

    return total > 0 and matches / total >= 0.8


def classify_strategy(
    actions: list[str],
    opponent_actions: list[str] | None = None,
    game_name: str | None = None,
) -> str:
    """Classify an action history into a coarse-grained strategy label."""
    if not actions:
        return "unknown"

    unique_actions = {str(action) for action in actions}
    if unique_actions == {"passive"}:
        return "passive"

    if game_name == "conversation":
        return "free_form"

    if _all_numeric(actions):
        numeric_actions = [float(action) for action in actions]
        normalized = mean(numeric_actions) / 10.0 if numeric_actions else 0.0
        if normalized >= 0.75:
            return "high_contributor"
        if normalized >= 0.45:
            return "moderate_contributor"
        if normalized <= 0.15:
            return "low_contributor"
        return "mixed_numeric"

    if unique_actions <= {"accept", "reject"}:
        if unique_actions == {"accept"}:
            return "always_accept"
        if unique_actions == {"reject"}:
            return "always_reject"
        return "conditional_acceptor"

    if len(unique_actions) == 1:
        return f"always_{next(iter(unique_actions))}"

    cooperative_actions = COOPERATIVE_ACTIONS_BY_GAME.get(game_name or "", set())
    if not cooperative_actions:
        return "mixed"

    coop_rate = cooperation_rate(actions, cooperative_actions)
    if coop_rate == 1.0:
        return "always_cooperate"
    if coop_rate == 0.0:
        return "always_defect"

    if opponent_actions:
        if _is_tit_for_tat(actions, opponent_actions, cooperative_actions):
            return "tit_for_tat"
        if _is_grim_trigger(actions, opponent_actions, cooperative_actions):
            return "grim_trigger"
        if _is_pavlov(actions, opponent_actions):
            return "pavlov"

    if coop_rate >= 0.75:
        return "mostly_cooperate"
    if coop_rate <= 0.25:
        return "mostly_defect"
    return "mixed"
