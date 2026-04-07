"""Metrics for pairwise games and multi-agent society simulations."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any


COOPERATIVE_ACTIONS_BY_GAME: dict[str, set[str]] = {
    "prisoners_dilemma": {"cooperate"},
    "chicken": {"swerve"},
    "stag_hunt": {"stag"},
    "battle_of_sexes": {"opera", "football"},
}

DEFECTIVE_ACTIONS_BY_GAME: dict[str, set[str]] = {
    "prisoners_dilemma": {"defect"},
    "chicken": {"straight"},
    "stag_hunt": {"hare"},
}


def safe_mean(values: list[float]) -> float:
    """Return the mean of a list, defaulting to 0.0 for empty input."""
    return mean(values) if values else 0.0


def cooperation_rate(actions: list[str], cooperative_actions: set[str]) -> float:
    """Return the share of actions that count as cooperative."""
    if not actions:
        return 0.0
    cooperative = sum(1 for action in actions if action in cooperative_actions)
    return cooperative / len(actions)


def defection_rate(actions: list[str], defective_actions: set[str]) -> float:
    """Return the share of actions that count as defection."""
    if not actions:
        return 0.0
    defects = sum(1 for action in actions if action in defective_actions)
    return defects / len(actions)


def reciprocity_index(actions: list[str], opponent_actions: list[str]) -> float:
    """Measure how often an agent mirrors the opponent's previous move."""
    comparable_rounds = min(len(actions), len(opponent_actions)) - 1
    if comparable_rounds <= 0:
        return 0.0

    mirrored = 0
    for idx in range(1, comparable_rounds + 1):
        if actions[idx] == opponent_actions[idx - 1]:
            mirrored += 1

    return mirrored / comparable_rounds


def forgiveness_index(
    actions: list[str],
    opponent_actions: list[str],
    cooperative_actions: set[str],
    defective_actions: set[str],
) -> float:
    """Measure how often the agent returns to cooperation after being defected on."""
    opportunities = 0
    forgiven = 0

    for idx in range(1, min(len(actions), len(opponent_actions))):
        if opponent_actions[idx - 1] in defective_actions:
            opportunities += 1
            if actions[idx] in cooperative_actions:
                forgiven += 1

    return forgiven / opportunities if opportunities else 0.0


def exploitation_index(
    actions: list[str],
    opponent_actions: list[str],
    cooperative_actions: set[str],
    defective_actions: set[str],
) -> float:
    """Measure how often the agent defects immediately after opponent cooperation."""
    opportunities = 0
    exploitations = 0

    for idx in range(1, min(len(actions), len(opponent_actions))):
        if opponent_actions[idx - 1] in cooperative_actions:
            opportunities += 1
            if actions[idx] in defective_actions:
                exploitations += 1

    return exploitations / opportunities if opportunities else 0.0


def gini_coefficient(values: list[float]) -> float:
    """Compute the Gini coefficient of a non-negative distribution."""
    cleaned = [max(0.0, float(value)) for value in values]
    if not cleaned:
        return 0.0
    if sum(cleaned) == 0:
        return 0.0

    sorted_values = sorted(cleaned)
    n = len(sorted_values)
    cumulative_weighted = sum((idx + 1) * value for idx, value in enumerate(sorted_values))
    total = sum(sorted_values)
    return (2 * cumulative_weighted) / (n * total) - (n + 1) / n


def resource_velocity(transfer_amounts: list[float]) -> float:
    """Average amount of resource moved per timestep."""
    return safe_mean([float(value) for value in transfer_amounts])


def commons_health(resource_levels: list[float], max_resource: float) -> float:
    """Average normalized health of the public resource pool."""
    if max_resource <= 0:
        return 0.0
    normalized = [max(0.0, min(level / max_resource, 1.0)) for level in resource_levels]
    return safe_mean(normalized)


def summarize_pairwise_trial(
    game_name: str,
    rounds: list[dict[str, Any]],
) -> dict[str, float]:
    """Summarize a classical two-player trial into standard metrics."""
    cooperative_actions = COOPERATIVE_ACTIONS_BY_GAME.get(game_name, set())
    defective_actions = DEFECTIVE_ACTIONS_BY_GAME.get(game_name, set())

    actions_a = [str(round_data.get("action_a", "")) for round_data in rounds if "action_a" in round_data]
    actions_b = [str(round_data.get("action_b", "")) for round_data in rounds if "action_b" in round_data]
    payoffs_a = [float(round_data.get("payoff_a", 0.0)) for round_data in rounds]
    payoffs_b = [float(round_data.get("payoff_b", 0.0)) for round_data in rounds]

    summary: dict[str, float] = {
        "total_payoff_a": sum(payoffs_a),
        "total_payoff_b": sum(payoffs_b),
        "average_payoff_a": safe_mean(payoffs_a),
        "average_payoff_b": safe_mean(payoffs_b),
    }

    if cooperative_actions:
        summary.update(
            {
                "cooperation_rate_a": cooperation_rate(actions_a, cooperative_actions),
                "cooperation_rate_b": cooperation_rate(actions_b, cooperative_actions),
                "reciprocity_index_a": reciprocity_index(actions_a, actions_b),
                "reciprocity_index_b": reciprocity_index(actions_b, actions_a),
            }
        )

    if cooperative_actions and defective_actions:
        summary.update(
            {
                "defection_rate_a": defection_rate(actions_a, defective_actions),
                "defection_rate_b": defection_rate(actions_b, defective_actions),
                "forgiveness_index_a": forgiveness_index(
                    actions_a, actions_b, cooperative_actions, defective_actions
                ),
                "forgiveness_index_b": forgiveness_index(
                    actions_b, actions_a, cooperative_actions, defective_actions
                ),
                "exploitation_index_a": exploitation_index(
                    actions_a, actions_b, cooperative_actions, defective_actions
                ),
                "exploitation_index_b": exploitation_index(
                    actions_b, actions_a, cooperative_actions, defective_actions
                ),
            }
        )

    return summary


def summarize_numeric_trial(rounds: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize allocation-style games such as public goods or dictator."""
    numeric_a: list[float] = []
    numeric_b: list[float] = []
    payoffs_a: list[float] = []
    payoffs_b: list[float] = []

    for round_data in rounds:
        if "action_a" in round_data:
            try:
                numeric_a.append(float(round_data["action_a"]))
            except (TypeError, ValueError):
                pass
        if "action_b" in round_data:
            try:
                numeric_b.append(float(round_data["action_b"]))
            except (TypeError, ValueError):
                pass
        payoffs_a.append(float(round_data.get("payoff_a", 0.0)))
        payoffs_b.append(float(round_data.get("payoff_b", 0.0)))

    summary = {
        "total_payoff_a": sum(payoffs_a),
        "total_payoff_b": sum(payoffs_b),
        "average_payoff_a": safe_mean(payoffs_a),
        "average_payoff_b": safe_mean(payoffs_b),
    }

    if numeric_a:
        summary["average_action_a"] = safe_mean(numeric_a)
    if numeric_b:
        summary["average_action_b"] = safe_mean(numeric_b)

    if numeric_a and numeric_b:
        summary["average_fairness_gap"] = safe_mean(
            [abs(a - b) for a, b in zip(numeric_a, numeric_b, strict=False)]
        )

    return summary


def summarize_society(
    rounds: list[dict[str, Any]],
    max_public_resources: float,
) -> dict[str, float]:
    """Summarize a society simulation over time."""
    if not rounds:
        return {
            "round_count": 0,
            "average_gini": 0.0,
            "average_trade_volume": 0.0,
            "commons_health": 0.0,
            "survival_rate": 0.0,
            "final_survival_rate": 0.0,
            "final_alive_count": 0.0,
            "final_total_agents": 0.0,
            "extinction_event": 0.0,
            "alliance_count": 0.0,
        }

    gini_values = []
    public_resources = []
    trade_volumes = []
    survival_rates = []
    alliance_round_counts: defaultdict[tuple[str, str], int] = defaultdict(int)

    for round_data in rounds:
        resources = list(round_data.get("agent_resources", {}).values())
        gini_values.append(gini_coefficient(resources))
        public_resources.append(float(round_data.get("public_resources", 0.0)))
        trade_volumes.append(float(round_data.get("trade_volume", 0.0)))
        alive_count = int(round_data.get("alive_count", 0))
        total_agents = int(round_data.get("total_agents", alive_count or 1))
        survival_rates.append(alive_count / total_agents if total_agents else 0.0)

        for event in round_data.get("events", []):
            if event.get("kind") in {"trade_completed", "share"}:
                actor = str(event.get("actor", ""))
                target = str(event.get("target", ""))
                if actor and target:
                    pair = tuple(sorted((actor, target)))
                    alliance_round_counts[pair] += 1

    alliance_count = sum(1 for count in alliance_round_counts.values() if count >= 2)
    final_round = rounds[-1]
    final_alive_count = float(final_round.get("alive_count", 0))
    final_total_agents = float(final_round.get("total_agents", final_alive_count or 1))
    final_survival_rate = final_alive_count / final_total_agents if final_total_agents else 0.0
    extinction_event = 1.0 if final_alive_count <= 0 else 0.0

    return {
        "round_count": float(len(rounds)),
        "average_gini": safe_mean(gini_values),
        "average_trade_volume": resource_velocity(trade_volumes),
        "commons_health": commons_health(public_resources, max_public_resources),
        "survival_rate": safe_mean(survival_rates),
        "final_survival_rate": final_survival_rate,
        "final_alive_count": final_alive_count,
        "final_total_agents": final_total_agents,
        "extinction_event": extinction_event,
        "alliance_count": float(alliance_count),
        "commons_depletion_rate": 1.0 - commons_health(public_resources, max_public_resources),
    }


def categorical_distribution(items: list[str]) -> dict[str, float]:
    """Return a normalized distribution over categorical items."""
    if not items:
        return {}
    counts = Counter(items)
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()}
