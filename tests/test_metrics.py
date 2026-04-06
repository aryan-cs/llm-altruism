"""Tests for metrics and strategy classification."""

from __future__ import annotations

from src.analysis import classify_strategy, gini_coefficient, summarize_pairwise_trial, summarize_society


def test_strategy_classifier_detects_tit_for_tat():
    actions = ["cooperate", "cooperate", "defect", "cooperate"]
    opponent = ["cooperate", "defect", "cooperate", "cooperate"]
    assert classify_strategy(actions, opponent, game_name="prisoners_dilemma") == "tit_for_tat"


def test_gini_coefficient_zero_for_equal_distribution():
    assert gini_coefficient([5, 5, 5, 5]) == 0.0


def test_pairwise_summary_contains_cooperation_metrics():
    rounds = [
        {"action_a": "cooperate", "action_b": "cooperate", "payoff_a": 3, "payoff_b": 3},
        {"action_a": "defect", "action_b": "cooperate", "payoff_a": 5, "payoff_b": 0},
    ]
    summary = summarize_pairwise_trial("prisoners_dilemma", rounds)
    assert summary["cooperation_rate_a"] == 0.5
    assert summary["total_payoff_a"] == 8


def test_society_summary_aggregates_core_metrics():
    rounds = [
        {
            "agent_resources": {"a": 5, "b": 5},
            "public_resources": 20,
            "trade_volume": 2,
            "alive_count": 2,
            "total_agents": 2,
            "events": [{"kind": "share", "actor": "a", "target": "b"}],
        },
        {
            "agent_resources": {"a": 4, "b": 6},
            "public_resources": 18,
            "trade_volume": 3,
            "alive_count": 2,
            "total_agents": 2,
            "events": [{"kind": "trade_completed", "actor": "a", "target": "b"}],
        },
    ]
    summary = summarize_society(rounds, max_public_resources=20)
    assert summary["average_trade_volume"] == 2.5
    assert summary["alliance_count"] == 1.0
    assert summary["final_survival_rate"] == 1.0
    assert summary["extinction_event"] == 0.0
