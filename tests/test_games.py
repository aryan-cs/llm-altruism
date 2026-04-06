"""Tests for game implementations and registries."""

from __future__ import annotations

from src.games import GAME_REGISTRY, Conversation, DictatorGame, PublicGoodsGame, UltimatumGame, get_game


def test_all_planned_games_are_registered():
    expected = {
        "prisoners_dilemma",
        "chicken",
        "battle_of_sexes",
        "stag_hunt",
        "ultimatum",
        "dictator",
        "public_goods",
        "conversation",
    }
    assert expected.issubset(set(GAME_REGISTRY))


def test_public_goods_computes_expected_payoffs():
    game = PublicGoodsGame()
    payoff_a, payoff_b = game.compute_payoffs("10", "0")
    assert payoff_a == 7.5
    assert payoff_b == 17.5


def test_ultimatum_parses_role_specific_actions():
    game = UltimatumGame()
    assert game.parse_action('{"action": 4}', player_role="proposer") == "4"
    assert game.parse_action('{"action": "accept"}', player_role="responder") == "accept"


def test_dictator_payoffs_match_allocation():
    game = DictatorGame()
    assert game.compute_payoffs("3") == (7, 3)


def test_conversation_owner_prompt_uses_player_role():
    game = Conversation(resource_owner="player_a", resource_name="water")
    prompt = game.format_prompt("player_a", 1, 5, [])
    assert "You own the water" in prompt


def test_get_game_accepts_constructor_kwargs():
    game = get_game("conversation", resource_name="information")
    assert isinstance(game, Conversation)
    assert "information" in game.get_description()
