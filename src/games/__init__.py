"""Game logic layer for llm-altruism.

This module exports all available games and provides a registry for easy access.
Games implement the Game abstract base class and can be used to generate prompts,
parse actions, and compute payoffs.
"""

from .base import Game
from .battle_of_sexes import BattleOfSexes
from .chicken import Chicken
from .conversation import Conversation
from .dictator import DictatorGame
from .prisoners_dilemma import PrisonersDilemma
from .public_goods import PublicGoodsGame
from .stag_hunt import StagHunt
from .ultimatum import UltimatumGame

__all__ = [
    "Game",
    "PrisonersDilemma",
    "Chicken",
    "BattleOfSexes",
    "StagHunt",
    "UltimatumGame",
    "DictatorGame",
    "PublicGoodsGame",
    "Conversation",
    "GAME_REGISTRY",
]

# Registry mapping game names to their class constructors
GAME_REGISTRY: dict[str, type[Game]] = {
    "prisoners_dilemma": PrisonersDilemma,
    "chicken": Chicken,
    "battle_of_sexes": BattleOfSexes,
    "stag_hunt": StagHunt,
    "ultimatum": UltimatumGame,
    "dictator": DictatorGame,
    "public_goods": PublicGoodsGame,
    "conversation": Conversation,
}


def get_game(game_name: str, **kwargs) -> Game:
    """Instantiate a game by name.

    Args:
        game_name: Name of the game (must be in GAME_REGISTRY)
        **kwargs: Additional arguments to pass to the game constructor
                  (e.g., resource_name for Conversation game)

    Returns:
        An instance of the requested game

    Raises:
        ValueError: If game_name is not in the registry
    """
    if game_name not in GAME_REGISTRY:
        available = ", ".join(sorted(GAME_REGISTRY.keys()))
        raise ValueError(
            f"Unknown game: {game_name}. Available games: {available}"
        )

    game_class = GAME_REGISTRY[game_name]
    return game_class(**kwargs)
