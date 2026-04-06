"""Battle of the Sexes game implementation."""

from typing import Optional

from src.agents.prompts import load_prompt, render_prompt_template

from .base import Game


class BattleOfSexes(Game):
    """The Battle of the Sexes coordination game.

    Two players (a couple) must coordinate on an activity: opera or football.
    Each prefers a different activity, but both prefer coordinating to going alone.

    - If both choose opera: one gets 3, the other gets 2
    - If both choose football: one gets 2, the other gets 3
    - If they choose differently: both get 0

    This game tests coordination under conflicting preferences. Unlike Prisoner's
    Dilemma, both players want to coordinate—they just disagree on which outcome.
    There are two pure Nash equilibria (opera, football) and one mixed equilibrium.
    """

    name = "battle_of_sexes"
    players = 2
    actions = ["opera", "football"]

    payoff_matrix = {
        ("opera", "opera"): (3, 2),
        ("opera", "football"): (0, 0),
        ("football", "opera"): (0, 0),
        ("football", "football"): (2, 3),
    }

    def get_description(self) -> str:
        """Return a natural language description of Battle of the Sexes."""
        return load_prompt("games/battle_of_sexes/description.txt")

    def format_prompt(
        self,
        player_id: str,
        round_num: int,
        total_rounds: Optional[int],
        history: list[dict],
        payoff_visible: bool,
        action_order_seed: Optional[int] = None,
    ) -> str:
        """Build the prompt for this round of Battle of the Sexes."""
        return render_prompt_template(
            "games/battle_of_sexes/round.txt",
            description_block=f"{self.get_description()}\n\n" if round_num == 1 else "",
            round_info=self._format_round_info(round_num, total_rounds),
            history_block=f"{self._format_history(history, payoff_visible)}\n\n" if history else "",
            actions=self._format_actions(action_order_seed),
        )
