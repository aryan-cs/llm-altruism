"""Chicken (Hawk-Dove) game implementation."""

from typing import Optional

from src.agents.prompts import load_prompt, render_prompt_template

from .base import Game


class Chicken(Game):
    """The Chicken or Hawk-Dove game.

    Two players drive toward each other. Each must choose to swerve or go straight.
    - If both swerve: each gets 3 points (both are cautious, both safe)
    - If one swerves and one goes straight: swerves gets 1, straight gets 5 (winner takes prestige)
    - If both go straight: each gets 0 points (crash, disaster)

    Unlike Prisoner's Dilemma, mutual defection (both straight) is the worst outcome.
    The game has two pure Nash equilibria (asymmetric) and one mixed Nash equilibrium.
    """

    name = "chicken"
    players = 2
    actions = ["swerve", "straight"]

    payoff_matrix = {
        ("swerve", "swerve"): (3, 3),
        ("swerve", "straight"): (1, 5),
        ("straight", "swerve"): (5, 1),
        ("straight", "straight"): (0, 0),
    }

    def get_description(self) -> str:
        """Return a natural language description of Chicken."""
        return load_prompt("games/chicken/description.txt")

    def format_prompt(
        self,
        player_id: str,
        round_num: int,
        total_rounds: Optional[int],
        history: list[dict],
        payoff_visible: bool,
        action_order_seed: Optional[int] = None,
    ) -> str:
        """Build the prompt for this round of Chicken."""
        return render_prompt_template(
            "games/chicken/round.txt",
            description_block=f"{self.get_description()}\n\n" if round_num == 1 else "",
            round_info=self._format_round_info(round_num, total_rounds),
            history_block=f"{self._format_history(history, payoff_visible)}\n\n" if history else "",
            actions=self._format_actions(action_order_seed),
        )
