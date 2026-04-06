"""Prisoner's Dilemma game implementation."""

from typing import Optional

from src.agents.prompts import load_prompt, render_prompt_template

from .base import Game


class PrisonersDilemma(Game):
    """The classic Prisoner's Dilemma game.

    Two players simultaneously choose to cooperate or defect.
    - If both cooperate: each gets 3 points
    - If one defects and one cooperates: defector gets 5, cooperator gets 0
    - If both defect: each gets 1 point

    This captures the tension between individual incentives (always defect)
    and collective benefit (both cooperate).
    """

    name = "prisoners_dilemma"
    players = 2
    actions = ["cooperate", "defect"]

    payoff_matrix = {
        ("cooperate", "cooperate"): (3, 3),
        ("cooperate", "defect"): (0, 5),
        ("defect", "cooperate"): (5, 0),
        ("defect", "defect"): (1, 1),
    }

    def get_description(self) -> str:
        """Return a natural language description of Prisoner's Dilemma."""
        return load_prompt("games/prisoners_dilemma/description.txt")

    def format_prompt(
        self,
        player_id: str,
        round_num: int,
        total_rounds: Optional[int],
        history: list[dict],
        payoff_visible: bool,
        action_order_seed: Optional[int] = None,
    ) -> str:
        """Build the prompt for this round of Prisoner's Dilemma."""
        return render_prompt_template(
            "games/prisoners_dilemma/round.txt",
            description_block=f"{self.get_description()}\n\n" if round_num == 1 else "",
            round_info=self._format_round_info(round_num, total_rounds),
            history_block=f"{self._format_history(history, payoff_visible)}\n\n" if history else "",
            actions=self._format_actions(action_order_seed),
        )
