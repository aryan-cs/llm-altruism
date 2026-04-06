"""Stag Hunt game implementation."""

from typing import Optional

from src.agents.prompts import load_prompt, render_prompt_template

from .base import Game


class StagHunt(Game):
    """The Stag Hunt coordination game.

    Two hunters can choose to hunt a stag (large, risky, requires cooperation)
    or hunt a hare (small, safe, can be done alone).

    - If both hunt stag: each gets 4 points (high payoff from cooperation)
    - If one hunts stag and one hunts hare: stag-hunter gets 0 (wasted effort), hare-hunter gets 3
    - If both hunt hare: each gets 2 points (guaranteed but low payoff)

    This game tests trust and risk preference. The stag outcome is payoff-dominant
    (best collective outcome) but risk-dominated (hare is the safe choice).
    It captures the tension between cooperation and safety.
    """

    name = "stag_hunt"
    players = 2
    actions = ["stag", "hare"]

    payoff_matrix = {
        ("stag", "stag"): (4, 4),
        ("stag", "hare"): (0, 3),
        ("hare", "stag"): (3, 0),
        ("hare", "hare"): (2, 2),
    }

    def get_description(self) -> str:
        """Return a natural language description of Stag Hunt."""
        return load_prompt("games/stag_hunt/description.txt")

    def format_prompt(
        self,
        player_id: str,
        round_num: int,
        total_rounds: Optional[int],
        history: list[dict],
        payoff_visible: bool,
        action_order_seed: Optional[int] = None,
    ) -> str:
        """Build the prompt for this round of Stag Hunt."""
        return render_prompt_template(
            "games/stag_hunt/round.txt",
            description_block=f"{self.get_description()}\n\n" if round_num == 1 else "",
            round_info=self._format_round_info(round_num, total_rounds),
            history_block=f"{self._format_history(history, payoff_visible)}\n\n" if history else "",
            actions=self._format_actions(action_order_seed),
        )
