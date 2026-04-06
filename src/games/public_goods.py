"""Public Goods Game implementation."""

from typing import Optional

from src.agents.prompts import load_prompt, render_prompt_template

from .base import Game


class PublicGoodsGame(Game):
    """The Public Goods Game (2-player variant).

    Each player has an endowment of 10 units. Both simultaneously decide how much
    to contribute to a shared pool. The pool is multiplied by 1.5, then split
    equally between the two players.

    Individual payoff = (10 - contribution) + (0.75 * total_contributions)

    This game captures the tension between self-interest (free-riding) and
    collective efficiency (mutual contribution).

    Actions are integers 0-10 representing units contributed to the pool.
    """

    name = "public_goods"
    players = 2
    actions = [str(i) for i in range(11)]  # "0", "1", ..., "10"

    payoff_matrix = None  # Payoffs depend on both actions; computed specially

    def get_description(self) -> str:
        """Return a natural language description of Public Goods Game."""
        return load_prompt(self.prompt_path("description", "games/public_goods/description.txt"))

    def format_prompt(
        self,
        player_id: str,
        round_num: int,
        total_rounds: Optional[int],
        history: list[dict],
        payoff_visible: bool,
        action_order_seed: Optional[int] = None,
    ) -> str:
        """Build the prompt for this round of Public Goods Game."""
        return render_prompt_template(
            self.prompt_path("round", "games/public_goods/round.txt"),
            description_block=f"{self.get_description()}\n\n" if round_num == 1 else "",
            round_info=self._format_round_info(round_num, total_rounds),
            history_block=f"{self._format_history_block(history, payoff_visible)}\n\n" if history else "",
            actions=self._format_actions(action_order_seed),
        )

    def _format_history_block(self, history: list[dict], payoff_visible: bool) -> str:
        """Format public goods history for the next prompt."""
        lines = ["Previous rounds:"]
        for round_data in history:
            action_a = int(round_data["action_a"])
            action_b = int(round_data["action_b"])
            payoff_line = ""
            if payoff_visible:
                payoff_line = render_prompt_template(
                    self.prompt_path("history_payoff", "games/public_goods/history_payoff.txt"),
                    payoff_a=round_data["payoff_a"],
                    payoff_b=round_data["payoff_b"],
                )

            lines.append(
                render_prompt_template(
                    self.prompt_path("history_entry", "games/public_goods/history_entry.txt"),
                    round_num=round_data["round"],
                    action_a=action_a,
                    action_b=action_b,
                    pool_total=action_a + action_b,
                    multiplied_pool=f"{1.5 * (action_a + action_b):.1f}",
                    payoff_block=f"\n{payoff_line}" if payoff_line else "",
                )
            )
        return "\n".join(lines)

    def parse_action(self, response: str) -> Optional[str]:
        """Parse the contribution amount from response.

        Args:
            response: LLM response text

        Returns:
            Integer string (0-10) representing contribution, or None if unparseable
        """
        response = response.strip()

        # Try JSON parsing first
        import json

        try:
            data = json.loads(response)
            if isinstance(data, dict) and "action" in data:
                action = str(data["action"]).strip()
                try:
                    num = int(action)
                    if 0 <= num <= 10:
                        return action
                except ValueError:
                    pass
        except json.JSONDecodeError:
            pass

        # Fallback: regex for any number 0-10
        import re

        match = re.search(r"\b(\d+)\b", response)
        if match:
            num = int(match.group(1))
            if 0 <= num <= 10:
                return str(num)

        return None

    def compute_payoffs(self, action_a: str, action_b: str) -> tuple[float, float]:
        """Compute payoffs for Public Goods Game.

        Args:
            action_a: Contribution by player A (0-10)
            action_b: Contribution by player B (0-10)

        Returns:
            Tuple (payoff_a, payoff_b)
        """
        try:
            contrib_a = int(action_a)
            contrib_b = int(action_b)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid actions: {action_a}, {action_b}")

        if not (0 <= contrib_a <= 10 and 0 <= contrib_b <= 10):
            raise ValueError(f"Contributions must be 0-10, got {contrib_a}, {contrib_b}")

        # Total pool is multiplied by 1.5
        pool_total = contrib_a + contrib_b
        multiplied_pool = 1.5 * pool_total
        per_player_share = multiplied_pool / 2.0

        # Each player gets: their remaining endowment + their share of the pool
        payoff_a = (10 - contrib_a) + per_player_share
        payoff_b = (10 - contrib_b) + per_player_share

        return (payoff_a, payoff_b)
