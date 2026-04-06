"""Dictator Game implementation."""

from typing import Optional

from src.agents.prompts import load_prompt, render_prompt_template

from .base import Game


class DictatorGame(Game):
    """The Dictator Game.

    One player (allocator) unilaterally decides how to split 10 units with
    another player (recipient). The recipient has no choice or power.

    Unlike the Ultimatum Game, there is no rejection mechanism. The allocator
    has absolute power.

    This game tests pure generosity without strategic incentive. Any allocation
    reflects the allocator's intrinsic fairness norms or altruism.

    Only the allocator makes a decision each round. The recipient role is
    passive—they receive the amount decided by the allocator.
    """

    name = "dictator"
    players = 2
    actions = ["allocate"]  # Meta-action; actual moves are integers 0-10

    payoff_matrix = None  # Handled specially

    def get_description(self) -> str:
        """Return a natural language description of Dictator Game."""
        return load_prompt("games/dictator/description.txt")

    def format_prompt(
        self,
        player_id: str,
        round_num: int,
        total_rounds: Optional[int],
        history: list[dict],
        payoff_visible: bool,
        action_order_seed: Optional[int] = None,
    ) -> str:
        """Build the prompt for Dictator Game.

        Note: Only the allocator gets a prompt. The recipient is passive.

        Args:
            player_id: Either 'allocator' or 'recipient'
            round_num: Current round number
            total_rounds: Total rounds
            history: Previous round results
            payoff_visible: Whether to show payoff details
            action_order_seed: Not used for this asymmetric game

        Returns:
            Formatted prompt for allocator; empty string for recipient
        """
        if player_id.lower() != "allocator":
            # Recipient doesn't make a decision; they just receive
            return ""

        return render_prompt_template(
            "games/dictator/allocator_round.txt",
            description_block=f"{self.get_description()}\n\n" if round_num == 1 else "",
            round_info=self._format_round_info(round_num, total_rounds),
            history_block=f"{self._format_history_block(history, payoff_visible)}\n\n" if history else "",
        )

    def _format_history_block(self, history: list[dict], payoff_visible: bool) -> str:
        """Format dictator-game history for the allocator prompt."""
        lines = ["Previous rounds:"]
        for round_data in history:
            payoff_block = ""
            if payoff_visible:
                payoff_block = render_prompt_template(
                    "games/dictator/history_payoff.txt",
                    payoff_allocator=round_data.get("payoff_allocator"),
                    payoff_recipient=round_data.get("payoff_recipient"),
                )
            lines.append(
                render_prompt_template(
                    "games/dictator/history_entry.txt",
                    round_num=round_data["round"],
                    allocator_action=round_data.get("allocator_action"),
                    payoff_block=f"\n{payoff_block}" if payoff_block else "",
                )
            )
        return "\n".join(lines)

    def parse_action(self, response: str) -> Optional[str]:
        """Parse the allocator's decision.

        Args:
            response: LLM response text

        Returns:
            Integer string (0-10) representing units to give, or None if unparseable
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

    def compute_payoffs(
        self,
        allocator_action: str,
        recipient_action: Optional[str] = None,
    ) -> tuple[float, float]:
        """Compute payoffs for Dictator Game.

        Args:
            allocator_action: Integer string (0-10) representing units given to recipient
            recipient_action: Not used (recipient has no action)

        Returns:
            Tuple (allocator_payoff, recipient_payoff)
        """
        try:
            given = int(allocator_action)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid allocator action: {allocator_action}")

        if not (0 <= given <= 10):
            raise ValueError(f"Allocation must be 0-10, got {given}")

        # Allocator keeps 10 - given, recipient gets given
        return (10 - given, given)
