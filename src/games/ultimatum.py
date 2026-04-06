"""Ultimatum Game implementation."""

from typing import Optional

from src.agents.prompts import load_prompt, render_prompt_template

from .base import Game


class UltimatumGame(Game):
    """The Ultimatum Game.

    One player (proposer) has 10 units to split with another player (responder).
    The proposer offers X units to the responder (keeping 10-X).
    The responder can either accept (both get their amounts) or reject (both get 0).

    This game tests fairness norms and the responder's willingness to punish
    unfair offers, even at personal cost.

    Since the proposer moves first and the responder moves second, we track roles
    and handle moves sequentially within a single round.
    """

    name = "ultimatum"
    players = 2
    actions = ["offer", "accept", "reject"]  # Meta-actions; actual moves are context-specific

    payoff_matrix = None  # Handled specially in compute_payoffs

    def get_description(self) -> str:
        """Return a natural language description of Ultimatum Game."""
        return load_prompt("games/ultimatum/description.txt")

    def format_prompt(
        self,
        player_id: str,
        round_num: int,
        total_rounds: Optional[int],
        history: list[dict],
        payoff_visible: bool,
        action_order_seed: Optional[int] = None,
    ) -> str:
        """Build the prompt for Ultimatum Game.

        Args:
            player_id: Either 'proposer' or 'responder'
            round_num: Current round number
            total_rounds: Total rounds
            history: Previous round results (if any)
            payoff_visible: Whether to show payoff details
            action_order_seed: Not used for this asymmetric game

        Returns:
            Formatted prompt based on role
        """
        template = "games/ultimatum/proposer_round.txt"
        if player_id.lower() == "proposer":
            template = "games/ultimatum/proposer_round.txt"
        else:
            template = "games/ultimatum/responder_round.txt"

        return render_prompt_template(
            template,
            description_block=f"{self.get_description()}\n\n" if round_num == 1 else "",
            round_info=self._format_round_info(round_num, total_rounds),
            history_block=f"{self._format_history(history, payoff_visible)}\n\n" if history else "",
        )

    def parse_action(self, response: str, player_role: str = "proposer") -> Optional[str]:
        """Parse action, with awareness of player role.

        Args:
            response: LLM response text
            player_role: Either 'proposer' or 'responder'

        Returns:
            The parsed action (integer string for proposer, 'accept'/'reject' for responder)
        """
        response = response.strip()

        # Try JSON parsing first
        import json

        try:
            data = json.loads(response)
            if isinstance(data, dict) and "action" in data:
                action = str(data["action"]).strip()

                if player_role.lower() == "proposer":
                    # Proposer must offer 0-10
                    try:
                        num = int(action)
                        if 0 <= num <= 10:
                            return action
                    except ValueError:
                        pass
                else:
                    # Responder must accept or reject
                    if action.lower() in ["accept", "reject"]:
                        return action.lower()
        except json.JSONDecodeError:
            pass

        # Fallback: regex matching
        if player_role.lower() == "proposer":
            # Look for any number 0-10
            import re

            match = re.search(r"\b(\d+)\b", response)
            if match:
                num = int(match.group(1))
                if 0 <= num <= 10:
                    return str(num)
        else:
            # Look for accept/reject
            import re

            if re.search(r"\baccept\b", response, re.IGNORECASE):
                return "accept"
            if re.search(r"\breject\b", response, re.IGNORECASE):
                return "reject"

        return None

    def compute_payoffs(
        self,
        proposer_action: str,
        responder_action: str,
    ) -> tuple[float, float]:
        """Compute payoffs for Ultimatum Game.

        Args:
            proposer_action: Integer string (0-10) indicating units offered
            responder_action: 'accept' or 'reject'

        Returns:
            Tuple (proposer_payoff, responder_payoff)
        """
        try:
            offered = int(proposer_action)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid proposer action: {proposer_action}")

        if not (0 <= offered <= 10):
            raise ValueError(f"Offer must be 0-10, got {offered}")

        if responder_action.lower() not in ["accept", "reject"]:
            raise ValueError(f"Responder action must be 'accept' or 'reject', got {responder_action}")

        if responder_action.lower() == "accept":
            # Proposer keeps 10 - offered, responder gets offered
            return (10 - offered, offered)
        else:  # reject
            # Both get 0
            return (0, 0)
