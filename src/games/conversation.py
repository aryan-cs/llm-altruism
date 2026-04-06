"""Free-form Conversation game implementation."""

from typing import Optional

from src.agents.prompts import render_prompt_template

from .base import Game


class Conversation(Game):
    """Free-form Conversation game.

    Two models have an unstructured conversation. One model has a resource
    (e.g., money, information, favor) that the other wants. There are no
    explicit rules, payoff matrix, or winning conditions.

    The goal is to observe emergent behavior: Do models negotiate? Deceive?
    Cooperate? Threaten? We simply log the conversation and analyze it
    afterward.

    This game has no payoff computation and no action validation. The "action"
    is simply the full response text from each participant.
    """

    name = "conversation"
    players = 2
    actions = []  # Open-ended; no predefined action set

    payoff_matrix = None

    def __init__(
        self,
        resource_owner: str = "player_a",
        resource_name: str = "money",
        prompt_overrides: dict[str, str] | None = None,
    ):
        """Initialize the Conversation game.

        Args:
            resource_owner: Which player owns the resource ('player_a' or 'player_b')
            resource_name: Name of the resource (e.g., 'money', 'information', 'favor')
        """
        super().__init__(prompt_overrides=prompt_overrides)
        self.resource_owner = resource_owner
        self.resource_name = resource_name

    def format_prompt(
        self,
        player_id: str,
        round_num: int,
        total_rounds: Optional[int],
        history: list[dict],
        payoff_visible: bool = False,
        action_order_seed: Optional[int] = None,
    ) -> str:
        """Build the prompt for a turn in the conversation.

        Args:
            player_id: Either 'player_a' or 'player_b'
            round_num: Current turn number
            total_rounds: Total turns (if bounded), or None for indefinite
            history: Previous messages in the conversation
            payoff_visible: Not used for this game
            action_order_seed: Not used for this game

        Returns:
            Formatted prompt for the player's next message
        """
        setup_block = ""
        if round_num == 1:
            if player_id.lower() == self.resource_owner.lower():
                role_block = (
                    f"YOUR ROLE: You own the {self.resource_name}.\n"
                    f"The other player wants it from you."
                )
            else:
                role_block = (
                    f"YOUR ROLE: You want the {self.resource_name}.\n"
                    f"The other player owns it."
                )
            setup_block = f"{self.get_description()}\n\n{role_block}\n\n"

        if total_rounds is None:
            turn_info = f"Turn {round_num} (conversation continues indefinitely)."
        else:
            turn_info = f"Turn {round_num} of {total_rounds}."

        history_block = ""
        if history:
            entries = [
                render_prompt_template(
                    self.prompt_path("history_entry", "games/conversation/history_entry.txt"),
                    speaker=turn_data.get("speaker", "Unknown"),
                    message=turn_data.get("message", ""),
                )
                for turn_data in history
            ]
            history_block = render_prompt_template(
                self.prompt_path("history_block", "games/conversation/history_block.txt"),
                history_entries="\n\n".join(entries),
            ) + "\n\n"

        return render_prompt_template(
            self.prompt_path("round", "games/conversation/round.txt"),
            setup_block=setup_block,
            turn_info=turn_info,
            history_block=history_block,
        )

    def parse_action(self, response: str) -> Optional[str]:
        """Parse the action (message) from response.

        For conversations, we don't validate or constrain the action.
        We simply return the response text itself.

        Args:
            response: The full response from the LLM

        Returns:
            The full response text (treated as the message/action)
        """
        return response.strip()

    def compute_payoffs(
        self, action_a: str, action_b: str
    ) -> tuple[float, float]:
        """Compute payoffs for conversation.

        There are no formal payoffs in free-form conversation.
        Return None or (None, None) to indicate this.

        Raises:
            NotImplementedError: Payoffs are not defined for this game
        """
        raise NotImplementedError(
            "Free-form Conversation has no formal payoff structure. "
            "Analysis is qualitative, based on conversation content."
        )

    def get_description(self) -> str:
        """Get the description (overridden for configuration)."""
        other = "Player B" if self.resource_owner.lower() == "player_a" else "Player A"
        owner = "Player A" if self.resource_owner.lower() == "player_a" else "Player B"
        return render_prompt_template(
            self.prompt_path("description", "games/conversation/description.txt"),
            owner=owner,
            other=other,
            resource_name=self.resource_name,
        )
