"""Abstract base class for all games in the llm-altruism project."""

import json
import random
import re
from abc import ABC, abstractmethod
from typing import Any, Optional


class Game(ABC):
    """Abstract base class for game-theoretic interactions.

    This class defines the interface for all games. Subclasses implement specific
    game rules, payoff matrices, and action parsing logic.
    """

    name: str
    """Unique identifier for this game (e.g., 'prisoners_dilemma', 'chicken')."""

    players: int = 2
    """Number of players in this game (default 2 for symmetric games)."""

    actions: list[str]
    """List of available actions as strings (e.g., ['cooperate', 'defect'])."""

    payoff_matrix: dict[tuple[str, str], tuple[float, float]]
    """Mapping from (action_a, action_b) tuples to (payoff_a, payoff_b) tuples.

    For games with asymmetric roles (ultimatum, dictator), this may be None
    or have special structure. Subclasses handle role-specific payoff logic.
    """

    def __init__(
        self,
        *,
        prompt_overrides: dict[str, str] | None = None,
        action_aliases: dict[str, str] | None = None,
    ):
        """Initialize the game.

        Args:
            prompt_overrides: Optional mapping from logical prompt key to prompt path.
                Example: {"description": "games_variants/prisoners_dilemma/unnamed_description.txt"}
            action_aliases: Optional mapping from canonical action to displayed alias.
        """
        if not hasattr(self, "name"):
            raise NotImplementedError(f"{self.__class__.__name__} must define 'name'")
        if not hasattr(self, "actions"):
            raise NotImplementedError(
                f"{self.__class__.__name__} must define 'actions' list"
            )
        self.prompt_overrides = dict(prompt_overrides or {})
        self.action_aliases = dict(action_aliases or {})
        unknown_alias_keys = set(self.action_aliases) - set(self.actions)
        if unknown_alias_keys:
            unknown = ", ".join(sorted(unknown_alias_keys))
            raise ValueError(f"Unknown action alias keys for {self.name}: {unknown}")
        self.alias_to_action = {
            alias.strip().lower(): action
            for action, alias in self.action_aliases.items()
            if isinstance(alias, str) and alias.strip()
        }

    def prompt_path(self, key: str, default_path: str) -> str:
        """Return the prompt path for a logical template key."""
        return self.prompt_overrides.get(key, default_path)

    def display_action(self, action: str) -> str:
        """Return the displayed label for an action."""
        return self.action_aliases.get(action, action)

    def canonicalize_action(self, action: str) -> str | None:
        """Map either a canonical action or an alias back to the canonical action."""
        action_lower = action.strip().lower()
        for candidate in self.actions:
            if candidate.lower() == action_lower:
                return candidate
        return self.alias_to_action.get(action_lower)

    def get_description(self) -> str:
        """Return a natural language description of the game rules.

        This is shown to players when they first encounter the game.
        Should explain what actions are available, what the payoffs represent,
        and any special rules.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_description()"
        )

    def format_prompt(
        self,
        player_id: str,
        round_num: int,
        total_rounds: Optional[int],
        history: list[dict],
        payoff_visible: bool,
        action_order_seed: Optional[int] = None,
    ) -> str:
        """Build the message prompt for a player in this round.

        Args:
            player_id: Identifier for the player (usually 'A' or 'B', or role name)
            round_num: Current round number (1-indexed)
            total_rounds: Total rounds in this game, or None if indefinite
            history: List of previous rounds. Each dict has:
                - 'round': round number
                - 'action_a': action taken by player A
                - 'action_b': action taken by player B
                - 'payoff_a': payoff to player A
                - 'payoff_b': payoff to player B
            payoff_visible: Whether to show full payoff details to players
            action_order_seed: Optional seed to randomize action order (avoids positional bias)

        Returns:
            A formatted string prompt for the LLM.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement format_prompt()"
        )

    def parse_action(self, response: str) -> Optional[str]:
        """Extract the chosen action from an LLM response.

        Tries multiple parsing strategies:
        1. JSON parsing (expects {"action": "..."})
        2. Regex matching against action names (case-insensitive)
        3. For numeric actions, try to extract an integer

        Args:
            response: Raw text response from the LLM

        Returns:
            The parsed action (as a string), or None if unparseable.
        """
        response = response.strip()

        # Strategy 1: Try JSON parsing
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "action" in data:
                action = str(data["action"]).strip()
                canonical = self.canonicalize_action(action)
                if canonical is not None:
                    return canonical
        except json.JSONDecodeError:
            pass

        # Strategy 2: Regex match against action names or aliases (case-insensitive)
        for action in self.actions:
            candidates = [action, self.display_action(action)]
            pattern = r"\b(" + "|".join(re.escape(candidate) for candidate in candidates) + r")\b"
            if re.search(pattern, response, re.IGNORECASE):
                return action

        # Strategy 3: Try to extract a numeric action (for allocation games)
        if all(action.isdigit() for action in self.actions):
            match = re.search(r"\b(\d+)\b", response)
            if match:
                num_str = match.group(1)
                if num_str in self.actions:
                    return num_str

        return None

    def compute_payoffs(self, action_a: str, action_b: str) -> tuple[float, float]:
        """Compute payoffs for both players given their actions.

        For standard matrix games, looks up in payoff_matrix.
        For asymmetric games (ultimatum, dictator), overridden in subclasses.

        Args:
            action_a: Action chosen by player A
            action_b: Action chosen by player B

        Returns:
            Tuple (payoff_a, payoff_b)

        Raises:
            KeyError if the action combination is not defined
        """
        if not hasattr(self, "payoff_matrix") or self.payoff_matrix is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement compute_payoffs()"
            )
        return self.payoff_matrix[(action_a, action_b)]

    def _is_valid_action(self, action: str) -> bool:
        """Check if a string is a valid action in this game (case-insensitive)."""
        action_lower = action.lower()
        return any(a.lower() == action_lower for a in self.actions)

    def _randomize_action_order(
        self, seed: int
    ) -> list[str]:
        """Return actions in randomized order for a given seed.

        Used to avoid positional bias when asking LLMs to choose.
        The randomization is deterministic: the same seed always produces
        the same order.

        Args:
            seed: Random seed (usually derived from player_id + round_num)

        Returns:
            Actions in randomized order
        """
        rng = random.Random(seed)
        actions = self.actions.copy()
        rng.shuffle(actions)
        return actions

    def _format_history(self, history: list[dict], payoff_visible: bool) -> str:
        """Format game history in a readable way.

        Args:
            history: List of previous round results
            payoff_visible: Whether to include payoff details

        Returns:
            Formatted history string
        """
        if not history:
            return "No previous rounds yet."

        lines = ["Previous rounds:"]
        for round_data in history:
            round_num = round_data["round"]
            action_a = self.display_action(round_data["action_a"])
            action_b = self.display_action(round_data["action_b"])
            lines.append(f"  Round {round_num}: You played {action_a}, opponent played {action_b}")

            if payoff_visible:
                payoff_a = round_data["payoff_a"]
                payoff_b = round_data["payoff_b"]
                lines.append(f"    You earned {payoff_a}, opponent earned {payoff_b}")

        return "\n".join(lines)

    def _format_actions(
        self, action_order_seed: Optional[int] = None
    ) -> str:
        """Format the list of available actions.

        Args:
            action_order_seed: If provided, randomize action order for this seed

        Returns:
            Formatted string listing actions
        """
        actions = (
            self._randomize_action_order(action_order_seed)
            if action_order_seed is not None
            else self.actions
        )
        formatted = ", ".join(f'"{self.display_action(a)}"' for a in actions)
        return f"Available actions: {formatted}"

    def _format_round_info(
        self, round_num: int, total_rounds: Optional[int]
    ) -> str:
        """Format information about the current round.

        Args:
            round_num: Current round number (1-indexed)
            total_rounds: Total rounds, or None if indefinite

        Returns:
            Formatted string describing the round
        """
        if total_rounds is None:
            return f"Round {round_num} of an indefinite series (will continue with some probability each round)."
        else:
            return f"Round {round_num} of {total_rounds}."
