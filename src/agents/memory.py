"""
Memory management for agents in repeated games and society simulations.

Supports different memory modes (NONE, FULL, WINDOWED, SUMMARIZED) to control
how agents retain and recall their history of past interactions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MemoryMode(Enum):
    """Enum for different memory retention strategies."""

    NONE = "none"  # Agent has no memory of past rounds
    FULL = "full"  # Agent retains all history
    WINDOWED = "windowed"  # Agent retains last N rounds
    SUMMARIZED = "summarized"  # Agent gets a summary of history


@dataclass
class MemoryEntry:
    """A single entry in an agent's memory (one round of play)."""

    round_num: int
    action: str
    payoff: float
    opponent_action: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "round_num": self.round_num,
            "action": self.action,
            "payoff": self.payoff,
            "opponent_action": self.opponent_action,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class Memory:
    """Memory system for agents with configurable retention strategy."""

    def __init__(
        self,
        mode: MemoryMode = MemoryMode.FULL,
        window_size: int = 10,
    ):
        """
        Initialize memory with specified mode.

        Args:
            mode: The memory retention strategy (MemoryMode enum)
            window_size: For WINDOWED mode, how many recent rounds to remember
        """
        self.mode = mode
        self.window_size = window_size
        self.entries: list[MemoryEntry] = []
        self._summary_cache: Optional[str] = None

    def add(self, entry: MemoryEntry) -> None:
        """
        Add a new memory entry (result of a round).

        Args:
            entry: The MemoryEntry to add
        """
        self.entries.append(entry)
        # Invalidate summary cache when new entry is added
        self._summary_cache = None

    def get_history(self) -> list[MemoryEntry]:
        """
        Get the history based on the current memory mode.

        Returns:
            List of MemoryEntry objects based on retention strategy
        """
        if self.mode == MemoryMode.NONE:
            return []

        if self.mode == MemoryMode.FULL:
            return self.entries.copy()

        if self.mode == MemoryMode.WINDOWED:
            # Return last N entries
            return self.entries[-self.window_size :].copy()

        if self.mode == MemoryMode.SUMMARIZED:
            # For summarized mode, we don't return raw entries
            # Instead, use get_summary() to get the text summary
            return []

        return []

    def get_summary(self, include_payoffs: bool = True) -> str:
        """
        Get a text summary of the memory.

        For NONE mode: returns empty string
        For FULL/WINDOWED/SUMMARIZED: returns a textual summary

        Returns:
            A string summary of the agent's history
        """
        if self.mode == MemoryMode.NONE:
            return ""

        if self._summary_cache is not None:
            if include_payoffs:
                return self._summary_cache

        # Build summary from history. SUMMARIZED mode should summarize all entries,
        # not the empty placeholder returned by get_history().
        history = self.entries.copy() if self.mode == MemoryMode.SUMMARIZED else self.get_history()

        if not history:
            return ""

        # Compile rounds
        rounds_text = []
        for entry in history:
            round_str = f"Round {entry.round_num}: I played {entry.action}"
            if entry.opponent_action:
                round_str += f", opponent played {entry.opponent_action}"
            if include_payoffs:
                round_str += f", payoff: {entry.payoff}"
            rounds_text.append(round_str)

        summary = "\n".join(rounds_text)

        # Add aggregate stats if we have enough history
        if len(history) > 0:
            summary += f"\n\nAggregate: {len(history)} rounds played"

            if include_payoffs:
                total_payoff = sum(e.payoff for e in history)
                avg_payoff = total_payoff / len(history)
                summary += (
                    f", total payoff: {total_payoff}, average: {avg_payoff:.2f}"
                )

            # Count action frequencies
            action_counts: dict[str, int] = {}
            for entry in history:
                action_counts[entry.action] = action_counts.get(entry.action, 0) + 1

            if action_counts:
                summary += "\nAction distribution: " + ", ".join(
                    f"{action}: {count}" for action, count in sorted(action_counts.items())
                )

        if include_payoffs:
            self._summary_cache = summary

        return summary

    def clear(self) -> None:
        """Clear all memory entries and cache."""
        self.entries.clear()
        self._summary_cache = None

    def to_dict(self) -> dict:
        """Convert memory to dictionary representation."""
        return {
            "mode": self.mode.value,
            "window_size": self.window_size,
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        """Create Memory instance from dictionary."""
        memory = cls(
            mode=MemoryMode(data.get("mode", "full")),
            window_size=data.get("window_size", 10),
        )
        for entry_dict in data.get("entries", []):
            entry = MemoryEntry(
                round_num=entry_dict["round_num"],
                action=entry_dict["action"],
                payoff=entry_dict["payoff"],
                opponent_action=entry_dict.get("opponent_action"),
                timestamp=entry_dict.get("timestamp"),
                metadata=entry_dict.get("metadata", {}),
            )
            memory.add(entry)
        return memory
