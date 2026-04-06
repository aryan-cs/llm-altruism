"""
Base Agent class for game-theoretic simulations.

Agents have configurable system prompts, framings, personas, and memory systems.
They can be prompted for actions and maintain a history of interactions.
"""

from dataclasses import dataclass, field
from typing import Optional

from .memory import Memory, MemoryEntry, MemoryMode


@dataclass
class Agent:
    """
    An agent that plays in game-theoretic simulations.

    Attributes:
        agent_id: Unique identifier for this agent
        model: The LLM model to use (e.g., "gpt-4o", "claude-3-5-sonnet")
        provider_name: The LLM provider (e.g., "openai", "anthropic")
        system_prompt: The core system prompt defining the agent's role
        framing: Optional additional framing to shape behavior
        persona: Optional persona describing the agent's character
        temperature: Sampling temperature for model responses (0.0-2.0)
        memory: Memory object for tracking past interactions
        scratchpad: List of private reasoning entries (for parts 2/3 of tasks)
    """

    agent_id: str
    model: str
    provider_name: str
    system_prompt: str
    temperature: float = 0.7
    framing: Optional[str] = None
    persona: Optional[str] = None
    memory: Memory = field(default_factory=lambda: Memory(mode=MemoryMode.FULL))
    scratchpad: list[str] = field(default_factory=list)

    def build_system_message(self) -> str:
        """
        Build the complete system message from components.

        Combines system_prompt + framing + persona into a single system message.
        Components are joined with blank lines.

        Returns:
            The complete system message string
        """
        parts = [self.system_prompt]

        if self.framing:
            parts.append(self.framing)

        if self.persona:
            parts.append(self.persona)

        return "\n\n".join(parts)

    def build_messages(
        self,
        user_content: str,
        include_payoffs: bool = True,
    ) -> list[dict[str, str]]:
        """
        Build a complete message list in OpenAI format.

        Includes the system message, relevant history from memory, and the current user message.

        Args:
            user_content: The user message content

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        messages = []

        # Add system message
        messages.append(
            {
                "role": "system",
                "content": self.build_system_message(),
            }
        )

        # Add relevant history based on memory mode
        if self.memory.mode == MemoryMode.NONE:
            # No history
            pass
        elif self.memory.mode == MemoryMode.SUMMARIZED:
            # Add summary as a system message or user message
            summary = self.memory.get_summary(include_payoffs=include_payoffs)
            if summary:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Here is your history so far:\n\n{summary}",
                    }
                )
        else:
            # FULL or WINDOWED: add individual round history
            history = self.memory.get_history()
            if history:
                history_text = "Your recent history:\n\n"
                for entry in history:
                    history_text += f"Round {entry.round_num}: You played {entry.action}"
                    if entry.opponent_action:
                        history_text += f", opponent played {entry.opponent_action}"
                    if include_payoffs:
                        history_text += f", you earned {entry.payoff}"
                    history_text += "\n"

                messages.append(
                    {
                        "role": "user",
                        "content": history_text,
                    }
                )

        # Add current user message
        messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )

        return messages

    def record_action(
        self,
        round_num: int,
        action: str,
        payoff: float,
        opponent_action: Optional[str] = None,
    ) -> None:
        """
        Record the result of a round in memory.

        Args:
            round_num: The round number
            action: The action taken by this agent
            payoff: The payoff received
            opponent_action: Optional action taken by opponent
        """
        entry = MemoryEntry(
            round_num=round_num,
            action=action,
            payoff=payoff,
            opponent_action=opponent_action,
        )
        self.memory.add(entry)

    def add_scratchpad_entry(self, thought: str) -> None:
        """
        Add a private reasoning entry to the scratchpad.

        Scratchpad is for internal reasoning and chain-of-thought tracking
        during multi-part tasks.

        Args:
            thought: The reasoning text to add
        """
        self.scratchpad.append(thought)

    def clear_scratchpad(self) -> None:
        """Clear all scratchpad entries."""
        self.scratchpad.clear()

    def get_scratchpad(self) -> str:
        """
        Get the complete scratchpad as a single string.

        Returns:
            Scratchpad entries joined with newlines
        """
        return "\n".join(self.scratchpad)

    def to_dict(self) -> dict:
        """
        Convert agent to a serializable dictionary representation.

        Returns:
            Dictionary with all agent state
        """
        return {
            "agent_id": self.agent_id,
            "model": self.model,
            "provider_name": self.provider_name,
            "system_prompt": self.system_prompt,
            "framing": self.framing,
            "persona": self.persona,
            "temperature": self.temperature,
            "memory": self.memory.to_dict(),
            "scratchpad": self.scratchpad,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Agent":
        """
        Create an Agent instance from a dictionary.

        Args:
            data: Dictionary with agent configuration

        Returns:
            New Agent instance
        """
        memory_data = data.get("memory", {})
        memory = Memory.from_dict(memory_data) if memory_data else Memory()

        return cls(
            agent_id=data["agent_id"],
            model=data["model"],
            provider_name=data["provider_name"],
            system_prompt=data["system_prompt"],
            temperature=data.get("temperature", 0.7),
            framing=data.get("framing"),
            persona=data.get("persona"),
            memory=memory,
            scratchpad=data.get("scratchpad", []),
        )
