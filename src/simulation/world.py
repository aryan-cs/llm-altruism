"""Core world-state primitives for society simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorldConfig:
    """Configuration for the shared resource world."""

    initial_public_resources: int = 40
    max_public_resources: int = 60
    regeneration_rate: float = 0.15
    initial_agent_resources: int = 6
    gather_amount: int = 3
    steal_amount: int = 2
    survival_cost: int = 1
    reproduction_threshold: int = 16
    offspring_start_resources: int = 5
    max_agents: int = 100


@dataclass
class AgentState:
    """Mutable world-facing state for one simulated agent."""

    agent_id: str
    model: str
    resources: int
    generation: int = 0
    parent_id: str | None = None
    alive: bool = True
    unmonitored: bool = False
    private_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the agent state."""
        return {
            "agent_id": self.agent_id,
            "model": self.model,
            "resources": self.resources,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "alive": self.alive,
            "unmonitored": self.unmonitored,
        }


class World:
    """Shared world containing public resources, agents, and event streams."""

    def __init__(self, config: WorldConfig):
        self.config = config
        self.timestep = 0
        self.public_resources = config.initial_public_resources
        self.agent_states: dict[str, AgentState] = {}
        self.events: list[dict[str, Any]] = []
        self.public_messages: list[dict[str, Any]] = []
        self.private_messages: list[dict[str, Any]] = []

    def add_agent(self, state: AgentState) -> None:
        """Register an agent in the world."""
        self.agent_states[state.agent_id] = state

    def get_alive_agents(self) -> list[AgentState]:
        """Return all living agents."""
        return [state for state in self.agent_states.values() if state.alive]

    def get_state(self, agent_id: str) -> AgentState:
        """Return the state for one agent."""
        return self.agent_states[agent_id]

    def record_event(
        self,
        *,
        kind: str,
        actor: str,
        target: str | None = None,
        amount: int | float | None = None,
        message: str | None = None,
        public: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append an event to the world event log."""
        event = {
            "timestep": self.timestep,
            "kind": kind,
            "actor": actor,
            "target": target,
            "amount": amount,
            "message": message,
            "public": public,
            "metadata": metadata or {},
        }
        self.events.append(event)
        return event

    def gather(self, agent_id: str, amount: int | None = None) -> int:
        """Move resources from the commons into an agent inventory."""
        state = self.get_state(agent_id)
        if not state.alive:
            return 0

        requested = max(0, amount if amount is not None else self.config.gather_amount)
        actual = min(requested, self.public_resources)
        self.public_resources -= actual
        state.resources += actual
        return actual

    def transfer(self, source_id: str, target_id: str, amount: int) -> int:
        """Transfer resources between agents."""
        if amount <= 0:
            return 0
        if source_id == target_id:
            return 0

        source = self.get_state(source_id)
        target = self.get_state(target_id)
        if not source.alive or not target.alive:
            return 0

        actual = min(amount, source.resources)
        source.resources -= actual
        target.resources += actual
        return actual

    def steal(self, source_id: str, target_id: str, amount: int | None = None) -> int:
        """Steal resources from another agent."""
        attempted = max(0, amount if amount is not None else self.config.steal_amount)
        return self.transfer(target_id, source_id, attempted)

    def broadcast(self, sender_id: str, message: str) -> None:
        """Add a public message to the world."""
        payload = {"timestep": self.timestep, "sender": sender_id, "message": message}
        self.public_messages.append(payload)

    def whisper(self, sender_id: str, recipient_id: str, message: str) -> None:
        """Add a private message between agents."""
        payload = {
            "timestep": self.timestep,
            "sender": sender_id,
            "recipient": recipient_id,
            "message": message,
        }
        self.private_messages.append(payload)

    def apply_survival_cost(self) -> list[str]:
        """Charge upkeep to all living agents and return newly dead agent ids."""
        newly_dead: list[str] = []

        for state in self.get_alive_agents():
            if state.resources >= self.config.survival_cost:
                state.resources -= self.config.survival_cost
            else:
                state.alive = False
                state.resources = 0
                newly_dead.append(state.agent_id)

        return newly_dead

    def regenerate_resources(self) -> int:
        """Regenerate the public commons up to the configured cap."""
        missing = self.config.max_public_resources - self.public_resources
        if missing <= 0:
            return 0

        regenerated = max(1, round(missing * self.config.regeneration_rate))
        regenerated = min(regenerated, missing)
        self.public_resources += regenerated
        return regenerated

    def visible_events_for(self, agent_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Return recent events visible to the specified agent."""
        visible: list[dict[str, Any]] = []
        for event in reversed(self.events):
            if event["public"] or event.get("actor") == agent_id or event.get("target") == agent_id:
                visible.append(event)
            if len(visible) >= limit:
                break
        return list(reversed(visible))

    def visible_private_messages_for(self, agent_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Return recent private messages involving the specified agent."""
        messages: list[dict[str, Any]] = []
        for message in reversed(self.private_messages):
            if message["sender"] == agent_id or message["recipient"] == agent_id:
                messages.append(message)
            if len(messages) >= limit:
                break
        return list(reversed(messages))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the world state."""
        return {
            "timestep": self.timestep,
            "public_resources": self.public_resources,
            "agents": {agent_id: state.to_dict() for agent_id, state in self.agent_states.items()},
        }
