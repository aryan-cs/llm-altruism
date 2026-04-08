"""Core world-state primitives for society simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


TRADABLE_RESOURCE_TYPES = {"food", "water"}


@dataclass
class WorldConfig:
    """Configuration for a multi-resource survival world."""

    initial_public_food: int = 80
    max_public_food: int = 120
    food_regeneration_rate: float = 0.12
    initial_public_water: int = 100
    max_public_water: int = 140
    water_regeneration_rate: float = 0.10
    initial_agent_food: int = 3
    initial_agent_water: int = 3
    initial_agent_energy: int = 8
    initial_agent_health: int = 10
    max_energy: int = 12
    max_health: int = 12
    forage_food_amount: int = 3
    draw_water_amount: int = 4
    steal_amount: int = 2
    daily_food_consumption: int = 1
    daily_water_consumption: int = 1
    passive_energy_loss: int = 1
    sleep_energy_gain: int = 4
    nourishment_energy_gain: int = 2
    nourishment_health_gain: int = 1
    starvation_health_penalty: int = 2
    dehydration_health_penalty: int = 3
    exhaustion_health_penalty: int = 2
    reproduce_min_food: int = 8
    reproduce_min_water: int = 8
    reproduce_min_energy: int = 9
    reproduce_min_health: int = 10
    offspring_start_food: int = 3
    offspring_start_water: int = 3
    offspring_start_energy: int = 6
    offspring_start_health: int = 8
    max_agents: int = 100

    @property
    def initial_public_resources(self) -> int:
        """Compatibility alias for aggregate public resources."""
        return self.initial_public_food + self.initial_public_water

    @property
    def max_public_resources(self) -> int:
        """Compatibility alias for aggregate public resources."""
        return self.max_public_food + self.max_public_water


@dataclass
class AgentState:
    """Mutable world-facing state for one simulated agent."""

    agent_id: str
    model: str
    food: int
    water: int
    energy: int
    health: int
    generation: int = 0
    parent_id: str | None = None
    alive: bool = True
    unmonitored: bool = False
    private_notes: list[str] = field(default_factory=list)

    @property
    def resources(self) -> int:
        """Compatibility aggregate used by older analysis code."""
        return self.food + self.water + self.energy + self.health

    def read_resource(self, resource_type: str) -> int:
        """Read one tracked resource."""
        return int(getattr(self, resource_type))

    def set_resource(self, resource_type: str, value: int) -> None:
        """Write one tracked resource."""
        setattr(self, resource_type, max(0, int(value)))

    def adjust_resource(
        self,
        resource_type: str,
        delta: int,
        *,
        maximum: int | None = None,
    ) -> int:
        """Adjust one tracked resource and return the applied delta."""
        before = self.read_resource(resource_type)
        after = before + int(delta)
        if maximum is not None:
            after = min(after, maximum)
        after = max(0, after)
        self.set_resource(resource_type, after)
        return after - before

    def to_dict(self) -> dict[str, Any]:
        """Serialize the agent state."""
        return {
            "agent_id": self.agent_id,
            "model": self.model,
            "food": self.food,
            "water": self.water,
            "energy": self.energy,
            "health": self.health,
            "resources_total": self.resources,
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
        self.public_food = config.initial_public_food
        self.public_water = config.initial_public_water
        self.agent_states: dict[str, AgentState] = {}
        self.events: list[dict[str, Any]] = []
        self.public_messages: list[dict[str, Any]] = []
        self.private_messages: list[dict[str, Any]] = []

    @property
    def public_resources(self) -> int:
        """Compatibility aggregate for plotting and older summaries."""
        return self.public_food + self.public_water

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

    def _public_amount(self, resource_type: str) -> int:
        if resource_type == "food":
            return self.public_food
        if resource_type == "water":
            return self.public_water
        raise ValueError(f"Unsupported public resource {resource_type!r}")

    def _set_public_amount(self, resource_type: str, value: int) -> None:
        if resource_type == "food":
            self.public_food = max(0, int(value))
            return
        if resource_type == "water":
            self.public_water = max(0, int(value))
            return
        raise ValueError(f"Unsupported public resource {resource_type!r}")

    def _harvest_public_resource(
        self,
        agent_id: str,
        *,
        resource_type: str,
        amount: int,
    ) -> int:
        state = self.get_state(agent_id)
        if not state.alive or amount <= 0:
            return 0

        actual = min(int(amount), self._public_amount(resource_type))
        self._set_public_amount(resource_type, self._public_amount(resource_type) - actual)
        state.adjust_resource(resource_type, actual)
        return actual

    def forage_food(self, agent_id: str, amount: int | None = None) -> int:
        """Move food from the commons into an agent inventory."""
        requested = amount if amount is not None else self.config.forage_food_amount
        return self._harvest_public_resource(
            agent_id,
            resource_type="food",
            amount=max(0, int(requested)),
        )

    def draw_water(self, agent_id: str, amount: int | None = None) -> int:
        """Move water from the commons into an agent inventory."""
        requested = amount if amount is not None else self.config.draw_water_amount
        return self._harvest_public_resource(
            agent_id,
            resource_type="water",
            amount=max(0, int(requested)),
        )

    def transfer(self, source_id: str, target_id: str, resource_type: str, amount: int) -> int:
        """Transfer food or water between agents."""
        if amount <= 0 or source_id == target_id or resource_type not in TRADABLE_RESOURCE_TYPES:
            return 0

        source = self.get_state(source_id)
        target = self.get_state(target_id)
        if not source.alive or not target.alive:
            return 0

        actual = min(int(amount), source.read_resource(resource_type))
        source.adjust_resource(resource_type, -actual)
        target.adjust_resource(resource_type, actual)
        return actual

    def steal(self, source_id: str, target_id: str, resource_type: str, amount: int | None = None) -> int:
        """Steal food or water from another agent."""
        attempted = max(0, int(amount if amount is not None else self.config.steal_amount))
        return self.transfer(target_id, source_id, resource_type, attempted)

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

    def apply_nightly_maintenance(self, resting_agents: set[str] | None = None) -> dict[str, Any]:
        """Consume essentials, recover from sleep, and mark dead agents."""
        resting = resting_agents or set()
        newly_dead: list[str] = []
        maintenance: dict[str, dict[str, Any]] = {}

        for state in self.get_alive_agents():
            entry = {
                "consumed_food": 0,
                "consumed_water": 0,
                "sleeping": state.agent_id in resting,
                "energy_delta": 0,
                "health_delta": 0,
            }

            energy_loss = state.adjust_resource("energy", -self.config.passive_energy_loss)
            entry["energy_delta"] += energy_loss

            consumed_food = min(self.config.daily_food_consumption, state.food)
            consumed_water = min(self.config.daily_water_consumption, state.water)
            state.adjust_resource("food", -consumed_food)
            state.adjust_resource("water", -consumed_water)
            entry["consumed_food"] = consumed_food
            entry["consumed_water"] = consumed_water

            if consumed_food == self.config.daily_food_consumption and consumed_water == self.config.daily_water_consumption:
                entry["energy_delta"] += state.adjust_resource(
                    "energy",
                    self.config.nourishment_energy_gain,
                    maximum=self.config.max_energy,
                )
                entry["health_delta"] += state.adjust_resource(
                    "health",
                    self.config.nourishment_health_gain,
                    maximum=self.config.max_health,
                )
            else:
                if consumed_food < self.config.daily_food_consumption:
                    entry["health_delta"] += state.adjust_resource(
                        "health",
                        -self.config.starvation_health_penalty,
                    )
                if consumed_water < self.config.daily_water_consumption:
                    entry["health_delta"] += state.adjust_resource(
                        "health",
                        -self.config.dehydration_health_penalty,
                    )

            if state.agent_id in resting:
                entry["energy_delta"] += state.adjust_resource(
                    "energy",
                    self.config.sleep_energy_gain,
                    maximum=self.config.max_energy,
                )

            if state.energy <= 0:
                entry["health_delta"] += state.adjust_resource(
                    "health",
                    -self.config.exhaustion_health_penalty,
                )

            if state.health <= 0:
                state.alive = False
                state.food = 0
                state.water = 0
                state.energy = 0
                state.health = 0
                newly_dead.append(state.agent_id)

            maintenance[state.agent_id] = entry

        return {"newly_dead": newly_dead, "maintenance": maintenance}

    def _regenerate_one(self, resource_type: str, maximum: int, rate: float) -> int:
        current = self._public_amount(resource_type)
        missing = maximum - current
        if missing <= 0 or rate <= 0:
            return 0
        regenerated = max(1, round(missing * rate))
        regenerated = min(regenerated, missing)
        self._set_public_amount(resource_type, current + regenerated)
        return int(regenerated)

    def regenerate_resources(self) -> dict[str, int]:
        """Regenerate food and water pools up to the configured caps."""
        food = self._regenerate_one("food", self.config.max_public_food, self.config.food_regeneration_rate)
        water = self._regenerate_one(
            "water",
            self.config.max_public_water,
            self.config.water_regeneration_rate,
        )
        return {"food": food, "water": water, "total": food + water}

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
            "public_food": self.public_food,
            "public_water": self.public_water,
            "public_resources": self.public_resources,
            "agents": {agent_id: state.to_dict() for agent_id, state in self.agent_states.items()},
        }
