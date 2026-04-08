"""Optional exogenous events for long-horizon society simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any

from .world import World


@dataclass
class EventConfig:
    """Configuration for optional random events."""

    enabled: bool = False
    event_probability: float = 0.0
    seed: int = 42
    allowed_events: list[str] = field(
        default_factory=lambda: ["drought", "blight", "heatwave", "disease"]
    )
    drought_water_loss_fraction: float = 0.2
    blight_food_loss_fraction: float = 0.2
    heatwave_energy_penalty: int = 2
    disease_health_penalty: int = 2


class RandomEventEngine:
    """Apply disabled-by-default exogenous shocks to the world."""

    def __init__(self, config: EventConfig | None = None):
        self.config = config or EventConfig()
        self.rng = random.Random(self.config.seed)

    def maybe_apply(self, world: World) -> list[dict[str, Any]]:
        """Apply at most one event this timestep when configured to do so."""
        if not self.config.enabled or self.config.event_probability <= 0:
            return []
        if self.rng.random() > self.config.event_probability:
            return []
        if not self.config.allowed_events:
            return []

        event_kind = self.rng.choice(self.config.allowed_events)
        if event_kind == "drought":
            lost = min(
                world.public_water,
                max(1, round(world.public_water * self.config.drought_water_loss_fraction)),
            )
            world.public_water -= lost
            return [
                world.record_event(
                    kind="drought",
                    actor="environment",
                    amount=lost,
                    public=True,
                    metadata={"resource_type": "water"},
                )
            ]
        if event_kind == "blight":
            lost = min(
                world.public_food,
                max(1, round(world.public_food * self.config.blight_food_loss_fraction)),
            )
            world.public_food -= lost
            return [
                world.record_event(
                    kind="blight",
                    actor="environment",
                    amount=lost,
                    public=True,
                    metadata={"resource_type": "food"},
                )
            ]
        if event_kind == "heatwave":
            affected = []
            for state in world.get_alive_agents():
                state.adjust_resource("energy", -self.config.heatwave_energy_penalty)
                affected.append(state.agent_id)
            return [
                world.record_event(
                    kind="heatwave",
                    actor="environment",
                    amount=self.config.heatwave_energy_penalty,
                    public=True,
                    metadata={"affected_agents": affected},
                )
            ]
        if event_kind == "disease":
            alive = world.get_alive_agents()
            if not alive:
                return []
            target = self.rng.choice(alive)
            target.adjust_resource("health", -self.config.disease_health_penalty)
            return [
                world.record_event(
                    kind="disease",
                    actor="environment",
                    target=target.agent_id,
                    amount=self.config.disease_health_penalty,
                    public=True,
                )
            ]
        return []
