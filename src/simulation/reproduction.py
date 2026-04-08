"""Agent spawning logic for society simulations."""

from __future__ import annotations

from src.agents.base import Agent
from src.agents.memory import Memory, MemoryMode

from .world import AgentState, World


class ReproductionEngine:
    """Clone new agents from viable parents once thresholds are met."""

    def can_reproduce(self, world: World, agent_id: str) -> bool:
        """Return whether an agent has enough food, water, energy, and health to spawn."""
        state = world.get_state(agent_id)
        return (
            state.alive
            and state.food >= world.config.reproduce_min_food
            and state.water >= world.config.reproduce_min_water
            and state.energy >= world.config.reproduce_min_energy
            and state.health >= world.config.reproduce_min_health
            and len(world.get_alive_agents()) < world.config.max_agents
        )

    def spawn(
        self,
        *,
        world: World,
        agents: dict[str, Agent],
        parent_id: str,
        child_id: str,
    ) -> Agent | None:
        """Spawn a new agent by inheriting prompt/persona/model from a parent."""
        if not self.can_reproduce(world, parent_id):
            return None

        parent_state = world.get_state(parent_id)
        parent_agent = agents[parent_id]
        parent_state.adjust_resource("food", -world.config.offspring_start_food)
        parent_state.adjust_resource("water", -world.config.offspring_start_water)
        parent_state.adjust_resource("energy", -max(1, world.config.offspring_start_energy // 2))
        parent_state.adjust_resource("health", -1)

        child_agent = Agent(
            agent_id=child_id,
            model=parent_agent.model,
            provider_name=parent_agent.provider_name,
            system_prompt=parent_agent.system_prompt,
            temperature=parent_agent.temperature,
            framing=parent_agent.framing,
            persona=parent_agent.persona,
            memory=Memory(
                mode=MemoryMode(parent_agent.memory.mode.value),
                window_size=parent_agent.memory.window_size,
            ),
        )
        agents[child_id] = child_agent
        world.add_agent(
            AgentState(
                agent_id=child_id,
                model=child_agent.model,
                food=world.config.offspring_start_food,
                water=world.config.offspring_start_water,
                energy=world.config.offspring_start_energy,
                health=world.config.offspring_start_health,
                generation=parent_state.generation + 1,
                parent_id=parent_id,
                alive=True,
                unmonitored=False,
            )
        )
        world.record_event(
            kind="reproduce",
            actor=parent_id,
            target=child_id,
            amount=world.get_state(child_id).resources,
            public=True,
            metadata={
                "food": world.config.offspring_start_food,
                "water": world.config.offspring_start_water,
                "energy": world.config.offspring_start_energy,
                "health": world.config.offspring_start_health,
            },
        )
        return child_agent
