"""Agent spawning logic for society simulations."""

from __future__ import annotations

from typing import Any

from src.agents.base import Agent
from src.agents.memory import Memory, MemoryMode

from .world import AgentState, World


class ReproductionEngine:
    """Clone new agents from successful parents once thresholds are met."""

    def can_reproduce(self, world: World, agent_id: str) -> bool:
        """Return whether an agent has enough resources to spawn."""
        state = world.get_state(agent_id)
        return (
            state.alive
            and state.resources >= world.config.reproduction_threshold
            and len(world.agent_states) < world.config.max_agents
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
        spawn_cost = world.config.offspring_start_resources
        parent_state.resources -= spawn_cost

        child_agent = Agent(
            agent_id=child_id,
            model=parent_agent.model,
            provider_name=parent_agent.provider_name,
            system_prompt=parent_agent.system_prompt,
            temperature=parent_agent.temperature,
            framing=parent_agent.framing,
            persona=parent_agent.persona,
            memory=Memory(mode=MemoryMode(parent_agent.memory.mode.value), window_size=parent_agent.memory.window_size),
        )
        agents[child_id] = child_agent
        world.add_agent(
            AgentState(
                agent_id=child_id,
                model=child_agent.model,
                resources=world.config.offspring_start_resources,
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
            amount=world.config.offspring_start_resources,
            public=True,
        )
        return child_agent
