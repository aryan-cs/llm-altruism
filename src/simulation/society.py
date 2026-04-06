"""Society simulation orchestration on top of world/economy/reputation primitives."""

from __future__ import annotations

from typing import Any

from src.agents.base import Agent

from .economy import EconomyEngine
from .reproduction import ReproductionEngine
from .reputation import ReputationSystem
from .world import World


class SocietySimulation:
    """Manage a multi-agent micro-society over discrete timesteps."""

    def __init__(
        self,
        *,
        world: World,
        agents: dict[str, Agent],
        economy: EconomyEngine,
        reproduction: ReproductionEngine,
        reputation: ReputationSystem | None = None,
        allow_private_messages: bool = True,
        allow_steal: bool = True,
    ):
        self.world = world
        self.agents = agents
        self.economy = economy
        self.reproduction = reproduction
        self.reputation = reputation
        self.allow_private_messages = allow_private_messages
        self.allow_steal = allow_steal
        self._child_counter = 0

    def alive_agent_ids(self) -> list[str]:
        """Return the ids of all living agents."""
        return [state.agent_id for state in self.world.get_alive_agents()]

    def build_agent_context(self, agent_id: str, recent_limit: int = 8) -> dict[str, Any]:
        """Build the visible world state for a specific agent."""
        state = self.world.get_state(agent_id)
        return {
            "timestep": self.world.timestep,
            "self": state.to_dict(),
            "public_resources": self.world.public_resources,
            "visible_events": self.world.visible_events_for(agent_id, limit=recent_limit),
            "public_messages": self.world.public_messages[-recent_limit:],
            "private_messages": self.world.visible_private_messages_for(agent_id, limit=recent_limit),
            "pending_offers": self.economy.get_visible_offers(agent_id),
            "other_agents": {
                other.agent_id: other.to_dict()
                for other in self.world.get_alive_agents()
                if other.agent_id != agent_id
            },
            "reputation": self.reputation.summary() if self.reputation else {},
        }

    def _next_child_id(self) -> str:
        self._child_counter += 1
        return f"offspring-{self._child_counter}"

    def apply_decisions(self, decisions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Advance the society by one timestep using the supplied decisions."""
        pre_resources = {
            agent_id: self.world.get_state(agent_id).resources
            for agent_id in self.alive_agent_ids()
        }

        for agent_id, decision in decisions.items():
            scratchpad = str(decision.get("scratchpad", "")).strip()
            if scratchpad:
                self.agents[agent_id].add_scratchpad_entry(scratchpad)

        events, trade_volume = self.economy.resolve(
            self.world,
            decisions,
            allow_private_messages=self.allow_private_messages,
            allow_steal=self.allow_steal,
        )

        ratings_logged: list[dict[str, Any]] = []
        if self.reputation:
            for agent_id, decision in decisions.items():
                for rating_payload in decision.get("ratings", []):
                    if not isinstance(rating_payload, dict):
                        continue
                    target = rating_payload.get("target")
                    score = rating_payload.get("score")
                    if not isinstance(target, str) or not isinstance(score, int):
                        continue
                    rating = self.reputation.add_rating(
                        rater=agent_id,
                        target=target,
                        score=score,
                        timestep=self.world.timestep,
                        note=str(rating_payload.get("note", "")).strip() or None,
                    )
                    if rating:
                        ratings_logged.append(rating.to_dict(anonymous=self.reputation.config.anonymous_ratings))
            self.reputation.apply_decay()

        spawned_agents: list[str] = []
        for agent_id, decision in decisions.items():
            if str(decision.get("action", "idle")) != "reproduce":
                continue
            child_id = self._next_child_id()
            child = self.reproduction.spawn(
                world=self.world,
                agents=self.agents,
                parent_id=agent_id,
                child_id=child_id,
            )
            if child is not None:
                spawned_agents.append(child.agent_id)

        newly_dead = self.world.apply_survival_cost()
        regenerated = self.world.regenerate_resources()

        post_resources = {
            agent_id: self.world.get_state(agent_id).resources
            for agent_id in self.world.agent_states
        }

        for agent_id, agent in self.agents.items():
            state = self.world.get_state(agent_id)
            if not state.alive and agent_id not in pre_resources:
                continue
            before = pre_resources.get(agent_id, 0)
            after = post_resources.get(agent_id, 0)
            decision = decisions.get(agent_id, {"action": "idle"})
            target = decision.get("target")
            opponent_action = str(target) if target else None
            agent.record_action(
                round_num=self.world.timestep,
                action=str(decision.get("action", "idle")),
                payoff=float(after - before),
                opponent_action=opponent_action,
            )

        agent_resources = {
            agent_id: state.resources
            for agent_id, state in self.world.agent_states.items()
            if state.alive
        }

        return {
            "timestep": self.world.timestep,
            "events": events,
            "ratings": ratings_logged,
            "trade_volume": trade_volume,
            "public_resources": self.world.public_resources,
            "agent_resources": agent_resources,
            "alive_count": len(self.alive_agent_ids()),
            "total_agents": len(self.world.agent_states),
            "spawned_agents": spawned_agents,
            "newly_dead": newly_dead,
            "regenerated": regenerated,
        }
