"""Part 2 runner for the main long-horizon society viability experiments."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from src.agents.prompts import render_prompt_template
from src.analysis import summarize_society
from src.simulation import (
    AgentState,
    EconomyEngine,
    EventConfig,
    RandomEventEngine,
    ReproductionEngine,
    SocietySimulation,
    World,
    WorldConfig,
)
from src.utils.parsing import parse_json_response

from .config import ModelSpec, PopulationSpec, PromptVariantConfig
from .runner import BaseExperimentRunner, ModelUnavailableError


class Part2Runner(BaseExperimentRunner):
    """Run the main scarce-resource society simulation without public reputation."""

    def enable_reputation(self) -> bool:
        return False

    async def run(self) -> dict[str, Any]:
        start = time.time()
        trials: list[dict[str, Any]] = []
        trial_id = 0

        for prompt_variant in self.config.prompt_variants:
            for temperature in self.config.parameters.temperature:
                for repetition in range(self.config.repetitions):
                    trial = await self._run_society_trial(
                        trial_id=trial_id,
                        prompt_variant=prompt_variant,
                        temperature=temperature,
                        repetition=repetition,
                    )
                    trials.append(trial)
                    self.logger.log_trial_summary(trial_id, trial["summary"])
                    if trial.get("status") == "skipped":
                        self.record_skipped_trial(
                            {
                                "trial_id": trial_id,
                                "prompt_variant": prompt_variant.name,
                                "temperature": temperature,
                                "repetition": repetition,
                                "reason": trial.get("skip_reason", "unavailable population"),
                                "status": "skipped",
                            }
                        )
                    trial_id += 1

        duration = time.time() - start
        aggregate = self._aggregate_trials(trials)
        return self.finalize({"trials": trials, "aggregate_summary": aggregate}, duration)

    async def _run_society_trial(
        self,
        *,
        trial_id: int,
        prompt_variant: PromptVariantConfig,
        temperature: float,
        repetition: int,
    ) -> dict[str, Any]:
        simulation = self._create_simulation(prompt_variant, temperature)
        if len(simulation.alive_agent_ids()) < 2:
            return {
                "trial_id": trial_id,
                "prompt_variant": prompt_variant.name,
                "temperature": temperature,
                "repetition": repetition,
                "rounds": [],
                "summary": {"status": "skipped", "completed_rounds": 0},
                "status": "skipped",
                "skip_reason": "fewer than two configured/available agents remained after env filtering",
            }
        rounds: list[dict[str, Any]] = []

        for timestep in range(1, self.config.rounds + 1):
            simulation.world.timestep = timestep
            decisions = await self._collect_decisions(simulation, temperature)
            removed_agents = []
            for agent_id, decision in list(decisions.items()):
                if decision.get("_unavailable"):
                    simulation.world.get_state(agent_id).alive = False
                    removed_agents.append(
                        {
                            "agent_id": agent_id,
                            "reason": decision.get("_unavailable_reason", "model unavailable"),
                        }
                    )
            if removed_agents:
                self.record_skipped_trial(
                    {
                        "trial_id": trial_id,
                        "timestep": timestep,
                        "status": "model_pruned",
                        "agents": removed_agents,
                    }
                )
            if len(simulation.alive_agent_ids()) < 2:
                break
            step = simulation.apply_decisions(decisions)
            rounds.append(step)
            self.logger.log_round(trial_id, timestep, step)

        summary = summarize_society(
            rounds,
            max_public_resources=simulation.world.config.max_public_resources,
        )
        return {
            "trial_id": trial_id,
            "prompt_variant": prompt_variant.name,
            "temperature": temperature,
            "repetition": repetition,
            "rounds": rounds,
            "summary": summary,
            "status": "completed",
        }

    def _create_simulation(self, prompt_variant: PromptVariantConfig, temperature: float) -> SocietySimulation:
        world = World(WorldConfig(**self.config.world.model_dump()))
        agents: dict[str, Any] = {}
        index = 0

        total_population = sum(population.count for population in self.config.agents)
        for population in self.config.agents:
            spec = ModelSpec(model=population.model, provider=population.provider)
            if self.get_skip_reason(spec) is not None:
                continue
            for _ in range(population.count):
                agent_id = f"agent-{index}"
                agent = self.build_agent(
                    agent_id=agent_id,
                    spec=spec,
                    prompt_variant=prompt_variant,
                    temperature=temperature,
                )
                agents[agent_id] = agent
                world.add_agent(
                    AgentState(
                        agent_id=agent_id,
                        model=population.model,
                        food=world.config.initial_agent_food,
                        water=world.config.initial_agent_water,
                        energy=world.config.initial_agent_energy,
                        health=world.config.initial_agent_health,
                        generation=0,
                        parent_id=None,
                        alive=True,
                        unmonitored=self._is_unmonitored(index, total_population),
                    )
                )
                index += 1

        event_engine = None
        if self.config.events is not None:
            event_engine = RandomEventEngine(EventConfig(**self.config.events.model_dump()))

        return SocietySimulation(
            world=world,
            agents=agents,
            economy=EconomyEngine(trade_offer_ttl=self.config.society.trade_offer_ttl),
            reproduction=ReproductionEngine(),
            reputation=None,
            event_engine=event_engine,
            allow_private_messages=self.config.society.allow_private_messages,
            allow_steal=self.config.society.allow_steal,
        )

    def _is_unmonitored(self, index: int, total_population: int) -> bool:
        if not self.config.society.allow_unmonitored_agents:
            return False
        fraction = self.config.society.unmonitored_fraction
        if fraction <= 0:
            return False
        threshold = max(1, round(total_population * fraction))
        return index < threshold

    async def _collect_decisions(
        self,
        simulation: SocietySimulation,
        temperature: float,
    ) -> dict[str, dict[str, Any]]:
        semaphore = asyncio.Semaphore(self.config.parameters.concurrency)

        async def get_decision(agent_id: str) -> tuple[str, dict[str, Any]]:
            agent = simulation.agents[agent_id]
            spec = ModelSpec(model=agent.model, provider=agent.provider_name)
            context = simulation.build_agent_context(agent_id)
            prompt = self._build_society_prompt(context, enable_reputation=self.enable_reputation())
            mock_content = self._mock_society_response(context)

            async with semaphore:
                try:
                    response = await self.request_completion(
                        spec=spec,
                        messages=agent.build_messages(prompt, include_payoffs=False),
                        temperature=temperature,
                        response_format={"type": "json_object"},
                        mock_content=mock_content,
                    )
                except ModelUnavailableError as exc:
                    return (
                        agent_id,
                        {
                            "action": "idle",
                            "_unavailable": True,
                            "_unavailable_reason": str(exc),
                        },
                    )

            parsed = parse_json_response(response.content) or {}
            if not isinstance(parsed, dict) or "action" not in parsed:
                parsed = {
                    "action": "forage_food",
                    "amount": simulation.world.config.forage_food_amount,
                }
            parsed["raw_response"] = response.content
            parsed["latency_ms"] = response.latency_ms
            parsed["tokens"] = response.usage.model_dump()
            parsed["cost_usd"] = response.cost_usd
            return (agent_id, parsed)

        results = await asyncio.gather(*(get_decision(agent_id) for agent_id in simulation.alive_agent_ids()))
        return {agent_id: decision for agent_id, decision in results}

    def _format_visible_event(self, event: dict[str, Any]) -> str:
        resource_type = event.get("metadata", {}).get("resource_type")
        resource_text = f" resource={resource_type}" if resource_type else ""
        amount_text = f" amount={event.get('amount')}" if event.get("amount") is not None else ""
        target_text = f" target={event.get('target')}" if event.get("target") else ""
        return f"- {event['kind']}: actor={event['actor']}{target_text}{amount_text}{resource_text}"

    def _build_society_prompt(self, context: dict[str, Any], *, enable_reputation: bool) -> str:
        visible_events = "\n".join(
            self._format_visible_event(event) for event in context["visible_events"][-6:]
        ) or "- No recent visible events."

        public_messages_section = ""
        if context["public_messages"]:
            public_messages_section = "\n\nRecent public messages:\n" + "\n".join(
                f"- {message['sender']}: {message['message']}"
                for message in context["public_messages"][-4:]
            )

        private_messages_section = ""
        if context["private_messages"]:
            private_messages_section = "\n\nRecent private messages involving you:\n" + "\n".join(
                f"- {message['sender']} -> {message['recipient']}: {message['message']}"
                for message in context["private_messages"][-4:]
            )

        pending_offers_section = ""
        if context["pending_offers"]:
            pending_offers_section = "\n\nPending trade offers:\n" + "\n".join(
                f"- {offer['offer_id']}: {offer['from_agent']} offers {offer['give_amount']} "
                f"{offer['give_resource']} for {offer['ask_amount']} {offer['ask_resource']} "
                f"from {offer['to_agent']}"
                for offer in context["pending_offers"]
            )

        other_agents_section = ""
        if context["other_agents"]:
            ranked_agents = sorted(
                context["other_agents"].items(),
                key=lambda item: (
                    item[1]["health"],
                    item[1]["food"] + item[1]["water"] + item[1]["energy"],
                    item[0],
                ),
            )
            visible_subset = ranked_agents[:10]
            other_agents_section = "\n\nVisible vulnerable agents:\n" + "\n".join(
                f"- {agent_id}: food={info['food']} water={info['water']} "
                f"energy={info['energy']} health={info['health']}"
                for agent_id, info in visible_subset
            )
            hidden_count = len(ranked_agents) - len(visible_subset)
            if hidden_count > 0:
                other_agents_section += f"\n- ... and {hidden_count} more agents not listed."

        reputation_section = ""
        if enable_reputation and context["reputation"]:
            reputation_section = "\n\nPublic reputation:\n" + "\n".join(
                f"- {agent_id}: score={info['score']:.2f}"
                for agent_id, info in sorted(context["reputation"].items())
            )

        template = (
            "simulation/society_decision_with_reputation.txt"
            if enable_reputation
            else "simulation/society_decision.txt"
        )
        return render_prompt_template(
            template,
            timestep=context["timestep"],
            self_food=context["self"]["food"],
            self_water=context["self"]["water"],
            self_energy=context["self"]["energy"],
            self_health=context["self"]["health"],
            self_resources=context["self"]["resources_total"],
            public_food=context["public_food"],
            public_water=context["public_water"],
            public_resources=context["public_resources"],
            visible_events_block=visible_events,
            public_messages_section=public_messages_section,
            private_messages_section=private_messages_section,
            pending_offers_section=pending_offers_section,
            other_agents_section=other_agents_section,
            reputation_section=reputation_section,
        )

    def _mock_society_response(self, context: dict[str, Any]) -> str:
        self_state = context["self"]
        own_food = int(self_state["food"])
        own_water = int(self_state["water"])
        own_energy = int(self_state["energy"])
        own_health = int(self_state["health"])
        public_food = int(context["public_food"])
        public_water = int(context["public_water"])
        pending_offers = context["pending_offers"]
        other_agents = context["other_agents"]

        if pending_offers:
            own_id = self_state["agent_id"]
            for offer in pending_offers:
                if offer["to_agent"] != own_id:
                    continue
                ask_resource = offer["ask_resource"]
                ask_amount = int(offer["ask_amount"])
                give_resource = offer["give_resource"]
                if ask_resource == "food" and own_food >= ask_amount and give_resource == "water" and own_water <= 2:
                    return json.dumps(
                        {
                            "scratchpad": "Accept water when I am short on it.",
                            "action": "accept_trade",
                            "offer_id": offer["offer_id"],
                            "ratings": [],
                        }
                    )
                if ask_resource == "water" and own_water >= ask_amount and give_resource == "food" and own_food <= 2:
                    return json.dumps(
                        {
                            "scratchpad": "Accept food when I am short on it.",
                            "action": "accept_trade",
                            "offer_id": offer["offer_id"],
                            "ratings": [],
                        }
                    )

        if own_health <= 4 or own_energy <= 2:
            return json.dumps(
                {
                    "scratchpad": "Prioritize recovery before collapse.",
                    "action": "sleep",
                    "ratings": [],
                }
            )

        if (
            own_food >= self.config.world.reproduce_min_food
            and own_water >= self.config.world.reproduce_min_water
            and own_energy >= self.config.world.reproduce_min_energy
            and own_health >= self.config.world.reproduce_min_health
        ):
            return json.dumps(
                {
                    "scratchpad": "Conditions are good enough to reproduce.",
                    "action": "reproduce",
                    "ratings": [],
                }
            )

        hungry_agents = [
            agent_id for agent_id, info in other_agents.items() if int(info["food"]) <= 1
        ]
        thirsty_agents = [
            agent_id for agent_id, info in other_agents.items() if int(info["water"]) <= 1
        ]
        if own_food >= 6 and hungry_agents:
            return json.dumps(
                {
                    "scratchpad": "I can spare food to stabilize a neighbor.",
                    "action": "share",
                    "target": hungry_agents[0],
                    "resource_type": "food",
                    "amount": 2,
                    "ratings": [],
                }
            )
        if own_water >= 6 and thirsty_agents:
            return json.dumps(
                {
                    "scratchpad": "I can spare water to stabilize a neighbor.",
                    "action": "share",
                    "target": thirsty_agents[0],
                    "resource_type": "water",
                    "amount": 2,
                    "ratings": [],
                }
            )

        if own_food <= 1 and public_food > 0:
            return json.dumps(
                {
                    "scratchpad": "Food is urgent.",
                    "action": "forage_food",
                    "amount": self.config.world.forage_food_amount,
                    "ratings": [],
                }
            )
        if own_water <= 1 and public_water > 0:
            return json.dumps(
                {
                    "scratchpad": "Water is urgent.",
                    "action": "draw_water",
                    "amount": self.config.world.draw_water_amount,
                    "ratings": [],
                }
            )

        water_rich_agents = [
            agent_id for agent_id, info in other_agents.items() if int(info["water"]) >= 5
        ]
        food_rich_agents = [
            agent_id for agent_id, info in other_agents.items() if int(info["food"]) >= 5
        ]
        if own_food >= 5 and own_water <= 2 and water_rich_agents:
            return json.dumps(
                {
                    "scratchpad": "Trade food for scarce water.",
                    "action": "offer_trade",
                    "target": water_rich_agents[0],
                    "give_resource": "food",
                    "give_amount": 2,
                    "ask_resource": "water",
                    "ask_amount": 2,
                    "message": "Food for water?",
                    "ratings": [],
                }
            )
        if own_water >= 5 and own_food <= 2 and food_rich_agents:
            return json.dumps(
                {
                    "scratchpad": "Trade water for scarce food.",
                    "action": "offer_trade",
                    "target": food_rich_agents[0],
                    "give_resource": "water",
                    "give_amount": 2,
                    "ask_resource": "food",
                    "ask_amount": 2,
                    "message": "Water for food?",
                    "ratings": [],
                }
            )

        if own_food <= 1 and food_rich_agents and self.config.society.allow_steal:
            return json.dumps(
                {
                    "scratchpad": "Food shortage is existential.",
                    "action": "steal",
                    "target": food_rich_agents[0],
                    "resource_type": "food",
                    "amount": self.config.world.steal_amount,
                    "ratings": [],
                }
            )
        if own_water <= 1 and water_rich_agents and self.config.society.allow_steal:
            return json.dumps(
                {
                    "scratchpad": "Water shortage is existential.",
                    "action": "steal",
                    "target": water_rich_agents[0],
                    "resource_type": "water",
                    "amount": self.config.world.steal_amount,
                    "ratings": [],
                }
            )

        if own_food < own_water and public_food > 0:
            return json.dumps(
                {
                    "scratchpad": "Rebalance toward food.",
                    "action": "forage_food",
                    "amount": self.config.world.forage_food_amount,
                    "ratings": [],
                }
            )
        if own_water < own_food and public_water > 0:
            return json.dumps(
                {
                    "scratchpad": "Rebalance toward water.",
                    "action": "draw_water",
                    "amount": self.config.world.draw_water_amount,
                    "ratings": [],
                }
            )
        if public_food > 0:
            return json.dumps(
                {
                    "scratchpad": "Build a food buffer.",
                    "action": "forage_food",
                    "amount": self.config.world.forage_food_amount,
                    "ratings": [],
                }
            )
        if public_water > 0:
            return json.dumps(
                {
                    "scratchpad": "Build a water buffer.",
                    "action": "draw_water",
                    "amount": self.config.world.draw_water_amount,
                    "ratings": [],
                }
            )

        return json.dumps(
            {
                "scratchpad": "No productive move remains, so recover.",
                "action": "sleep",
                "ratings": [],
            }
        )

    def _aggregate_trials(self, trials: list[dict[str, Any]]) -> dict[str, float]:
        completed_trials = [trial for trial in trials if trial.get("status") != "skipped"]
        if not completed_trials:
            return {}
        numeric_fields: dict[str, list[float]] = {}
        for trial in completed_trials:
            for key, value in trial["summary"].items():
                if isinstance(value, (int, float)):
                    numeric_fields.setdefault(key, []).append(float(value))
        return {key: sum(values) / len(values) for key, values in numeric_fields.items() if values}
