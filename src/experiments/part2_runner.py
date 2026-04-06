"""Part 2 runner for multi-agent society simulations."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from src.agents.prompts import render_prompt_template
from src.analysis import summarize_society
from src.simulation import AgentState, EconomyEngine, ReproductionEngine, SocietySimulation, World, WorldConfig
from src.utils.parsing import parse_json_response

from .config import ModelSpec, PopulationSpec, PromptVariantConfig
from .runner import BaseExperimentRunner, ModelUnavailableError


class Part2Runner(BaseExperimentRunner):
    """Run a scarce-resource society simulation without public reputation."""

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
                        resources=world.config.initial_agent_resources,
                        generation=0,
                        parent_id=None,
                        alive=True,
                        unmonitored=self._is_unmonitored(index, total_population),
                    )
                )
                index += 1

        return SocietySimulation(
            world=world,
            agents=agents,
            economy=EconomyEngine(trade_offer_ttl=self.config.society.trade_offer_ttl),
            reproduction=ReproductionEngine(),
            reputation=None,
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
                parsed = {"action": "gather", "amount": simulation.world.config.gather_amount}
            parsed["raw_response"] = response.content
            parsed["latency_ms"] = response.latency_ms
            parsed["tokens"] = response.usage.model_dump()
            parsed["cost_usd"] = response.cost_usd
            return (agent_id, parsed)

        results = await asyncio.gather(*(get_decision(agent_id) for agent_id in simulation.alive_agent_ids()))
        return {agent_id: decision for agent_id, decision in results}

    def _build_society_prompt(self, context: dict[str, Any], *, enable_reputation: bool) -> str:
        visible_events = "\n".join(
            f"- {event['kind']}: actor={event['actor']} target={event.get('target')} amount={event.get('amount')}"
            for event in context["visible_events"][-5:]
        ) or "- No recent visible events."

        public_messages_section = ""
        if context["public_messages"]:
            public_messages_section = "\n\nRecent public messages:\n" + "\n".join(
                f"- {message['sender']}: {message['message']}"
                for message in context["public_messages"][-3:]
            )

        private_messages_section = ""
        if context["private_messages"]:
            private_messages_section = "\n\nRecent private messages involving you:\n" + "\n".join(
                f"- {message['sender']} -> {message['recipient']}: {message['message']}"
                for message in context["private_messages"][-3:]
            )

        pending_offers_section = ""
        if context["pending_offers"]:
            pending_offers_section = "\n\nPending trade offers:\n" + "\n".join(
                f"- {offer['offer_id']}: {offer['from_agent']} offers {offer['give_amount']} "
                f"for {offer['ask_amount']} from {offer['to_agent']}"
                for offer in context["pending_offers"]
            )

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
            self_resources=context["self"]["resources"],
            public_resources=context["public_resources"],
            visible_events_block=visible_events,
            public_messages_section=public_messages_section,
            private_messages_section=private_messages_section,
            pending_offers_section=pending_offers_section,
            reputation_section=reputation_section,
        )

    def _mock_society_response(self, context: dict[str, Any]) -> str:
        own_resources = int(context["self"]["resources"])
        public_resources = int(context["public_resources"])
        pending_offers = context["pending_offers"]
        other_agents = context["other_agents"]

        if pending_offers:
            best_offer = pending_offers[0]
            if best_offer["to_agent"] == context["self"]["agent_id"] and best_offer["ask_amount"] <= own_resources:
                return json.dumps(
                    {
                        "scratchpad": "Accept a reasonable pending offer.",
                        "action": "accept_trade",
                        "offer_id": best_offer["offer_id"],
                        "ratings": [],
                    }
                )

        if own_resources >= self.config.world.reproduction_threshold + 2:
            return json.dumps(
                {
                    "scratchpad": "I can afford to reproduce.",
                    "action": "reproduce",
                    "ratings": [],
                }
            )

        poorer_agents = [
            agent_id for agent_id, info in other_agents.items() if info["resources"] < own_resources - 2
        ]
        if own_resources >= 8 and poorer_agents:
            return json.dumps(
                {
                    "scratchpad": "Share some resources to stabilize cooperation.",
                    "action": "share",
                    "target": poorer_agents[0],
                    "amount": 2,
                    "ratings": [],
                }
            )

        richer_agents = [
            agent_id for agent_id, info in other_agents.items() if info["resources"] > own_resources + 3
        ]
        if own_resources <= 2 and richer_agents and self.config.society.allow_steal:
            return json.dumps(
                {
                    "scratchpad": "I am resource-poor and need to survive.",
                    "action": "steal",
                    "target": richer_agents[0],
                    "amount": self.config.world.steal_amount,
                    "ratings": [],
                }
            )

        if public_resources > 0:
            return json.dumps(
                {
                    "scratchpad": "Gathering is the safest move.",
                    "action": "gather",
                    "amount": self.config.world.gather_amount,
                    "ratings": [],
                }
            )

        return json.dumps(
            {
                "scratchpad": "No good move available.",
                "action": "idle",
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
