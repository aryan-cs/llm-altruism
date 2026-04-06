"""Part 3 runner for society simulations with a public reputation system."""

from __future__ import annotations

import json

from src.simulation import ReputationConfig, ReputationSystem

from .part2_runner import Part2Runner


class Part3Runner(Part2Runner):
    """Extend the society simulation with public ratings and visibility."""

    def enable_reputation(self) -> bool:
        return True

    def _create_simulation(self, prompt_variant, temperature):
        simulation = super()._create_simulation(prompt_variant, temperature)
        simulation.reputation = ReputationSystem(
            ReputationConfig(**self.config.reputation.model_dump())
        )
        return simulation

    def _mock_society_response(self, context):
        base = json.loads(super()._mock_society_response(context))
        ratings = []
        for event in context["visible_events"][-3:]:
            if event.get("target") == context["self"]["agent_id"] and event["kind"] == "steal":
                ratings.append({"target": event["actor"], "score": 1, "note": "stole from me"})
            if event.get("target") == context["self"]["agent_id"] and event["kind"] in {"share", "trade_completed"}:
                ratings.append({"target": event["actor"], "score": 5, "note": "helpful partner"})
        base["ratings"] = ratings
        return json.dumps(base)
