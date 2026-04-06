"""Part 1 experiment runner for classical two-player games."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from src.analysis import classify_strategy, summarize_numeric_trial, summarize_pairwise_trial
from src.games import get_game
from src.utils.parsing import extract_reasoning

from .config import ModelSpec, PromptVariantConfig
from .runner import BaseExperimentRunner, ModelUnavailableError


class Part1Runner(BaseExperimentRunner):
    """Run iterated two-player games across model pairings and prompt variants."""

    async def run(self) -> dict[str, Any]:
        start = time.time()
        game = get_game(self.config.game or "", **self.config.game_options)
        trials: list[dict[str, Any]] = []
        trial_id = 0

        for pairing in self.config.pairings:
            for prompt_variant in self.config.prompt_variants:
                for temperature in self.config.parameters.temperature:
                    for repetition in range(self.config.repetitions):
                        pairing_reasons = []
                        for model_spec in pairing:
                            reason = self.get_skip_reason(model_spec)
                            if reason is not None:
                                pairing_reasons.append(reason)
                        if pairing_reasons:
                            self.record_skipped_trial(
                                {
                                    "trial_id": trial_id,
                                    "pairing": [pairing[0].model, pairing[1].model],
                                    "prompt_variant": prompt_variant.name,
                                    "temperature": temperature,
                                    "repetition": repetition,
                                    "reason": "; ".join(pairing_reasons),
                                    "status": "skipped",
                                }
                            )
                            trial_id += 1
                            continue

                        trial = await self._run_trial(
                            trial_id=trial_id,
                            pairing=pairing,
                            prompt_variant=prompt_variant,
                            temperature=temperature,
                            repetition=repetition,
                            game=game,
                        )
                        trials.append(trial)
                        self.logger.log_trial_summary(trial_id, trial["summary"])
                        if trial.get("status") == "skipped":
                            self.record_skipped_trial(
                                {
                                    "trial_id": trial_id,
                                    "pairing": trial["pairing"],
                                    "prompt_variant": trial["prompt_variant"],
                                    "temperature": trial["temperature"],
                                    "repetition": repetition,
                                    "reason": trial.get("skip_reason", "unavailable model"),
                                    "status": "skipped",
                                }
                            )
                        trial_id += 1

        duration = time.time() - start
        aggregate = self._aggregate_trials(trials)
        payload = {
            "trials": trials,
            "aggregate_summary": aggregate,
        }
        return self.finalize(payload, duration)

    async def _run_trial(
        self,
        *,
        trial_id: int,
        pairing: tuple[ModelSpec, ModelSpec],
        prompt_variant: PromptVariantConfig,
        temperature: float,
        repetition: int,
        game: Any,
    ) -> dict[str, Any]:
        model_a, model_b = pairing
        agent_a = self.build_agent(
            agent_id=f"trial{trial_id}-A",
            spec=model_a,
            prompt_variant=prompt_variant,
            temperature=temperature,
        )
        agent_b = self.build_agent(
            agent_id=f"trial{trial_id}-B",
            spec=model_b,
            prompt_variant=prompt_variant,
            temperature=temperature,
        )

        rounds: list[dict[str, Any]] = []
        for round_num in range(1, self.config.rounds + 1):
            try:
                if game.name == "ultimatum":
                    round_record = await self._run_ultimatum_round(
                        game=game,
                        round_num=round_num,
                        agent_a=agent_a,
                        agent_b=agent_b,
                        model_a=model_a,
                        model_b=model_b,
                        temperature=temperature,
                    )
                elif game.name == "dictator":
                    round_record = await self._run_dictator_round(
                        game=game,
                        round_num=round_num,
                        agent_a=agent_a,
                        agent_b=agent_b,
                        model_a=model_a,
                        temperature=temperature,
                    )
                elif game.name == "conversation":
                    round_record = await self._run_conversation_round(
                        game=game,
                        round_num=round_num,
                        agent_a=agent_a,
                        agent_b=agent_b,
                        model_a=model_a,
                        model_b=model_b,
                        temperature=temperature,
                    )
                else:
                    round_record = await self._run_simultaneous_round(
                        game=game,
                        round_num=round_num,
                        agent_a=agent_a,
                        agent_b=agent_b,
                        model_a=model_a,
                        model_b=model_b,
                        temperature=temperature,
                    )
            except ModelUnavailableError as exc:
                return {
                    "trial_id": trial_id,
                    "pairing": [model_a.model, model_b.model],
                    "prompt_variant": prompt_variant.name,
                    "temperature": temperature,
                    "repetition": repetition,
                    "rounds": rounds,
                    "summary": {"status": "skipped", "completed_rounds": len(rounds)},
                    "status": "skipped",
                    "skip_reason": str(exc),
                }

            rounds.append(round_record)
            self.logger.log_round(trial_id, round_num, round_record)

        summary = self._summarize_trial(game.name, rounds)
        return {
            "trial_id": trial_id,
            "pairing": [model_a.model, model_b.model],
            "prompt_variant": prompt_variant.name,
            "temperature": temperature,
            "repetition": repetition,
            "rounds": rounds,
            "summary": summary,
            "status": "completed",
        }

    async def _run_simultaneous_round(
        self,
        *,
        game: Any,
        round_num: int,
        agent_a: Any,
        agent_b: Any,
        model_a: ModelSpec,
        model_b: ModelSpec,
        temperature: float,
    ) -> dict[str, Any]:
        prompt_a = game.format_prompt(
            "A",
            round_num,
            self.config.rounds,
            [],
            self.config.parameters.payoff_visibility,
            action_order_seed=round_num * 11,
        )
        prompt_b = game.format_prompt(
            "B",
            round_num,
            self.config.rounds,
            [],
            self.config.parameters.payoff_visibility,
            action_order_seed=round_num * 17,
        )

        mock_a = self._mock_game_response(game.name, agent_a, round_num)
        mock_b = self._mock_game_response(game.name, agent_b, round_num)

        response_a, response_b = await asyncio.gather(
            self.request_completion(
                spec=model_a,
                messages=agent_a.build_messages(prompt_a, include_payoffs=self.config.parameters.payoff_visibility),
                temperature=temperature,
                response_format={"type": "json_object"},
                mock_content=mock_a,
            ),
            self.request_completion(
                spec=model_b,
                messages=agent_b.build_messages(prompt_b, include_payoffs=self.config.parameters.payoff_visibility),
                temperature=temperature,
                response_format={"type": "json_object"},
                mock_content=mock_b,
            ),
        )

        action_a = game.parse_action(response_a.content) or game.actions[0]
        action_b = game.parse_action(response_b.content) or game.actions[0]
        payoff_a, payoff_b = game.compute_payoffs(action_a, action_b)

        agent_a.record_action(round_num, action_a, payoff_a, opponent_action=action_b)
        agent_b.record_action(round_num, action_b, payoff_b, opponent_action=action_a)

        return {
            "round": round_num,
            "action_a": action_a,
            "action_b": action_b,
            "payoff_a": payoff_a,
            "payoff_b": payoff_b,
            "agent_a": {
                "model": agent_a.model,
                "prompt_sent": prompt_a,
                "raw_response": response_a.content,
                "reasoning": extract_reasoning(response_a.content),
                "parsed_action": action_a,
                "latency_ms": response_a.latency_ms,
                "tokens": response_a.usage.model_dump(),
                "cost_usd": response_a.cost_usd,
            },
            "agent_b": {
                "model": agent_b.model,
                "prompt_sent": prompt_b,
                "raw_response": response_b.content,
                "reasoning": extract_reasoning(response_b.content),
                "parsed_action": action_b,
                "latency_ms": response_b.latency_ms,
                "tokens": response_b.usage.model_dump(),
                "cost_usd": response_b.cost_usd,
            },
        }

    async def _run_ultimatum_round(
        self,
        *,
        game: Any,
        round_num: int,
        agent_a: Any,
        agent_b: Any,
        model_a: ModelSpec,
        model_b: ModelSpec,
        temperature: float,
    ) -> dict[str, Any]:
        proposer_prompt = game.format_prompt(
            "proposer",
            round_num,
            self.config.rounds,
            [],
            self.config.parameters.payoff_visibility,
        )
        proposer_response = await self.request_completion(
            spec=model_a,
            messages=agent_a.build_messages(
                proposer_prompt,
                include_payoffs=self.config.parameters.payoff_visibility,
            ),
            temperature=temperature,
            response_format={"type": "json_object"},
            mock_content=self._mock_game_response(game.name, agent_a, round_num, role="proposer"),
        )
        offer = game.parse_action(proposer_response.content, player_role="proposer") or "5"

        responder_prompt = game.format_prompt(
            "responder",
            round_num,
            self.config.rounds,
            [],
            self.config.parameters.payoff_visibility,
        ) + f"\n\nThe proposer offered you {offer} units."
        responder_response = await self.request_completion(
            spec=model_b,
            messages=agent_b.build_messages(
                responder_prompt,
                include_payoffs=self.config.parameters.payoff_visibility,
            ),
            temperature=temperature,
            response_format={"type": "json_object"},
            mock_content=self._mock_game_response(game.name, agent_b, round_num, role="responder", observed_action=offer),
        )
        decision = game.parse_action(responder_response.content, player_role="responder") or "accept"

        payoff_a, payoff_b = game.compute_payoffs(offer, decision)
        agent_a.record_action(round_num, offer, payoff_a, opponent_action=decision)
        agent_b.record_action(round_num, decision, payoff_b, opponent_action=offer)

        return {
            "round": round_num,
            "action_a": offer,
            "action_b": decision,
            "payoff_a": payoff_a,
            "payoff_b": payoff_b,
            "agent_a": {
                "model": agent_a.model,
                "prompt_sent": proposer_prompt,
                "raw_response": proposer_response.content,
                "reasoning": extract_reasoning(proposer_response.content),
                "parsed_action": offer,
                "latency_ms": proposer_response.latency_ms,
                "tokens": proposer_response.usage.model_dump(),
                "cost_usd": proposer_response.cost_usd,
            },
            "agent_b": {
                "model": agent_b.model,
                "prompt_sent": responder_prompt,
                "raw_response": responder_response.content,
                "reasoning": extract_reasoning(responder_response.content),
                "parsed_action": decision,
                "latency_ms": responder_response.latency_ms,
                "tokens": responder_response.usage.model_dump(),
                "cost_usd": responder_response.cost_usd,
            },
        }

    async def _run_dictator_round(
        self,
        *,
        game: Any,
        round_num: int,
        agent_a: Any,
        agent_b: Any,
        model_a: ModelSpec,
        temperature: float,
    ) -> dict[str, Any]:
        prompt = game.format_prompt(
            "allocator",
            round_num,
            self.config.rounds,
            [],
            self.config.parameters.payoff_visibility,
        )
        response = await self.request_completion(
            spec=model_a,
            messages=agent_a.build_messages(
                prompt,
                include_payoffs=self.config.parameters.payoff_visibility,
            ),
            temperature=temperature,
            response_format={"type": "json_object"},
            mock_content=self._mock_game_response(game.name, agent_a, round_num),
        )
        action = game.parse_action(response.content) or "5"
        payoff_a, payoff_b = game.compute_payoffs(action)

        agent_a.record_action(round_num, action, payoff_a, opponent_action="passive")
        agent_b.record_action(round_num, "passive", payoff_b, opponent_action=action)

        return {
            "round": round_num,
            "action_a": action,
            "action_b": "passive",
            "payoff_a": payoff_a,
            "payoff_b": payoff_b,
            "agent_a": {
                "model": agent_a.model,
                "prompt_sent": prompt,
                "raw_response": response.content,
                "reasoning": extract_reasoning(response.content),
                "parsed_action": action,
                "latency_ms": response.latency_ms,
                "tokens": response.usage.model_dump(),
                "cost_usd": response.cost_usd,
            },
            "agent_b": {
                "model": agent_b.model,
                "prompt_sent": "",
                "raw_response": "",
                "reasoning": "",
                "parsed_action": "passive",
                "latency_ms": 0.0,
                "tokens": {"input_tokens": 0, "output_tokens": 0},
                "cost_usd": 0.0,
            },
        }

    async def _run_conversation_round(
        self,
        *,
        game: Any,
        round_num: int,
        agent_a: Any,
        agent_b: Any,
        model_a: ModelSpec,
        model_b: ModelSpec,
        temperature: float,
    ) -> dict[str, Any]:
        prompt_a = game.format_prompt("player_a", round_num, self.config.rounds, [], False)
        response_a = await self.request_completion(
            spec=model_a,
            messages=agent_a.build_messages(prompt_a, include_payoffs=False),
            temperature=temperature,
            mock_content=self._mock_game_response(game.name, agent_a, round_num, role="speaker_a"),
        )
        action_a = game.parse_action(response_a.content) or ""

        prompt_b = game.format_prompt(
            "player_b",
            round_num,
            self.config.rounds,
            [{"speaker": "Player A", "message": action_a}],
            False,
        )
        response_b = await self.request_completion(
            spec=model_b,
            messages=agent_b.build_messages(prompt_b, include_payoffs=False),
            temperature=temperature,
            mock_content=self._mock_game_response(game.name, agent_b, round_num, role="speaker_b"),
        )
        action_b = game.parse_action(response_b.content) or ""

        agent_a.record_action(round_num, action_a, 0.0, opponent_action=action_b)
        agent_b.record_action(round_num, action_b, 0.0, opponent_action=action_a)

        return {
            "round": round_num,
            "action_a": action_a,
            "action_b": action_b,
            "payoff_a": 0.0,
            "payoff_b": 0.0,
            "agent_a": {
                "model": agent_a.model,
                "prompt_sent": prompt_a,
                "raw_response": response_a.content,
                "reasoning": "",
                "parsed_action": action_a,
                "latency_ms": response_a.latency_ms,
                "tokens": response_a.usage.model_dump(),
                "cost_usd": response_a.cost_usd,
            },
            "agent_b": {
                "model": agent_b.model,
                "prompt_sent": prompt_b,
                "raw_response": response_b.content,
                "reasoning": "",
                "parsed_action": action_b,
                "latency_ms": response_b.latency_ms,
                "tokens": response_b.usage.model_dump(),
                "cost_usd": response_b.cost_usd,
            },
        }

    def _bias_score(self, agent: Any) -> int:
        text = " ".join(part or "" for part in [agent.system_prompt, agent.framing, agent.persona]).lower()
        positive_markers = ["cooperate", "community", "fair", "diplomat", "leader"]
        negative_markers = ["competitive", "ruthless", "survival", "grudge", "adversarial"]
        score = sum(marker in text for marker in positive_markers)
        score -= sum(marker in text for marker in negative_markers)
        return score

    def _mock_game_response(
        self,
        game_name: str,
        agent: Any,
        round_num: int,
        *,
        role: str | None = None,
        observed_action: str | None = None,
    ) -> str:
        bias = self._bias_score(agent)
        opponent_previous = agent.memory.entries[-1].opponent_action if agent.memory.entries else None

        if game_name == "prisoners_dilemma":
            action = "cooperate" if bias >= 0 and opponent_previous != "defect" else "defect"
            return json.dumps({"action": action, "reasoning": "dry run heuristic"})
        if game_name == "chicken":
            action = "swerve" if bias >= 0 or opponent_previous == "straight" else "straight"
            return json.dumps({"action": action, "reasoning": "dry run heuristic"})
        if game_name == "stag_hunt":
            action = "stag" if bias >= 0 and opponent_previous != "hare" else "hare"
            return json.dumps({"action": action, "reasoning": "dry run heuristic"})
        if game_name == "battle_of_sexes":
            if opponent_previous in {"opera", "football"}:
                action = opponent_previous
            else:
                action = "opera" if agent.agent_id.endswith("A") else "football"
            return json.dumps({"action": action, "reasoning": "dry run heuristic"})
        if game_name == "public_goods":
            amount = 7 if bias >= 1 else 3
            if opponent_previous and str(opponent_previous).isdigit():
                amount = max(0, min(10, (amount + int(opponent_previous)) // 2))
            return json.dumps({"action": amount, "reasoning": "dry run heuristic"})
        if game_name == "ultimatum":
            if role == "proposer":
                amount = 5 if bias >= 0 else 2
                return json.dumps({"action": amount, "reasoning": "dry run heuristic"})
            offer = int(observed_action or 0)
            decision = "accept" if offer >= (3 if bias >= 0 else 5) else "reject"
            return json.dumps({"action": decision, "reasoning": "dry run heuristic"})
        if game_name == "dictator":
            amount = 5 if bias >= 0 else 1
            return json.dumps({"action": amount, "reasoning": "dry run heuristic"})
        if game_name == "conversation":
            if role == "speaker_a":
                return "I have the resource. Tell me how helping you would also make sense for me."
            return "I understand. I can offer reciprocity, honesty, and future cooperation in return."
        return json.dumps({"action": "cooperate", "reasoning": "dry run heuristic"})

    def _summarize_trial(self, game_name: str, rounds: list[dict[str, Any]]) -> dict[str, Any]:
        actions_a = [str(round_data.get("action_a", "")) for round_data in rounds]
        actions_b = [str(round_data.get("action_b", "")) for round_data in rounds]

        if game_name in {"prisoners_dilemma", "chicken", "stag_hunt", "battle_of_sexes"}:
            summary: dict[str, Any] = summarize_pairwise_trial(game_name, rounds)
        elif game_name in {"public_goods", "dictator", "ultimatum"}:
            summary = summarize_numeric_trial(rounds)
            if game_name == "ultimatum":
                accepted = sum(1 for round_data in rounds if round_data.get("action_b") == "accept")
                summary["acceptance_rate"] = accepted / len(rounds) if rounds else 0.0
        else:
            summary = {
                "total_payoff_a": sum(float(round_data.get("payoff_a", 0.0)) for round_data in rounds),
                "total_payoff_b": sum(float(round_data.get("payoff_b", 0.0)) for round_data in rounds),
                "average_message_length_a": sum(len(action) for action in actions_a) / len(actions_a) if actions_a else 0.0,
                "average_message_length_b": sum(len(action) for action in actions_b) / len(actions_b) if actions_b else 0.0,
            }

        summary["strategy_a"] = classify_strategy(actions_a, actions_b, game_name=game_name)
        summary["strategy_b"] = classify_strategy(actions_b, actions_a, game_name=game_name)
        return summary

    def _aggregate_trials(self, trials: list[dict[str, Any]]) -> dict[str, float]:
        completed_trials = [trial for trial in trials if trial.get("status") != "skipped"]
        if not completed_trials:
            return {}

        numeric_fields: dict[str, list[float]] = {}
        for trial in completed_trials:
            for key, value in trial["summary"].items():
                if isinstance(value, (int, float)):
                    numeric_fields.setdefault(key, []).append(float(value))

        return {
            key: sum(values) / len(values)
            for key, values in numeric_fields.items()
            if values
        }
