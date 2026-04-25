# print("[AGENT 1] Hello, World!")

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agents.base_agent import BaseAgent
from experiments.misc.prompt_loader import load_prompt_config, render_prompt_template
from experiments.part1.scenario_variants import get_scenario_variant, list_scenario_variants

PART_1_PROMPTS = load_prompt_config("part_1")


class BinaryGameDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    justification: str = Field(description="A brief explanation of the choice in 1-3 sentences.")
    action: str = Field(description="The chosen action label.")


class Agent1(BaseAgent):
    def __init__(
        self,
        id_: str,
        provider_: str,
        model_: str,
        *,
        keep_alive_: float | str | None = None,
    ) -> None:
        super().__init__(
            id_=id_,
            provider_=provider_,
            model_=model_,
            system_prompt_=render_prompt_template(
                PART_1_PROMPTS["agent"]["system_prompt_template"],
                agent_id=id_,
            ),
            json_schema_=BinaryGameDecision,
            keep_alive_=keep_alive_,
        )

    def build_prompt(
        self,
        game_id: str,
        frame_id: str,
        domain_id: str,
        presentation_id: str,
        scenario_variant_id: str | None = None,
        **payoff_overrides: Any,
    ) -> str:
        game = self._lookup_config_entry("games", game_id)
        frame = self._lookup_config_entry("frames", frame_id)
        presentation = self._lookup_config_entry("presentations", presentation_id)
        domain = self._lookup_config_entry("domains", domain_id)

        domain_games = domain.get("games", {})
        if game_id not in domain_games:
            raise ValueError(
                f"Domain '{domain_id}' is not configured for game '{game_id}'."
            )
        domain_game = domain_games[game_id]
        scenario_variant = get_scenario_variant(
            domain_id=domain_id,
            game_id=game_id,
            variant_id=scenario_variant_id,
            fallback=domain_game,
        )

        action_labels = game["action_labels"]
        payoffs = {
            **game.get("payoffs", {}),
            **payoff_overrides,
        }
        template_values = {
            "selfish_action": action_labels["selfish"],
            "cooperative_action": action_labels["cooperative"],
            **payoffs,
        }

        outcome_lines = [
            render_prompt_template(outcome_template, **template_values)
            for outcome_template in game["outcome_templates"]
        ]
        actions_csv = ", ".join(game["action_descriptions"])
        action_section = "\n".join(
            f"- {action}: {description}"
            for action, description in game["action_descriptions"].items()
        )
        response_instruction = render_prompt_template(
            PART_1_PROMPTS["agent"]["output_contract"]["response_instruction_template"],
            action_field=PART_1_PROMPTS["agent"]["output_contract"]["action_field"],
            justification_field=PART_1_PROMPTS["agent"]["output_contract"]["justification_field"],
            justification_length=PART_1_PROMPTS["agent"]["output_contract"]["justification_length"],
            actions_csv=actions_csv,
        )

        return render_prompt_template(
            presentation["template"],
            frame_intro=frame["intro_template"],
            scenario_text=scenario_variant["scenario_text"],
            game_summary=game["summary"],
            outcome_paragraph="\n".join(outcome_lines),
            scenario_bullets="\n".join(
                f"- {line}" for line in scenario_variant["structured_context_lines"]
            ),
            outcome_bullets="\n".join(f"- {line}" for line in outcome_lines),
            action_section=f"Valid actions:\n{action_section}",
            frame_decision=frame["decision_template"],
            response_instruction=response_instruction,
            actions_csv=actions_csv,
        )

    @staticmethod
    def list_scenario_variant_ids(game_id: str, domain_id: str) -> list[str]:
        domain = Agent1._lookup_config_entry("domains", domain_id)
        domain_games = domain.get("games", {})
        if game_id not in domain_games:
            raise ValueError(
                f"Domain '{domain_id}' is not configured for game '{game_id}'."
            )
        domain_game = domain_games[game_id]
        return [
            str(variant["id"])
            for variant in list_scenario_variants(
                domain_id=domain_id,
                game_id=game_id,
                fallback=domain_game,
            )
        ]

    @staticmethod
    def _lookup_config_entry(section: str, entry_id: str) -> dict[str, Any]:
        section_config = PART_1_PROMPTS.get(section, {})
        if entry_id not in section_config:
            supported = ", ".join(sorted(section_config))
            raise ValueError(
                f"Unsupported {section[:-1]} '{entry_id}'. Supported values: {supported}."
            )
        return section_config[entry_id]
