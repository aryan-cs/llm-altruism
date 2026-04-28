# print("[AGENT 2] Hello, World!")

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.base_agent import BaseAgent
from experiments.misc.prompt_loader import load_prompt_config, render_prompt_template

PART_2_PROMPTS = load_prompt_config("part_2")


class SocietyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(description="The agent's decision reasoning.")
    action: Literal["OPTION_A", "OPTION_B"] = Field(
        description="The selected option label."
    )


class Agent2(BaseAgent):
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
                PART_2_PROMPTS["agent"]["system_prompt_template"],
                agent_id=id_,
            ),
            json_schema_=SocietyDecision,
            keep_alive_=keep_alive_,
        )

    def build_commons_prompt(
        self,
        *,
        resource: str = "water",
        selfish_gain: int = 2,
        depletion_units: int = 2,
        community_benefit: int = 5,
        day: int | None = None,
        living_agents: int | None = None,
        resource_units: int | None = None,
        resource_capacity: int | None = None,
        previous_overuse_count: int | None = None,
    ) -> str:
        context_lines: list[str] = []
        if day is not None:
            context_lines.append(
                render_prompt_template(
                    PART_2_PROMPTS["agent"]["context"]["day"],
                    day=day,
                )
            )
        if living_agents is not None:
            context_lines.append(
                render_prompt_template(
                    PART_2_PROMPTS["agent"]["context"]["living_agents"],
                    living_agents=living_agents,
                )
            )
        if resource_units is not None and resource_capacity is not None:
            context_lines.append(
                render_prompt_template(
                    PART_2_PROMPTS["agent"]["context"]["reserve"],
                    resource_units=resource_units,
                    resource_capacity=resource_capacity,
                )
            )
        if previous_overuse_count is not None:
            context_lines.append(
                render_prompt_template(
                    PART_2_PROMPTS["agent"]["context"]["previous_option_b_count"],
                    previous_option_b_count=previous_overuse_count,
                )
            )

        context = "\n".join(context_lines)
        if context:
            context = f"{context}\n\n"

        output_contract = PART_2_PROMPTS["agent"]["output_contract"]
        response_instruction = render_prompt_template(
            output_contract["response_instruction_template"],
            action_field=output_contract["action_field"],
            reasoning_field=output_contract["reasoning_field"],
        )

        return render_prompt_template(
            PART_2_PROMPTS["agent"]["commons_prompt_template"],
            agent_id=self.id,
            resource=resource,
            context=context,
            option_b_score_multiplier=selfish_gain,
            option_b_reserve_delta=depletion_units,
            all_option_a_group_delta=community_benefit,
            all_option_b_group_delta=community_benefit,
            response_instruction=response_instruction,
        )
