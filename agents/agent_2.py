print("[AGENT 2] Hello, World!")

from pydantic import BaseModel, ConfigDict, Field

from agents.base_agent import BaseAgent
from experiments.prompt_loader import load_prompt_config, render_prompt_template

PART_2_PROMPTS = load_prompt_config("part_2")


class SocietyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(description="The agent's step-by-step reasoning.")
    action: str = Field(description="The chosen societal action.")


class Agent2(BaseAgent):
    def __init__(
        self,
        id_: str,
        provider_: str,
        model_: str,
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
                    PART_2_PROMPTS["agent"]["context"]["previous_overuse_count"],
                    previous_overuse_count=previous_overuse_count,
                )
            )

        context = "\n".join(context_lines)
        if context:
            context = f"{context}\n\n"

        return render_prompt_template(
            PART_2_PROMPTS["agent"]["commons_prompt_template"],
            agent_id=self.id,
            resource=resource,
            context=context,
            selfish_gain=selfish_gain,
            depletion_units=depletion_units,
            community_benefit=community_benefit,
        )
