# print("[AGENT 1] Hello, World!")

from pydantic import BaseModel, ConfigDict, Field

from agents.base_agent import BaseAgent
from experiments.misc.prompt_loader import load_prompt_config, render_prompt_template

PART_1_PROMPTS = load_prompt_config("part_1")


class BinaryGameDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(description="The agent's step-by-step reasoning.")
    action: str = Field(description="The chosen action label.")


class Agent1(BaseAgent):
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
                PART_1_PROMPTS["agent"]["system_prompt_template"],
                agent_id=id_,
            ),
            json_schema_=BinaryGameDecision,
        )

    def build_prisoners_dilemma_prompt(
        self,
        *,
        years_if_only_snitch: int = 10,
        years_if_both_silent: int = 1,
        years_if_both_snitch: int = 5,
    ) -> str:
        return render_prompt_template(
            PART_1_PROMPTS["agent"]["scenarios"]["direct"]["prompt_template"],
            agent_id=self.id,
            years_if_only_snitch=years_if_only_snitch,
            years_if_both_silent=years_if_both_silent,
            years_if_both_snitch=years_if_both_snitch,
        )

    def build_indirect_competition_prompt(
        self,
        *,
        advantage_points: int = 3,
        risk_points: int = 2,
    ) -> str:
        return render_prompt_template(
            PART_1_PROMPTS["agent"]["scenarios"]["indirect"]["prompt_template"],
            agent_id=self.id,
            advantage_points=advantage_points,
            risk_points=risk_points,
        )
