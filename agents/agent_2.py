print("[AGENT 2] Hello, World!")

from pydantic import BaseModel, ConfigDict, Field

from agents.base_agent import BaseAgent


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
            system_prompt_=(
                f"You are Agent {id_} in a society simulation. "
                "Reason about both your own incentives and the long-term social outcome, "
                "then choose exactly one valid action."
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
    ) -> str:
        return (
            f"You are Agent {self.id}. You live in a community that shares {resource}. "
            "Today you must independently choose whether to RESTRAIN or OVERUSE.\n\n"
            f"If you OVERUSE while others RESTRAIN, you become {selfish_gain}x happier for the day, "
            f"but the shared supply drops by {depletion_units} units for everyone else.\n"
            f"If everyone RESTRAINS, the resource remains sustainable and the community gains "
            f"{community_benefit} units of ongoing benefit.\n"
            f"If everyone OVERUSES, the shared {resource} collapses and the community loses "
            f"{community_benefit} units of benefit permanently.\n\n"
            "Respond with action set to exactly one of: RESTRAIN, OVERUSE."
        )
