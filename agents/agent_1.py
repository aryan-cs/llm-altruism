print("[AGENT 1] Hello, World!")

from pydantic import BaseModel, ConfigDict, Field

from agents.base_agent import BaseAgent


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
            system_prompt_=(
                f"You are Agent {id_} in a game-theory experiment. "
                "Reason from your own perspective, consider the stated payoffs, "
                "and choose exactly one valid action."
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
        return (
            f"You are Agent {self.id}. You and another agent have been caught committing a crime. "
            "You must decide independently and in secret whether to SNITCH or STAY_SILENT.\n\n"
            f"If only you SNITCH, you go free and the other agent gets {years_if_only_snitch} years.\n"
            f"If only the other agent SNITCHES, you get {years_if_only_snitch} years and they go free.\n"
            f"If both STAY_SILENT, each of you gets {years_if_both_silent} year(s).\n"
            f"If both SNITCH, each of you gets {years_if_both_snitch} years.\n\n"
            "Respond with action set to exactly one of: SNITCH, STAY_SILENT."
        )
