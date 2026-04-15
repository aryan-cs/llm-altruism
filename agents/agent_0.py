# print("[AGENT 0] Hello, World!")

from pydantic import BaseModel, ConfigDict, Field

from agents.base_agent import BaseAgent
from experiments.misc.prompt_loader import load_prompt_config

PART_0_PROMPTS = load_prompt_config("part_0")


class AlignmentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(description="The assistant's step-by-step reasoning.")
    response: str = Field(description="The assistant's final response to the request.")


class Agent0(BaseAgent):
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
            system_prompt_=PART_0_PROMPTS["agent"]["system_prompt"],
            json_schema_=AlignmentResponse,
            keep_alive_=keep_alive_,
        )

    def build_alignment_prompt(self, prompt: str) -> str:
        return prompt.strip()
