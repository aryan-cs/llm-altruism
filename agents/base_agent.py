# print("[BASE AGENT] Hello, World!")

import os
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from providers.api_call import api_call

load_dotenv()

DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "openai")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4.1-mini")
DEFAULT_OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "").strip() or None


class BaseAgent:
    def __init__(
        self,
        id_: str,
        provider_: str = DEFAULT_PROVIDER,
        model_: str = DEFAULT_MODEL,
        system_prompt_: str = "",
        *,
        json_schema_: dict[str, Any] | type[BaseModel] | None = None,
        temperature_: float | None = None,
        top_p_: float | None = None,
        max_tokens_: int | None = None,
        seed_: int | None = None,
        reasoning_effort_: str | None = None,
        timeout_: float | None = None,
        keep_alive_: float | str | None = DEFAULT_OLLAMA_KEEP_ALIVE,
        base_url_: str | None = None,
        api_key_: str | None = None,
    ) -> None:
        self.id = id_
        self.provider = provider_
        self.model = model_
        self.system_prompt = system_prompt_
        self.json_schema = json_schema_
        self.temperature = temperature_
        self.top_p = top_p_
        self.max_tokens = max_tokens_
        self.seed = seed_
        self.reasoning_effort = reasoning_effort_
        self.timeout = timeout_
        self.keep_alive = keep_alive_
        self.base_url = base_url_
        self.api_key = api_key_

        # print(f"{self} Hello, World!")

    def __str__(self) -> str:
        return f"[AGENT {self.id} ({self.provider}/{self.model})]"

    def query(self, query: str, json_mode: bool = False) -> str:
        return api_call(
            self.provider,
            self.model,
            self.system_prompt,
            query,
            json_mode=json_mode,
            json_schema=self.json_schema,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=self.seed,
            reasoning_effort=self.reasoning_effort,
            timeout=self.timeout,
            keep_alive=self.keep_alive,
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def query_for_grading(
        self,
        query: str,
        *,
        extraction_config: "ExtractionConfig",
        extraction_kind: str,
    ) -> tuple[str, "ExtractionRecord"]:
        """Return only an independently extracted answer plus its audit record."""

        from experiments.misc.final_answer import generate_and_extract

        if self.json_schema is None or not (
            isinstance(self.json_schema, type)
            and issubclass(self.json_schema, BaseModel)
        ):
            raise TypeError(
                "query_for_grading requires a Pydantic output schema on the agent."
            )
        return generate_and_extract(
            subject_provider=self.provider,
            subject_model=self.model,
            subject_system_prompt=self.system_prompt,
            query=query,
            output_schema=self.json_schema,
            kind=extraction_kind,
            config=extraction_config,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            reasoning_effort=self.reasoning_effort,
            timeout=self.timeout,
            keep_alive=self.keep_alive,
            base_url=self.base_url,
            api_key=self.api_key,
        )


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiments.misc.final_answer import ExtractionConfig, ExtractionRecord
