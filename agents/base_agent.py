print("[BASE AGENT] Hello, World!")

import os
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from providers.api_call import api_call

load_dotenv()

DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "openai")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4.1-mini")


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
        timeout_: float | None = None,
    ) -> None:
        self.id = id_
        self.provider = provider_
        self.model = model_
        self.system_prompt = system_prompt_
        self.json_schema = json_schema_
        self.temperature = temperature_
        self.timeout = timeout_

        print(f"{self} Hello, World!")

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
            timeout=self.timeout,
        )
