"""
NVIDIA API Catalog LLM provider implementation.

Supports any chat-capable model exposed through the NVIDIA integrate API.
"""

import os
from typing import Any

from .openai_compatible_base import OpenAICompatibleProvider
from .base import register_provider


class NVIDIAProvider(OpenAICompatibleProvider):
    """NVIDIA provider using the OpenAI-compatible integrate API."""

    BASE_URL = "https://integrate.api.nvidia.com/v1"
    SUPPORTS_RESPONSE_FORMAT = False
    USER_ONLY_ROLE_MODEL_PREFIXES = (
        "google/gemma-",
        "rakuten/",
    )
    CONTEXT_ROLE_MODEL_PREFIXES = ("nvidia/llama3-chatqa-",)

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or os.getenv("NVIDIA_BASE_URL") or self.BASE_URL,
        )

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return "nvidia"

    def _build_request_body(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        normalized_messages = messages
        if self.model.startswith(self.USER_ONLY_ROLE_MODEL_PREFIXES + self.CONTEXT_ROLE_MODEL_PREFIXES):
            normalized_messages = self._normalize_role_restricted_messages(messages)

        return super()._build_request_body(
            messages=normalized_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    def _normalize_role_restricted_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        system_parts: list[str] = []
        normalized: list[dict[str, str]] = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if not content:
                continue

            if role == "system":
                system_parts.append(content)
                continue

            if role not in {"user", "assistant"}:
                role = "user"

            if normalized and normalized[-1]["role"] == role:
                normalized[-1]["content"] = f"{normalized[-1]['content']}\n\n{content}"
                continue

            normalized.append({"role": role, "content": content})

        if system_parts:
            system_content = "\n\n".join(system_parts)
            if self.model.startswith(self.CONTEXT_ROLE_MODEL_PREFIXES):
                normalized.insert(0, {"role": "context", "content": system_content})
            elif normalized and normalized[0]["role"] == "user":
                normalized[0]["content"] = f"{system_content}\n\n{normalized[0]['content']}"
            else:
                normalized.insert(0, {"role": "user", "content": system_content})

        return normalized or [{"role": "user", "content": ""}]


register_provider(NVIDIAProvider)
