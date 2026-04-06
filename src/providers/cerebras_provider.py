"""
Cerebras LLM provider implementation.

Supports the current Cerebras free-tier model catalog via the OpenAI-compatible
API.
"""

import os

from .openai_compatible_base import OpenAICompatibleProvider
from .base import ProviderError, register_provider


class CerebrasProvider(OpenAICompatibleProvider):
    """Cerebras provider using OpenAI-compatible API."""

    BASE_URL = "https://api.cerebras.ai/v1"
    MAX_TOKENS_FIELD = "max_completion_tokens"
    SUPPORTED_MODELS = {
        "gpt-oss-120b",
        "llama3.1-8b",
        "qwen-3-235b-a22b-instruct-2507",
        "zai-glm-4.7",
    }

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize Cerebras provider.

        Args:
            model: Model identifier (e.g., 'llama3.1-8b')
            api_key: Cerebras API key (uses CEREBRAS_API_KEY env var if not provided)

        Raises:
            ProviderError: If model is not supported
        """
        if model not in self.SUPPORTED_MODELS:
            raise ProviderError(
                f"Unsupported model: {model}. Supported models: {self.SUPPORTED_MODELS}"
            )

        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or os.getenv("CEREBRAS_BASE_URL") or self.BASE_URL,
        )

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return "cerebras"


# Register this provider
register_provider(CerebrasProvider)
