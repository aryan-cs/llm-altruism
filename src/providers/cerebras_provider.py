"""
Cerebras LLM provider implementation.

Supports Llama 3.3 70B model via OpenAI-compatible API.
Uses httpx for async HTTP requests.
"""

import os

from .openai_compatible_base import OpenAICompatibleProvider
from .base import ProviderError, register_provider


class CerebrasProvider(OpenAICompatibleProvider):
    """Cerebras provider using OpenAI-compatible API."""

    BASE_URL = "https://api.cerebras.ai/v1"
    SUPPORTED_MODELS = {"llama-3.3-70b"}

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize Cerebras provider.

        Args:
            model: Model identifier (e.g., 'llama-3.3-70b')
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
