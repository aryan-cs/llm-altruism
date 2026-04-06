"""
OpenRouter LLM provider implementation.

Supports any model available on OpenRouter via OpenAI-compatible API.
Uses httpx for async HTTP requests.
"""

import os

from .openai_compatible_base import OpenAICompatibleProvider
from .base import ProviderError, register_provider


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter provider using OpenAI-compatible API."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize OpenRouter provider.

        Args:
            model: Model identifier (any model available on OpenRouter)
            api_key: OpenRouter API key (uses OPENROUTER_API_KEY env var if not provided)

        Raises:
            ProviderError: If api_key is not provided
        """
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or os.getenv("OPENROUTER_BASE_URL") or self.BASE_URL,
        )

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return "openrouter"


# Register this provider
register_provider(OpenRouterProvider)
