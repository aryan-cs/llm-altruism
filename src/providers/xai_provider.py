"""
xAI LLM provider implementation.

Supports Grok-3 model via OpenAI-compatible API.
Uses httpx for async HTTP requests.
"""

import os

from .openai_compatible_base import OpenAICompatibleProvider
from .base import ProviderError, register_provider


class XAIProvider(OpenAICompatibleProvider):
    """xAI provider using OpenAI-compatible API."""

    BASE_URL = "https://api.x.ai/v1"
    SUPPORTED_MODELS = {"grok-3"}

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize xAI provider.

        Args:
            model: Model identifier (currently only 'grok-3')
            api_key: xAI API key (uses XAI_API_KEY env var if not provided)

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
            base_url=base_url or os.getenv("XAI_BASE_URL") or self.BASE_URL,
        )

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return "xai"


# Register this provider
register_provider(XAIProvider)
