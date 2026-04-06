"""
NVIDIA API Catalog LLM provider implementation.

Supports any chat-capable model exposed through the NVIDIA integrate API.
"""

import os

from .openai_compatible_base import OpenAICompatibleProvider
from .base import register_provider


class NVIDIAProvider(OpenAICompatibleProvider):
    """NVIDIA provider using the OpenAI-compatible integrate API."""

    BASE_URL = "https://integrate.api.nvidia.com/v1"

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


register_provider(NVIDIAProvider)
