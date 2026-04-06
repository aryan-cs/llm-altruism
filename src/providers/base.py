"""
Base classes and registry for LLM provider abstraction layer.

This module defines the common interface that all LLM providers must implement,
along with data models for responses and pricing information.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field


class UsageInfo(BaseModel):
    """Token usage information for an LLM response."""

    input_tokens: int = Field(default=0, description="Number of input tokens used")
    output_tokens: int = Field(default=0, description="Number of output tokens used")

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens used."""
        return self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    """Response from an LLM provider."""

    content: str = Field(description="The generated content")
    model: str = Field(description="The model that generated the response")
    provider: str = Field(description="The provider name")
    usage: UsageInfo = Field(description="Token usage information")
    latency_ms: float = Field(description="Request latency in milliseconds")
    cost_usd: float = Field(description="Estimated cost in USD")


class ProviderError(Exception):
    """Base exception for provider-related errors."""

    pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the provider.

        Args:
            model: The model identifier for this provider
            api_key: Optional API key for authentication
            base_url: Optional endpoint override
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Generate a completion based on the provided messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            response_format: Optional response format specification (e.g., JSON schema)

        Returns:
            LLMResponse object with generated content and metadata

        Raises:
            ProviderError: If the API call fails
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider name."""
        pass

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Estimate the cost of a completion.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        # Override in subclasses with actual pricing
        return 0.0


# Pricing information per model (in USD)
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "o1": {"input": 0.015, "output": 0.06},
    "o3-mini": {"input": 0.002, "output": 0.008},
    # Anthropic
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku-20241022": {"input": 0.00080, "output": 0.004},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-opus-4-20250514": {"input": 0.015, "output": 0.06},
    # Google
    "gemini-2.0-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-2.5-pro": {"input": 0.001, "output": 0.004},
    # xAI
    "grok-3": {"input": 0.002, "output": 0.008},
    # Cerebras
    "llama-3.3-70b": {"input": 0.000625, "output": 0.001},
}

# Provider registry
PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {}


def register_provider(provider_class: type[LLMProvider]) -> None:
    """Register a provider in the global registry."""
    # Provider name is derived from class name
    provider_name = provider_class.__module__.split(".")[-1].replace("_provider", "")
    PROVIDER_REGISTRY[provider_name] = provider_class


def get_provider(
    provider_name: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMProvider:
    """
    Factory function to get a provider instance.

    Args:
        provider_name: The provider name (e.g., 'openai', 'anthropic')
        model: The model identifier
        api_key: Optional API key for authentication
        base_url: Optional base URL override for the provider

    Returns:
        An instance of the appropriate LLMProvider

    Raises:
        ProviderError: If the provider is not found or model is not supported
    """
    if provider_name not in PROVIDER_REGISTRY:
        raise ProviderError(
            f"Unknown provider: {provider_name}. "
            f"Available providers: {list(PROVIDER_REGISTRY.keys())}"
        )

    provider_class = PROVIDER_REGISTRY[provider_name]
    return provider_class(model=model, api_key=api_key, base_url=base_url)


def get_provider_for_model(
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMProvider:
    """
    Auto-detect provider from model name and return provider instance.

    Args:
        model: The model identifier
        api_key: Optional API key for authentication
        base_url: Optional base URL override for the provider

    Returns:
        An instance of the appropriate LLMProvider

    Raises:
        ProviderError: If provider cannot be auto-detected
    """
    # Model name to provider mapping
    model_to_provider = {
        "gpt-": "openai",
        "o1": "openai",
        "o3": "openai",
        "claude-": "anthropic",
        "gemini-": "google",
        "grok-": "xai",
        "llama-": "cerebras",
    }

    for prefix, provider in model_to_provider.items():
        if model.startswith(prefix):
            return get_provider(provider, model=model, api_key=api_key, base_url=base_url)

    raise ProviderError(
        f"Could not auto-detect provider for model: {model}. "
        f"Please specify provider explicitly."
    )
