"""
LLM Provider abstraction layer for the llm-altruism project.

This module provides a unified interface for interacting with various LLM providers
including OpenAI, Anthropic, Google, xAI, Cerebras, OpenRouter, NVIDIA, and Ollama.

All providers implement a common async interface through the LLMProvider abstract base class.
"""

# Import base classes and registry
from .base import (
    LLMProvider,
    LLMResponse,
    MODEL_PRICING,
    ProviderError,
    PROVIDER_REGISTRY,
    RateLimitProviderError,
    TemporaryProviderError,
    UsageInfo,
    get_provider,
    get_provider_for_model,
)

# Import all provider implementations (order matters for registration)
from . import openai_provider  # noqa: F401
from . import anthropic_provider  # noqa: F401
from . import google_provider  # noqa: F401
from . import xai_provider  # noqa: F401
from . import cerebras_provider  # noqa: F401
from . import openrouter_provider  # noqa: F401
from . import nvidia_provider  # noqa: F401
from . import ollama_provider  # noqa: F401

__all__ = [
    # Base classes
    "LLMProvider",
    "LLMResponse",
    "UsageInfo",
    "ProviderError",
    "RateLimitProviderError",
    # Registry and factory
    "PROVIDER_REGISTRY",
    "TemporaryProviderError",
    "get_provider",
    "get_provider_for_model",
    # Pricing
    "MODEL_PRICING",
]
