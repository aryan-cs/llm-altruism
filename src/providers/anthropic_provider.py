"""
Anthropic LLM provider implementation.

Supports Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude Sonnet 4, Claude Opus 4.
Uses the official Anthropic Python SDK with async support.
"""

import json
import os
import time
from typing import Any, Optional

from anthropic import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncAnthropic,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from .base import (
    LLMProvider,
    LLMResponse,
    MODEL_PRICING,
    ProviderError,
    RateLimitProviderError,
    TemporaryProviderError,
    UsageInfo,
    register_provider,
)


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider using the official Anthropic SDK."""

    SUPPORTED_MODELS = {
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
    }

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize Anthropic provider.

        Args:
            model: Model identifier (e.g., 'claude-3-5-sonnet-20241022')
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)

        Raises:
            ProviderError: If model is not supported
        """
        resolved_base_url = base_url or os.getenv("ANTHROPIC_BASE_URL")
        super().__init__(model=model, api_key=api_key, base_url=resolved_base_url)

        if model not in self.SUPPORTED_MODELS:
            raise ProviderError(
                f"Unsupported model: {model}. Supported models: {self.SUPPORTED_MODELS}"
            )

        self.client = AsyncAnthropic(api_key=api_key, base_url=resolved_base_url)

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return "anthropic"

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Generate a completion using Anthropic API.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0 for Claude, clipped)
            max_tokens: Maximum tokens to generate
            response_format: Optional response format specification

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            ProviderError: If the API call fails
        """
        start_time = time.time()

        try:
            # Anthropic's temperature range is 0.0 to 1.0
            temperature = min(max(temperature, 0.0), 1.0)

            # Build request kwargs
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            # Handle response format if provided
            # Anthropic uses "json" mode for structured output
            if response_format:
                # If it's a JSON schema format, add system prompt hint
                if isinstance(response_format, dict) and "json_schema" in response_format:
                    kwargs["system"] = (
                        "You must respond with valid JSON that matches the provided schema. "
                        "Ensure all output is valid JSON."
                    )

            response = await self.client.messages.create(**kwargs)

            latency_ms = (time.time() - start_time) * 1000

            # Extract content (Claude returns a list of content blocks)
            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text

            # Extract usage information
            usage = UsageInfo(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

            # Estimate cost
            cost = self.estimate_cost(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )

            return LLMResponse(
                content=content,
                model=self.model,
                provider=self.get_provider_name(),
                usage=usage,
                latency_ms=latency_ms,
                cost_usd=cost,
            )

        except RateLimitError as e:
            retry_after = getattr(e, "retry_after", None)
            retry_after_seconds = None
            if retry_after is not None:
                try:
                    retry_after_seconds = float(retry_after)
                except (TypeError, ValueError):
                    retry_after_seconds = None
            raise RateLimitProviderError(
                f"Anthropic rate limit: {str(e)}",
                retry_after_seconds=retry_after_seconds,
            ) from e
        except (APIConnectionError, APITimeoutError, InternalServerError) as e:
            raise TemporaryProviderError(f"Anthropic temporary error: {str(e)}") from e
        except (AuthenticationError, PermissionDeniedError, NotFoundError, BadRequestError) as e:
            raise ProviderError(f"Anthropic API error: {str(e)}") from e
        except APIError as e:
            if getattr(e, "status_code", None) in {408, 409, 423, 425, 500, 502, 503, 504}:
                raise TemporaryProviderError(f"Anthropic temporary error: {str(e)}") from e
            raise ProviderError(f"Anthropic API error: {str(e)}") from e
        except Exception as e:
            raise ProviderError(f"Unexpected error calling Anthropic API: {str(e)}") from e

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
        pricing = MODEL_PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


# Register this provider
register_provider(AnthropicProvider)
