"""
OpenAI LLM provider implementation.

Supports GPT-4o, GPT-4o-mini, o1, o3-mini models.
Uses the official OpenAI Python SDK with async support.
"""

import time
import os
from typing import Any, Optional

from openai import AsyncOpenAI, BadRequestError, RateLimitError

from .base import LLMProvider, LLMResponse, MODEL_PRICING, ProviderError, UsageInfo, register_provider


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider using the official OpenAI SDK."""

    SUPPORTED_MODELS = {"gpt-4o", "gpt-4o-mini", "o1", "o3-mini"}

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: Model identifier (e.g., 'gpt-4o', 'o1')
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)

        Raises:
            ProviderError: If model is not supported
        """
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")
        super().__init__(model=model, api_key=api_key, base_url=resolved_base_url)

        if model not in self.SUPPORTED_MODELS:
            raise ProviderError(
                f"Unsupported model: {model}. Supported models: {self.SUPPORTED_MODELS}"
            )

        self.client = AsyncOpenAI(api_key=api_key, base_url=resolved_base_url)

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return "openai"

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Generate a completion using OpenAI API.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            response_format: Optional response format (e.g., {"type": "json_object"})

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            ProviderError: If the API call fails
        """
        start_time = time.time()

        try:
            # o1 and o3-mini models have specific temperature constraints
            if self.model.startswith("o1") or self.model.startswith("o3"):
                temperature = 1.0  # o1/o3 models use temperature=1
                max_tokens = None  # o1 models don't accept max_tokens

            # Build request kwargs
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }

            # o1/o3 models don't support max_tokens parameter
            if not (self.model.startswith("o1") or self.model.startswith("o3")):
                kwargs["max_tokens"] = max_tokens

            # Add response format if provided
            if response_format:
                kwargs["response_format"] = response_format

            response = await self.client.chat.completions.create(**kwargs)

            latency_ms = (time.time() - start_time) * 1000

            # Extract usage information
            usage = UsageInfo(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

            # Estimate cost
            cost = self.estimate_cost(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=self.model,
                provider=self.get_provider_name(),
                usage=usage,
                latency_ms=latency_ms,
                cost_usd=cost,
            )

        except (RateLimitError, BadRequestError) as e:
            raise ProviderError(f"OpenAI API error: {str(e)}") from e
        except Exception as e:
            raise ProviderError(f"Unexpected error calling OpenAI API: {str(e)}") from e

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
register_provider(OpenAIProvider)
