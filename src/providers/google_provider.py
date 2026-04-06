"""
Google Generative AI LLM provider implementation.

Supports Gemini 2.0 Flash and Gemini 2.5 Pro models.
Uses the official Google GenAI Python SDK with async support.
"""

import time
from typing import Any, Optional

try:
    import google.genai as genai
except ImportError:
    # Fallback to deprecated package for compatibility
    import google.generativeai as genai

try:
    from google.api_core.exceptions import GoogleAPIError, RetryError
except ImportError:
    class GoogleAPIError(Exception):
        """Fallback Google API error type when api-core is unavailable."""

    class RetryError(Exception):
        """Fallback retry error type when api-core is unavailable."""

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


class GoogleProvider(LLMProvider):
    """Google Generative AI provider using the official GenAI SDK."""

    SUPPORTED_MODELS = {
        "gemini-2.0-flash",
        "gemini-2.5-pro",
    }

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize Google provider.

        Args:
            model: Model identifier (e.g., 'gemini-2.0-flash')
            api_key: Google API key (uses GOOGLE_API_KEY env var if not provided)

        Raises:
            ProviderError: If model is not supported
        """
        super().__init__(model=model, api_key=api_key, base_url=base_url)

        if model not in self.SUPPORTED_MODELS:
            raise ProviderError(
                f"Unsupported model: {model}. Supported models: {self.SUPPORTED_MODELS}"
            )

        self._uses_client_sdk = hasattr(genai, "Client")

        if self._uses_client_sdk:
            self.client = genai.Client(api_key=api_key)
        else:
            if api_key and hasattr(genai, "configure"):
                genai.configure(api_key=api_key)

            self.client = genai.GenerativeModel(
                model_name=model,
            )

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return "google"

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Generate a completion using Google Generative AI API.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            response_format: Optional response format specification

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            ProviderError: If the API call fails
        """
        start_time = time.time()

        try:
            # Convert messages to Google format
            # Google uses "user" and "model" instead of "user" and "assistant"
            google_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                if role == "assistant":
                    role = "model"
                google_messages.append(
                    {
                        "role": role,
                        "parts": [msg.get("content", "")],
                    }
                )

            # Build generation config
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            # Add response mime type if JSON format is requested
            if response_format:
                if isinstance(response_format, dict):
                    if response_format.get("type") == "json_object":
                        generation_config["response_mime_type"] = "application/json"

            # Call the API synchronously (Google SDK is synchronous)
            # This is wrapped to work in async context
            response = await self._call_generate_content(google_messages, generation_config)

            latency_ms = (time.time() - start_time) * 1000

            # Extract content
            content = response.text if response.text else ""

            # Google doesn't provide detailed token counts in free tier,
            # so we estimate based on response
            input_tokens = self._estimate_tokens(
                "\n".join([msg.get("content", "") for msg in messages])
            )
            output_tokens = self._estimate_tokens(content)

            usage = UsageInfo(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
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

        except RetryError as e:
            raise TemporaryProviderError(f"Google Generative AI retryable error: {str(e)}") from e
        except GoogleAPIError as e:
            message = str(e)
            lowered = message.lower()
            if "429" in lowered or "rate limit" in lowered or "resource_exhausted" in lowered:
                raise RateLimitProviderError(f"Google rate limit: {message}") from e
            if any(code in lowered for code in ("408", "500", "502", "503", "504", "unavailable", "timeout")):
                raise TemporaryProviderError(f"Google Generative AI temporary error: {message}") from e
            raise ProviderError(f"Google Generative AI error: {message}") from e
        except Exception as e:
            raise ProviderError(f"Unexpected error calling Google API: {str(e)}") from e

    async def _call_generate_content(self, messages: list[dict], generation_config: dict):
        """Call the synchronous API in an async-safe way."""
        if self._uses_client_sdk:
            contents = []
            for message in messages:
                role = message.get("role", "user")
                parts = message.get("parts", [])
                text = "\n".join(str(part) for part in parts)
                contents.append({"role": role, "parts": [{"text": text}]})

            config = {
                "temperature": generation_config.get("temperature"),
                "max_output_tokens": generation_config.get("max_output_tokens"),
            }
            if "response_mime_type" in generation_config:
                config["response_mime_type"] = generation_config["response_mime_type"]

            return self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

        return self.client.generate_content(
            messages,
            generation_config=generation_config,
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 1 token per 4 characters)."""
        return max(1, len(text) // 4)

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
register_provider(GoogleProvider)
