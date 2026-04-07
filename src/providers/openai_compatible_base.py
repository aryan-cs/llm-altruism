"""
Base class for OpenAI-compatible API providers.

This module provides a shared implementation for providers that expose
OpenAI-compatible APIs (xAI, Cerebras, OpenRouter, etc.).
"""

import time
from typing import Any, Optional

import httpx

from .base import (
    LLMProvider,
    LLMResponse,
    MODEL_PRICING,
    ProviderError,
    RateLimitProviderError,
    TemporaryProviderError,
    UsageInfo,
)


class OpenAICompatibleProvider(LLMProvider):
    """Base class for OpenAI-compatible API providers."""

    BASE_URL: str = ""  # Override in subclasses
    DEFAULT_TIMEOUT: float = 60.0
    MAX_TOKENS_FIELD: str = "max_tokens"
    SUPPORTS_RESPONSE_FORMAT: bool = True

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize OpenAI-compatible provider.

        Args:
            model: Model identifier
            api_key: API key for authentication
        """
        resolved_base_url = (base_url or self.BASE_URL).rstrip("/") + "/"
        super().__init__(model=model, api_key=api_key, base_url=resolved_base_url)

        if not resolved_base_url:
            raise ProviderError("BASE_URL must be set in subclass")

        if not api_key:
            raise ProviderError(f"API key is required for {self.get_provider_name()}")

        self.async_client = httpx.AsyncClient(
            base_url=resolved_base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.DEFAULT_TIMEOUT,
        )

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Generate a completion using OpenAI-compatible API.

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
            request_body = self._build_request_body(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )

            response = await self.async_client.post(
                "chat/completions",
                json=request_body,
            )

            if response.status_code != 200:
                raise self._classify_error(response)

            data = response.json()
            latency_ms = (time.time() - start_time) * 1000

            # Extract content
            content = self._extract_content(data)

            # Extract usage
            usage = UsageInfo(
                input_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                output_tokens=data.get("usage", {}).get("completion_tokens", 0),
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

        except (RateLimitProviderError, TemporaryProviderError, ProviderError):
            raise
        except httpx.RequestError as e:
            raise TemporaryProviderError(
                f"Network error calling {self.get_provider_name()}: {str(e)}"
            ) from e
        except (KeyError, ValueError) as e:
            raise ProviderError(f"Invalid response from {self.get_provider_name()}: {str(e)}") from e
        except Exception as e:
            raise ProviderError(
                f"Unexpected error calling {self.get_provider_name()}: {str(e)}"
            ) from e

    def _extract_content(self, data: dict[str, Any]) -> str:
        """Extract text content from common OpenAI-compatible response shapes."""
        choices = data["choices"]
        if not choices:
            raise KeyError("choices")

        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")

        if isinstance(content, str) and content.strip():
            return content

        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, str) and part.strip():
                    text_parts.append(part)
                    continue
                if not isinstance(part, dict):
                    continue
                text_value = part.get("text") or part.get("content")
                if isinstance(text_value, str) and text_value.strip():
                    text_parts.append(text_value)
            if text_parts:
                return "\n".join(text_parts)

        for fallback_key in ("reasoning_content", "reasoning"):
            fallback_value = message.get(fallback_key)
            if isinstance(fallback_value, str) and fallback_value.strip():
                return fallback_value

        text = choice.get("text")
        if isinstance(text, str) and text.strip():
            return text

        raise KeyError("content")

    def _build_request_body(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build a provider request body, allowing subclasses to tweak fields."""
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            self.MAX_TOKENS_FIELD: max_tokens,
        }

        if response_format and self.SUPPORTS_RESPONSE_FORMAT:
            request_body["response_format"] = response_format

        return request_body

    def _classify_error(self, response: httpx.Response) -> ProviderError:
        """Map an HTTP error response to a permanent or retryable provider error."""
        status_code = response.status_code
        message = f"API returned status {status_code}: {response.text}"
        retry_after_seconds = self._parse_retry_after_seconds(response)

        if status_code == 429:
            return RateLimitProviderError(message, retry_after_seconds=retry_after_seconds)

        if status_code in {408, 409, 423, 425, 500, 502, 503, 504}:
            return TemporaryProviderError(message, retry_after_seconds=retry_after_seconds)

        return ProviderError(message)

    def _parse_retry_after_seconds(self, response: httpx.Response) -> float | None:
        """Extract a retry wait time from common rate-limit headers when present."""
        for header_name in (
            "retry-after",
            "x-ratelimit-reset",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset-requests-day",
            "x-ratelimit-reset-tokens",
            "x-ratelimit-reset-tokens-minute",
        ):
            header_value = response.headers.get(header_name)
            if not header_value:
                continue
            try:
                wait_seconds = float(header_value)
            except ValueError:
                continue
            if wait_seconds >= 0:
                return wait_seconds
        return None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.async_client.aclose()

    async def close(self):
        """Close the async client."""
        await self.async_client.aclose()

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
