"""
Ollama LLM provider implementation.

Supports any model available locally via Ollama.
Uses httpx for async HTTP requests to local Ollama instance.
"""

import time
import os
from typing import Any, Optional

import httpx

from .base import (
    LLMProvider,
    LLMResponse,
    ProviderError,
    TemporaryProviderError,
    UsageInfo,
    register_provider,
)


class OllamaProvider(LLMProvider):
    """Ollama provider for local model inference."""

    BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 300.0  # Longer timeout for local inference

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Model identifier (any model available in Ollama)
            api_key: Not used for Ollama (local inference)
        """
        resolved_base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or self.BASE_URL).rstrip("/") + "/"
        super().__init__(model=model, api_key=api_key, base_url=resolved_base_url)

        self.async_client = httpx.AsyncClient(
            base_url=resolved_base_url,
            timeout=self.DEFAULT_TIMEOUT,
        )

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return "ollama"

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Generate a completion using local Ollama instance.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            response_format: Optional response format specification (limited support)

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            ProviderError: If the API call fails
        """
        start_time = time.time()

        try:
            # Build request for Ollama chat endpoint
            request_body = {
                "model": self.model,
                "messages": messages,
                "stream": False,
            }

            # Add options
            options = {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
            request_body["options"] = options

            # Format string for JSON output if requested
            if response_format and isinstance(response_format, dict):
                if response_format.get("type") == "json_object":
                    request_body["format"] = "json"

            response = await self.async_client.post(
                "api/chat",
                json=request_body,
            )

            if response.status_code != 200:
                if response.status_code in {408, 409, 423, 425, 500, 502, 503, 504}:
                    raise TemporaryProviderError(
                        f"Ollama returned status {response.status_code}: {response.text}"
                    )
                raise ProviderError(f"Ollama returned status {response.status_code}: {response.text}")

            data = response.json()
            latency_ms = (time.time() - start_time) * 1000

            # Extract content
            content = data.get("message", {}).get("content", "")

            # Estimate tokens (Ollama doesn't always return token counts)
            input_tokens = self._estimate_tokens(
                "\n".join([msg.get("content", "") for msg in messages])
            )
            output_tokens = self._estimate_tokens(content)

            usage = UsageInfo(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            # Cost is 0 for local inference
            cost = 0.0

            return LLMResponse(
                content=content,
                model=self.model,
                provider=self.get_provider_name(),
                usage=usage,
                latency_ms=latency_ms,
                cost_usd=cost,
            )

        except (TemporaryProviderError, ProviderError):
            raise
        except httpx.RequestError as e:
            raise TemporaryProviderError(
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Ensure Ollama is running: {str(e)}"
            ) from e
        except (KeyError, ValueError) as e:
            raise ProviderError(f"Invalid response from Ollama: {str(e)}") from e
        except Exception as e:
            raise ProviderError(f"Unexpected error calling Ollama: {str(e)}") from e

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 1 token per 4 characters)."""
        return max(1, len(text) // 4)

    async def close(self):
        """Close the async client."""
        await self.async_client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Register this provider
register_provider(OllamaProvider)
