"""
Comprehensive tests for the LLM provider abstraction layer.

Tests the factory functions, registry, base classes, and error handling.
"""

import pytest
import asyncio
from src.providers import (
    LLMProvider,
    LLMResponse,
    UsageInfo,
    ProviderError,
    PROVIDER_REGISTRY,
    RateLimitProviderError,
    get_provider,
    get_provider_for_model,
    MODEL_PRICING,
    TemporaryProviderError,
)
from src.providers.openai_compatible_base import OpenAICompatibleProvider


class TestRegistry:
    """Tests for provider registry and factory functions."""

    def test_all_providers_registered(self):
        """Verify all providers are registered."""
        expected = {"openai", "anthropic", "google", "xai", "cerebras", "openrouter", "nvidia", "ollama"}
        assert set(PROVIDER_REGISTRY.keys()) == expected

    def test_get_provider_openai(self):
        """Test getting OpenAI provider."""
        provider = get_provider("openai", "gpt-4o", api_key="test-key")
        assert provider.model == "gpt-4o"
        assert provider.get_provider_name() == "openai"

    def test_get_provider_anthropic(self):
        """Test getting Anthropic provider."""
        provider = get_provider("anthropic", "claude-3-5-sonnet-20241022", api_key="test-key")
        assert provider.model == "claude-3-5-sonnet-20241022"
        assert provider.get_provider_name() == "anthropic"

    def test_get_provider_unknown(self):
        """Test error handling for unknown provider."""
        with pytest.raises(ProviderError) as exc:
            get_provider("unknown-provider", "test-model")
        assert "Unknown provider" in str(exc.value)

    def test_get_provider_for_model_gpt(self):
        """Test auto-detection of OpenAI from model name."""
        provider = get_provider_for_model("gpt-4o", api_key="test-key")
        assert provider.get_provider_name() == "openai"

    def test_get_provider_for_model_o1(self):
        """Test auto-detection of OpenAI o1 model."""
        provider = get_provider_for_model("o1", api_key="test-key")
        assert provider.get_provider_name() == "openai"

    def test_get_provider_for_model_claude(self):
        """Test auto-detection of Anthropic from model name."""
        provider = get_provider_for_model("claude-3-5-sonnet-20241022", api_key="test-key")
        assert provider.get_provider_name() == "anthropic"

    def test_get_provider_for_model_gemini(self):
        """Test auto-detection of Google from model name."""
        provider = get_provider_for_model("gemini-2.0-flash", api_key="test-key")
        assert provider.get_provider_name() == "google"

    def test_get_provider_for_model_grok(self):
        """Test auto-detection of xAI from model name."""
        provider = get_provider_for_model("grok-3", api_key="test-key")
        assert provider.get_provider_name() == "xai"

    def test_get_provider_for_model_unknown(self):
        """Test error handling for unknown model in auto-detection."""
        with pytest.raises(ProviderError) as exc:
            get_provider_for_model("unknown-model-xyz")
        assert "Could not auto-detect" in str(exc.value)


class TestModels:
    """Tests for data models."""

    def test_usage_info_creation(self):
        """Test UsageInfo model creation."""
        usage = UsageInfo(input_tokens=100, output_tokens=50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_usage_info_default_values(self):
        """Test UsageInfo default values."""
        usage = UsageInfo()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_llm_response_creation(self):
        """Test LLMResponse model creation."""
        usage = UsageInfo(input_tokens=100, output_tokens=50)
        response = LLMResponse(
            content="Hello world",
            model="gpt-4o",
            provider="openai",
            usage=usage,
            latency_ms=1234.5,
            cost_usd=0.00123,
        )
        assert response.content == "Hello world"
        assert response.model == "gpt-4o"
        assert response.provider == "openai"
        assert response.usage.total_tokens == 150
        assert response.latency_ms == 1234.5
        assert response.cost_usd == 0.00123

    def test_llm_response_json_serialization(self):
        """Test LLMResponse JSON serialization."""
        usage = UsageInfo(input_tokens=100, output_tokens=50)
        response = LLMResponse(
            content="Hello",
            model="gpt-4o",
            provider="openai",
            usage=usage,
            latency_ms=100.0,
            cost_usd=0.001,
        )
        json_str = response.model_dump_json()
        assert "Hello" in json_str
        assert "gpt-4o" in json_str


class TestPricing:
    """Tests for pricing functionality."""

    def test_pricing_constants_exist(self):
        """Verify pricing constants are defined for all supported models."""
        # Sample of models that should have pricing
        assert "gpt-4o" in MODEL_PRICING
        assert "claude-3-5-sonnet-20241022" in MODEL_PRICING
        assert "gemini-2.0-flash" in MODEL_PRICING
        assert "grok-3" in MODEL_PRICING
        assert "llama3.1-8b" in MODEL_PRICING

    def test_pricing_structure(self):
        """Verify pricing entries have correct structure."""
        for model, pricing in MODEL_PRICING.items():
            assert "input" in pricing
            assert "output" in pricing
            assert isinstance(pricing["input"], (int, float))
            assert isinstance(pricing["output"], (int, float))

    def test_openai_provider_cost_estimation(self):
        """Test cost estimation for OpenAI provider."""
        provider = get_provider("openai", "gpt-4o", api_key="test-key")
        # GPT-4o: input=$0.005/1M, output=$0.015/1M
        cost = provider.estimate_cost(input_tokens=1_000_000, output_tokens=1_000_000)
        assert abs(cost - 0.020) < 0.0001  # Should be $0.020

    def test_anthropic_provider_cost_estimation(self):
        """Test cost estimation for Anthropic provider."""
        provider = get_provider("anthropic", "claude-3-5-sonnet-20241022", api_key="test-key")
        # Claude 3.5 Sonnet: input=$0.003/1M, output=$0.015/1M
        cost = provider.estimate_cost(input_tokens=1_000_000, output_tokens=1_000_000)
        assert abs(cost - 0.018) < 0.0001  # Should be $0.018


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    def test_openai_supported_models(self):
        """Test OpenAI provider with supported models."""
        from src.providers.openai_provider import OpenAIProvider

        supported = OpenAIProvider.SUPPORTED_MODELS
        assert "gpt-4o" in supported
        assert "gpt-4o-mini" in supported
        assert "o1" in supported
        assert "o3-mini" in supported

    def test_openai_unsupported_model(self):
        """Test error handling for unsupported OpenAI model."""
        with pytest.raises(ProviderError) as exc:
            get_provider("openai", "gpt-3.5-turbo", api_key="test-key")
        assert "Unsupported model" in str(exc.value)


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    def test_anthropic_supported_models(self):
        """Test Anthropic provider with supported models."""
        from src.providers.anthropic_provider import AnthropicProvider

        supported = AnthropicProvider.SUPPORTED_MODELS
        assert "claude-3-5-sonnet-20241022" in supported
        assert "claude-3-5-haiku-20241022" in supported
        assert "claude-sonnet-4-20250514" in supported
        assert "claude-opus-4-20250514" in supported

    def test_anthropic_unsupported_model(self):
        """Test error handling for unsupported Anthropic model."""
        with pytest.raises(ProviderError) as exc:
            get_provider("anthropic", "claude-2", api_key="test-key")
        assert "Unsupported model" in str(exc.value)


class TestGoogleProvider:
    """Tests for Google provider."""

    def test_google_supported_models(self):
        """Test Google provider with supported models."""
        from src.providers.google_provider import GoogleProvider

        supported = GoogleProvider.SUPPORTED_MODELS
        assert "gemini-2.0-flash" in supported
        assert "gemini-2.5-pro" in supported


class TestXAIProvider:
    """Tests for xAI provider."""

    def test_xai_supported_models(self):
        """Test xAI provider initialization."""
        provider = get_provider("xai", "grok-3", api_key="test-key")
        assert provider.model == "grok-3"
        assert provider.get_provider_name() == "xai"

    def test_xai_unsupported_model(self):
        """Test error handling for unsupported xAI model."""
        with pytest.raises(ProviderError) as exc:
            get_provider("xai", "grok-1", api_key="test-key")
        assert "Unsupported model" in str(exc.value)


class TestCerebrasProvider:
    """Tests for Cerebras provider."""

    def test_cerebras_supported_models(self):
        """Test Cerebras provider initialization."""
        from src.providers.cerebras_provider import CerebrasProvider

        assert CerebrasProvider.SUPPORTED_MODELS == {
            "gpt-oss-120b",
            "llama3.1-8b",
            "qwen-3-235b-a22b-instruct-2507",
            "zai-glm-4.7",
        }
        for model in CerebrasProvider.SUPPORTED_MODELS:
            provider = get_provider("cerebras", model, api_key="test-key")
            assert provider.model == model
            assert provider.get_provider_name() == "cerebras"

    def test_cerebras_unsupported_model(self):
        """Test error handling for unsupported Cerebras model."""
        with pytest.raises(ProviderError) as exc:
            get_provider("cerebras", "mistral-7b", api_key="test-key")
        assert "Unsupported model" in str(exc.value)


class TestOpenRouterProvider:
    """Tests for OpenRouter provider."""

    def test_openrouter_any_model(self):
        """Test OpenRouter provider with arbitrary model names."""
        provider = get_provider("openrouter", "meta-llama/llama-2-70b", api_key="test-key")
        assert provider.model == "meta-llama/llama-2-70b"
        assert provider.get_provider_name() == "openrouter"

    def test_openrouter_requires_api_key(self):
        """Test that OpenRouter requires an API key."""
        with pytest.raises(ProviderError):
            get_provider("openrouter", "any-model")


class TestNVIDIAProvider:
    """Tests for NVIDIA provider."""

    def test_nvidia_any_model(self):
        """Test NVIDIA provider with an arbitrary catalog model name."""
        provider = get_provider("nvidia", "meta/llama-3.1-8b-instruct", api_key="test-key")
        assert provider.model == "meta/llama-3.1-8b-instruct"
        assert provider.get_provider_name() == "nvidia"

    def test_nvidia_requires_api_key(self):
        """Test that NVIDIA API usage requires an API key."""
        with pytest.raises(ProviderError):
            get_provider("nvidia", "meta/llama-3.1-8b-instruct")


class TestOllamaProvider:
    """Tests for Ollama provider."""

    def test_ollama_any_model(self):
        """Test Ollama provider with arbitrary model names."""
        provider = get_provider("ollama", "llama2", api_key=None)
        assert provider.model == "llama2"
        assert provider.get_provider_name() == "ollama"

    def test_ollama_cost_zero(self):
        """Test that Ollama local inference has zero cost."""
        provider = get_provider("ollama", "llama2")
        cost = provider.estimate_cost(input_tokens=1000, output_tokens=500)
        assert cost == 0.0


class TestProviderErrors:
    """Tests for error handling."""

    def test_provider_error_inheritance(self):
        """Test ProviderError is an Exception."""
        assert issubclass(ProviderError, Exception)

    def test_provider_error_message(self):
        """Test ProviderError message handling."""
        error = ProviderError("Test error message")
        assert "Test error message" in str(error)

    def test_temporary_provider_error_tracks_retry_after(self):
        """Retryable errors can carry provider-suggested wait times."""
        error = TemporaryProviderError("retry later", retry_after_seconds=12.5)
        assert "retry later" in str(error)
        assert error.retry_after_seconds == 12.5

    def test_rate_limit_provider_error_inherits_from_temporary_error(self):
        """Rate-limit errors are a specialized retryable provider error."""
        error = RateLimitProviderError("too many requests", retry_after_seconds=30)
        assert isinstance(error, TemporaryProviderError)
        assert error.retry_after_seconds == 30


class TestOpenAICompatibleResponseParsing:
    """Tests for flexible OpenAI-compatible response parsing."""

    class DummyProvider(OpenAICompatibleProvider):
        BASE_URL = "https://example.invalid/v1"

        def get_provider_name(self) -> str:
            return "dummy"

    def test_extract_content_prefers_message_content(self):
        """Standard chat-completion content should be returned unchanged."""
        provider = self.DummyProvider(model="dummy-model", api_key="test-key")
        data = {
            "choices": [
                {
                    "message": {
                        "content": "hello",
                    }
                }
            ]
        }

        assert provider._extract_content(data) == "hello"
        asyncio.run(provider.close())

    def test_extract_content_accepts_reasoning_content_fallback(self):
        """Reasoning-only NVIDIA-style responses should still surface text."""
        provider = self.DummyProvider(model="dummy-model", api_key="test-key")
        data = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning_content": "thinking output",
                    }
                }
            ]
        }

        assert provider._extract_content(data) == "thinking output"
        asyncio.run(provider.close())

    def test_extract_content_joins_text_parts(self):
        """Content arrays with text parts should be flattened into one string."""
        provider = self.DummyProvider(model="dummy-model", api_key="test-key")
        data = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "hello"},
                            {"type": "text", "text": "world"},
                        ],
                    }
                }
            ]
        }

        assert provider._extract_content(data) == "hello\nworld"
        asyncio.run(provider.close())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
