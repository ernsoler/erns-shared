import sys
import pytest
from unittest.mock import MagicMock, patch

from erns_shared.ai.client import (
    AIResponse,
    AnthropicClient,
    GeminiClient,
    OllamaClient,
    OpenAIClient,
    ProviderError,
    estimate_cost,
    get_ai_client,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_anthropic_stream(text="hello", input_tokens=100, output_tokens=50, stop_reason="end_turn"):
    response = MagicMock()
    response.content = [MagicMock(text=text)]
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    response.stop_reason = stop_reason

    stream = MagicMock()
    stream.__enter__ = MagicMock(return_value=stream)
    stream.__exit__ = MagicMock(return_value=False)
    stream.get_final_message.return_value = response
    return stream


class _AnthropicAuthenticationError(Exception): pass
class _AnthropicPermissionDeniedError(Exception): pass
class _AnthropicRateLimitError(Exception): pass
class _AnthropicAPITimeoutError(Exception): pass
class _AnthropicBadRequestError(Exception): pass
class _AnthropicAPIStatusError(Exception):
    def __init__(self, msg, status_code=503, body=None):
        super().__init__(msg)
        self.status_code = status_code
        self.body = body or {}


def _anthropic_client() -> AnthropicClient:
    """Instantiate AnthropicClient with a mocked SDK module."""
    mock_sdk = MagicMock()
    # Attach real exception classes so except clauses in complete() work
    mock_sdk.AuthenticationError = _AnthropicAuthenticationError
    mock_sdk.PermissionDeniedError = _AnthropicPermissionDeniedError
    mock_sdk.RateLimitError = _AnthropicRateLimitError
    mock_sdk.APITimeoutError = _AnthropicAPITimeoutError
    mock_sdk.BadRequestError = _AnthropicBadRequestError
    mock_sdk.APIStatusError = _AnthropicAPIStatusError
    with patch.dict(sys.modules, {"anthropic": mock_sdk}):
        client = AnthropicClient(model="claude-sonnet-4-6", api_key="test-key")
    client._anthropic = mock_sdk
    client._client = MagicMock()
    return client


def _openai_client() -> OpenAIClient:
    mock_sdk = MagicMock()
    with patch.dict(sys.modules, {"openai": mock_sdk}):
        client = OpenAIClient(model="gpt-4o", api_key="test-key")
    client._client = MagicMock()
    return client


def _gemini_client() -> GeminiClient:
    mock_sdk = MagicMock()
    with patch.dict(sys.modules, {"google.generativeai": mock_sdk, "google": MagicMock()}):
        client = GeminiClient(model="gemini-2.0-flash", api_key="test-key")
    client._genai = mock_sdk
    return client


def _ollama_client() -> OllamaClient:
    mock_httpx = MagicMock()
    with patch.dict(sys.modules, {"httpx": mock_httpx}):
        client = OllamaClient(model="llama3", base_url="http://localhost:11434")
    client._http = MagicMock()
    return client


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------

class TestEstimateCost:
    def test_known_model(self):
        cost = estimate_cost("claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == pytest.approx(3.00 + 15.00)

    def test_unknown_model_returns_zero(self):
        assert estimate_cost("unknown-model", 1_000_000, 1_000_000) == 0.0

    def test_zero_tokens(self):
        assert estimate_cost("claude-sonnet-4-6", 0, 0) == 0.0

    def test_fractional_cost(self):
        cost = estimate_cost("gpt-4o", input_tokens=500_000, output_tokens=0)
        assert cost == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# AIResponse
# ---------------------------------------------------------------------------

class TestAIResponse:
    def test_estimated_cost_usd(self):
        r = AIResponse(
            text="hi", input_tokens=1_000_000, output_tokens=1_000_000,
            model="claude-sonnet-4-6", provider="anthropic",
        )
        assert r.estimated_cost_usd == pytest.approx(18.00)

    def test_free_model_cost_is_zero(self):
        r = AIResponse(text="hi", input_tokens=999, output_tokens=999, model="llama3", provider="ollama")
        assert r.estimated_cost_usd == 0.0


# ---------------------------------------------------------------------------
# AnthropicClient
# ---------------------------------------------------------------------------

class TestAnthropicClient:
    def test_successful_completion(self):
        client = _anthropic_client()
        client._client.messages.stream.return_value = _make_anthropic_stream("hello world")
        result = client.complete(system="sys", user="hi")
        assert result.text == "hello world"
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.provider == "anthropic"

    def test_raises_provider_error_on_auth_failure(self):
        client = _anthropic_client()
        client._client.messages.stream.side_effect = client._anthropic.AuthenticationError("bad key")
        with pytest.raises(ProviderError) as exc:
            client.complete(system="sys", user="hi")
        assert exc.value.status_code == 503

    def test_raises_provider_error_on_rate_limit(self):
        client = _anthropic_client()
        client._client.messages.stream.side_effect = client._anthropic.RateLimitError("rate limit")
        with pytest.raises(ProviderError) as exc:
            client.complete(system="sys", user="hi")
        assert exc.value.status_code == 429
        assert exc.value.retryable is True

    def test_raises_provider_error_on_timeout(self):
        client = _anthropic_client()
        client._client.messages.stream.side_effect = client._anthropic.APITimeoutError("timeout")
        with pytest.raises(ProviderError) as exc:
            client.complete(system="sys", user="hi")
        assert exc.value.retryable is True

    def test_raises_provider_error_on_max_tokens(self):
        client = _anthropic_client()
        client._client.messages.stream.return_value = _make_anthropic_stream(
            stop_reason="max_tokens"
        )
        with pytest.raises(ProviderError) as exc:
            client.complete(system="sys", user="hi")
        assert exc.value.status_code == 422
        assert exc.value.retryable is False

    def test_raises_provider_error_on_empty_response(self):
        client = _anthropic_client()
        client._client.messages.stream.return_value = _make_anthropic_stream(text="   ")
        with pytest.raises(ProviderError) as exc:
            client.complete(system="sys", user="hi")
        assert exc.value.retryable is True

    def test_raises_provider_error_on_overloaded(self):
        client = _anthropic_client()
        error = client._anthropic.APIStatusError(
            "overloaded", status_code=529, body={"error": {"type": "overloaded_error"}}
        )
        client._client.messages.stream.side_effect = error
        with pytest.raises(ProviderError) as exc:
            client.complete(system="sys", user="hi")
        assert exc.value.retryable is True


# ---------------------------------------------------------------------------
# OpenAIClient
# ---------------------------------------------------------------------------

class TestOpenAIClient:
    def test_successful_completion(self):
        client = _openai_client()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "gpt says hi"
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10
        client._client.chat.completions.create.return_value = mock_response

        result = client.complete(system="sys", user="hi")
        assert result.text == "gpt says hi"
        assert result.input_tokens == 20
        assert result.output_tokens == 10
        assert result.provider == "openai"

    def test_raises_provider_error_on_sdk_exception(self):
        client = _openai_client()
        client._client.chat.completions.create.side_effect = Exception("network error")
        with pytest.raises(ProviderError) as exc:
            client.complete(system="sys", user="hi")
        assert exc.value.retryable is True

    def test_raises_provider_error_on_empty_response(self):
        client = _openai_client()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "  "
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 1
        client._client.chat.completions.create.return_value = mock_response
        with pytest.raises(ProviderError):
            client.complete(system="sys", user="hi")


# ---------------------------------------------------------------------------
# GeminiClient
# ---------------------------------------------------------------------------

class TestGeminiClient:
    def test_successful_completion(self):
        client = _gemini_client()
        mock_response = MagicMock()
        mock_response.text = "gemini says hi"
        mock_response.usage_metadata.prompt_token_count = 30
        mock_response.usage_metadata.candidates_token_count = 15
        client._genai.GenerativeModel.return_value.generate_content.return_value = mock_response

        result = client.complete(system="sys", user="hi")
        assert result.text == "gemini says hi"
        assert result.provider == "google"

    def test_raises_provider_error_on_sdk_exception(self):
        client = _gemini_client()
        client._genai.GenerativeModel.return_value.generate_content.side_effect = Exception("api error")
        with pytest.raises(ProviderError) as exc:
            client.complete(system="sys", user="hi")
        assert exc.value.retryable is True

    def test_handles_missing_usage_metadata(self):
        client = _gemini_client()
        mock_response = MagicMock()
        mock_response.text = "hello"
        mock_response.usage_metadata = None
        client._genai.GenerativeModel.return_value.generate_content.return_value = mock_response

        result = client.complete(system="sys", user="hi")
        assert result.input_tokens == 0
        assert result.output_tokens == 0


# ---------------------------------------------------------------------------
# OllamaClient
# ---------------------------------------------------------------------------

class TestOllamaClient:
    def test_successful_completion(self):
        client = _ollama_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "ollama says hi"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        client._http.post.return_value = mock_resp

        result = client.complete(system="sys", user="hi")
        assert result.text == "ollama says hi"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.provider == "ollama"

    def test_raises_provider_error_on_http_failure(self):
        client = _ollama_client()
        client._http.post.side_effect = Exception("connection refused")
        with pytest.raises(ProviderError) as exc:
            client.complete(system="sys", user="hi")
        assert exc.value.retryable is True

    def test_raises_provider_error_on_empty_response(self):
        client = _ollama_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": ""}}
        client._http.post.return_value = mock_resp
        with pytest.raises(ProviderError):
            client.complete(system="sys", user="hi")


# ---------------------------------------------------------------------------
# get_ai_client factory
# ---------------------------------------------------------------------------

class TestGetAiClient:
    def test_returns_anthropic_client(self):
        mock_sdk = MagicMock()
        with patch.dict(sys.modules, {"anthropic": mock_sdk}):
            client = get_ai_client("anthropic", "claude-sonnet-4-6", api_key="key")
        assert isinstance(client, AnthropicClient)

    def test_returns_openai_client(self):
        mock_sdk = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_sdk}):
            client = get_ai_client("openai", "gpt-4o", api_key="key")
        assert isinstance(client, OpenAIClient)

    def test_returns_ollama_client_without_api_key(self):
        mock_httpx = MagicMock()
        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            client = get_ai_client("ollama", "llama3")
        assert isinstance(client, OllamaClient)

    def test_provider_case_insensitive(self):
        mock_sdk = MagicMock()
        with patch.dict(sys.modules, {"anthropic": mock_sdk}):
            client = get_ai_client("ANTHROPIC", "claude-sonnet-4-6", api_key="key")
        assert isinstance(client, AnthropicClient)

    def test_raises_on_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown AI provider"):
            get_ai_client("cohere", "command", api_key="key")

    def test_api_key_falls_back_to_env_var(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        mock_sdk = MagicMock()
        with patch.dict(sys.modules, {"anthropic": mock_sdk}):
            client = get_ai_client("anthropic", "claude-sonnet-4-6")
        assert isinstance(client, AnthropicClient)
