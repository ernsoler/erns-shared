"""
AI client abstraction layer

Provides a single AIClient interface that normalises responses from:
  - Anthropic  (Claude)
  - OpenAI     (GPT-4o, etc.)
  - Google     (Gemini)
  - Ollama     (local models via HTTP)

Provider selected via env var AI_PROVIDER (default: anthropic).
Ollama needs no key — just OLLAMA_BASE_URL (default: http://localhost:11434).

Usage:
    from erns_shared.ai.client import get_ai_client
    client = get_ai_client(provider="anthropic", model="claude-sonnet-4-6")
    response = client.complete(system="...", user="...", max_tokens=4096)
    print(response.text, response.input_tokens, response.output_tokens)
"""

from __future__ import annotations

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost table  (USD per 1 000 000 tokens — update as pricing changes)
# ---------------------------------------------------------------------------
MODEL_COSTS: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-7": {"input": 15.00, "output": 75.00},
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "o1": {"input": 15.00, "output": 60.00},
    # Google
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    # Ollama — local, treat as free
    "llama3": {"input": 0.00, "output": 0.00},
    "mistral": {"input": 0.00, "output": 0.00},
    "llama3.1:8b": {"input": 0.00, "output": 0.00},
    "llama3.3:70b": {"input": 0.00, "output": 0.00},
    "phi3": {"input": 0.00, "output": 0.00},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for a completion call."""
    costs = MODEL_COSTS.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens / 1_000_000 * costs["input"]) + (
        output_tokens / 1_000_000 * costs["output"]
    )


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------


@dataclass
class AIResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str

    @property
    def estimated_cost_usd(self) -> float:
        return estimate_cost(self.model, self.input_tokens, self.output_tokens)


# ---------------------------------------------------------------------------
# Provider error — wraps all SDK errors into a single provider-agnostic type
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Raised by AIClient.complete() for any provider-side failure.

    Attributes:
        status_code:    Suggested HTTP status code for the API response.
        retryable:      True if the caller should suggest the user retry.
        public_message: Safe message to return to the end user (no internal detail).
    """

    def __init__(
        self, public_message: str, status_code: int = 503, retryable: bool = False
    ) -> None:
        super().__init__(public_message)
        self.status_code = status_code
        self.retryable = retryable
        self.public_message = public_message


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AIClient(ABC):
    """All concrete clients implement this single interface."""

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 4096) -> AIResponse:
        """Send a completion request and return a normalised AIResponse.

        Args:
            system:     System / instruction prompt.
            user:       User-turn message.
            max_tokens: Maximum tokens in the response.

        Returns:
            AIResponse with text, token counts, model, and provider.
        """

    @abstractmethod
    async def stream(
        self, system: str, user: str, max_tokens: int = 4096
    ) -> AsyncGenerator[str, None]:
        """Stream a completion, yielding text chunks as they arrive.

        Designed to feed directly into sse_stream() from erns_shared.http::

            async def generate():
                async for chunk in client.stream(system=..., user=...):
                    yield SSEEvent(data={"text": chunk}, event="delta")
                yield SSEEvent(data="[DONE]", event="done")

            return sse_stream(generate())

        Args:
            system:     System / instruction prompt.
            user:       User-turn message.
            max_tokens: Maximum tokens in the response.

        Yields:
            str — text chunks as they arrive from the provider.
        """
        # make this an async generator at the abstract level
        return
        yield  # noqa: unreachable


# ---------------------------------------------------------------------------
# Anthropic  (Claude)
# ---------------------------------------------------------------------------


class AnthropicClient(AIClient):
    """Wraps the official anthropic SDK."""

    PROVIDER = "anthropic"

    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model)
        try:
            import anthropic as _anthropic  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "anthropic package is not installed. Run: pip install anthropic"
            ) from exc
        self._anthropic = _anthropic
        self._api_key = api_key
        # 600s timeout — process Lambda has 900s; leaves ~300s headroom for internal
        # backoff retries before the Lambda itself times out.
        self._client = _anthropic.Anthropic(api_key=api_key, timeout=600.0)
        self._async_client = _anthropic.AsyncAnthropic(api_key=api_key, timeout=600.0)

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> AIResponse:
        # Stream so tokens are consumed as they arrive — prevents empty responses
        # on large outputs where the non-streaming API can drop the connection.
        try:
            with self._client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            ) as stream:
                response = stream.get_final_message()
        except self._anthropic.AuthenticationError as e:
            logger.error("Anthropic auth error: %s", e)
            raise ProviderError(
                "AI service configuration error.", status_code=503
            ) from e
        except self._anthropic.PermissionDeniedError as e:
            logger.error("Anthropic permission error: %s", e)
            raise ProviderError(
                "AI service configuration error.", status_code=503
            ) from e
        except self._anthropic.RateLimitError as e:
            logger.warning("Anthropic rate limit: %s", e)
            raise ProviderError(
                "AI service is busy. Please try again shortly.",
                status_code=429,
                retryable=True,
            ) from e
        except self._anthropic.APITimeoutError as e:
            logger.warning("Anthropic timeout: %s", e)
            raise ProviderError(
                "Analysis timed out. Please try again.", status_code=503, retryable=True
            ) from e
        except self._anthropic.BadRequestError as e:
            logger.error("Anthropic bad request: %s", e)
            raise ProviderError(
                "AI service is temporarily unavailable.", status_code=503
            ) from e
        except self._anthropic.APIStatusError as e:
            # overloaded_error can arrive in the stream body after an HTTP 200
            error_type = ""
            if isinstance(getattr(e, "body", None), dict):
                error_type = e.body.get("error", {}).get("type", "")
            if error_type == "overloaded_error" or "overloaded" in str(e).lower():
                logger.warning("Anthropic overloaded (status=%s): %s", e.status_code, e)
                raise ProviderError(
                    "AI service is busy. Please try again shortly.",
                    status_code=503,
                    retryable=True,
                ) from e
            logger.error("Anthropic API error status=%s: %s", e.status_code, e)
            raise ProviderError(
                "AI service is temporarily unavailable.",
                status_code=503,
                retryable=True,
            ) from e

        text = response.content[0].text if response.content else ""
        logger.info(
            "Anthropic response stop_reason=%s output_tokens=%d text_len=%d",
            response.stop_reason,
            response.usage.output_tokens,
            len(text),
        )
        if response.stop_reason == "max_tokens":
            logger.error(
                "Anthropic hit max_tokens limit model=%s output_tokens=%d — response truncated",
                self.model,
                response.usage.output_tokens,
            )
            raise ProviderError(
                "The document is too large to analyse. Please try a shorter document.",
                status_code=422,
                retryable=False,
            )
        if not text.strip():
            raise ProviderError(
                "Analysis service unavailable. Please try again.",
                status_code=503,
                retryable=True,
            )
        return AIResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self.model,
            provider=self.PROVIDER,
        )

    async def stream(
        self, system: str, user: str, max_tokens: int = 4096
    ) -> AsyncGenerator[str, None]:
        try:
            async with self._async_client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            ) as s:
                async for text in s.text_stream:
                    yield text
        except Exception as e:
            logger.error("Anthropic stream error: %s", e)
            raise ProviderError(
                "AI service is temporarily unavailable. Please try again.",
                status_code=503,
                retryable=True,
            ) from e


# ---------------------------------------------------------------------------
# OpenAI  (GPT-4o, etc.)
# ---------------------------------------------------------------------------


class OpenAIClient(AIClient):
    """Wraps the official openai SDK (v1+)."""

    PROVIDER = "openai"

    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model)
        try:
            from openai import (
                OpenAI,
                AuthenticationError,
                RateLimitError,
                APITimeoutError,
                APIStatusError,
            )  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "openai package is not installed. Run: pip install openai"
            ) from exc
        self._api_key = api_key
        self._client = OpenAI(api_key=api_key)
        self._errors = (
            AuthenticationError,
            RateLimitError,
            APITimeoutError,
            APIStatusError,
        )

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> AIResponse:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as e:
            logger.error("OpenAI error: %s", e)
            raise ProviderError(
                "AI service is temporarily unavailable. Please try again.",
                status_code=503,
                retryable=True,
            ) from e

        text = response.choices[0].message.content or ""
        if not text.strip():
            raise ProviderError(
                "AI service returned an empty response. Please try again.",
                status_code=503,
                retryable=True,
            )
        usage = response.usage
        return AIResponse(
            text=text,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            model=self.model,
            provider=self.PROVIDER,
        )

    async def stream(
        self, system: str, user: str, max_tokens: int = 4096
    ) -> AsyncGenerator[str, None]:
        try:
            from openai import AsyncOpenAI  # noqa: PLC0415

            async_client = AsyncOpenAI(api_key=self._api_key)
            async with async_client.chat.completions.stream(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            ) as s:
                async for text in s.text_stream:
                    if text:
                        yield text
        except Exception as e:
            logger.error("OpenAI stream error: %s", e)
            raise ProviderError(
                "AI service is temporarily unavailable. Please try again.",
                status_code=503,
                retryable=True,
            ) from e


# ---------------------------------------------------------------------------
# Google  (Gemini)
# ---------------------------------------------------------------------------


class GeminiClient(AIClient):
    """Wraps the google-generativeai SDK."""

    PROVIDER = "google"

    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model)
        try:
            import google.generativeai as genai  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "google-generativeai package is not installed. Run: pip install google-generativeai"
            ) from exc
        genai.configure(api_key=api_key)
        self._genai = genai

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> AIResponse:
        try:
            model_instance = self._genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system,
                generation_config=self._genai.GenerationConfig(
                    max_output_tokens=max_tokens
                ),
            )
            response = model_instance.generate_content(user)
        except Exception as e:
            logger.error("Gemini error: %s", e)
            raise ProviderError(
                "AI service is temporarily unavailable. Please try again.",
                status_code=503,
                retryable=True,
            ) from e

        text = response.text or ""
        if not text.strip():
            raise ProviderError(
                "AI service returned an empty response. Please try again.",
                status_code=503,
                retryable=True,
            )
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            meta = response.usage_metadata
            input_tokens = getattr(meta, "prompt_token_count", 0) or 0
            output_tokens = getattr(meta, "candidates_token_count", 0) or 0

        return AIResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            provider=self.PROVIDER,
        )

    async def stream(
        self, system: str, user: str, max_tokens: int = 4096
    ) -> AsyncGenerator[str, None]:
        # google-generativeai sync SDK — run in thread, yield full text as one chunk
        import asyncio

        response = await asyncio.to_thread(self.complete, system, user, max_tokens)
        yield response.text


# ---------------------------------------------------------------------------
# Ollama  (local HTTP API)
# ---------------------------------------------------------------------------


class OllamaClient(AIClient):
    """Calls a locally-running Ollama instance via its HTTP API.

    No API key required. Set OLLAMA_BASE_URL if Ollama is not on localhost.
    """

    PROVIDER = "ollama"

    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        super().__init__(model)
        try:
            import httpx  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "httpx package is not installed. Run: pip install httpx"
            ) from exc
        # 10-minute timeout — local models can be slow on first run
        self._http = httpx.Client(timeout=600.0)
        self._base_url = base_url.rstrip("/")

    def complete(self, system: str, user: str, max_tokens: int = 4096) -> AIResponse:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"num_predict": max_tokens},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            resp = self._http.post(f"{self._base_url}/api/chat", json=payload)
            resp.raise_for_status()
        except Exception as e:
            logger.error("Ollama error: %s", e)
            raise ProviderError(
                "Local AI service is unavailable. Make sure Ollama is running.",
                status_code=503,
                retryable=True,
            ) from e

        data = resp.json()
        text = data.get("message", {}).get("content", "")
        if not text.strip():
            raise ProviderError(
                "AI service returned an empty response. Please try again.",
                status_code=503,
                retryable=True,
            )
        return AIResponse(
            text=text,
            input_tokens=data.get("prompt_eval_count", 0) or 0,
            output_tokens=data.get("eval_count", 0) or 0,
            model=self.model,
            provider=self.PROVIDER,
        )

    async def stream(
        self, system: str, user: str, max_tokens: int = 4096
    ) -> AsyncGenerator[str, None]:
        import json
        import httpx  # noqa: PLC0415

        payload = {
            "model": self.model,
            "stream": True,
            "options": {"num_predict": max_tokens},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                async with client.stream(
                    "POST", f"{self._base_url}/api/chat", json=payload
                ) as response:
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        if content := data.get("message", {}).get("content"):
                            yield content
                        if data.get("done"):
                            break
        except Exception as e:
            logger.error("Ollama stream error: %s", e)
            raise ProviderError(
                "Local AI service is unavailable. Make sure Ollama is running.",
                status_code=503,
                retryable=True,
            ) from e


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_ai_client(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
) -> AIClient:
    """Instantiate the correct AIClient for the given provider.

    API keys fall back to environment variables when not passed explicitly:
        ANTHROPIC_API_KEY  — for provider=anthropic
        OPENAI_API_KEY     — for provider=openai
        GOOGLE_API_KEY     — for provider=google
        OLLAMA_BASE_URL    — optional for provider=ollama (default: http://localhost:11434)

    Args:
        provider: One of "anthropic", "openai", "google", "ollama".
        model:    Provider-specific model identifier string.
        api_key:  Optional API key. Falls back to env var if omitted.

    Returns:
        Concrete AIClient instance.

    Raises:
        ValueError:  Unknown provider string.
        ImportError: Provider SDK not installed.
    """
    provider = provider.lower().strip()
    if provider == "anthropic":
        return AnthropicClient(
            model=model, api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        )
    if provider == "openai":
        return OpenAIClient(
            model=model, api_key=api_key or os.environ.get("OPENAI_API_KEY", "")
        )
    if provider == "google":
        return GeminiClient(
            model=model, api_key=api_key or os.environ.get("GOOGLE_API_KEY", "")
        )
    if provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaClient(model=model, base_url=base_url)
    raise ValueError(
        f"Unknown AI provider: '{provider}'. Valid options: anthropic, openai, google, ollama"
    )
