"""OpenAI LLM Client for RAG pipeline.

WARNING: This client uses external OpenAI API and should only be used
for testing purposes. For production use, prefer local serving frameworks
like vLLM, sglang, or ollama via LocalLLMClient.
"""

import logging
import os

import httpx
from openai import OpenAI

from .base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI API client for chat completions.

    WARNING: This client sends data to external OpenAI servers.
    For production use with sensitive data, use LocalLLMClient instead.

    Attributes:
        model: Model name to use (default: gpt-4o-mini)
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens in response
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = BaseLLMClient.DEFAULT_TEMPERATURE,
        max_tokens: int = BaseLLMClient.DEFAULT_MAX_TOKENS,
        base_url: str | None = None,
    ) -> None:
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            base_url: Optional base URL for API (for compatible endpoints)
        """
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )

        # Use explicit httpx client to avoid proxy parameter conflicts
        # in certain httpx/openai version combinations
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            http_client=httpx.Client(),
        )

        logger.warning(
            f"OpenAI client initialized with model: {model}. "
            "NOTE: Using external API - for testing only."
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Generate a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            model: Override default model

        Returns:
            LLMResponse with generated content
        """
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=max_tokens or self.max_tokens,
            )

            message = response.choices[0].message
            usage = response.usage

            return LLMResponse(
                content=message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                },
                finish_reason=response.choices[0].finish_reason or "stop",
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

        # generate_with_context is inherited from BaseLLMClient
