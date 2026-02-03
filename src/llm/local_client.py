"""Local LLM Client for vLLM, sglang, and ollama.

These frameworks all provide OpenAI-compatible APIs, allowing us to use
the same client implementation with different base URLs.

Supported frameworks:
- vLLM: High-throughput serving with PagedAttention
- sglang: Fast serving with RadixAttention
- ollama: Easy local model deployment

Security Note:
    This client is designed for local/private deployment to ensure
    no data leaves your infrastructure.
"""

import logging
import os

import httpx
from openai import OpenAI

from .base import BaseLLMClient, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


# Default endpoints for each provider
DEFAULT_ENDPOINTS = {
    LLMProvider.VLLM: "http://localhost:8000/v1",
    LLMProvider.SGLANG: "http://localhost:30000/v1",
    LLMProvider.OLLAMA: "http://localhost:11434/v1",
}

# Default models for each provider
DEFAULT_MODELS = {
    LLMProvider.VLLM: "meta-llama/Llama-3.1-8B-Instruct",
    LLMProvider.SGLANG: "meta-llama/Llama-3.1-8B-Instruct",
    LLMProvider.OLLAMA: "llama3.1:8b",
}


class LocalLLMClient(BaseLLMClient):
    """LLM Client for local serving frameworks (vLLM, sglang, ollama).

    All these frameworks provide OpenAI-compatible APIs, so we use the
    OpenAI client library with a custom base_url.

    Example usage:
        # Using vLLM
        client = LocalLLMClient(provider=LLMProvider.VLLM)

        # Using ollama with custom model
        client = LocalLLMClient(
            provider=LLMProvider.OLLAMA,
            model="mistral:7b"
        )

        # Using sglang with custom endpoint
        client = LocalLLMClient(
            provider=LLMProvider.SGLANG,
            base_url="http://gpu-server:30000/v1"
        )
    """

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.VLLM,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = BaseLLMClient.DEFAULT_TEMPERATURE,
        max_tokens: int = BaseLLMClient.DEFAULT_MAX_TOKENS,
    ) -> None:
        """Initialize local LLM client.

        Args:
            provider: Which serving framework to use
            model: Model name (uses provider default if not specified)
            base_url: Custom API endpoint (uses provider default if not specified)
            api_key: API key if required (most local deployments don't need this)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        if provider == LLMProvider.OPENAI:
            raise ValueError(
                "Use OpenAIClient for OpenAI API. "
                "LocalLLMClient is for local serving frameworks only."
            )

        self.provider = provider

        # Use environment variable or provider default for base_url
        env_var_name = f"{provider.value.upper()}_API_BASE"
        self.base_url = base_url or os.getenv(env_var_name) or DEFAULT_ENDPOINTS[provider]

        # Use provider default model if not specified
        resolved_model = model or DEFAULT_MODELS[provider]

        super().__init__(
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Local deployments typically don't require API key
        # but some setups might use one for authentication
        self.api_key = api_key or os.getenv(f"{provider.value.upper()}_API_KEY") or "not-required"

        # Use explicit httpx client to avoid proxy parameter conflicts
        # in certain httpx/openai version combinations
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=httpx.Client(),
        )

        logger.info(
            f"LocalLLMClient initialized: provider={provider.value}, "
            f"model={resolved_model}, base_url={self.base_url}"
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Generate a chat completion using local LLM.

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
            error_msg = str(e)
            # Provide more helpful error messages for common issues
            if "Connection refused" in error_msg or "Connection error" in error_msg.lower():
                raise ConnectionError(
                    f"{self.provider.value} server is not running at {self.base_url}. "
                    f"Please start the server first. "
                    f"For vLLM: python -m vllm.entrypoints.openai.api_server --model {self.model}"
                ) from e
            elif "timeout" in error_msg.lower():
                raise TimeoutError(
                    f"{self.provider.value} server timed out at {self.base_url}. "
                    f"The model may be loading or the server is overloaded."
                ) from e
            elif "404" in error_msg or "not found" in error_msg.lower():
                raise ValueError(
                    f"Model '{model or self.model}' not found on {self.provider.value} server. "
                    f"Check available models with: curl {self.base_url}/models"
                ) from e
            logger.error(f"Local LLM API error ({self.provider.value}): {e}")
            raise

    def health_check(self) -> bool:
        """Check if the local LLM server is running and accessible.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            # Try to list models as a health check
            self.client.models.list()
            return True
        except Exception as e:
            logger.warning(f"Health check failed for {self.provider.value}: {e}")
            return False
