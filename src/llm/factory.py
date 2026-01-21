"""Factory for creating LLM clients.

This module provides a convenient way to create LLM clients based on
configuration or environment variables.

Usage:
    # From environment variable (LLM_PROVIDER)
    client = create_llm_client()

    # Explicit provider for production (local)
    client = create_llm_client(provider="vllm")

    # For testing only (uses external API)
    client = create_llm_client(provider="openai")
"""

import logging
import os
from typing import Any

from .base import BaseLLMClient, LLMProvider

logger = logging.getLogger(__name__)


def create_llm_client(
    provider: str | LLMProvider | None = None,
    **kwargs: Any,
) -> BaseLLMClient:
    """Create an LLM client based on the specified provider.

    Args:
        provider: Provider name or LLMProvider enum.
                  If not specified, uses LLM_PROVIDER env var.
                  Defaults to "vllm" for production use.
        **kwargs: Additional arguments passed to the client constructor.
                  Common options:
                  - model: Model name
                  - temperature: Sampling temperature
                  - max_tokens: Maximum tokens in response
                  - base_url: Custom API endpoint (for local clients)
                  - api_key: API key (for OpenAI client)

    Returns:
        An LLM client instance

    Raises:
        ValueError: If an unknown provider is specified

    Examples:
        # Production: Use local vLLM server
        client = create_llm_client(provider="vllm")

        # Production: Use ollama with specific model
        client = create_llm_client(provider="ollama", model="mistral:7b")

        # Testing only: Use OpenAI API
        client = create_llm_client(provider="openai", model="gpt-4o-mini")
    """
    # Resolve provider
    if provider is None:
        provider_str = os.getenv("LLM_PROVIDER", "vllm")
    elif isinstance(provider, LLMProvider):
        provider_str = provider.value
    else:
        provider_str = provider.lower()

    # Convert string to enum
    try:
        provider_enum = LLMProvider(provider_str)
    except ValueError:
        valid_providers = [p.value for p in LLMProvider]
        raise ValueError(
            f"Unknown provider: {provider_str}. "
            f"Valid providers: {valid_providers}"
        )

    # Create appropriate client
    if provider_enum == LLMProvider.OPENAI:
        logger.warning(
            "Using OpenAI provider. This should only be used for testing. "
            "For production, use local serving frameworks (vllm, sglang, ollama)."
        )
        from .openai_client import OpenAIClient

        return OpenAIClient(**kwargs)
    else:
        from .local_client import LocalLLMClient

        return LocalLLMClient(provider=provider_enum, **kwargs)


def get_default_provider() -> LLMProvider:
    """Get the default LLM provider from environment.

    Returns:
        LLMProvider enum value
    """
    provider_str = os.getenv("LLM_PROVIDER", "vllm")
    try:
        return LLMProvider(provider_str)
    except ValueError:
        logger.warning(f"Unknown LLM_PROVIDER: {provider_str}, defaulting to vllm")
        return LLMProvider.VLLM
