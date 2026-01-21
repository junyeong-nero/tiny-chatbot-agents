"""LLM Client modules.

Provides LLM client implementations for both testing and production use.

For Testing:
    Use OpenAIClient (requires OPENAI_API_KEY)

For Production:
    Use LocalLLMClient with vLLM, sglang, or ollama
    This ensures data stays within your infrastructure.

Quick Start:
    # Production (recommended)
    from src.llm import create_llm_client, LLMProvider
    client = create_llm_client(provider=LLMProvider.VLLM)

    # Testing only
    client = create_llm_client(provider=LLMProvider.OPENAI)
"""

from .base import BaseLLMClient, LLMProvider, LLMResponse
from .factory import create_llm_client, get_default_provider
from .local_client import LocalLLMClient
from .openai_client import OpenAIClient

__all__ = [
    # Base classes
    "BaseLLMClient",
    "LLMResponse",
    "LLMProvider",
    # Clients
    "OpenAIClient",
    "LocalLLMClient",
    # Factory
    "create_llm_client",
    "get_default_provider",
]
