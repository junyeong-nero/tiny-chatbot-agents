"""Unified client for frontier LLM models (Claude, GPT, Gemini).

This module provides a unified interface for calling frontier model APIs
for golden answer generation and LLM-as-a-Judge evaluation.
"""

import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class FrontierProvider(Enum):
    """Supported frontier model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class FrontierModelConfig:
    """Configuration for a frontier model."""

    provider: FrontierProvider
    model: str
    api_key_env: str
    temperature: float = 0.0
    max_tokens: int = 2048

    @classmethod
    def default_openai(cls) -> "FrontierModelConfig":
        return cls(
            provider=FrontierProvider.OPENAI,
            model="gpt-4o",
            api_key_env="OPENAI_API_KEY",
        )

    @classmethod
    def default_anthropic(cls) -> "FrontierModelConfig":
        return cls(
            provider=FrontierProvider.ANTHROPIC,
            model="claude-sonnet-4-20250514",
            api_key_env="ANTHROPIC_API_KEY",
        )

    @classmethod
    def default_google(cls) -> "FrontierModelConfig":
        return cls(
            provider=FrontierProvider.GOOGLE,
            model="gemini-1.5-pro",
            api_key_env="GOOGLE_API_KEY",
        )


class FrontierClient:
    """Unified client for frontier models (Claude, GPT, Gemini).

    Provides a consistent interface for generating completions across
    different frontier model providers.
    """

    def __init__(self, config: FrontierModelConfig | None = None) -> None:
        """Initialize the frontier client.

        Args:
            config: Model configuration. Defaults to OpenAI GPT-4o.
        """
        self.config = config or FrontierModelConfig.default_openai()
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the provider-specific client."""
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found. Set {self.config.api_key_env} environment variable."
            )

        if self.config.provider == FrontierProvider.OPENAI:
            self._init_openai(api_key)
        elif self.config.provider == FrontierProvider.ANTHROPIC:
            self._init_anthropic(api_key)
        elif self.config.provider == FrontierProvider.GOOGLE:
            self._init_google(api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _init_openai(self, api_key: str) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key)
            logger.info(f"Initialized OpenAI client with model: {self.config.model}")
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

    def _init_anthropic(self, api_key: str) -> None:
        """Initialize Anthropic client."""
        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key)
            logger.info(f"Initialized Anthropic client with model: {self.config.model}")
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

    def _init_google(self, api_key: str) -> None:
        """Initialize Google Gemini client."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.config.model)
            logger.info(f"Initialized Google client with model: {self.config.model}")
        except ImportError:
            raise ImportError(
                "google-generativeai package required. "
                "Install with: pip install google-generativeai"
            )

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a completion using the configured frontier model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Generated text content
        """
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens or self.config.max_tokens

        if self.config.provider == FrontierProvider.OPENAI:
            return self._generate_openai(messages, temp, tokens)
        elif self.config.provider == FrontierProvider.ANTHROPIC:
            return self._generate_anthropic(messages, temp, tokens)
        elif self.config.provider == FrontierProvider.GOOGLE:
            return self._generate_google(messages, temp, tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _generate_openai(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate using OpenAI API."""
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def _generate_anthropic(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate using Anthropic API."""
        # Extract system message if present
        system_content = ""
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                filtered_messages.append(msg)

        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_content if system_content else None,
            messages=filtered_messages,
        )
        return response.content[0].text

    def _generate_google(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate using Google Gemini API."""
        # Convert messages to Gemini format
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt = "\n\n".join(prompt_parts)
        prompt += "\n\nAssistant:"

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        response = self._client.generate_content(
            prompt,
            generation_config=generation_config,
        )
        return response.text

    def generate_json(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Generate a JSON response using the configured frontier model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Parsed JSON dict
        """
        response = self.generate(messages, temperature, max_tokens)
        return self._parse_json(response)

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Parse JSON from text, handling markdown code blocks."""
        # Try to extract JSON from code block
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            json_str = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            json_str = text[start:end].strip()
        else:
            json_str = text.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            logger.debug(f"Raw response: {text}")
            return {}

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.model

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self.config.provider.value


def create_frontier_client(
    provider: str = "openai",
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> FrontierClient:
    """Factory function to create a frontier client.

    Args:
        provider: Provider name ('openai', 'anthropic', 'google')
        model: Optional model override
        temperature: Sampling temperature
        max_tokens: Maximum tokens

    Returns:
        Configured FrontierClient
    """
    provider_enum = FrontierProvider(provider.lower())

    if provider_enum == FrontierProvider.OPENAI:
        config = FrontierModelConfig(
            provider=provider_enum,
            model=model or "gpt-4o",
            api_key_env="OPENAI_API_KEY",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider_enum == FrontierProvider.ANTHROPIC:
        config = FrontierModelConfig(
            provider=provider_enum,
            model=model or "claude-sonnet-4-20250514",
            api_key_env="ANTHROPIC_API_KEY",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider_enum == FrontierProvider.GOOGLE:
        config = FrontierModelConfig(
            provider=provider_enum,
            model=model or "gemini-1.5-pro",
            api_key_env="GOOGLE_API_KEY",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return FrontierClient(config)
