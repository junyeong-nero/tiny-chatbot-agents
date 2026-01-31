"""Unified client for frontier LLM models (Claude, GPT, Gemini).

This module provides a unified interface for calling frontier model APIs
for golden answer generation and LLM-as-a-Judge evaluation.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import EvaluationConfig

logger = logging.getLogger(__name__)


def _get_config() -> "EvaluationConfig":
    """Get evaluation config (lazy import to avoid circular dependency)."""
    from .config import get_config

    return get_config()


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


class JudgeModelSelector:
    """Selects a different model for judging than was used for generation.

    This class helps prevent circular evaluation bias by ensuring the judge
    model is different from the model that generated the golden answers.

    Uses configuration from evaluation_config.yaml for diversity pairs.
    """

    # Fallback mapping by provider (used when config not available)
    PROVIDER_FALLBACKS: dict[str, tuple[str, str]] = {
        "openai": ("anthropic", "claude-sonnet-4-20250514"),
        "anthropic": ("openai", "gpt-4o"),
        "google": ("openai", "gpt-4o"),
    }

    @classmethod
    def _get_diversity_pairs_from_config(cls) -> dict[str, tuple[str, str]]:
        """Get diversity pairs from config file.

        Returns:
            Dictionary mapping generator model -> (provider, model)
        """
        try:
            config = _get_config()
            pairs = {}
            for model, mapping in config.diversity_pairs.items():
                pairs[model] = (mapping.provider, mapping.model)
            return pairs
        except Exception as e:
            logger.debug(f"Could not load diversity pairs from config: {e}")
            return {}

    @classmethod
    def get_diverse_judge(
        cls,
        generator_model: str,
        generator_provider: str | None = None,
    ) -> tuple[str, str]:
        """Get a different model for judging to avoid circular bias.

        Args:
            generator_model: The model used to generate golden answers
            generator_provider: Optional provider name for fallback lookup

        Returns:
            Tuple of (provider, model) for the judge
        """
        # Try to get diversity pairs from config
        diversity_pairs = cls._get_diversity_pairs_from_config()

        # Try exact model match first
        if generator_model in diversity_pairs:
            return diversity_pairs[generator_model]

        # Try provider-based fallback
        if generator_provider and generator_provider in cls.PROVIDER_FALLBACKS:
            return cls.PROVIDER_FALLBACKS[generator_provider]

        # Default fallback: use GPT-4o
        logger.warning(
            f"Unknown generator model '{generator_model}', defaulting to GPT-4o as judge"
        )
        return ("openai", "gpt-4o")

    @classmethod
    def is_same_model(
        cls,
        model1: str,
        model2: str,
        provider1: str | None = None,
        provider2: str | None = None,
    ) -> bool:
        """Check if two models are effectively the same.

        Args:
            model1: First model name
            model2: Second model name
            provider1: Optional provider for first model
            provider2: Optional provider for second model

        Returns:
            True if models are the same or from the same family
        """
        # Exact match
        if model1 == model2:
            return True

        # Normalize model names for comparison
        m1_normalized = model1.lower().replace("-", "").replace("_", "")
        m2_normalized = model2.lower().replace("-", "").replace("_", "")

        if m1_normalized == m2_normalized:
            return True

        # Check if same provider (different models from same provider may have similar biases)
        if provider1 and provider2 and provider1 == provider2:
            logger.warning(
                f"Models '{model1}' and '{model2}' are from the same provider "
                f"'{provider1}', which may introduce similar biases"
            )

        return False

    @classmethod
    def validate_diversity(
        cls,
        generator_model: str,
        judge_model: str,
        generator_provider: str | None = None,
        judge_provider: str | None = None,
        strict: bool = False,
    ) -> bool:
        """Validate that generator and judge models are sufficiently different.

        Args:
            generator_model: Model used to generate golden answers
            judge_model: Model used for judging
            generator_provider: Provider of generator model
            judge_provider: Provider of judge model
            strict: If True, raise error on same model; if False, just warn

        Returns:
            True if models are different, False otherwise

        Raises:
            ValueError: If strict=True and models are the same
        """
        if cls.is_same_model(generator_model, judge_model, generator_provider, judge_provider):
            msg = (
                f"Circular evaluation bias detected: generator model '{generator_model}' "
                f"and judge model '{judge_model}' are the same or similar. "
                "This may lead to biased evaluation results."
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
            return False

        return True


def _get_provider_defaults(provider: str) -> tuple[str, str]:
    """Get default model and API key env for a provider from config.

    Args:
        provider: Provider name

    Returns:
        Tuple of (default_model, api_key_env)
    """
    # Hardcoded fallbacks
    fallbacks = {
        "openai": ("gpt-4o", "OPENAI_API_KEY"),
        "anthropic": ("claude-sonnet-4-20250514", "ANTHROPIC_API_KEY"),
        "google": ("gemini-1.5-pro", "GOOGLE_API_KEY"),
    }

    try:
        config = _get_config()
        provider_default = config.get_provider_default(provider)
        if provider_default:
            return (provider_default.model, provider_default.api_key_env)
    except Exception as e:
        logger.debug(f"Could not load provider defaults from config: {e}")

    return fallbacks.get(provider, ("gpt-4o", "OPENAI_API_KEY"))


def create_frontier_client(
    provider: str = "openai",
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> FrontierClient:
    """Factory function to create a frontier client.

    Args:
        provider: Provider name ('openai', 'anthropic', 'google')
        model: Optional model override (uses config default if not provided)
        temperature: Sampling temperature (uses config default if not provided)
        max_tokens: Maximum tokens (uses config default if not provided)

    Returns:
        Configured FrontierClient
    """
    provider_enum = FrontierProvider(provider.lower())

    # Get defaults from config
    default_model, api_key_env = _get_provider_defaults(provider)

    # Get temperature and max_tokens from config if not provided
    try:
        eval_config = _get_config()
        config_temperature = eval_config.judge.temperature
        config_max_tokens = eval_config.judge.max_tokens
    except Exception:
        config_temperature = 0.0
        config_max_tokens = 2048

    final_temperature = temperature if temperature is not None else config_temperature
    final_max_tokens = max_tokens if max_tokens is not None else config_max_tokens

    model_config = FrontierModelConfig(
        provider=provider_enum,
        model=model or default_model,
        api_key_env=api_key_env,
        temperature=final_temperature,
        max_tokens=final_max_tokens,
    )

    return FrontierClient(model_config)
