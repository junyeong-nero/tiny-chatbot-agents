"""Base LLM Client interface."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"  # For testing only
    VLLM = "vllm"
    SGLANG = "sglang"
    OLLAMA = "ollama"


@dataclass
class LLMResponse:
    """Response from LLM."""

    content: str
    model: str
    usage: dict[str, int]
    finish_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
        }


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients.

    All LLM implementations must inherit from this class.
    This ensures consistent interface across different providers.
    """

    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        model: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        """Initialize LLM client.

        Args:
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
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
        pass

    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate response with RAG context.

        Args:
            query: User's question
            context: Retrieved context to use
            system_prompt: Optional system prompt override

        Returns:
            LLMResponse with generated answer
        """
        if system_prompt is None:
            system_prompt = """당신은 금융 서비스 고객 상담 AI입니다.

규칙:
1. 제공된 컨텍스트 정보만을 기반으로 정확하게 답변하세요.
2. 컨텍스트에 없는 내용은 절대 지어내지 마세요.
3. 확실하지 않으면 "해당 내용은 확인되지 않습니다"라고 답변하세요.
4. 친절하고 전문적인 톤으로 답변하세요.
5. 답변 끝에 참조한 출처가 있다면 명시하세요."""

        user_prompt = f"""다음 정보를 참고하여 질문에 답변해주세요.

[참고 정보]
{context}

[질문]
{query}

[답변]"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self.generate(messages)
