"""OpenAI LLM Client for RAG pipeline."""

import logging
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


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


class OpenAIClient:
    """OpenAI API client for chat completions.

    Attributes:
        model: Model name to use (default: gpt-4o-mini)
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens in response
    """

    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
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
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
        )

        logger.info(f"OpenAI client initialized with model: {model}")

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
                temperature=temperature or self.temperature,
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
