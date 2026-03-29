from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResponseSource(Enum):
    QNA = "qna"
    TOS = "tos"
    NO_CONTEXT = "no_context"


@dataclass
class PipelineResponse:
    query: str
    answer: str
    source: ResponseSource
    confidence: float
    response_mode: str = "answer"
    context: list[dict[str, Any]] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    verified: bool = True
    verification_score: float = 1.0
    verification_issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "source": self.source.value,
            "confidence": self.confidence,
            "response_mode": self.response_mode,
            "context": self.context,
            "citations": self.citations,
            "metadata": self.metadata,
            "verified": self.verified,
            "verification_score": self.verification_score,
            "verification_issues": self.verification_issues,
        }
