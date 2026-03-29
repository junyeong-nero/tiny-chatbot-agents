from dataclasses import dataclass, field
from typing import Any, Literal

from src.pipeline.models import PipelineResponse, ResponseSource


@dataclass
class GraphState:
    query: str
    qna_results: list[dict[str, Any]] = field(default_factory=list)
    qna_score: float = 0.0
    tos_results: list[dict[str, Any]] = field(default_factory=list)
    tos_score: float = 0.0
    answer: str = ""
    source: ResponseSource | None = None
    confidence: float = 0.0
    response_mode: str = "answer"
    context: list[dict[str, Any]] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    verified: bool = True
    verification_score: float = 1.0
    verification_issues: list[str] = field(default_factory=list)
    route: (
        Literal[
            "qna_ok",
            "qna_mid",
            "tos",
            "tos_ok",
            "tos_mid",
            "tos_low",
            "no_context",
        ]
        | None
    ) = None
    section_reference: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    response: PipelineResponse | None = None
