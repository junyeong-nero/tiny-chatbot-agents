"""Query Router for hierarchical question handling.

Implements the 3-layer fallback system:
1. QnA DB Search → 2. ToS RAG → 3. Human Agent
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ResponseSource(Enum):
    """Source of the response."""
    QNA = "qna"
    TOS = "tos"
    HUMAN_AGENT = "human_agent"
    ERROR = "error"


@dataclass
class RouterResponse:
    """Response from the Query Router."""

    query: str
    answer: str | None
    source: ResponseSource
    confidence: float
    citations: list[str]
    metadata: dict[str, Any]
    needs_human: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "source": self.source.value,
            "confidence": self.confidence,
            "citations": self.citations,
            "metadata": self.metadata,
            "needs_human": self.needs_human,
        }


class QueryRouter:
    """Hierarchical Query Router.

    Routes user queries through the 3-layer fallback system:
    1. QnA Vector DB: Fast lookup for FAQ matches
    2. ToS RAG: Semantic search with LLM generation
    3. Human Agent: Fallback for unanswerable queries

    Each layer has a confidence threshold. If the threshold is not met,
    the query is passed to the next layer.

    Attributes:
        qna_retriever: QnA Retriever instance
        tos_retriever: ToS Retriever instance
        qna_threshold: Threshold for QnA matching
        tos_threshold: Threshold for ToS RAG
        feedback_handler: Optional feedback handler for auto-learning
    """

    DEFAULT_QNA_THRESHOLD = 0.85
    DEFAULT_TOS_THRESHOLD = 0.7

    def __init__(
        self,
        qna_retriever: Any = None,
        tos_retriever: Any = None,
        qna_threshold: float = DEFAULT_QNA_THRESHOLD,
        tos_threshold: float = DEFAULT_TOS_THRESHOLD,
        feedback_handler: Any = None,
    ) -> None:
        """Initialize the Query Router.

        Args:
            qna_retriever: QnA Retriever instance
            tos_retriever: ToS Retriever instance
            qna_threshold: Threshold for QnA matching
            tos_threshold: Threshold for ToS RAG
            feedback_handler: Feedback handler for auto-learning
        """
        self.qna_retriever = qna_retriever
        self.tos_retriever = tos_retriever
        self.qna_threshold = qna_threshold
        self.tos_threshold = tos_threshold
        self.feedback_handler = feedback_handler

        logger.info(
            f"Query Router initialized. "
            f"QnA threshold: {qna_threshold}, ToS threshold: {tos_threshold}"
        )

    def handle_query(self, query: str) -> RouterResponse:
        """Handle a user query through the hierarchical pipeline.

        Args:
            query: User's question

        Returns:
            RouterResponse with answer and metadata
        """
        logger.info(f"Handling query: {query[:50]}...")

        # Step 1: QnA Search
        if self.qna_retriever:
            qna_result = self._try_qna(query)
            if qna_result:
                return qna_result

        # Step 2: ToS RAG
        if self.tos_retriever:
            tos_result = self._try_tos(query)
            if tos_result:
                return tos_result

        # Step 3: Human Agent Fallback
        return self._transfer_to_human(query)

    def _try_qna(self, query: str) -> RouterResponse | None:
        """Try to answer using QnA DB.

        Args:
            query: User query

        Returns:
            RouterResponse if match found, None otherwise
        """
        try:
            result = self.qna_retriever.retrieve(query)

            if result.is_match and result.score >= self.qna_threshold:
                logger.info(f"QnA match found with score {result.score:.2f}")

                return RouterResponse(
                    query=query,
                    answer=self._format_qna_answer(result.answer, result.question),
                    source=ResponseSource.QNA,
                    confidence=result.score,
                    citations=[f"FAQ: {result.question}"],
                    metadata={
                        "matched_question": result.question,
                        "category": result.metadata.get("category"),
                        "source_url": result.source_url,
                    },
                    needs_human=False,
                )

            logger.debug(f"QnA score {result.score:.2f} below threshold {self.qna_threshold}")
            return None

        except Exception as e:
            logger.error(f"QnA retrieval failed: {e}")
            return None

    def _try_tos(self, query: str) -> RouterResponse | None:
        """Try to answer using ToS RAG.

        Args:
            query: User query

        Returns:
            RouterResponse if verified answer generated, None otherwise
        """
        try:
            result = self.tos_retriever.retrieve(query)

            if result.verified and result.score >= self.tos_threshold:
                logger.info(f"ToS RAG answer verified with score {result.score:.2f}")

                return RouterResponse(
                    query=query,
                    answer=result.answer,
                    source=ResponseSource.TOS,
                    confidence=result.score,
                    citations=result.citations,
                    metadata={
                        "context": result.context,
                    },
                    needs_human=False,
                )

            if not result.verified:
                logger.warning("ToS RAG answer failed verification")
            else:
                logger.debug(f"ToS score {result.score:.2f} below threshold {self.tos_threshold}")

            return None

        except Exception as e:
            logger.error(f"ToS RAG failed: {e}")
            return None

    def _transfer_to_human(self, query: str) -> RouterResponse:
        """Transfer query to human agent.

        Args:
            query: User query

        Returns:
            RouterResponse indicating human transfer
        """
        logger.info("Transferring to human agent")

        fallback_message = (
            "죄송합니다. 해당 질문에 대해 정확한 답변을 드리기 어렵습니다.\n\n"
            "상담원에게 연결해 드리겠습니다. 잠시만 기다려주세요.\n"
            "(상담 가능 시간: 평일 09:00-18:00)"
        )

        return RouterResponse(
            query=query,
            answer=fallback_message,
            source=ResponseSource.HUMAN_AGENT,
            confidence=0.0,
            citations=[],
            metadata={
                "reason": "No confident answer found",
            },
            needs_human=True,
        )

    def _format_qna_answer(self, answer: str, question: str) -> str:
        """Format QnA answer with source attribution.

        Args:
            answer: The answer text
            question: The matched question

        Returns:
            Formatted answer with attribution
        """
        return f"{answer}\n\n[출처: FAQ]"

    def on_human_response(self, query: str, response: str) -> None:
        """Handle human agent's response for auto-learning.

        Args:
            query: Original user query
            response: Human agent's response
        """
        if self.feedback_handler:
            self.feedback_handler.add_answer(query=query, answer=response)
            logger.info("Human response added to QnA DB")

    def get_stats(self) -> dict[str, Any]:
        """Get router statistics.

        Returns:
            Dictionary with router stats
        """
        return {
            "qna_threshold": self.qna_threshold,
            "tos_threshold": self.tos_threshold,
            "qna_available": self.qna_retriever is not None,
            "tos_available": self.tos_retriever is not None,
            "feedback_enabled": self.feedback_handler is not None,
        }
