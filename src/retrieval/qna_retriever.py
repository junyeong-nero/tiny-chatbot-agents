"""QnA Retriever - First layer search for FAQ matching.

This retriever searches the QnA Vector DB for similar questions
and returns answers with confidence scores.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.vectorstore import QnAVectorStore

logger = logging.getLogger(__name__)


@dataclass
class QnARetrievalResult:
    """Result from QnA retrieval."""

    query: str
    answer: str | None
    question: str | None
    score: float
    source: str
    source_url: str
    is_match: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "question": self.question,
            "score": self.score,
            "source": self.source,
            "source_url": self.source_url,
            "is_match": self.is_match,
            "metadata": self.metadata,
        }


class QnARetriever:
    """QnA Retriever for FAQ matching.

    This is the first layer in the hierarchical search pipeline.
    It searches for similar questions in the QnA database and
    returns matching answers if the similarity score exceeds the threshold.

    Attributes:
        store: QnA Vector Store instance
        threshold: Minimum similarity score for a match (default: 0.85)
    """

    DEFAULT_THRESHOLD = 0.85

    def __init__(
        self,
        store: QnAVectorStore | None = None,
        persist_directory: str | Path = "data/vectordb/qna",
        threshold: float = DEFAULT_THRESHOLD,
        embedding_model: str | None = None,
    ) -> None:
        """Initialize the QnA Retriever.

        Args:
            store: Pre-initialized QnA Vector Store
            persist_directory: Directory for vector store (if store not provided)
            threshold: Minimum similarity score for matching
            embedding_model: Embedding model key from config
        """
        self.threshold = threshold

        if store is not None:
            self.store = store
        else:
            self.store = QnAVectorStore(
                persist_directory=persist_directory,
                embedding_model=embedding_model,
            )

        logger.info(
            f"QnA Retriever initialized. Threshold: {threshold}, "
            f"Documents: {self.store.count()}"
        )

    def retrieve(
        self,
        query: str,
        n_results: int = 1,
        category_filter: str | None = None,
    ) -> QnARetrievalResult:
        """Retrieve the best matching QnA for a query.

        Args:
            query: User's question
            n_results: Number of results to consider
            category_filter: Optional category filter

        Returns:
            QnARetrievalResult with match status and answer
        """
        results = self.store.search(
            query=query,
            n_results=n_results,
            category_filter=category_filter,
        )

        if not results:
            return QnARetrievalResult(
                query=query,
                answer=None,
                question=None,
                score=0.0,
                source="",
                source_url="",
                is_match=False,
                metadata={},
            )

        best = results[0]
        is_match = best.score >= self.threshold

        return QnARetrievalResult(
            query=query,
            answer=best.answer if is_match else None,
            question=best.question,
            score=best.score,
            source=best.source,
            source_url=best.source_url,
            is_match=is_match,
            metadata={
                "category": best.category,
                "sub_category": best.sub_category,
                "id": best.id,
                "all_results": [r.to_dict() for r in results],
            },
        )

    def retrieve_many(
        self,
        query: str,
        n_results: int = 5,
        category_filter: str | None = None,
    ) -> list[QnARetrievalResult]:
        """Retrieve multiple matching QnAs for a query.

        Args:
            query: User's question
            n_results: Number of results to return
            category_filter: Optional category filter

        Returns:
            List of QnARetrievalResult objects
        """
        results = self.store.search(
            query=query,
            n_results=n_results,
            category_filter=category_filter,
        )

        return [
            QnARetrievalResult(
                query=query,
                answer=r.answer if r.score >= self.threshold else None,
                question=r.question,
                score=r.score,
                source=r.source,
                source_url=r.source_url,
                is_match=r.score >= self.threshold,
                metadata={
                    "category": r.category,
                    "sub_category": r.sub_category,
                    "id": r.id,
                },
            )
            for r in results
        ]

    def set_threshold(self, threshold: float) -> None:
        """Update the matching threshold.

        Args:
            threshold: New threshold value (0.0 - 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.threshold = threshold
        logger.info(f"QnA Retriever threshold updated to {threshold}")
