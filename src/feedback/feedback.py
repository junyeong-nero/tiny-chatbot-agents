"""Feedback Handler for automatic QnA expansion.

This module captures human agent responses and automatically
adds them to the QnA database for future reference.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEntry:
    """A feedback entry from human agent response."""

    query: str
    answer: str
    source: str = "human_agent"
    created_at: str = ""
    quality_score: float | None = None
    is_duplicate: bool = False

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.query,
            "answer": self.answer,
            "source": self.source,
            "created_at": self.created_at,
            "quality_score": self.quality_score,
            "is_duplicate": self.is_duplicate,
        }


class FeedbackHandler:
    """Handler for capturing and storing human agent responses.

    Implements auto-learning by:
    1. Capturing human agent responses
    2. Checking for duplicates
    3. Optionally validating quality
    4. Adding to QnA Vector DB

    Attributes:
        qna_store: QnA Vector Store instance
        duplicate_threshold: Similarity threshold for duplicate detection
        require_quality_check: Whether to require quality validation
    """

    DEFAULT_DUPLICATE_THRESHOLD = 0.95

    def __init__(
        self,
        qna_store: Any = None,
        duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
        require_quality_check: bool = False,
        min_answer_length: int = 10,
    ) -> None:
        """Initialize the Feedback Handler.

        Args:
            qna_store: QnA Vector Store instance
            duplicate_threshold: Threshold for duplicate detection
            require_quality_check: Whether to validate answer quality
            min_answer_length: Minimum answer length to accept
        """
        self.qna_store = qna_store
        self.duplicate_threshold = duplicate_threshold
        self.require_quality_check = require_quality_check
        self.min_answer_length = min_answer_length

        # Statistics
        self._stats = {
            "total_received": 0,
            "duplicates_skipped": 0,
            "quality_failed": 0,
            "successfully_added": 0,
        }

        logger.info(
            f"Feedback Handler initialized. "
            f"Duplicate threshold: {duplicate_threshold}"
        )

    def add_answer(
        self,
        query: str,
        answer: str,
        source: str = "human_agent",
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackEntry:
        """Add a human agent answer to the QnA database.

        Args:
            query: User's original question
            answer: Human agent's response
            source: Source identifier (default: "human_agent")
            metadata: Optional additional metadata

        Returns:
            FeedbackEntry with addition status
        """
        self._stats["total_received"] += 1

        entry = FeedbackEntry(
            query=query,
            answer=answer,
            source=source,
        )

        # Validate answer length
        if len(answer.strip()) < self.min_answer_length:
            logger.warning(f"Answer too short: {len(answer)} chars")
            return entry

        # Check for duplicates
        if self.qna_store:
            is_duplicate = self._check_duplicate(query)
            if is_duplicate:
                entry.is_duplicate = True
                self._stats["duplicates_skipped"] += 1
                logger.info(f"Duplicate question detected, skipping: {query[:50]}...")
                return entry

        # Quality check (if enabled)
        if self.require_quality_check:
            quality_score = self._check_quality(answer)
            entry.quality_score = quality_score
            if quality_score < 0.5:  # Quality threshold
                self._stats["quality_failed"] += 1
                logger.warning(f"Answer quality check failed: {quality_score:.2f}")
                return entry

        # Add to QnA store
        if self.qna_store:
            try:
                qna_id = self.qna_store.add_qna(
                    question=query,
                    answer=answer,
                    source=source,
                    human_verified=True,
                    category=metadata.get("category", "") if metadata else "",
                )
                self._stats["successfully_added"] += 1
                logger.info(f"Added feedback to QnA DB: {qna_id}")

            except Exception as e:
                logger.error(f"Failed to add to QnA store: {e}")

        return entry

    def add_batch(
        self,
        entries: list[dict[str, Any]],
    ) -> list[FeedbackEntry]:
        """Add multiple answers in batch.

        Args:
            entries: List of dicts with 'query' and 'answer' keys

        Returns:
            List of FeedbackEntry results
        """
        results = []
        for entry in entries:
            result = self.add_answer(
                query=entry.get("query", ""),
                answer=entry.get("answer", ""),
                source=entry.get("source", "human_agent"),
                metadata=entry.get("metadata"),
            )
            results.append(result)
        return results

    def _check_duplicate(self, query: str) -> bool:
        """Check if a similar question already exists.

        Args:
            query: Question to check

        Returns:
            True if duplicate found
        """
        if not self.qna_store:
            return False

        try:
            results = self.qna_store.search(
                query=query,
                n_results=1,
            )

            if results and results[0].score >= self.duplicate_threshold:
                return True

            return False

        except Exception as e:
            logger.error(f"Duplicate check failed: {e}")
            return False

    def _check_quality(self, answer: str) -> float:
        """Check the quality of an answer.

        Basic quality checks:
        - Non-empty content
        - Reasonable length
        - Contains actual information

        Args:
            answer: Answer to check

        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 1.0

        # Check length
        if len(answer) < 20:
            score -= 0.3
        elif len(answer) > 2000:
            score -= 0.1

        # Check for placeholder content
        placeholders = ["죄송합니다", "모르겠습니다", "확인 후", "알 수 없습니다"]
        for p in placeholders:
            if p in answer:
                score -= 0.2

        # Check for actual content (not just greetings)
        if len(answer.split()) < 5:
            score -= 0.3

        return max(0.0, min(1.0, score))

    def merge_duplicates(self) -> int:
        """Merge duplicate questions in the store.

        Groups similar questions and keeps the best answer.

        Returns:
            Number of merged entries
        """
        if not self.qna_store:
            return 0

        # TODO: Implement duplicate merging logic
        # This would:
        # 1. Find clusters of similar questions
        # 2. Keep the best-quality answer
        # 3. Remove duplicates
        logger.info("Duplicate merging not yet implemented")
        return 0

    def get_stats(self) -> dict[str, Any]:
        """Get feedback handler statistics.

        Returns:
            Dictionary with stats
        """
        return {
            **self._stats,
            "duplicate_threshold": self.duplicate_threshold,
            "require_quality_check": self.require_quality_check,
        }

    def clear_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            "total_received": 0,
            "duplicates_skipped": 0,
            "quality_failed": 0,
            "successfully_added": 0,
        }
