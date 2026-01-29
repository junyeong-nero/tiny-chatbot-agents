"""Human-in-the-loop QnA Backfill Pipeline.

This module provides functionality to append human agent (상담원) answers
into the QnA vector store, enabling continuous learning from live support interactions.

Usage:
    from src.vectorstore.backfill import HumanAgentBackfill

    backfill = HumanAgentBackfill()
    backfill.add_agent_answer(
        question="계좌 해지 방법이 뭐야?",
        answer="고객센터 또는 앱에서 해지 신청 가능합니다.",
        category="계좌",
    )
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .qna_store import QnAVectorStore

logger = logging.getLogger(__name__)


@dataclass
class BackfillResult:
    """Result of a backfill operation."""

    success: bool
    added_count: int
    skipped_count: int
    errors: list[str]
    added_ids: list[str]

    def __str__(self) -> str:
        return (
            f"BackfillResult(success={self.success}, "
            f"added={self.added_count}, skipped={self.skipped_count}, "
            f"errors={len(self.errors)})"
        )


class HumanAgentBackfill:
    """Backfill pipeline for adding human agent answers to QnA store.

    This class implements the "상담원 답변 자동 추가" feature described in README,
    enabling the system to learn from live customer support interactions.

    Agent answers are stored with:
    - source: "human_agent" (distinguishes from FAQ)
    - human_verified: True (agent-provided answers are considered verified)

    Attributes:
        qna_store: The QnA vector store instance
    """

    SOURCE_HUMAN_AGENT = "human_agent"

    def __init__(
        self,
        qna_store: QnAVectorStore | None = None,
        persist_directory: str | Path = "data/vectordb/qna",
        embedding_model: str | None = None,
    ) -> None:
        """Initialize the backfill pipeline.

        Args:
            qna_store: Existing QnA store instance (created if None)
            persist_directory: Directory for ChromaDB data (if creating new store)
            embedding_model: Model key from embedding_config.yaml
        """
        if qna_store is not None:
            self.qna_store = qna_store
        else:
            self.qna_store = QnAVectorStore(
                persist_directory=persist_directory,
                embedding_model=embedding_model,
            )

    def add_agent_answer(
        self,
        question: str,
        answer: str,
        category: str = "",
        sub_category: str = "",
        source_url: str = "",
        agent_id: str | None = None,
        session_id: str | None = None,
        created_at: str | None = None,
    ) -> str:
        """Add a single agent answer to the QnA store.

        Args:
            question: Customer's question
            answer: Agent's answer
            category: Question category (e.g., "계좌", "환불")
            sub_category: Question sub-category
            source_url: Reference URL if applicable
            agent_id: Optional identifier of the responding agent
            session_id: Optional session identifier for tracking
            created_at: Timestamp string (auto-generated if None)

        Returns:
            The ID of the added entry
        """
        question = question.strip()
        answer = answer.strip()

        if not question or not answer:
            raise ValueError("질문/답변이 비어있습니다.")

        if created_at is None:
            created_at = datetime.now().isoformat()

        qna_id = None
        if session_id:
            import hashlib

            hash_input = f"{question}_{session_id}_{created_at}"
            qna_id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        qna_id = self.qna_store.add_qna(
            question=question,
            answer=answer,
            category=category,
            sub_category=sub_category,
            source=self.SOURCE_HUMAN_AGENT,
            source_url=source_url,
            human_verified=True,
            created_at=created_at,
            qna_id=qna_id,
            agent_id=agent_id,
            session_id=session_id,
        )

        logger.info(f"상담원 답변 추가 완료: {qna_id} (category={category}, agent_id={agent_id})")
        return qna_id

    def add_agent_answers_batch(
        self,
        items: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> BackfillResult:
        """Add multiple agent answers in batches.

        Args:
            items: List of agent answer dictionaries with keys:
                - question (required): Customer's question
                - answer (required): Agent's answer
                - category, sub_category, source_url (optional)
                - agent_id, session_id, created_at (optional)
            batch_size: Number of items to process per batch

        Returns:
            BackfillResult with operation details
        """
        added_ids: list[str] = []
        skipped_count = 0
        errors: list[str] = []

        total = len(items)
        for start in range(0, total, batch_size):
            batch = items[start : start + batch_size]
            for i, item in enumerate(batch, start=start):
                try:
                    question = item.get("question", "").strip()
                    answer = item.get("answer", "").strip()

                    if not question or not answer:
                        logger.warning(f"항목 {i}: 질문 또는 답변이 비어있어 건너뜁니다.")
                        skipped_count += 1
                        continue

                    qna_id = self.add_agent_answer(
                        question=question,
                        answer=answer,
                        category=item.get("category", ""),
                        sub_category=item.get("sub_category", ""),
                        source_url=item.get("source_url", ""),
                        agent_id=item.get("agent_id"),
                        session_id=item.get("session_id"),
                        created_at=item.get("created_at"),
                    )
                    added_ids.append(qna_id)

                except Exception as e:
                    error_msg = f"항목 {i}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)

        return BackfillResult(
            success=len(errors) == 0,
            added_count=len(added_ids),
            skipped_count=skipped_count,
            errors=errors,
            added_ids=added_ids,
        )

    def load_from_json(self, json_path: str | Path) -> BackfillResult:
        """Load agent answers from a JSON file.

        Expected JSON format (same as QnA crawl format):
        [
            {
                "question": "계좌 해지 방법이 뭐야?",
                "answer": "고객센터 또는 앱에서 해지 신청 가능합니다.",
                "category": "계좌",
                "sub_category": "",
                "agent_id": "agent_001",
                "session_id": "sess_12345",
                "created_at": "2026-01-28T10:30:00"
            }
        ]

        Args:
            json_path: Path to JSON file with agent answers

        Returns:
            BackfillResult with operation details
        """
        json_path = Path(json_path)

        if not json_path.exists():
            return BackfillResult(
                success=False,
                added_count=0,
                skipped_count=0,
                errors=[f"파일을 찾을 수 없습니다: {json_path}"],
                added_ids=[],
            )

        try:
            data_text = json_path.read_text(encoding="utf-8")
            data = json.loads(data_text)
        except json.JSONDecodeError as e:
            return BackfillResult(
                success=False,
                added_count=0,
                skipped_count=0,
                errors=[f"JSON 파싱 오류: {e}"],
                added_ids=[],
            )

        if not isinstance(data, list):
            return BackfillResult(
                success=False,
                added_count=0,
                skipped_count=0,
                errors=["JSON 형식 오류: 배열이 아닙니다."],
                added_ids=[],
            )

        logger.info(f"상담원 답변 {len(data)}개 로드 중: {json_path}")
        return self.add_agent_answers_batch(data)

    def search_existing(
        self,
        question: str,
        n_results: int = 3,
        score_threshold: float = 0.85,
    ) -> list[dict[str, Any]]:
        """Search for existing similar questions to avoid duplicates.

        Args:
            question: Question to search for
            n_results: Maximum number of results
            score_threshold: Minimum similarity score to consider as duplicate

        Returns:
            List of similar existing entries
        """
        results = self.qna_store.search(
            query=question,
            n_results=n_results,
            score_threshold=score_threshold,
        )
        return [r.to_dict() for r in results]

    def add_if_not_duplicate(
        self,
        question: str,
        answer: str,
        category: str = "",
        sub_category: str = "",
        duplicate_threshold: float = 0.90,
        **kwargs: Any,
    ) -> tuple[str | None, bool]:
        """Add agent answer only if no near-duplicate exists.

        Args:
            question: Customer's question
            answer: Agent's answer
            category: Question category
            sub_category: Question sub-category
            duplicate_threshold: Similarity threshold to consider as duplicate
            **kwargs: Additional arguments passed to add_agent_answer

        Returns:
            Tuple of (qna_id or None, is_duplicate)
        """
        question = question.strip()
        answer = answer.strip()

        if not question or not answer:
            raise ValueError("질문/답변이 비어있습니다.")

        existing = self.search_existing(
            question=question,
            n_results=1,
            score_threshold=duplicate_threshold,
        )

        if existing:
            logger.info(
                f"중복 질문 발견 (score={existing[0]['score']:.3f}): "
                f"{existing[0]['question'][:50]}..."
            )
            return None, True

        qna_id = self.add_agent_answer(
            question=question,
            answer=answer,
            category=category,
            sub_category=sub_category,
            **kwargs,
        )
        return qna_id, False

    def get_stats(self) -> dict[str, Any]:
        """Get basic statistics about stored entries.

        Returns:
            Dictionary with total entry count
        """
        total = self.qna_store.count()
        return {"total_entries": total}
