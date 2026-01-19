"""ToS Retriever - Second layer RAG search for Terms of Service.

This retriever searches the ToS Vector DB and optionally Graph DB
for relevant sections and generates answers using LLM.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.vectorstore import ToSVectorStore

logger = logging.getLogger(__name__)


@dataclass
class ToSRetrievalResult:
    """Result from ToS RAG retrieval."""

    query: str
    answer: str | None
    context: list[dict[str, Any]]
    citations: list[str]
    score: float
    verified: bool
    source: str = "ToS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "context": self.context,
            "citations": self.citations,
            "score": self.score,
            "verified": self.verified,
            "source": self.source,
        }


class ToSRetriever:
    """ToS Retriever with RAG for Terms of Service queries.

    This is the second layer in the hierarchical search pipeline.
    It searches for relevant ToS sections and generates answers
    using a local LLM.

    Attributes:
        store: ToS Vector Store instance
        llm_client: LLM client for answer generation
        threshold: Minimum similarity score for context
        verifier: Optional answer verifier
    """

    DEFAULT_THRESHOLD = 0.7

    def __init__(
        self,
        store: ToSVectorStore | None = None,
        persist_directory: str | Path = "data/vectordb/tos",
        threshold: float = DEFAULT_THRESHOLD,
        embedding_model: str | None = None,
        llm_client: Any = None,
        verifier: Any = None,
    ) -> None:
        """Initialize the ToS Retriever.

        Args:
            store: Pre-initialized ToS Vector Store
            persist_directory: Directory for vector store
            threshold: Minimum similarity score for context
            embedding_model: Embedding model key from config
            llm_client: LLM client for answer generation
            verifier: Answer verifier instance
        """
        self.threshold = threshold
        self.llm_client = llm_client
        self.verifier = verifier

        if store is not None:
            self.store = store
        else:
            self.store = ToSVectorStore(
                persist_directory=persist_directory,
                embedding_model=embedding_model,
            )

        logger.info(
            f"ToS Retriever initialized. Threshold: {threshold}, "
            f"Documents: {self.store.count()}"
        )

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        generate_answer: bool = True,
    ) -> ToSRetrievalResult:
        """Retrieve relevant ToS context and generate answer.

        Args:
            query: User's question
            n_results: Number of context chunks to retrieve
            generate_answer: Whether to generate LLM answer

        Returns:
            ToSRetrievalResult with answer and citations
        """
        # Search for relevant ToS sections
        results = self.store.search(
            query=query,
            n_results=n_results,
            score_threshold=self.threshold,
        )

        if not results:
            return ToSRetrievalResult(
                query=query,
                answer=None,
                context=[],
                citations=[],
                score=0.0,
                verified=False,
            )

        # Build context from results
        context = [
            {
                "section_title": r.section_title,
                "section_content": r.section_content,
                "document_title": r.document_title,
                "score": r.score,
            }
            for r in results
        ]

        avg_score = sum(r.score for r in results) / len(results)

        # Generate answer if LLM client available
        answer = None
        citations = []
        verified = False

        if generate_answer and self.llm_client:
            answer, citations = self._generate_answer(query, context)

            # Verify answer if verifier available
            if self.verifier and answer:
                verified = self.verifier.verify(
                    question=query,
                    answer=answer,
                    context=context,
                )
            else:
                # Without verifier, assume verified if answer generated
                verified = answer is not None
        else:
            # Without LLM, return context only
            verified = len(context) > 0

        return ToSRetrievalResult(
            query=query,
            answer=answer,
            context=context,
            citations=citations,
            score=avg_score,
            verified=verified,
        )

    def _generate_answer(
        self,
        query: str,
        context: list[dict[str, Any]],
    ) -> tuple[str | None, list[str]]:
        """Generate answer using LLM based on context.

        Args:
            query: User's question
            context: Retrieved ToS sections

        Returns:
            Tuple of (answer, citations)
        """
        if not self.llm_client:
            return None, []

        # Build context string
        context_str = self._format_context(context)

        # System prompt for answer generation
        system_prompt = """당신은 약관 및 이용조건에 대해 답변하는 AI 상담원입니다.
        
규칙:
1. 제공된 약관 내용만을 기반으로 답변하세요.
2. 약관에 없는 내용은 절대 지어내지 마세요.
3. 확실하지 않으면 "해당 내용은 약관에서 확인되지 않습니다"라고 답변하세요.
4. 답변 마지막에 반드시 참조한 조항을 명시하세요. 예: [참조: 제3조 2항]"""

        user_prompt = f"""다음 약관 내용을 참고하여 질문에 답변해주세요.

[약관 내용]
{context_str}

[질문]
{query}

[답변]"""

        try:
            response = self.llm_client.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            answer = response.get("content", "")

            # Extract citations from answer
            citations = self._extract_citations(answer)

            return answer, citations

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None, []

    def _format_context(self, context: list[dict[str, Any]]) -> str:
        """Format context for LLM prompt."""
        parts = []
        for i, c in enumerate(context, 1):
            parts.append(
                f"[{i}] {c['section_title']}\n{c['section_content']}"
            )
        return "\n\n".join(parts)

    def _extract_citations(self, answer: str) -> list[str]:
        """Extract citations from answer text."""
        import re

        citations = []
        # Pattern: [참조: 제N조 N항]
        pattern = r"\[참조:\s*([^\]]+)\]"
        matches = re.findall(pattern, answer)
        citations.extend(matches)
        return citations

    def set_threshold(self, threshold: float) -> None:
        """Update the context threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.threshold = threshold
        logger.info(f"ToS Retriever threshold updated to {threshold}")
