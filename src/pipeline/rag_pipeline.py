"""Unified RAG Pipeline for QnA and ToS retrieval.

Flow:
1. User query → Search QnA DB
2. If similar question found (score >= threshold) → Use QnA as context for LLM
3. If no match → Search ToS DB → Use ToS sections as context for LLM
4. Verify answer using AnswerVerifier (hallucination detection)
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.llm import BaseLLMClient, create_llm_client
from src.vectorstore import QnAVectorStore, ToSVectorStore
from src.verifier import AnswerVerifier, VerificationResult

logger = logging.getLogger(__name__)


class ResponseSource(Enum):
    """Source of the response."""

    QNA = "qna"
    TOS = "tos"
    NO_CONTEXT = "no_context"


@dataclass
class PipelineResponse:
    """Response from RAG Pipeline."""

    query: str
    answer: str
    source: ResponseSource
    confidence: float
    response_mode: str = "answer"
    context: list[dict[str, Any]] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Verification fields
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


class RAGPipeline:
    """Unified RAG Pipeline with QnA-first, ToS-fallback strategy.

    The pipeline follows this flow:
    1. Search QnA DB for similar questions
    2. If score >= qna_threshold: Use matched Q&A as context for LLM answer
    3. If score < qna_threshold: Search ToS DB for relevant sections
    4. Use ToS sections as context for LLM answer
    5. Verify answer using AnswerVerifier (hallucination detection)

    Attributes:
        qna_store: QnA Vector Store
        tos_store: ToS Vector Store
        llm: OpenAI LLM client
        verifier: Answer verifier for hallucination detection
        qna_threshold: Threshold for QnA matching
        tos_threshold: Threshold for ToS retrieval
    """

    DEFAULT_QNA_THRESHOLD = 0.80
    DEFAULT_TOS_THRESHOLD = 0.65
    DEFAULT_QNA_MID_THRESHOLD = 0.70
    DEFAULT_TOS_MID_THRESHOLD = 0.55
    DEFAULT_TOS_LOW_THRESHOLD = 0.40
    DEFAULT_VERIFICATION_THRESHOLD = 0.7

    def __init__(
        self,
        llm: BaseLLMClient | None = None,
        qna_store: QnAVectorStore | None = None,
        tos_store: ToSVectorStore | None = None,
        verifier: AnswerVerifier | None = None,
        qna_db_path: str | Path = "data/vectordb/qna",
        tos_db_path: str | Path = "data/vectordb/tos",
        embedding_model: str | None = None,
        qna_threshold: float = DEFAULT_QNA_THRESHOLD,
        tos_threshold: float = DEFAULT_TOS_THRESHOLD,
        qna_mid_threshold: float = DEFAULT_QNA_MID_THRESHOLD,
        tos_mid_threshold: float = DEFAULT_TOS_MID_THRESHOLD,
        tos_low_threshold: float = DEFAULT_TOS_LOW_THRESHOLD,
        enable_verification: bool = True,
        verification_threshold: float = DEFAULT_VERIFICATION_THRESHOLD,
        enable_hybrid_tos_search: bool = False,
    ) -> None:
        """Initialize RAG Pipeline.

        Args:
            llm: OpenAI client (created if not provided)
            qna_store: Pre-initialized QnA store
            tos_store: Pre-initialized ToS store
            verifier: Answer verifier (created if not provided and enabled)
            qna_db_path: Path to QnA vector DB
            tos_db_path: Path to ToS vector DB
            embedding_model: Embedding model key
            qna_threshold: Minimum score for QnA match
            tos_threshold: Minimum score for ToS retrieval
            qna_mid_threshold: Mid-band score for QnA limited answer
            tos_mid_threshold: Mid-band score for ToS limited answer
            tos_low_threshold: Low-band score for ToS clarification
            enable_verification: Whether to enable hallucination verification
            verification_threshold: Minimum verification score to pass
            enable_hybrid_tos_search: Enable rule-based and triplet search for ToS
        """
        # Initialize LLM
        # Uses LLM_PROVIDER env var (default: vllm for production)
        # Set LLM_PROVIDER=openai for testing only
        self.llm = llm or create_llm_client()

        # Initialize stores
        self.qna_store = qna_store or QnAVectorStore(
            persist_directory=qna_db_path,
            embedding_model=embedding_model,
        )
        self.tos_store = tos_store or ToSVectorStore(
            persist_directory=tos_db_path,
            embedding_model=embedding_model,
            enable_hybrid_search=enable_hybrid_tos_search,
        )

        # Track hybrid search setting
        self.enable_hybrid_tos_search = enable_hybrid_tos_search

        # Initialize verifier
        self.enable_verification = enable_verification
        self.verification_threshold = verification_threshold
        if enable_verification:
            self.verifier = verifier or AnswerVerifier(
                llm_client=self.llm,
                confidence_threshold=verification_threshold,
                require_citations=True,
                use_llm_verification=True,
            )
        else:
            self.verifier = None

        self.qna_threshold = qna_threshold
        self.tos_threshold = tos_threshold
        self.qna_mid_threshold = qna_mid_threshold
        self.tos_mid_threshold = tos_mid_threshold
        self.tos_low_threshold = tos_low_threshold

        if not 0.0 <= self.qna_mid_threshold <= self.qna_threshold <= 1.0:
            raise ValueError("qna_mid_threshold must be <= qna_threshold and within [0, 1]")
        if not 0.0 <= self.tos_low_threshold <= self.tos_mid_threshold <= self.tos_threshold <= 1.0:
            raise ValueError(
                "tos_low_threshold must be <= tos_mid_threshold <= tos_threshold within [0, 1]"
            )

        logger.info(
            f"RAG Pipeline initialized. "
            f"QnA: {self.qna_store.count()} docs, "
            f"ToS: {self.tos_store.count()} docs, "
            f"Verification: {'enabled' if enable_verification else 'disabled'}, "
            f"Hybrid ToS: {'enabled' if enable_hybrid_tos_search else 'disabled'}"
        )

    def query(self, user_query: str) -> PipelineResponse:
        """Process a user query through the RAG pipeline.

        Args:
            user_query: User's question

        Returns:
            PipelineResponse with answer and metadata
        """
        logger.info(f"Processing query: {user_query[:50]}...")

        # Step 1: Try QnA DB
        qna_response = self._try_qna(user_query)
        if qna_response:
            return qna_response

        # Step 2: Try ToS DB
        tos_response = self._try_tos(user_query)
        if tos_response:
            return tos_response

        # Step 3: No context found - answer without context
        return self._answer_without_context(user_query)

    def _try_qna(self, query: str) -> PipelineResponse | None:
        """Try to answer using QnA DB.

        Args:
            query: User query

        Returns:
            PipelineResponse if match found, None otherwise
        """
        results = self.qna_store.search(query, n_results=3)

        if not results:
            logger.debug("No QnA results found")
            return None

        best = results[0]
        if best.score < self.qna_threshold:
            if best.score >= self.qna_mid_threshold:
                logger.info(
                    f"QnA mid-band score {best.score:.3f} below threshold {self.qna_threshold}"
                )
                qna_limited = self._build_qna_limited_response(
                    query=query, results=results, best=best
                )
                tos_response = self._try_tos(query)
                if (
                    tos_response
                    and tos_response.source == ResponseSource.TOS
                    and tos_response.response_mode in {"answer", "limited_answer"}
                    and tos_response.confidence >= qna_limited.confidence
                ):
                    return tos_response
                return qna_limited
            logger.debug(f"QnA score {best.score:.3f} below threshold {self.qna_threshold}")
            return None

        # Build context from matched Q&A
        context_parts = []
        for r in results:
            if r.score >= self.qna_threshold * 0.9:  # Include close matches
                context_parts.append(f"Q: {r.question}\nA: {r.answer}")

        context_str = "\n\n".join(context_parts)

        # Generate answer using LLM with QnA context
        system_prompt = """당신은 금융 서비스 고객 상담 AI입니다.

제공된 FAQ 정보를 바탕으로 고객 질문에 답변합니다.

규칙:
1. FAQ 답변을 기반으로 하되, 자연스럽게 재구성하여 답변하세요.
2. FAQ에 없는 내용은 추가하지 마세요.
3. 친절하고 전문적인 톤을 유지하세요.
4. 필요시 "자세한 사항은 고객센터로 문의해주세요"를 안내하세요."""

        response = self.llm.generate_with_context(
            query=query,
            context=context_str,
            system_prompt=system_prompt,
        )

        logger.info(f"QnA match found with score {best.score:.3f}")

        return PipelineResponse(
            query=query,
            answer=response.content,
            source=ResponseSource.QNA,
            confidence=best.score,
            response_mode="answer",
            context=[
                {"question": r.question, "answer": r.answer, "score": r.score}
                for r in results
                if r.score >= self.qna_threshold * 0.9
            ],
            citations=[f"FAQ: {best.question}"],
            metadata={
                "matched_question": best.question,
                "category": best.category,
                "sub_category": best.sub_category,
                "llm_model": response.model,
                "prompt_tokens": response.usage.get("prompt_tokens", 0),
                "completion_tokens": response.usage.get("completion_tokens", 0),
                "total_tokens": response.usage.get("total_tokens", 0),
                "tokens_used": response.usage.get("total_tokens", 0),
            },
        )

    def _build_qna_limited_response(
        self,
        query: str,
        results: list[Any],
        best: Any,
    ) -> PipelineResponse:
        context_parts = []
        for r in results:
            if r.score >= self.qna_mid_threshold * 0.9:
                context_parts.append(f"Q: {r.question}\nA: {r.answer}")

        if not context_parts:
            context_parts.append(f"Q: {best.question}\nA: {best.answer}")

        context_str = "\n\n".join(context_parts)

        context_items = [
            {"question": r.question, "answer": r.answer, "score": r.score}
            for r in results
            if r.score >= self.qna_mid_threshold * 0.9
        ]
        if not context_items:
            context_items = [
                {"question": best.question, "answer": best.answer, "score": best.score}
            ]

        system_prompt = """당신은 금융 서비스 고객 상담 AI입니다.

FAQ에서 일부 유사한 항목이 발견되었지만 정확한 일치는 아닙니다.

규칙:
1. 제공된 FAQ 내용을 근거로 제한적으로 답변하세요.
2. 확신이 없는 내용은 "확실하지 않습니다"라고 밝혀 주세요.
3. 추가 정보가 필요하면 질문을 더 구체적으로 요청하세요.
4. FAQ에 없는 내용은 지어내지 마세요."""

        response = self.llm.generate_with_context(
            query=query,
            context=context_str,
            system_prompt=system_prompt,
        )

        return PipelineResponse(
            query=query,
            answer=response.content,
            source=ResponseSource.QNA,
            confidence=best.score,
            response_mode="limited_answer",
            context=context_items,
            citations=[f"FAQ: {best.question}"],
            metadata={
                "matched_question": best.question,
                "category": best.category,
                "sub_category": best.sub_category,
                "confidence_band": "mid",
                "qna_threshold": self.qna_threshold,
                "qna_mid_threshold": self.qna_mid_threshold,
                "llm_model": response.model,
                "prompt_tokens": response.usage.get("prompt_tokens", 0),
                "completion_tokens": response.usage.get("completion_tokens", 0),
                "total_tokens": response.usage.get("total_tokens", 0),
                "tokens_used": response.usage.get("total_tokens", 0),
            },
        )

    def _try_tos(self, query: str) -> PipelineResponse | None:
        """Try to answer using ToS DB.

        Args:
            query: User query

        Returns:
            PipelineResponse if relevant sections found, None otherwise
        """
        # Check if query explicitly references a specific section
        section_match = self._extract_section_reference(query)

        n_results = 5

        # Use hybrid search if enabled
        if self.enable_hybrid_tos_search:
            hybrid_results = self.tos_store.search_hybrid(query, n_results=n_results)
            if not hybrid_results:
                logger.debug("No ToS results found (hybrid)")
                return None

            # Use combined_score for hybrid, score for regular
            if "final_score" in hybrid_results[0]:
                score_key = "final_score"
            else:
                score_key = "combined_score" if "combined_score" in hybrid_results[0] else "score"
            relevant = [r for r in hybrid_results if r.get(score_key, 0) >= self.tos_threshold]

            if not relevant:
                best_score = hybrid_results[0].get(score_key, 0)
                if best_score >= self.tos_mid_threshold:
                    limited = [
                        r for r in hybrid_results if r.get(score_key, 0) >= self.tos_mid_threshold
                    ]
                    return self._build_tos_response_from_hybrid(
                        query=query,
                        results=limited,
                        section_match=section_match,
                        limited=True,
                    )
                if best_score >= self.tos_low_threshold:
                    logger.debug(
                        f"ToS scores below mid threshold {self.tos_mid_threshold}. "
                        f"Best: {best_score:.3f} (hybrid)"
                    )
                    return self._build_clarification_response(
                        query=query,
                        confidence=best_score,
                        metadata={
                            "confidence_band": "low",
                            "tos_threshold": self.tos_threshold,
                            "tos_mid_threshold": self.tos_mid_threshold,
                            "tos_low_threshold": self.tos_low_threshold,
                            "hybrid_search": True,
                        },
                    )
                logger.debug(
                    f"ToS scores below low threshold {self.tos_low_threshold}. "
                    f"Best: {best_score:.3f} (hybrid)"
                )
                return None

            return self._build_tos_response_from_hybrid(
                query=query,
                results=relevant,
                section_match=section_match,
                limited=False,
            )

        # Regular vector-only search
        results = self.tos_store.search(query, n_results=n_results)

        if not results:
            logger.debug("No ToS results found")
            return None

        # Filter by threshold
        relevant = [r for r in results if r.score >= self.tos_threshold]

        if not relevant:
            best_score = results[0].score
            if best_score >= self.tos_mid_threshold:
                limited = [r for r in results if r.score >= self.tos_mid_threshold]
                return self._build_tos_response(
                    query=query,
                    results=limited,
                    section_match=section_match,
                    limited=True,
                )
            if best_score >= self.tos_low_threshold:
                logger.debug(
                    f"ToS scores below mid threshold {self.tos_mid_threshold}. Best: {best_score:.3f}"
                )
                return self._build_clarification_response(
                    query=query,
                    confidence=best_score,
                    metadata={
                        "confidence_band": "low",
                        "tos_threshold": self.tos_threshold,
                        "tos_mid_threshold": self.tos_mid_threshold,
                        "tos_low_threshold": self.tos_low_threshold,
                    },
                )
            logger.debug(
                f"ToS scores below low threshold {self.tos_low_threshold}. Best: {best_score:.3f}"
            )
            return None

        return self._build_tos_response(
            query=query,
            results=relevant,
            section_match=section_match,
            limited=False,
        )

    def _build_tos_response(
        self,
        query: str,
        results: list[Any],
        section_match: str | None,
        limited: bool,
    ) -> PipelineResponse:
        context_parts = []
        for r in results:
            section_text = f"[{r.document_title}]\n"
            if r.section_title:
                section_text += f"{r.section_title}\n"
            section_text += r.section_content
            context_parts.append(section_text)

        context_str = "\n\n---\n\n".join(context_parts)

        if limited:
            system_prompt = """당신은 금융 서비스 약관 전문 상담 AI입니다.

관련 조항이 부분적으로 매칭되었지만 확실하지 않습니다.

규칙:
1. 제공된 약관 내용만을 근거로 제한적으로 답변하세요.
2. 불확실한 부분은 명시하고 단정하지 마세요.
3. 약관에 없는 내용은 지어내지 마세요.
4. 답변 마지막에 참조한 약관/조항을 명시하세요. 예: [참조: OO약관 제N조]
5. 추가로 필요한 정보나 정확한 조항을 요청하세요."""
        else:
            system_prompt = """당신은 금융 서비스 약관 전문 상담 AI입니다.

제공된 약관 내용을 바탕으로 고객 질문에 답변합니다.

규칙:
1. 약관 내용만을 기반으로 정확하게 답변하세요.
2. 약관에 없는 내용은 절대 지어내지 마세요.
3. 확실하지 않으면 "해당 내용은 약관에서 확인되지 않습니다"라고 답변하세요.
4. 답변 마지막에 참조한 약관/조항을 명시하세요. 예: [참조: OO약관 제N조]
5. 전문 용어는 쉽게 풀어서 설명하세요."""

        response = self.llm.generate_with_context(
            query=query,
            context=context_str,
            system_prompt=system_prompt,
        )

        citations = self._extract_citations(response.content)
        if not citations:
            citations = [f"{r.document_title} - {r.section_title}" for r in results[:2]]

        avg_score = sum(r.score for r in results) / len(results)
        logger.info(
            f"ToS context found with avg score {avg_score:.3f}{' (limited)' if limited else ''}"
        )

        verification_context = [
            {
                "section_title": r.section_title,
                "section_content": r.section_content,
            }
            for r in results
        ]

        verification_result = self._verify_answer(
            question=query,
            answer=response.content,
            context=verification_context,
        )

        metadata = {
            "section_reference": section_match,
            "llm_model": response.model,
            "prompt_tokens": response.usage.get("prompt_tokens", 0),
            "completion_tokens": response.usage.get("completion_tokens", 0),
            "total_tokens": response.usage.get("total_tokens", 0),
            "tokens_used": response.usage.get("total_tokens", 0),
            "verification_reasoning": verification_result.reasoning
            if verification_result
            else None,
        }
        if limited:
            metadata.update(
                {
                    "confidence_band": "mid",
                    "tos_threshold": self.tos_threshold,
                    "tos_mid_threshold": self.tos_mid_threshold,
                }
            )

        return PipelineResponse(
            query=query,
            answer=response.content,
            source=ResponseSource.TOS,
            confidence=avg_score,
            response_mode="limited_answer" if limited else "answer",
            context=[
                {
                    "document_title": r.document_title,
                    "section_title": r.section_title,
                    "section_content": r.section_content[:500],
                    "score": r.score,
                }
                for r in results
            ],
            citations=citations,
            metadata=metadata,
            verified=verification_result.verified if verification_result else True,
            verification_score=verification_result.confidence if verification_result else 1.0,
            verification_issues=verification_result.issues if verification_result else [],
        )

    def _build_tos_response_from_hybrid(
        self,
        query: str,
        results: list[dict[str, Any]],
        section_match: str | None,
        limited: bool = False,
    ) -> PipelineResponse:
        """Build ToS response from hybrid search results.

        Args:
            query: User query
            results: Hybrid search results with combined scores
            section_match: Extracted section reference from query

        Returns:
            PipelineResponse
        """
        # Build context from hybrid results
        context_parts = []
        for r in results:
            section_text = f"[{r.get('document_title', '')}]\n"
            if r.get("section_title"):
                section_text += f"{r['section_title']}\n"
            section_text += r.get("section_content", "")
            context_parts.append(section_text)

        context_str = "\n\n---\n\n".join(context_parts)

        if limited:
            system_prompt = """당신은 금융 서비스 약관 전문 상담 AI입니다.

관련 조항이 부분적으로 매칭되었지만 확실하지 않습니다.

규칙:
1. 제공된 약관 내용만을 근거로 제한적으로 답변하세요.
2. 불확실한 부분은 명시하고 단정하지 마세요.
3. 약관에 없는 내용은 지어내지 마세요.
4. 답변 마지막에 참조한 약관/조항을 명시하세요. 예: [참조: OO약관 제N조]
5. 추가로 필요한 정보나 정확한 조항을 요청하세요."""
        else:
            system_prompt = """당신은 금융 서비스 약관 전문 상담 AI입니다.

제공된 약관 내용을 바탕으로 고객 질문에 답변합니다.

규칙:
1. 약관 내용만을 기반으로 정확하게 답변하세요.
2. 약관에 없는 내용은 절대 지어내지 마세요.
3. 확실하지 않으면 "해당 내용은 약관에서 확인되지 않습니다"라고 답변하세요.
4. 답변 마지막에 참조한 약관/조항을 명시하세요. 예: [참조: OO약관 제N조]
5. 전문 용어는 쉽게 풀어서 설명하세요."""

        response = self.llm.generate_with_context(
            query=query,
            context=context_str,
            system_prompt=system_prompt,
        )

        # Extract citations from answer
        citations = self._extract_citations(response.content)
        if not citations:
            citations = [
                f"{r.get('document_title', '')} - {r.get('section_title', '')}" for r in results[:2]
            ]

        # Use combined_score if available
        if "final_score" in results[0]:
            score_key = "final_score"
        else:
            score_key = "combined_score" if "combined_score" in results[0] else "score"
        avg_score = sum(r.get(score_key, 0) for r in results) / len(results)
        logger.info(
            f"ToS context found with avg score {avg_score:.3f} (hybrid)"
            f"{' (limited)' if limited else ''}"
        )

        # Build context for verification
        verification_context = [
            {
                "section_title": r.get("section_title", ""),
                "section_content": r.get("section_content", ""),
            }
            for r in results
        ]

        # Verify answer
        verification_result = self._verify_answer(
            question=query,
            answer=response.content,
            context=verification_context,
        )

        # Collect hybrid-specific metadata
        matched_keywords = []
        matched_triplets = []
        for r in results:
            matched_keywords.extend(r.get("matched_keywords", []))
            matched_triplets.extend(r.get("matched_triplets", []))

        metadata = {
            "section_reference": section_match,
            "llm_model": response.model,
            "prompt_tokens": response.usage.get("prompt_tokens", 0),
            "completion_tokens": response.usage.get("completion_tokens", 0),
            "total_tokens": response.usage.get("total_tokens", 0),
            "tokens_used": response.usage.get("total_tokens", 0),
            "verification_reasoning": verification_result.reasoning
            if verification_result
            else None,
            "hybrid_search": True,
            "matched_keywords": list(set(matched_keywords)),
            "matched_triplets": matched_triplets[:5],
        }
        if limited:
            metadata.update(
                {
                    "confidence_band": "mid",
                    "tos_threshold": self.tos_threshold,
                    "tos_mid_threshold": self.tos_mid_threshold,
                }
            )

        return PipelineResponse(
            query=query,
            answer=response.content,
            source=ResponseSource.TOS,
            confidence=avg_score,
            response_mode="limited_answer" if limited else "answer",
            context=[
                {
                    "document_title": r.get("document_title", ""),
                    "section_title": r.get("section_title", ""),
                    "section_content": r.get("section_content", "")[:500],
                    "combined_score": r.get("combined_score", 0),
                    "final_score": r.get("final_score"),
                    "rerank_score": r.get("rerank_score"),
                    "vector_score": r.get("vector_score", 0),
                    "rule_score": r.get("rule_score", 0),
                    "triplet_score": r.get("triplet_score", 0),
                }
                for r in results
            ],
            citations=citations,
            metadata=metadata,
            verified=verification_result.verified if verification_result else True,
            verification_score=verification_result.confidence if verification_result else 1.0,
            verification_issues=verification_result.issues if verification_result else [],
        )

    def _build_clarification_response(
        self,
        query: str,
        confidence: float,
        metadata: dict[str, Any] | None = None,
    ) -> PipelineResponse:
        system_prompt = """당신은 금융 서비스 고객 상담 AI입니다.

규칙:
1. 현재 정보로는 정확한 답변이 어렵다고 안내하세요.
2. 정확한 답변에 필요한 추가 정보를 1-2개 질문하세요.
3. 필요 시 상담원 연결 안내를 포함하세요.
4. 간결하고 정중하게 작성하세요."""

        response = self.llm.generate_with_context(
            query=query,
            context="관련 근거가 부족합니다.",
            system_prompt=system_prompt,
        )

        return PipelineResponse(
            query=query,
            answer=response.content,
            source=ResponseSource.NO_CONTEXT,
            confidence=confidence,
            response_mode="clarification",
            context=[],
            citations=[],
            metadata={
                **(metadata or {}),
                "llm_model": response.model,
                "prompt_tokens": response.usage.get("prompt_tokens", 0),
                "completion_tokens": response.usage.get("completion_tokens", 0),
                "total_tokens": response.usage.get("total_tokens", 0),
                "tokens_used": response.usage.get("total_tokens", 0),
            },
        )

    def _answer_without_context(self, query: str) -> PipelineResponse:
        """Generate answer when no context is found.

        Args:
            query: User query

        Returns:
            PipelineResponse with general answer
        """
        system_prompt = """당신은 금융 서비스 고객 상담 AI입니다.

질문에 대한 관련 정보를 찾을 수 없습니다.

규칙:
1. 정보를 찾을 수 없다고 정중하게 안내하세요.
2. 고객센터 연락처나 추가 도움 방법을 안내하세요.
3. 절대로 정보를 지어내지 마세요."""

        response = self.llm.generate_with_context(
            query=query,
            context="관련 정보를 찾을 수 없습니다.",
            system_prompt=system_prompt,
        )

        return PipelineResponse(
            query=query,
            answer=response.content,
            source=ResponseSource.NO_CONTEXT,
            confidence=0.0,
            response_mode="handoff",
            context=[],
            citations=[],
            metadata={
                "llm_model": response.model,
                "prompt_tokens": response.usage.get("prompt_tokens", 0),
                "completion_tokens": response.usage.get("completion_tokens", 0),
                "total_tokens": response.usage.get("total_tokens", 0),
                "tokens_used": response.usage.get("total_tokens", 0),
            },
        )

    def _extract_section_reference(self, query: str) -> str | None:
        """Extract section reference from query (e.g., '제1조', '1조 1항').

        Args:
            query: User query

        Returns:
            Extracted section reference or None
        """
        patterns = [
            r"제?\s*(\d+)\s*조\s*(?:제?\s*(\d+)\s*항)?",  # 제1조 제2항, 1조 2항
            r"(\d+)\s*조\s*(\d+)\s*항",  # 1조 1항
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                article = match.group(1)
                paragraph = match.group(2) if len(match.groups()) > 1 else None
                if paragraph:
                    return f"제{article}조 제{paragraph}항"
                return f"제{article}조"

        return None

    def _extract_citations(self, answer: str) -> list[str]:
        """Extract citations from LLM answer.

        Args:
            answer: LLM generated answer

        Returns:
            List of citation strings
        """
        citations = []
        # Pattern: [참조: 제N조 N항] or [참조: OO약관]
        pattern = r"\[참조:\s*([^\]]+)\]"
        matches = re.findall(pattern, answer)
        citations.extend(matches)
        return citations

    def search_qna(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Direct QnA search.

        Args:
            query: Search query
            n_results: Number of results

        Returns:
            List of QnA results as dicts
        """
        results = self.qna_store.search(query, n_results=n_results)
        return [
            {
                "question": r.question,
                "answer": r.answer,
                "category": r.category,
                "sub_category": r.sub_category,
                "score": r.score,
            }
            for r in results
        ]

    def search_tos(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Direct ToS search.

        Args:
            query: Search query
            n_results: Number of results

        Returns:
            List of ToS results as dicts
        """
        results = self.tos_store.search(query, n_results=n_results)
        return [
            {
                "document_title": r.document_title,
                "section_title": r.section_title,
                "section_content": r.section_content,
                "category": r.category,
                "score": r.score,
            }
            for r in results
        ]

    def _verify_answer(
        self,
        question: str,
        answer: str,
        context: list[dict[str, Any]],
    ) -> VerificationResult | None:
        """Verify answer using the hallucination verifier.

        Args:
            question: Original question
            answer: Generated answer
            context: Source context used for generation

        Returns:
            VerificationResult or None if verification is disabled
        """
        if not self.enable_verification or not self.verifier:
            return None

        try:
            result = self.verifier.verify(
                question=question,
                answer=answer,
                context=context,
            )

            if result.verified:
                logger.info(f"Answer verified. Score: {result.confidence:.2f}")
            else:
                logger.warning(
                    f"Answer verification failed. Score: {result.confidence:.2f}, "
                    f"Issues: {result.issues}"
                )

            return result

        except Exception as e:
            logger.error(f"Verification failed with error: {e}")
            return None
