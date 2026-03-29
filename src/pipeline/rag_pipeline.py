import logging
from pathlib import Path
from typing import Any

from src.graph import GraphState, build_graph
from src.graph.utils import extract_citations, extract_section_reference
from src.llm import BaseLLMClient, create_llm_client
from src.pipeline.models import PipelineResponse, ResponseSource
from src.vectorstore import QnAVectorStore, ToSVectorStore
from src.verifier import AnswerVerifier, VerificationResult

logger = logging.getLogger(__name__)


class RAGPipeline:
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
        self.llm = llm or create_llm_client()
        self.qna_store = qna_store or QnAVectorStore(
            persist_directory=qna_db_path,
            embedding_model=embedding_model,
        )
        self.tos_store = tos_store or ToSVectorStore(
            persist_directory=tos_db_path,
            embedding_model=embedding_model,
            enable_hybrid_search=enable_hybrid_tos_search,
        )

        self.enable_hybrid_tos_search = enable_hybrid_tos_search
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

        self._graph = build_graph(
            self.qna_store,
            self.tos_store,
            self.llm,
            self.verifier,
            qna_threshold=self.qna_threshold,
            qna_mid_threshold=self.qna_mid_threshold,
            tos_threshold=self.tos_threshold,
            tos_mid_threshold=self.tos_mid_threshold,
            tos_low_threshold=self.tos_low_threshold,
            enable_verification=self.enable_verification,
            enable_hybrid_tos_search=self.enable_hybrid_tos_search,
        )

        logger.info(
            "RAG Pipeline initialized. QnA: %s docs, ToS: %s docs, Verification: %s, Hybrid ToS: %s",
            self.qna_store.count(),
            self.tos_store.count(),
            "enabled" if enable_verification else "disabled",
            "enabled" if enable_hybrid_tos_search else "disabled",
        )

    def query(self, user_query: str) -> PipelineResponse:
        logger.info("Processing query: %s...", user_query[:50])
        final_state = self._graph.invoke({"query": user_query})
        state = final_state if isinstance(final_state, GraphState) else GraphState(**final_state)

        if state.response is not None:
            return state.response

        return PipelineResponse(
            query=state.query,
            answer=state.answer,
            source=state.source or ResponseSource.NO_CONTEXT,
            confidence=state.confidence,
            response_mode=state.response_mode,
            context=state.context,
            citations=state.citations,
            metadata=state.metadata,
            verified=state.verified,
            verification_score=state.verification_score,
            verification_issues=state.verification_issues,
        )

    def search_qna(
        self,
        query: str,
        n_results: int = 5,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        result_count = top_k if top_k is not None else n_results
        results = self.qna_store.search(query, n_results=result_count)
        return [
            {
                "question": item.question,
                "answer": item.answer,
                "category": item.category,
                "sub_category": item.sub_category,
                "score": item.score,
            }
            for item in results
        ]

    def search_tos(
        self,
        query: str,
        n_results: int = 5,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        result_count = top_k if top_k is not None else n_results
        results = self.tos_store.search(query, n_results=result_count)
        return [
            {
                "document_title": item.document_title,
                "section_title": item.section_title,
                "section_content": item.section_content,
                "category": item.category,
                "score": item.score,
            }
            for item in results
        ]

    def _extract_section_reference(self, query: str) -> str | None:
        return extract_section_reference(query)

    def _extract_citations(self, answer: str) -> list[str]:
        return extract_citations(answer)

    def _verify_answer(
        self,
        question: str,
        answer: str,
        context: list[dict[str, Any]],
    ) -> VerificationResult | None:
        if not self.enable_verification or not self.verifier:
            return None

        try:
            return self.verifier.verify(question=question, answer=answer, context=context)
        except Exception as exc:
            logger.error("Verification failed with error: %s", exc)
            return None


__all__ = ["PipelineResponse", "RAGPipeline", "ResponseSource"]
