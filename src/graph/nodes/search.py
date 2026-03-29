import logging
from typing import Any

from src.graph.state import GraphState
from src.graph.utils import extract_section_reference

logger = logging.getLogger(__name__)


def make_search_qna(qna_store: Any):
    def search_qna(state: GraphState) -> dict[str, Any]:
        results = qna_store.search(state.query, n_results=5)
        normalized = [
            {
                "question": item.question,
                "answer": item.answer,
                "category": item.category,
                "sub_category": item.sub_category,
                "score": item.score,
                "source": getattr(item, "source", "FAQ"),
                "source_url": getattr(item, "source_url", ""),
                "id": getattr(item, "id", ""),
            }
            for item in results
        ]
        score = normalized[0]["score"] if normalized else 0.0
        logger.debug("QnA search completed with top score %.3f", score)
        return {"qna_results": normalized, "qna_score": score}

    return search_qna


def make_search_tos(tos_store: Any, enable_hybrid_tos_search: bool):
    def search_tos(state: GraphState) -> dict[str, Any]:
        section_reference = extract_section_reference(state.query)

        if enable_hybrid_tos_search:
            results = tos_store.search_hybrid(state.query, n_results=5)
            normalized = []
            for item in results:
                score = item.get("final_score")
                if score is None:
                    score = item.get("combined_score", item.get("score", 0.0))
                normalized.append(
                    {
                        "document_title": item.get("document_title", ""),
                        "section_title": item.get("section_title", ""),
                        "section_content": item.get("section_content", ""),
                        "category": item.get("category", ""),
                        "parent_content": item.get("parent_content", ""),
                        "effective_date": item.get("effective_date", ""),
                        "source_url": item.get("source_url", ""),
                        "score": score,
                        "combined_score": item.get("combined_score"),
                        "final_score": item.get("final_score"),
                        "rerank_score": item.get("rerank_score"),
                        "vector_score": item.get("vector_score", 0.0),
                        "rule_score": item.get("rule_score", 0.0),
                        "triplet_score": item.get("triplet_score", 0.0),
                        "matched_keywords": item.get("matched_keywords", []),
                        "matched_triplets": item.get("matched_triplets", []),
                        "hybrid_search": True,
                    }
                )
        else:
            results = tos_store.search(state.query, n_results=5)
            normalized = [
                {
                    "document_title": item.document_title,
                    "section_title": item.section_title,
                    "section_content": item.section_content,
                    "category": item.category,
                    "parent_content": item.parent_content,
                    "effective_date": item.effective_date,
                    "source_url": item.source_url,
                    "score": item.score,
                    "id": item.id,
                    "hybrid_search": False,
                }
                for item in results
            ]

        score = normalized[0]["score"] if normalized else 0.0
        logger.debug("ToS search completed with top score %.3f", score)
        return {
            "tos_results": normalized,
            "tos_score": score,
            "section_reference": section_reference,
        }

    return search_tos
