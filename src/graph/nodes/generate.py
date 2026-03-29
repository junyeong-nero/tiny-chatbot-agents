import logging
from typing import Any

from src.graph.state import GraphState
from src.graph.utils import extract_citations, extract_usage_metadata
from src.pipeline.models import ResponseSource

logger = logging.getLogger(__name__)

QNA_ANSWER_PROMPT = """당신은 금융 서비스 고객 상담 AI입니다.

제공된 FAQ 정보를 바탕으로 고객 질문에 답변합니다.

규칙:
1. FAQ 답변을 기반으로 하되, 자연스럽게 재구성하여 답변하세요.
2. FAQ에 없는 내용은 추가하지 마세요.
3. 친절하고 전문적인 톤을 유지하세요.
4. 필요시 \"자세한 사항은 고객센터로 문의해주세요\"를 안내하세요."""

QNA_LIMITED_PROMPT = """당신은 금융 서비스 고객 상담 AI입니다.

FAQ에서 일부 유사한 항목이 발견되었지만 정확한 일치는 아닙니다.

규칙:
1. 제공된 FAQ 내용을 근거로 제한적으로 답변하세요.
2. 확신이 없는 내용은 \"확실하지 않습니다\"라고 밝혀 주세요.
3. 추가 정보가 필요하면 질문을 더 구체적으로 요청하세요.
4. FAQ에 없는 내용은 지어내지 마세요."""

TOS_ANSWER_PROMPT = """당신은 금융 서비스 약관 전문 상담 AI입니다.

제공된 약관 내용을 바탕으로 고객 질문에 답변합니다.

규칙:
1. 약관 내용만을 기반으로 정확하게 답변하세요.
2. 약관에 없는 내용은 절대 지어내지 마세요.
3. 확실하지 않으면 \"해당 내용은 약관에서 확인되지 않습니다\"라고 답변하세요.
4. 답변 마지막에 참조한 약관/조항을 명시하세요. 예: [참조: OO약관 제N조]
5. 전문 용어는 쉽게 풀어서 설명하세요."""

TOS_LIMITED_PROMPT = """당신은 금융 서비스 약관 전문 상담 AI입니다.

관련 조항이 부분적으로 매칭되었지만 확실하지 않습니다.

규칙:
1. 제공된 약관 내용만을 근거로 제한적으로 답변하세요.
2. 불확실한 부분은 명시하고 단정하지 마세요.
3. 약관에 없는 내용은 지어내지 마세요.
4. 답변 마지막에 참조한 약관/조항을 명시하세요. 예: [참조: OO약관 제N조]
5. 추가로 필요한 정보나 정확한 조항을 요청하세요."""

CLARIFICATION_PROMPT = """당신은 금융 서비스 고객 상담 AI입니다.

규칙:
1. 현재 정보로는 정확한 답변이 어렵다고 안내하세요.
2. 정확한 답변에 필요한 추가 정보를 1-2개 질문하세요.
3. 필요 시 상담원 연결 안내를 포함하세요.
4. 간결하고 정중하게 작성하세요."""

NO_CONTEXT_PROMPT = """당신은 금융 서비스 고객 상담 AI입니다.

질문에 대한 관련 정보를 찾을 수 없습니다.

규칙:
1. 정보를 찾을 수 없다고 정중하게 안내하세요.
2. 고객센터 연락처나 추가 도움 방법을 안내하세요.
3. 절대로 정보를 지어내지 마세요."""


def _generate_with_context(llm: Any, query: str, context: str, system_prompt: str) -> Any:
    return llm.generate_with_context(query=query, context=context, system_prompt=system_prompt)


def make_generate_qna_answer(llm: Any, qna_threshold: float):
    def generate_qna_answer(state: GraphState) -> dict[str, Any]:
        results = [item for item in state.qna_results if item["score"] >= qna_threshold * 0.9]
        if not results and state.qna_results:
            results = [state.qna_results[0]]

        context_str = "\n\n".join(f"Q: {item['question']}\nA: {item['answer']}" for item in results)
        response = _generate_with_context(llm, state.query, context_str, QNA_ANSWER_PROMPT)
        best = results[0] if results else {}

        return {
            "answer": response.content,
            "source": ResponseSource.QNA,
            "confidence": state.qna_score,
            "response_mode": "answer",
            "context": results,
            "citations": [f"FAQ: {best.get('question', '')}"] if best else [],
            "metadata": {
                "matched_question": best.get("question", ""),
                "category": best.get("category", ""),
                "sub_category": best.get("sub_category", ""),
                **extract_usage_metadata(response),
            },
            "route": "qna_ok",
        }

    return generate_qna_answer


def make_generate_qna_limited(llm: Any, qna_threshold: float, qna_mid_threshold: float):
    def generate_qna_limited(state: GraphState) -> dict[str, Any]:
        results = [item for item in state.qna_results if item["score"] >= qna_mid_threshold * 0.9]
        if not results and state.qna_results:
            results = [state.qna_results[0]]

        context_str = "\n\n".join(f"Q: {item['question']}\nA: {item['answer']}" for item in results)
        response = _generate_with_context(llm, state.query, context_str, QNA_LIMITED_PROMPT)
        best = results[0] if results else {}

        return {
            "answer": response.content,
            "source": ResponseSource.QNA,
            "confidence": state.qna_score,
            "response_mode": "limited_answer",
            "context": results,
            "citations": [f"FAQ: {best.get('question', '')}"] if best else [],
            "metadata": {
                "matched_question": best.get("question", ""),
                "category": best.get("category", ""),
                "sub_category": best.get("sub_category", ""),
                "confidence_band": "mid",
                "qna_threshold": qna_threshold,
                "qna_mid_threshold": qna_mid_threshold,
                **extract_usage_metadata(response),
            },
            "route": "qna_mid",
        }

    return generate_qna_limited


def _build_tos_context(results: list[dict[str, Any]]) -> str:
    context_parts = []
    for item in results:
        section_text = f"[{item.get('document_title', '')}]\n"
        if item.get("section_title"):
            section_text += f"{item['section_title']}\n"
        section_text += item.get("section_content", "")
        context_parts.append(section_text)
    return "\n\n---\n\n".join(context_parts)


def _build_tos_context_items(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    context_items = []
    for item in results:
        context_item = {
            "document_title": item.get("document_title", ""),
            "section_title": item.get("section_title", ""),
            "section_content": item.get("section_content", "")[:500],
            "score": item.get("score", 0.0),
        }
        for key in [
            "combined_score",
            "final_score",
            "rerank_score",
            "vector_score",
            "rule_score",
            "triplet_score",
        ]:
            if key in item and item.get(key) is not None:
                context_item[key] = item.get(key)
        context_items.append(context_item)
    return context_items


def _build_tos_metadata(
    response: Any,
    state: GraphState,
    results: list[dict[str, Any]],
    limited: bool,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "section_reference": state.section_reference,
        **extract_usage_metadata(response),
    }
    if limited:
        metadata["confidence_band"] = "mid"

    if results and results[0].get("hybrid_search"):
        matched_keywords = []
        matched_triplets = []
        for item in results:
            matched_keywords.extend(item.get("matched_keywords", []))
            matched_triplets.extend(item.get("matched_triplets", []))
        metadata.update(
            {
                "hybrid_search": True,
                "matched_keywords": list(set(matched_keywords)),
                "matched_triplets": matched_triplets[:5],
            }
        )

    return metadata


def _generate_tos_response(
    llm: Any,
    state: GraphState,
    limited: bool,
) -> dict[str, Any]:
    results = state.tos_results
    response = _generate_with_context(
        llm,
        state.query,
        _build_tos_context(results),
        TOS_LIMITED_PROMPT if limited else TOS_ANSWER_PROMPT,
    )
    citations = extract_citations(response.content)
    if not citations:
        citations = [
            f"{item.get('document_title', '')} - {item.get('section_title', '')}"
            for item in results[:2]
        ]

    avg_score = sum(item.get("score", 0.0) for item in results) / len(results) if results else 0.0
    logger.info(
        "ToS context found with avg score %.3f%s",
        avg_score,
        " (limited)" if limited else "",
    )

    return {
        "answer": response.content,
        "source": ResponseSource.TOS,
        "confidence": avg_score,
        "response_mode": "limited_answer" if limited else "answer",
        "context": _build_tos_context_items(results),
        "citations": citations,
        "metadata": _build_tos_metadata(response, state, results, limited),
        "route": "tos_mid" if limited else "tos_ok",
    }


def make_generate_tos_answer(llm: Any):
    def generate_tos_answer(state: GraphState) -> dict[str, Any]:
        return _generate_tos_response(llm, state, limited=False)

    return generate_tos_answer


def make_generate_tos_limited(llm: Any, tos_threshold: float, tos_mid_threshold: float):
    def generate_tos_limited(state: GraphState) -> dict[str, Any]:
        updates = _generate_tos_response(llm, state, limited=True)
        updates["metadata"] = {
            **updates["metadata"],
            "tos_threshold": tos_threshold,
            "tos_mid_threshold": tos_mid_threshold,
        }
        return updates

    return generate_tos_limited


def make_generate_clarification(
    llm: Any, tos_threshold: float, tos_mid_threshold: float, tos_low_threshold: float
):
    def generate_clarification(state: GraphState) -> dict[str, Any]:
        response = _generate_with_context(
            llm,
            state.query,
            "관련 근거가 부족합니다.",
            CLARIFICATION_PROMPT,
        )
        return {
            "answer": response.content,
            "source": ResponseSource.NO_CONTEXT,
            "confidence": state.tos_score,
            "response_mode": "clarification",
            "context": [],
            "citations": [],
            "metadata": {
                "confidence_band": "low",
                "tos_threshold": tos_threshold,
                "tos_mid_threshold": tos_mid_threshold,
                "tos_low_threshold": tos_low_threshold,
                **extract_usage_metadata(response),
            },
            "route": "tos_low",
        }

    return generate_clarification


def make_generate_no_context(llm: Any):
    def generate_no_context(state: GraphState) -> dict[str, Any]:
        response = _generate_with_context(
            llm,
            state.query,
            "관련 정보를 찾을 수 없습니다.",
            NO_CONTEXT_PROMPT,
        )
        return {
            "answer": response.content,
            "source": ResponseSource.NO_CONTEXT,
            "confidence": 0.0,
            "response_mode": "handoff",
            "context": [],
            "citations": [],
            "metadata": extract_usage_metadata(response),
            "route": "no_context",
        }

    return generate_no_context
