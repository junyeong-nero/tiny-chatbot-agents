from unittest.mock import Mock

from src.graph.graph import build_graph
from src.graph.nodes.format import format_response
from src.graph.nodes.generate import (
    make_generate_clarification,
    make_generate_no_context,
    make_generate_qna_answer,
    make_generate_qna_limited,
    make_generate_tos_answer,
    make_generate_tos_limited,
)
from src.graph.nodes.search import make_search_qna, make_search_tos
from src.graph.nodes.verify import make_verify_answer
from src.graph.state import GraphState
from src.pipeline.models import ResponseSource


class MockQnAResult:
    def __init__(self, question: str, answer: str, score: float):
        self.question = question
        self.answer = answer
        self.score = score
        self.category = "FAQ"
        self.sub_category = "일반"
        self.source = "FAQ"
        self.source_url = "http://example.com"
        self.id = "qna-1"


class MockToSResult:
    def __init__(self, section_title: str, score: float):
        self.document_title = "이용약관"
        self.section_title = section_title
        self.section_content = "본 약관은 테스트용입니다."
        self.category = "약관"
        self.parent_content = ""
        self.effective_date = "2024-01-01"
        self.source_url = "http://example.com"
        self.score = score
        self.id = "tos-1"


class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content
        self.model = "mock-llm"
        self.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_search_qna_node_normalizes_store_results():
    store = Mock()
    store.search.return_value = [MockQnAResult("질문", "답변", 0.88)]

    node = make_search_qna(store)
    updates = node(GraphState(query="질문"))

    assert updates["qna_score"] == 0.88
    assert updates["qna_results"][0]["question"] == "질문"
    store.search.assert_called_with("질문", n_results=5)


def test_search_tos_node_extracts_section_reference():
    store = Mock()
    store.search.return_value = [MockToSResult("제1조 (목적)", 0.72)]

    node = make_search_tos(store, enable_hybrid_tos_search=False)
    updates = node(GraphState(query="제1조에 대해 알려줘"))

    assert updates["tos_score"] == 0.72
    assert updates["section_reference"] == "제1조"


def test_format_response_node_builds_pipeline_response():
    updates = format_response(
        GraphState(
            query="질문",
            answer="답변",
            source=ResponseSource.TOS,
            confidence=0.7,
            response_mode="answer",
            context=[{"section_title": "제1조"}],
            citations=["제1조"],
        )
    )

    response = updates["response"]

    assert response.query == "질문"
    assert response.source == ResponseSource.TOS
    assert response.citations == ["제1조"]


def test_generate_qna_limited_node_returns_limited_answer_metadata():
    llm = Mock()
    llm.generate_with_context.return_value = MockLLMResponse("제한 답변")

    node = make_generate_qna_limited(llm, qna_threshold=0.8, qna_mid_threshold=0.7)
    updates = node(
        GraphState(
            query="질문",
            qna_score=0.75,
            qna_results=[{"question": "유사 질문", "answer": "유사 답변", "score": 0.75}],
        )
    )

    assert updates["source"] == ResponseSource.QNA
    assert updates["response_mode"] == "limited_answer"
    assert updates["metadata"]["qna_mid_threshold"] == 0.7


def test_generate_qna_answer_node_returns_answer_metadata():
    llm = Mock()
    llm.generate_with_context.return_value = MockLLMResponse("일반 답변")

    node = make_generate_qna_answer(llm, qna_threshold=0.8)
    updates = node(
        GraphState(
            query="질문",
            qna_score=0.85,
            qna_results=[{"question": "FAQ 질문", "answer": "FAQ 답변", "score": 0.85}],
        )
    )

    assert updates["source"] == ResponseSource.QNA
    assert updates["response_mode"] == "answer"
    assert updates["citations"] == ["FAQ: FAQ 질문"]


def test_generate_tos_answer_node_extracts_citations():
    llm = Mock()
    llm.generate_with_context.return_value = MockLLMResponse("약관 답변 [참조: 제1조]")

    node = make_generate_tos_answer(llm)
    updates = node(
        GraphState(
            query="질문",
            tos_results=[
                {
                    "document_title": "이용약관",
                    "section_title": "제1조",
                    "section_content": "내용",
                    "score": 0.82,
                    "hybrid_search": False,
                }
            ],
            section_reference="제1조",
        )
    )

    assert updates["source"] == ResponseSource.TOS
    assert updates["citations"] == ["제1조"]
    assert updates["metadata"]["section_reference"] == "제1조"


def test_generate_tos_limited_node_marks_mid_band_metadata():
    llm = Mock()
    llm.generate_with_context.return_value = MockLLMResponse("약관 제한 답변")

    node = make_generate_tos_limited(llm, tos_threshold=0.65, tos_mid_threshold=0.55)
    updates = node(
        GraphState(
            query="질문",
            tos_results=[
                {
                    "document_title": "이용약관",
                    "section_title": "제1조",
                    "section_content": "내용",
                    "score": 0.58,
                    "hybrid_search": False,
                }
            ],
        )
    )

    assert updates["source"] == ResponseSource.TOS
    assert updates["response_mode"] == "limited_answer"
    assert updates["metadata"]["confidence_band"] == "mid"
    assert updates["metadata"]["tos_mid_threshold"] == 0.55


def test_generate_clarification_node_returns_no_context_clarification():
    llm = Mock()
    llm.generate_with_context.return_value = MockLLMResponse("추가 정보가 필요합니다.")

    node = make_generate_clarification(
        llm, tos_threshold=0.65, tos_mid_threshold=0.55, tos_low_threshold=0.4
    )
    updates = node(GraphState(query="질문", tos_score=0.45))

    assert updates["source"] == ResponseSource.NO_CONTEXT
    assert updates["response_mode"] == "clarification"
    assert updates["metadata"]["tos_low_threshold"] == 0.4


def test_generate_no_context_node_returns_handoff():
    llm = Mock()
    llm.generate_with_context.return_value = MockLLMResponse("고객센터로 문의해주세요.")

    node = make_generate_no_context(llm)
    updates = node(GraphState(query="질문"))

    assert updates["source"] == ResponseSource.NO_CONTEXT
    assert updates["response_mode"] == "handoff"
    assert updates["confidence"] == 0.0


def test_verify_node_attaches_verification_fields_for_tos_answers():
    verifier = Mock()
    verifier.verify.return_value = Mock(
        verified=True,
        confidence=0.95,
        issues=[],
        reasoning="ok",
    )

    node = make_verify_answer(verifier, enable_verification=True)
    updates = node(
        GraphState(
            query="질문",
            answer="답변 [참조: 제1조]",
            source=ResponseSource.TOS,
            response_mode="answer",
            context=[{"section_title": "제1조", "section_content": "내용"}],
            metadata={"llm_model": "mock"},
        )
    )

    assert updates["verified"] is True
    assert updates["verification_score"] == 0.95
    assert updates["metadata"]["verification_reasoning"] == "ok"


def test_compiled_graph_routes_qna_mid_band_to_limited_answer():
    llm = Mock()
    llm.generate_with_context.return_value = MockLLMResponse("제한 답변")

    qna_store = Mock()
    qna_store.search.return_value = [MockQnAResult("유사 질문", "유사 답변", 0.75)]

    tos_store = Mock()
    tos_store.search.return_value = [MockToSResult("제1조", 0.95)]

    graph = build_graph(
        qna_store,
        tos_store,
        llm,
        verifier=None,
        qna_threshold=0.8,
        qna_mid_threshold=0.7,
        tos_threshold=0.65,
        tos_mid_threshold=0.55,
        tos_low_threshold=0.4,
        enable_verification=False,
        enable_hybrid_tos_search=False,
    )

    result = graph.invoke({"query": "질문"})

    assert result["response"].source == ResponseSource.QNA
    assert result["response"].response_mode == "limited_answer"
    tos_store.search.assert_not_called()
