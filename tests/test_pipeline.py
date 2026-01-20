"""Tests for RAG Pipeline."""

import pytest
from unittest.mock import Mock, patch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.rag_pipeline import RAGPipeline, PipelineResponse, ResponseSource


class MockQnAResult:
    def __init__(self, question, answer, score, category="FAQ", sub_category="일반"):
        self.question = question
        self.answer = answer
        self.score = score
        self.category = category
        self.sub_category = sub_category
        self.source = "QnA"
        self.source_url = "http://example.com"
        self.id = "test-id"


class MockToSResult:
    def __init__(self, document_title, section_title, section_content, score, category="약관"):
        self.document_title = document_title
        self.section_title = section_title
        self.section_content = section_content
        self.score = score
        self.category = category
        self.parent_content = ""
        self.effective_date = "2024-01-01"
        self.source_url = "http://example.com"
        self.id = "test-id"


class MockLLMResponse:
    def __init__(self, content):
        self.content = content
        self.model = "gpt-4o-mini"
        self.usage = {"total_tokens": 100}


@pytest.fixture
def mock_qna_store():
    store = Mock()
    store.count.return_value = 10
    return store


@pytest.fixture
def mock_tos_store():
    store = Mock()
    store.count.return_value = 20
    return store


@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.generate_with_context.return_value = MockLLMResponse("테스트 답변입니다.")
    return llm


class TestRAGPipeline:
    def test_qna_match_above_threshold(self, mock_qna_store, mock_tos_store, mock_llm):
        """QnA score가 threshold 이상이면 QnA 답변 반환."""
        mock_qna_store.search.return_value = [
            MockQnAResult("비밀번호 변경 방법", "설정에서 변경 가능합니다.", 0.90)
        ]

        pipeline = RAGPipeline(
            llm=mock_llm,
            qna_store=mock_qna_store,
            tos_store=mock_tos_store,
            qna_threshold=0.85,
        )

        response = pipeline.query("비밀번호 어떻게 바꿔요?")

        assert response.source == ResponseSource.QNA
        assert response.confidence >= 0.85
        mock_llm.generate_with_context.assert_called_once()

    def test_qna_below_threshold_fallback_to_tos(self, mock_qna_store, mock_tos_store, mock_llm):
        """QnA score가 threshold 미만이면 ToS로 fallback."""
        mock_qna_store.search.return_value = [
            MockQnAResult("관련없는 질문", "관련없는 답변", 0.50)
        ]
        mock_tos_store.search.return_value = [
            MockToSResult("이용약관", "제1조 (목적)", "본 약관은...", 0.75)
        ]

        pipeline = RAGPipeline(
            llm=mock_llm,
            qna_store=mock_qna_store,
            tos_store=mock_tos_store,
            qna_threshold=0.80,
            tos_threshold=0.65,
        )

        response = pipeline.query("제1조에 대해 알려주세요")

        assert response.source == ResponseSource.TOS

    def test_no_context_found(self, mock_qna_store, mock_tos_store, mock_llm):
        """QnA와 ToS 모두 결과 없으면 NO_CONTEXT."""
        mock_qna_store.search.return_value = []
        mock_tos_store.search.return_value = []

        pipeline = RAGPipeline(
            llm=mock_llm,
            qna_store=mock_qna_store,
            tos_store=mock_tos_store,
        )

        response = pipeline.query("완전히 관련없는 질문")

        assert response.source == ResponseSource.NO_CONTEXT
        assert response.confidence == 0.0

    def test_extract_section_reference(self, mock_qna_store, mock_tos_store, mock_llm):
        """조항 참조 추출 테스트."""
        pipeline = RAGPipeline(
            llm=mock_llm,
            qna_store=mock_qna_store,
            tos_store=mock_tos_store,
        )

        assert pipeline._extract_section_reference("제1조에 대해") == "제1조"
        assert pipeline._extract_section_reference("1조 2항 알려줘") == "제1조 제2항"
        assert pipeline._extract_section_reference("제3조 제1항") == "제3조 제1항"
        assert pipeline._extract_section_reference("일반 질문") is None

    def test_search_qna_direct(self, mock_qna_store, mock_tos_store, mock_llm):
        """Direct QnA search."""
        mock_qna_store.search.return_value = [
            MockQnAResult("Q1", "A1", 0.9),
            MockQnAResult("Q2", "A2", 0.8),
        ]

        pipeline = RAGPipeline(
            llm=mock_llm,
            qna_store=mock_qna_store,
            tos_store=mock_tos_store,
        )

        results = pipeline.search_qna("test", n_results=2)

        assert len(results) == 2
        assert results[0]["question"] == "Q1"

    def test_search_tos_direct(self, mock_qna_store, mock_tos_store, mock_llm):
        """Direct ToS search."""
        mock_tos_store.search.return_value = [
            MockToSResult("약관A", "제1조", "내용1", 0.9),
        ]

        pipeline = RAGPipeline(
            llm=mock_llm,
            qna_store=mock_qna_store,
            tos_store=mock_tos_store,
        )

        results = pipeline.search_tos("제1조", n_results=1)

        assert len(results) == 1
        assert results[0]["section_title"] == "제1조"


    def test_tos_response_includes_verification(self, mock_qna_store, mock_tos_store, mock_llm):
        """ToS 응답에 verification 결과가 포함되는지 테스트."""
        mock_qna_store.search.return_value = []
        mock_tos_store.search.return_value = [
            MockToSResult("이용약관", "제1조 (목적)", "본 약관은...", 0.75)
        ]

        # Verification 비활성화 상태로 테스트
        pipeline = RAGPipeline(
            llm=mock_llm,
            qna_store=mock_qna_store,
            tos_store=mock_tos_store,
            enable_verification=False,
        )

        response = pipeline.query("제1조에 대해 알려주세요")

        assert response.source == ResponseSource.TOS
        assert response.verified is True  # 비활성화 시 기본값

    def test_verification_enabled(self, mock_qna_store, mock_tos_store, mock_llm):
        """Verification 활성화 테스트."""
        mock_qna_store.search.return_value = []
        mock_tos_store.search.return_value = [
            MockToSResult("이용약관", "제1조 (목적)", "본 약관은 서비스 이용에 관한...", 0.75)
        ]
        mock_llm.generate_with_context.return_value = MockLLMResponse(
            "본 약관은 서비스 이용에 관한 조건을 규정합니다. [참조: 제1조]"
        )
        mock_llm.generate.return_value = MockLLMResponse(
            '{"verified": true, "confidence": 0.9, "issues": [], "reasoning": "정확함"}'
        )

        pipeline = RAGPipeline(
            llm=mock_llm,
            qna_store=mock_qna_store,
            tos_store=mock_tos_store,
            enable_verification=True,
        )

        response = pipeline.query("제1조에 대해 알려주세요")

        assert response.source == ResponseSource.TOS
        # verification이 실행됨
        assert "verification_reasoning" in response.metadata


class TestPipelineResponse:
    def test_to_dict(self):
        """PipelineResponse to_dict 테스트."""
        response = PipelineResponse(
            query="test",
            answer="answer",
            source=ResponseSource.QNA,
            confidence=0.9,
            context=[{"q": "test"}],
            citations=["ref"],
            metadata={"key": "value"},
        )

        d = response.to_dict()

        assert d["query"] == "test"
        assert d["source"] == "qna"
        assert d["confidence"] == 0.9

    def test_to_dict_includes_verification(self):
        """PipelineResponse to_dict에 verification 필드 포함 테스트."""
        response = PipelineResponse(
            query="test",
            answer="answer",
            source=ResponseSource.TOS,
            confidence=0.9,
            verified=False,
            verification_score=0.5,
            verification_issues=["출처 없음"],
        )

        d = response.to_dict()

        assert d["verified"] is False
        assert d["verification_score"] == 0.5
        assert d["verification_issues"] == ["출처 없음"]
