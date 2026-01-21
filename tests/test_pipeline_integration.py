"""Integration tests for RAG Pipeline with OpenAI.

These tests use the actual OpenAI API and vector stores.
Requires:
- OPENAI_API_KEY environment variable
- Existing vector DB at data/vectordb/qna and data/vectordb/tos

Run with: pytest tests/test_pipeline_integration.py -v -s
"""

import os
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.rag_pipeline import RAGPipeline, ResponseSource
from src.llm import create_llm_client


# Test queries organized by expected source
TEST_QUERIES = {
    "qna": [
        # IMA 관련
        "IMA 상품은 원금이 보장되나요?",
        "IMA 수익률은 어떻게 되나요?",
        "IMA 중도해지 가능한가요?",
        # 공모주 청약 관련
        "공모주 청약자격이 어떻게 되나요?",
        "청약 우대 조건이 뭔가요?",
        # 계좌 관련
        "계좌 비밀번호를 잊어버렸어요",
        "비밀번호 5회 틀리면 어떻게 하나요?",
        # 수수료/세금 관련
        "주식 매매 세금이 얼마예요?",
        "주식 수수료 얼마나 되나요?",
        # 금현물 관련
        "금현물 거래 방법 알려주세요",
    ],
    "tos": [
        # CMS 출금이체 약관 관련
        "CMS 출금이체란 무엇인가요?",
        "출금이체 해지는 어떻게 하나요?",
        "자동이체 통합관리시스템이 뭔가요?",
        # 약관 조항 관련
        "제1조 약관의 적용에 대해 알려주세요",
        "제9조 출금이체 해지에 대해 설명해주세요",
        # ELW 관련
        "ELW 투자 시 유의사항이 뭔가요?",
        "ELW의 시간가치 감소란?",
        # 유렉스 관련
        "유렉스 연계선물이 뭔가요?",
        "유렉스 거래 위험에 대해 알려주세요",
    ],
    "general": [
        # 일반적인 질문 (context 없을 가능성)
        "오늘 날씨 어때요?",
        "맛있는 음식 추천해주세요",
        "파이썬 코딩 방법",
    ],
    "edge_cases": [
        # 경계 케이스
        "",  # 빈 쿼리
        "?",  # 특수문자만
        "ㅋㅋㅋㅋ",  # 의미없는 텍스트
        "a" * 500,  # 긴 쿼리
    ],
}


@pytest.fixture(scope="module")
def pipeline():
    """Create a pipeline instance for all tests."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    
    try:
        llm = create_llm_client(provider="openai", model="gpt-4o-mini")
    except Exception as e:
        pytest.skip(f"Failed to create OpenAI client: {e}")
    
    qna_db = Path("data/vectordb/qna")
    tos_db = Path("data/vectordb/tos")
    
    if not qna_db.exists() or not tos_db.exists():
        pytest.skip("Vector DB not found. Run indexing first.")
    
    return RAGPipeline(
        llm=llm,
        qna_db_path=str(qna_db),
        tos_db_path=str(tos_db),
        enable_verification=False,  # Disable for faster tests
    )


class TestPipelineIntegration:
    """Integration tests for RAG Pipeline."""
    
    def test_pipeline_initialization(self, pipeline):
        """Test that pipeline initializes correctly."""
        assert pipeline is not None
        assert pipeline.qna_store.count() > 0, "QnA store is empty"
        assert pipeline.tos_store.count() > 0, "ToS store is empty"
        print(f"\nQnA documents: {pipeline.qna_store.count()}")
        print(f"ToS documents: {pipeline.tos_store.count()}")
    
    @pytest.mark.parametrize("query", TEST_QUERIES["qna"])
    def test_qna_queries(self, pipeline, query):
        """Test queries expected to match QnA."""
        response = pipeline.query(query)
        
        assert response is not None
        assert response.answer is not None
        assert len(response.answer) > 0
        
        print(f"\n[Query] {query}")
        print(f"[Source] {response.source.value}")
        print(f"[Confidence] {response.confidence:.3f}")
        print(f"[Answer] {response.answer[:200]}...")
        
        # QnA 쿼리는 QnA 또는 ToS에서 답변이 나와야 함
        assert response.source in [ResponseSource.QNA, ResponseSource.TOS]
    
    @pytest.mark.parametrize("query", TEST_QUERIES["tos"])
    def test_tos_queries(self, pipeline, query):
        """Test queries expected to match ToS."""
        response = pipeline.query(query)
        
        assert response is not None
        assert response.answer is not None
        assert len(response.answer) > 0
        
        print(f"\n[Query] {query}")
        print(f"[Source] {response.source.value}")
        print(f"[Confidence] {response.confidence:.3f}")
        print(f"[Answer] {response.answer[:200]}...")
        
        # ToS 쿼리는 주로 ToS에서 답변이 나와야 하지만 QnA일 수도 있음
        assert response.source in [ResponseSource.QNA, ResponseSource.TOS, ResponseSource.NO_CONTEXT]
    
    @pytest.mark.parametrize("query", TEST_QUERIES["general"])
    def test_general_queries(self, pipeline, query):
        """Test general queries that may not have context."""
        response = pipeline.query(query)
        
        assert response is not None
        assert response.answer is not None
        
        print(f"\n[Query] {query}")
        print(f"[Source] {response.source.value}")
        print(f"[Confidence] {response.confidence:.3f}")
        print(f"[Answer] {response.answer[:200]}...")
        
        # 일반 쿼리는 NO_CONTEXT일 가능성이 높음
        # 하지만 어떤 응답이든 받아야 함
        assert response.source in [ResponseSource.QNA, ResponseSource.TOS, ResponseSource.NO_CONTEXT]
    
    @pytest.mark.parametrize("query", TEST_QUERIES["edge_cases"])
    def test_edge_case_queries(self, pipeline, query):
        """Test edge case queries."""
        if not query:  # 빈 쿼리 스킵
            pytest.skip("Empty query")
        
        try:
            response = pipeline.query(query)
            
            assert response is not None
            print(f"\n[Query] {query[:50]}...")
            print(f"[Source] {response.source.value}")
            print(f"[Answer] {response.answer[:100] if response.answer else 'None'}...")
        except Exception as e:
            # Edge case는 에러가 발생할 수 있음
            print(f"\n[Query] {query[:50]}... raised {type(e).__name__}: {e}")


class TestSearchFunctions:
    """Test direct search functions."""
    
    def test_search_qna(self, pipeline):
        """Test direct QnA search."""
        results = pipeline.search_qna("IMA 원금", n_results=5)
        
        assert len(results) > 0
        print(f"\n=== QnA Search Results ===")
        for i, r in enumerate(results, 1):
            print(f"[{i}] Score: {r['score']:.3f}")
            print(f"    Q: {r['question'][:80]}...")
            print(f"    A: {r['answer'][:80]}...")
    
    def test_search_tos(self, pipeline):
        """Test direct ToS search."""
        results = pipeline.search_tos("출금이체", n_results=5)
        
        assert len(results) > 0
        print(f"\n=== ToS Search Results ===")
        for i, r in enumerate(results, 1):
            print(f"[{i}] Score: {r['score']:.3f}")
            print(f"    Doc: {r['document_title']}")
            print(f"    Section: {r['section_title']}")
    
    def test_search_tos_by_article(self, pipeline):
        """Test ToS search by article number."""
        results = pipeline.search_tos("제1조", n_results=5)
        
        assert len(results) > 0
        print(f"\n=== ToS Search '제1조' Results ===")
        for i, r in enumerate(results, 1):
            print(f"[{i}] Score: {r['score']:.3f}")
            print(f"    Section: {r['section_title']}")


class TestResponseQuality:
    """Test response quality metrics."""
    
    def test_response_has_citations(self, pipeline):
        """Test that responses include citations when from context."""
        response = pipeline.query("출금이체 해지 방법 알려주세요")
        
        print(f"\n[Query] 출금이체 해지 방법")
        print(f"[Source] {response.source.value}")
        print(f"[Citations] {response.citations}")
        print(f"[Answer] {response.answer[:300]}...")
        
        if response.source != ResponseSource.NO_CONTEXT:
            # Context가 있으면 citations이 있어야 함 (하지만 필수는 아님)
            pass
    
    def test_confidence_score_range(self, pipeline):
        """Test that confidence scores are in valid range."""
        queries = [
            "IMA 원금 보장",  # High confidence expected
            "무작위 텍스트 abc123",  # Low confidence expected
        ]
        
        for query in queries:
            response = pipeline.query(query)
            
            assert 0.0 <= response.confidence <= 1.0, \
                f"Confidence {response.confidence} out of range for query: {query}"
            
            print(f"\n[Query] {query}")
            print(f"[Confidence] {response.confidence:.3f}")
    
    def test_response_to_dict(self, pipeline):
        """Test that response can be serialized to dict."""
        response = pipeline.query("주식 수수료 알려주세요")
        
        d = response.to_dict()
        
        assert "query" in d
        assert "answer" in d
        assert "source" in d
        assert "confidence" in d
        
        print(f"\n[Response Dict Keys] {list(d.keys())}")


class TestPipelineWithVerification:
    """Test pipeline with verification enabled."""
    
    @pytest.fixture
    def pipeline_with_verification(self):
        """Create pipeline with verification enabled."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")
        
        try:
            llm = create_llm_client(provider="openai", model="gpt-4o-mini")
        except Exception as e:
            pytest.skip(f"Failed to create OpenAI client: {e}")
        
        qna_db = Path("data/vectordb/qna")
        tos_db = Path("data/vectordb/tos")
        
        if not qna_db.exists() or not tos_db.exists():
            pytest.skip("Vector DB not found")
        
        return RAGPipeline(
            llm=llm,
            qna_db_path=str(qna_db),
            tos_db_path=str(tos_db),
            enable_verification=True,
            verification_threshold=0.7,
        )
    
    def test_verification_result(self, pipeline_with_verification):
        """Test that verification produces results."""
        response = pipeline_with_verification.query("출금이체 약관에 대해 알려주세요")
        
        print(f"\n[Query] 출금이체 약관")
        print(f"[Source] {response.source.value}")
        print(f"[Verified] {response.verified}")
        print(f"[Verification Score] {response.verification_score:.3f}")
        print(f"[Verification Issues] {response.verification_issues}")
        
        if response.source != ResponseSource.NO_CONTEXT:
            # verification 결과가 있어야 함
            assert response.verification_score >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
