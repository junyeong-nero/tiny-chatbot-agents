"""Tests for ToS Search module (rule-based, triplet, hybrid)."""

import pytest

import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tos_search.hybrid_search import HybridSearchConfig, ToSHybridSearch
from src.tos_search.rule_matcher import ToSRuleMatcher, SectionRef
from src.tos_search.triplet_store import TripletExtractor, TripletStore, Triplet


class TestToSRuleMatcher:
    """Tests for rule-based ToS matcher."""

    @pytest.fixture
    def matcher(self):
        return ToSRuleMatcher()

    def test_extract_section_reference_basic(self, matcher):
        """Test extraction of basic section references."""
        ref = matcher.extract_section_reference("제1조가 뭐야?")
        assert ref is not None
        assert ref.article_num == 1

    def test_extract_section_reference_with_clause(self, matcher):
        """Test extraction with clause number."""
        ref = matcher.extract_section_reference("제3조 2항 알려줘")
        assert ref is not None
        assert ref.article_num == 3
        assert ref.clause_num == 2

    def test_extract_section_reference_with_title(self, matcher):
        """Test extraction with section title."""
        ref = matcher.extract_section_reference("제1조(목적) 내용 알려줘")
        assert ref is not None
        assert ref.article_num == 1
        assert ref.title == "목적"

    def test_extract_section_reference_none(self, matcher):
        """Test when no section reference exists."""
        ref = matcher.extract_section_reference("환불 규정이 어떻게 되나요?")
        assert ref is None

    def test_calculate_keyword_score(self, matcher):
        """Test keyword score calculation."""
        query = "환불 받고 싶어요"
        content = "환불 신청은 마이페이지에서 가능합니다."

        score, keywords = matcher.calculate_keyword_score(query, content)

        assert score > 0
        assert "환불" in keywords

    def test_calculate_keyword_score_no_match(self, matcher):
        """Test keyword score when no keywords match."""
        query = "시간이 얼마나 걸려요"
        content = "처리 기간은 영업일 기준 3일입니다."

        score, keywords = matcher.calculate_keyword_score(query, content)

        assert score == 0
        assert len(keywords) == 0

    def test_match_section(self, matcher):
        """Test section matching."""
        ref = SectionRef(article_num=1, title="목적")

        assert matcher.match_section(ref, "제1조 (목적)")
        assert matcher.match_section(ref, "제1조(목적)")
        assert not matcher.match_section(ref, "제2조 (적용)")

    def test_match_documents(self, matcher):
        """Test matching against documents."""
        documents = [
            {
                "id": "doc1",
                "section_title": "제1조 (목적)",
                "section_content": "이 약관은 서비스 이용에 관한 사항을 규정합니다.",
                "document_title": "서비스 이용약관",
                "category": "서비스약관",
            },
            {
                "id": "doc2",
                "section_title": "제5조 (환불)",
                "section_content": "회원은 환불을 신청할 수 있습니다. 위약금이 부과될 수 있습니다.",
                "document_title": "서비스 이용약관",
                "category": "서비스약관",
            },
        ]

        # Query with section reference
        results = matcher.match(query="제1조 내용 알려줘", documents=documents)
        assert len(results) >= 1
        assert results[0].section_ref_match is True

        # Query with keyword
        results = matcher.match(query="환불 규정이 뭐야", documents=documents)
        assert len(results) >= 1
        assert "환불" in results[0].matched_keywords


class TestTripletExtractor:
    """Tests for triplet extraction."""

    @pytest.fixture
    def extractor(self):
        return TripletExtractor()

    def test_extract_subject_predicate_pattern(self, extractor):
        """Test extraction of subject-predicate patterns."""
        text = "회사는 서비스 제공을 거부할 수 있습니다."
        triplets = extractor.extract(text)

        # May or may not match depending on exact pattern
        # Just ensure no errors
        assert isinstance(triplets, list)

    def test_extract_from_tos_text(self, extractor):
        """Test extraction from realistic ToS text."""
        text = """
        제3조(출금)
        ① 회사는 출금 요청을 거부할 수 있습니다.
        ② 고객은 출금 신청서를 제출해야 합니다.
        """

        triplets = extractor.extract(text, source_chunk_id="chunk_001")

        # Check that extraction doesn't fail
        assert isinstance(triplets, list)
        for t in triplets:
            assert isinstance(t, Triplet)
            assert t.source_chunk_id == "chunk_001"


class TestTripletStore:
    """Tests for triplet storage."""

    @pytest.fixture
    def store(self, tmp_path):
        return TripletStore(persist_path=tmp_path / "triplets.json")

    def test_add_and_retrieve(self, store):
        """Test adding and retrieving triplets."""
        triplet = Triplet(
            subject="회사",
            predicate="가능",
            obj="환불 거부",
            source_chunk_id="chunk_001",
        )

        triplet_id = store.add(triplet)

        assert store.count() == 1
        assert triplet_id == triplet.id

    def test_search_by_subject(self, store):
        """Test searching by subject."""
        store.add(Triplet(subject="회사", predicate="가능", obj="환불"))
        store.add(Triplet(subject="고객", predicate="의무", obj="서류 제출"))

        results = store.search_by_subject("회사")

        assert len(results) == 1
        assert results[0].triplet.subject == "회사"

    def test_search_by_predicate(self, store):
        """Test searching by predicate."""
        store.add(Triplet(subject="회사", predicate="가능", obj="환불"))
        store.add(Triplet(subject="고객", predicate="가능", obj="해지"))

        results = store.search_by_predicate("가능")

        assert len(results) == 2

    def test_search_fuzzy(self, store):
        """Test fuzzy search."""
        store.add(Triplet(subject="회사", predicate="거부_가능", obj="환불"))

        results = store.search("거부")

        assert len(results) >= 1

    def test_persistence(self, tmp_path):
        """Test save and load."""
        persist_path = tmp_path / "triplets.json"

        # Create and save
        store1 = TripletStore(persist_path=persist_path)
        store1.add(Triplet(subject="회사", predicate="가능", obj="환불"))
        store1.save()

        # Load in new instance
        store2 = TripletStore(persist_path=persist_path)

        assert store2.count() == 1

    def test_clear(self, store):
        """Test clearing store."""
        store.add(Triplet(subject="회사", predicate="가능", obj="환불"))
        assert store.count() == 1

        store.clear()

        assert store.count() == 0


class TestSectionRef:
    """Tests for SectionRef dataclass."""

    def test_to_pattern(self):
        """Test pattern generation."""
        ref = SectionRef(article_num=1)
        pattern = ref.to_pattern()

        assert "1" in pattern
        assert "조" in pattern

    def test_str_representation(self):
        """Test string representation."""
        ref = SectionRef(article_num=1, title="목적", clause_num=2)
        s = str(ref)

        assert "제1조" in s
        assert "목적" in s
        assert "2항" in s


class MockVectorResult:
    def __init__(self, chunk_id: str, score: float):
        self.id = chunk_id
        self.score = score
        self.section_title = "제1조 (목적)"
        self.section_content = "약관의 목적을 설명합니다."
        self.document_title = "서비스 이용약관"
        self.category = "약관"
        self.effective_date = "2024-01-01"
        self.source_url = "http://example.com"


class MockVectorStore:
    def __init__(self, results):
        self._results = results

    def search(self, query, n_results=5, category_filter=None):
        return self._results[:n_results]


class DummyReranker:
    def rerank(self, query, candidates, top_k=5, score_key="rerank_score"):
        for item in candidates:
            item[score_key] = 0.9 if item.get("chunk_id") == "c2" else 0.1
        return sorted(candidates, key=lambda x: x.get(score_key, 0), reverse=True)[:top_k]


class TestHybridSearchRerank:
    def test_rerank_disabled_final_score_matches_combined(self):
        vector_store = MockVectorStore(
            [
                MockVectorResult("c1", 0.8),
                MockVectorResult("c2", 0.6),
            ]
        )

        config = HybridSearchConfig(rerank_enabled=False)
        hybrid = ToSHybridSearch(vector_store=cast(Any, vector_store), config=config)

        results = hybrid.search("제1조 목적", n_results=2)

        assert len(results) == 2
        for r in results:
            assert r.final_score == r.combined_score

    def test_rerank_enabled_orders_and_fills_results(self):
        vector_store = MockVectorStore(
            [
                MockVectorResult("c1", 0.8),
                MockVectorResult("c2", 0.6),
            ]
        )

        config = HybridSearchConfig(
            rerank_enabled=True,
            rerank_candidates=2,
            rerank_top_k=1,
            rerank_weight=0.5,
        )
        hybrid = ToSHybridSearch(
            vector_store=cast(Any, vector_store),
            config=config,
            reranker=cast(Any, DummyReranker()),
        )

        results = hybrid.search("제1조 목적", n_results=2)

        assert len(results) == 2
        assert results[0].chunk_id == "c2"
        assert results[0].final_score is not None
        assert results[1].final_score is not None
        assert results[0].final_score >= results[1].final_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
