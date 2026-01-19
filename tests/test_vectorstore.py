import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

MOCK_CONFIG = {
    "default_model": "test-model",
    "models": {
        "test-model": {
            "name": "test-model",
            "type": "e5",
            "dimension": 384,
            "normalize": True,
        },
        "multilingual-e5-large": {
            "name": "intfloat/multilingual-e5-large",
            "type": "e5",
            "dimension": 1024,
            "normalize": True,
        },
    },
    "device": None,
}


def create_mock_embedding_fn():
    def encode_side_effect(texts, **kwargs):
        if isinstance(texts, str):
            return np.array([0.1] * 384)
        return np.array([[0.1] * 384 for _ in range(len(texts))])

    mock_instance = MagicMock()
    mock_instance.encode.side_effect = encode_side_effect
    mock_instance.get_sentence_embedding_dimension.return_value = 384
    return mock_instance


@pytest.fixture(autouse=True)
def mock_config():
    with patch("src.vectorstore.embeddings.load_embedding_config", return_value=MOCK_CONFIG):
        yield


class TestLocalEmbeddingFunction:
    def test_embedding_function_call(self, mock_config):
        with patch("src.vectorstore.embeddings.SentenceTransformer") as MockModel:
            MockModel.return_value = create_mock_embedding_fn()

            from src.vectorstore.embeddings import LocalEmbeddingFunction

            embed_fn = LocalEmbeddingFunction(model_name="test-model")
            result = embed_fn(["hello", "world"])

            assert len(result) == 2
            assert len(result[0]) == 384

    def test_embedding_with_prefix(self, mock_config):
        with patch("src.vectorstore.embeddings.SentenceTransformer") as MockModel:
            mock_instance = create_mock_embedding_fn()
            MockModel.return_value = mock_instance

            from src.vectorstore.embeddings import LocalEmbeddingFunction

            embed_fn = LocalEmbeddingFunction(model_name="test-model", prefix="query: ")
            embed_fn(["hello"])

            call_args = mock_instance.encode.call_args[0][0]
            assert call_args == ["query: hello"]


class TestE5EmbeddingFunction:
    def test_embed_documents_uses_passage_prefix(self, mock_config):
        with patch("src.vectorstore.embeddings.SentenceTransformer") as MockModel:
            mock_instance = create_mock_embedding_fn()
            MockModel.return_value = mock_instance

            from src.vectorstore.embeddings import E5EmbeddingFunction

            embed_fn = E5EmbeddingFunction(model_name="test-model")
            embed_fn.embed_documents(["hello"])

            call_args = mock_instance.encode.call_args[0][0]
            assert call_args == ["passage: hello"]

    def test_embed_query_uses_query_prefix(self, mock_config):
        with patch("src.vectorstore.embeddings.SentenceTransformer") as MockModel:
            mock_instance = create_mock_embedding_fn()
            MockModel.return_value = mock_instance

            from src.vectorstore.embeddings import E5EmbeddingFunction

            embed_fn = E5EmbeddingFunction(model_name="test-model")
            embed_fn.embed_query("hello")

            call_args = mock_instance.encode.call_args[0][0]
            assert call_args == "query: hello"


class TestQnAVectorStore:
    @pytest.fixture
    def qna_store(self, tmp_path, mock_config):
        with patch("src.vectorstore.embeddings.SentenceTransformer") as MockModel:
            MockModel.return_value = create_mock_embedding_fn()

            from src.vectorstore import QnAVectorStore

            store = QnAVectorStore(
                persist_directory=tmp_path / "test_vectordb",
                embedding_model="test-model",
            )
            yield store

    def test_add_single_qna(self, qna_store):
        qna_id = qna_store.add_qna(
            question="환불은 어떻게 하나요?",
            answer="마이페이지에서 환불 신청하세요.",
            category="환불",
            source="FAQ",
        )

        assert qna_id is not None
        assert qna_store.count() == 1

    def test_add_qna_batch(self, qna_store):
        items = [
            {"question": "질문1", "answer": "답변1", "category": "카테고리1"},
            {"question": "질문2", "answer": "답변2", "category": "카테고리2"},
            {"question": "질문3", "answer": "답변3", "category": "카테고리3"},
        ]

        ids = qna_store.add_qna_batch(items)

        assert len(ids) == 3
        assert qna_store.count() == 3

    def test_add_qna_batch_skips_empty(self, qna_store):
        items = [
            {"question": "질문1", "answer": "답변1"},
            {"question": "", "answer": "답변2"},
            {"question": "질문3", "answer": ""},
        ]

        ids = qna_store.add_qna_batch(items)

        assert len(ids) == 1
        assert qna_store.count() == 1

    def test_get_by_id(self, qna_store):
        qna_id = qna_store.add_qna(
            question="테스트 질문",
            answer="테스트 답변",
            category="테스트",
        )

        result = qna_store.get_by_id(qna_id)

        assert result is not None
        assert result.question == "테스트 질문"
        assert result.answer == "테스트 답변"
        assert result.category == "테스트"

    def test_get_by_id_not_found(self, qna_store):
        result = qna_store.get_by_id("nonexistent-id")

        assert result is None

    def test_delete(self, qna_store):
        qna_id = qna_store.add_qna(
            question="삭제할 질문",
            answer="삭제할 답변",
        )

        assert qna_store.count() == 1

        qna_store.delete(qna_id)

        assert qna_store.count() == 0

    def test_clear(self, qna_store):
        qna_store.add_qna(question="질문1", answer="답변1")
        qna_store.add_qna(question="질문2", answer="답변2")

        assert qna_store.count() == 2

        qna_store.clear()

        assert qna_store.count() == 0

    def test_load_from_json(self, qna_store, tmp_path):
        json_data = [
            {
                "question": "IMA는 원금 보장 상품 인가요?",
                "answer": "회사는 만기 시 IMA의 세전평가금액...",
                "category": "전체",
                "sub_category": "",
                "source": "FAQ",
                "source_url": "https://example.com",
                "crawled_at": "2024-01-19T15:50:09",
            },
            {
                "question": "IMA는 수익률은 어떻게 되나요?",
                "answer": "IMA는 실적배당형 상품으로...",
                "category": "전체",
                "sub_category": "",
                "source": "FAQ",
                "source_url": "https://example.com",
                "crawled_at": "2024-01-19T15:50:10",
            },
        ]

        json_path = tmp_path / "test_qna.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False)

        ids = qna_store.load_from_json(json_path)

        assert len(ids) == 2
        assert qna_store.count() == 2


class TestQnASearchResult:
    def test_to_dict(self):
        from src.vectorstore.qna_store import QnASearchResult

        result = QnASearchResult(
            question="질문",
            answer="답변",
            category="카테고리",
            sub_category="서브카테고리",
            source="FAQ",
            source_url="https://example.com",
            score=0.95,
            id="test-id",
        )

        d = result.to_dict()

        assert d["question"] == "질문"
        assert d["answer"] == "답변"
        assert d["score"] == 0.95
        assert d["id"] == "test-id"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
