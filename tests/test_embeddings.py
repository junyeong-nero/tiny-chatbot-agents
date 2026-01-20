# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0

import pytest
from sentence_transformers import SentenceTransformer


@pytest.fixture(scope="module")
def model():
    """Load the embedding model once for all tests."""
    return SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")


@pytest.fixture
def sample_queries():
    return [
        "What is the capital of China?",
        "Explain gravity",
    ]


@pytest.fixture
def sample_documents():
    return [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]


class TestQwen3Embedding:
    def test_model_loads(self, model):
        """Test that the model loads successfully."""
        assert model is not None

    def test_encode_queries(self, model, sample_queries):
        """Test encoding queries with prompt."""
        query_embeddings = model.encode(sample_queries, prompt_name="query")
        assert query_embeddings.shape[0] == len(sample_queries)
        assert query_embeddings.shape[1] > 0

    def test_encode_documents(self, model, sample_documents):
        """Test encoding documents."""
        document_embeddings = model.encode(sample_documents)
        assert document_embeddings.shape[0] == len(sample_documents)
        assert document_embeddings.shape[1] > 0

    def test_similarity_matrix(self, model, sample_queries, sample_documents):
        """Test computing similarity between queries and documents."""
        query_embeddings = model.encode(sample_queries, prompt_name="query")
        document_embeddings = model.encode(sample_documents)

        similarity = model.similarity(query_embeddings, document_embeddings)

        # Check shape: (num_queries, num_documents)
        assert similarity.shape == (len(sample_queries), len(sample_documents))

    def test_relevant_pairs_have_higher_similarity(self, model, sample_queries, sample_documents):
        """Test that relevant query-document pairs have higher similarity."""
        query_embeddings = model.encode(sample_queries, prompt_name="query")
        document_embeddings = model.encode(sample_documents)

        similarity = model.similarity(query_embeddings, document_embeddings)

        # Query 0 ("capital of China") should be more similar to Document 0 ("Beijing")
        assert similarity[0, 0] > similarity[0, 1]

        # Query 1 ("Explain gravity") should be more similar to Document 1 ("Gravity...")
        assert similarity[1, 1] > similarity[1, 0]
