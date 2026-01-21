"""QnA Vector Store using ChromaDB with local embeddings.

This module provides a vector store specifically designed for QnA (FAQ) data,
supporting similarity search on questions to find relevant answers.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.config import Settings

from .embeddings import E5EmbeddingFunction

logger = logging.getLogger(__name__)


@dataclass
class QnASearchResult:
    """Search result from QnA vector store."""

    question: str
    answer: str
    category: str
    sub_category: str
    source: str
    source_url: str
    score: float  # Similarity score (higher is better for cosine similarity)
    id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "category": self.category,
            "sub_category": self.sub_category,
            "source": self.source,
            "source_url": self.source_url,
            "score": self.score,
            "id": self.id,
        }


class QnAVectorStore:
    """Vector store for QnA data using ChromaDB.

    Stores question-answer pairs and enables similarity search on questions
    to find relevant FAQ entries. Uses local embedding models only.

    The store follows the metadata schema from README.md:
    - question: The FAQ question text
    - answer: The answer text
    - source: "FAQ" or "human_agent"
    - category: Question category
    - created_at: Timestamp
    - human_verified: Whether verified by human

    Attributes:
        collection: ChromaDB collection for QnA data
        embedding_fn: Local embedding function
    """

    COLLECTION_NAME = "qna"

    def __init__(
        self,
        persist_directory: str | Path = "data/vectordb/qna",
        embedding_model: str | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize the QnA vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data
            embedding_model: Model key from embedding_config.yaml (uses default if None)
            device: Device for embedding model ('cpu', 'cuda', 'mps')
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Use E5 embedding function for proper query/document prefixing
        self.embedding_fn = E5EmbeddingFunction(
            model_name=embedding_model,
            device=device,
        )

        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,  # type: ignore[arg-type]
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"QnA Vector Store initialized. Collection '{self.COLLECTION_NAME}' "
            f"has {self.collection.count()} documents."
        )

    def add_qna(
        self,
        question: str,
        answer: str,
        category: str = "",
        sub_category: str = "",
        source: str = "FAQ",
        source_url: str = "",
        human_verified: bool = True,
        created_at: str | None = None,
        qna_id: str | None = None,
    ) -> str:
        """Add a single QnA entry to the store.

        Args:
            question: The question text (will be embedded for similarity search)
            answer: The answer text
            category: Question category
            sub_category: Question sub-category
            source: Source of QnA ("FAQ" or "human_agent")
            source_url: Original URL source
            human_verified: Whether the answer is verified by human
            created_at: Timestamp string (auto-generated if None)
            qna_id: Unique ID (auto-generated if None)

        Returns:
            The ID of the added entry
        """
        if created_at is None:
            created_at = datetime.now().isoformat()

        if qna_id is None:
            # Generate ID from question hash and timestamp
            hash_input = f"{question}_{created_at}"
            qna_id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        metadata = {
            "answer": answer,
            "category": category,
            "sub_category": sub_category,
            "source": source,
            "source_url": source_url,
            "human_verified": human_verified,
            "created_at": created_at,
        }

        # Add to collection - question is embedded, answer stored in metadata
        self.collection.add(
            ids=[qna_id],
            documents=[question],  # Embed the question for similarity search
            metadatas=[metadata],
        )

        logger.debug(f"Added QnA entry: {qna_id}")
        return qna_id

    def add_qna_batch(
        self,
        qna_items: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> list[str]:
        """Add multiple QnA entries in batches.

        Args:
            qna_items: List of QnA dictionaries with keys:
                - question (required)
                - answer (required)
                - category, sub_category, source, source_url, crawled_at (optional)
            batch_size: Number of items to add per batch

        Returns:
            List of added IDs
        """
        all_ids = []
        total = len(qna_items)

        for i in range(0, total, batch_size):
            batch = qna_items[i : i + batch_size]

            ids = []
            documents = []
            metadatas = []

            for item in batch:
                question = item.get("question", "")
                answer = item.get("answer", "")

                if not question or not answer:
                    logger.warning(f"Skipping item with empty question/answer: {item}")
                    continue

                created_at = item.get("crawled_at") or datetime.now().isoformat()

                # Generate ID
                hash_input = f"{question}_{created_at}"
                qna_id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

                ids.append(qna_id)
                documents.append(question)
                metadatas.append(
                    {
                        "answer": answer,
                        "category": item.get("category", ""),
                        "sub_category": item.get("sub_category", ""),
                        "source": item.get("source", "FAQ"),
                        "source_url": item.get("source_url", ""),
                        "human_verified": item.get("human_verified", True),
                        "created_at": created_at,
                    }
                )

            if ids:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )
                all_ids.extend(ids)

            logger.info(f"Added batch {i // batch_size + 1}: {len(ids)} items")

        logger.info(f"Total added: {len(all_ids)} QnA entries")
        return all_ids

    def search(
        self,
        query: str,
        n_results: int = 5,
        category_filter: str | None = None,
        score_threshold: float | None = None,
    ) -> list[QnASearchResult]:
        """Search for similar questions.

        Args:
            query: Search query (question to find similar matches for)
            n_results: Maximum number of results to return
            category_filter: Optional category to filter results
            score_threshold: Minimum similarity score (0-1, higher is better)

        Returns:
            List of QnASearchResult sorted by similarity (highest first)
        """
        # Build where filter if category specified
        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}

        # Use query embedding (with "query: " prefix for E5)
        query_embedding = self.embedding_fn.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,  # type: ignore[arg-type]
            include=["documents", "metadatas", "distances"],  # type: ignore[arg-type]
        )

        search_results = []

        if results["ids"] and results["ids"][0]:
            for i, qna_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances, convert to similarity score
                # For cosine distance: similarity = 1 - distance
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance

                if score_threshold is not None and score < score_threshold:
                    continue

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                question = results["documents"][0][i] if results["documents"] else ""

                search_results.append(
                    QnASearchResult(
                        question=question,
                        answer=str(metadata.get("answer", "")),
                        category=str(metadata.get("category", "")),
                        sub_category=str(metadata.get("sub_category", "")),
                        source=str(metadata.get("source", "")),
                        source_url=str(metadata.get("source_url", "")),
                        score=score,
                        id=qna_id,
                    )
                )

        return search_results

    def get_by_id(self, qna_id: str) -> QnASearchResult | None:
        """Get a specific QnA entry by ID.

        Args:
            qna_id: The ID of the entry to retrieve

        Returns:
            QnASearchResult if found, None otherwise
        """
        result = self.collection.get(
            ids=[qna_id],
            include=["documents", "metadatas"],  # type: ignore[arg-type]
        )

        if result["ids"]:
            metadata = result["metadatas"][0] if result["metadatas"] else {}
            question = result["documents"][0] if result["documents"] else ""

            return QnASearchResult(
                question=question,
                answer=str(metadata.get("answer", "")),
                category=str(metadata.get("category", "")),
                sub_category=str(metadata.get("sub_category", "")),
                source=str(metadata.get("source", "")),
                source_url=str(metadata.get("source_url", "")),
                score=1.0,
                id=qna_id,
            )

        return None

    def delete(self, qna_id: str) -> None:
        """Delete a QnA entry by ID."""
        self.collection.delete(ids=[qna_id])
        logger.debug(f"Deleted QnA entry: {qna_id}")

    def count(self) -> int:
        """Return the total number of QnA entries."""
        return self.collection.count()

    def load_from_json(self, json_path: str | Path) -> list[str]:
        """Load QnA data from a crawled JSON file.

        Args:
            json_path: Path to JSON file with crawled QnA data

        Returns:
            List of added IDs
        """
        json_path = Path(json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loading {len(data)} QnA items from {json_path}")
        return self.add_qna_batch(data)

    def clear(self) -> None:
        """Clear all entries from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,  # type: ignore[arg-type]
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Cleared all QnA entries")
