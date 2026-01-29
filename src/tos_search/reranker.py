"""Cross-Encoder Reranker for ToS Search.

This module implements a re-ranking stage using a Cross-Encoder model.
It takes a query and a list of candidate documents (retrieved by Bi-Encoder/Hybrid search)
and re-scores them by processing query and document pairs together.
"""

import logging
from typing import Any

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class ToSReranker:
    """Reranker for Terms of Service documents using Cross-Encoder."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        batch_size: int = 32,
        max_seq_length: int | None = None,
    ) -> None:
        """Initialize the reranker.

        Args:
            model_name: HuggingFace model ID for Cross-Encoder
            device: Device to run model on ('cpu', 'cuda', 'mps'). None for auto.
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        try:
            logger.info(f"Loading Cross-Encoder model: {model_name}")
            self.model = CrossEncoder(model_name, device=device)
            if self.max_seq_length:
                self.model.max_length = self.max_seq_length
        except Exception as e:
            logger.error(f"Failed to load Cross-Encoder model: {e}")
            self.model = None

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
        score_key: str = "rerank_score",
    ) -> list[dict[str, Any]]:
        """Rerank candidate documents based on query relevance.

        Args:
            query: User query string
            candidates: List of candidate documents (must contain 'section_content')
            top_k: Number of top results to return
            score_key: Key to store the new reranking score

        Returns:
            Top-k reranked candidates with added score field
        """
        if not self.model or not candidates:
            return candidates[:top_k]

        # Prepare pairs for Cross-Encoder
        # Format: (query, document_text)
        # We combine title and content for better context
        pairs = []
        for doc in candidates:
            content = doc.get("section_content", "")
            title = doc.get("section_title", "")
            doc_text = f"{title}\n{content}" if title else content
            pairs.append((query, doc_text))

        try:
            # Predict scores
            scores = self.model.predict(pairs, batch_size=self.batch_size)

            # Update candidates with new scores
            for i, doc in enumerate(candidates):
                doc[score_key] = float(scores[i])

            # Sort by new score descending
            reranked = sorted(
                candidates,
                key=lambda x: x.get(score_key, -float("inf")),
                reverse=True,
            )

            return reranked[:top_k]

        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Returning original order.")
            return candidates[:top_k]
