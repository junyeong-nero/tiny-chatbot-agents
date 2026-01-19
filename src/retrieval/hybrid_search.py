"""Hybrid Search combining Vector and Graph retrieval.

This module provides hybrid search capabilities that combine:
1. Vector-based semantic search (ToS Vector Store)
2. Graph-based relationship search (ToS Graph Store)
"""

import logging
from dataclasses import dataclass
from typing import Any

from src.vectorstore import ToSVectorStore

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """Result from hybrid search."""

    section_id: str
    section_title: str
    section_content: str
    document_title: str
    vector_score: float
    graph_score: float
    combined_score: float
    related_sections: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "section_title": self.section_title,
            "section_content": self.section_content,
            "document_title": self.document_title,
            "vector_score": self.vector_score,
            "graph_score": self.graph_score,
            "combined_score": self.combined_score,
            "related_sections": self.related_sections,
        }


class HybridSearch:
    """Hybrid search combining Vector and Graph retrieval.

    Uses score fusion to combine results from:
    1. Vector search: Semantic similarity based on embeddings
    2. Graph search: Relationship-based traversal

    Attributes:
        vector_store: ToS Vector Store instance
        graph_store: ToS Graph Store instance (optional)
        alpha: Weight for vector score (0-1), graph weight = 1 - alpha
    """

    def __init__(
        self,
        vector_store: ToSVectorStore,
        graph_store: Any = None,
        alpha: float = 0.7,
    ) -> None:
        """Initialize Hybrid Search.

        Args:
            vector_store: ToS Vector Store instance
            graph_store: ToS Graph Store instance (optional)
            alpha: Weight for vector score (default: 0.7)
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.alpha = alpha

        logger.info(
            f"Hybrid Search initialized. "
            f"Alpha: {alpha}, Graph available: {graph_store is not None}"
        )

    def search(
        self,
        query: str,
        n_results: int = 5,
        include_related: bool = True,
    ) -> list[HybridSearchResult]:
        """Perform hybrid search combining vector and graph results.

        Args:
            query: Search query
            n_results: Maximum number of results
            include_related: Whether to include related sections from graph

        Returns:
            List of HybridSearchResult sorted by combined score
        """
        # Vector search
        vector_results = self.vector_store.search(
            query=query,
            n_results=n_results * 2,  # Get more for fusion
        )

        # Graph search (if available)
        graph_results = []
        if self.graph_store and self.graph_store.is_available():
            graph_results = self.graph_store.search(
                query=query,
                n_results=n_results * 2,
            )

        # Fuse results
        fused = self._fuse_results(
            vector_results=vector_results,
            graph_results=graph_results,
            include_related=include_related,
        )

        # Sort by combined score and limit
        fused.sort(key=lambda x: x.combined_score, reverse=True)
        return fused[:n_results]

    def _fuse_results(
        self,
        vector_results: list[Any],
        graph_results: list[Any],
        include_related: bool,
    ) -> list[HybridSearchResult]:
        """Fuse vector and graph search results.

        Uses weighted score fusion:
        combined_score = alpha * vector_score + (1 - alpha) * graph_score
        """
        results_map: dict[str, HybridSearchResult] = {}

        # Process vector results
        for vr in vector_results:
            section_id = vr.id
            results_map[section_id] = HybridSearchResult(
                section_id=section_id,
                section_title=vr.section_title,
                section_content=vr.section_content,
                document_title=vr.document_title,
                vector_score=vr.score,
                graph_score=0.0,
                combined_score=self.alpha * vr.score,
                related_sections=[],
            )

        # Process graph results
        for gr in graph_results:
            section_id = gr.section_id

            if section_id in results_map:
                # Update existing result
                existing = results_map[section_id]
                results_map[section_id] = HybridSearchResult(
                    section_id=section_id,
                    section_title=existing.section_title,
                    section_content=existing.section_content,
                    document_title=existing.document_title,
                    vector_score=existing.vector_score,
                    graph_score=gr.score,
                    combined_score=(
                        self.alpha * existing.vector_score +
                        (1 - self.alpha) * gr.score
                    ),
                    related_sections=gr.related_sections if include_related else [],
                )
            else:
                # New result from graph only
                results_map[section_id] = HybridSearchResult(
                    section_id=section_id,
                    section_title=gr.section_title,
                    section_content=gr.section_content,
                    document_title=gr.document_title,
                    vector_score=0.0,
                    graph_score=gr.score,
                    combined_score=(1 - self.alpha) * gr.score,
                    related_sections=gr.related_sections if include_related else [],
                )

        return list(results_map.values())

    def set_alpha(self, alpha: float) -> None:
        """Update the fusion weight.

        Args:
            alpha: New alpha value (0.0 - 1.0)
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0")
        self.alpha = alpha
        logger.info(f"Hybrid Search alpha updated to {alpha}")
