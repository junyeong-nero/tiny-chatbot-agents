"""Hybrid search combining Vector, Rule-based, and Triplet search.

Final score = α * vector_score + β * rule_score + γ * triplet_score
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from src.tos_search.reranker import ToSReranker
from src.tos_search.rule_matcher import ToSRuleMatcher
from src.tos_search.triplet_store import TripletExtractor, TripletStore
from src.vectorstore.embeddings import load_embedding_config

if TYPE_CHECKING:
    from src.vectorstore import ToSVectorStore

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search weights."""

    vector_weight: float = 0.5  # α
    rule_weight: float = 0.3  # β
    triplet_weight: float = 0.2  # γ

    # Minimum scores to include component
    vector_threshold: float = 0.0
    rule_threshold: float = 0.0
    triplet_threshold: float = 0.0

    # Reranking options
    rerank_enabled: bool = False
    rerank_candidates: int = 20
    rerank_top_k: int | None = None
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_batch_size: int = 32
    rerank_device: str | None = None
    rerank_max_seq_length: int | None = None
    rerank_weight: float = 0.3

    @classmethod
    def from_embedding_config(cls) -> "HybridSearchConfig":
        config = load_embedding_config()
        reranker_config = config.get("reranker", {}) if isinstance(config, dict) else {}
        model_key = reranker_config.get("default_model")
        models = reranker_config.get("models", {}) if isinstance(reranker_config, dict) else {}

        model_name = cls.rerank_model
        max_seq_length = None

        if model_key and model_key in models:
            model_entry = models[model_key]
            if isinstance(model_entry, dict):
                model_name = model_entry.get("name", model_name)
                max_seq_length = model_entry.get("max_seq_length")
        elif model_key:
            logger.warning(f"Reranker default_model '{model_key}' not found; using '{model_name}'")

        return cls(
            rerank_enabled=reranker_config.get("enabled", cls.rerank_enabled),
            rerank_candidates=reranker_config.get("candidates", cls.rerank_candidates),
            rerank_top_k=reranker_config.get("top_k", cls.rerank_top_k),
            rerank_model=model_name,
            rerank_batch_size=reranker_config.get("batch_size", cls.rerank_batch_size),
            rerank_device=reranker_config.get("device", cls.rerank_device),
            rerank_max_seq_length=max_seq_length,
            rerank_weight=reranker_config.get("weight", cls.rerank_weight),
        )

    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.vector_weight + self.rule_weight + self.triplet_weight
        if total <= 0:
            raise ValueError("Hybrid search weights must sum to > 0")
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Hybrid search weights sum to {total}, normalizing to 1.0")
            self.vector_weight /= total
            self.rule_weight /= total
            self.triplet_weight /= total

        if not 0 <= self.rerank_weight <= 1:
            raise ValueError("rerank_weight must be between 0 and 1")


@dataclass
class HybridSearchResult:
    """Result from hybrid search."""

    chunk_id: str
    section_title: str
    section_content: str
    document_title: str
    category: str
    effective_date: str
    source_url: str

    # Scores
    combined_score: float
    vector_score: float = 0.0
    rule_score: float = 0.0
    triplet_score: float = 0.0
    rerank_score: float | None = None
    final_score: float | None = None

    # Additional info
    matched_keywords: list[str] = field(default_factory=list)
    section_ref_match: bool = False
    matched_triplets: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "section_title": self.section_title,
            "section_content": self.section_content,
            "document_title": self.document_title,
            "category": self.category,
            "effective_date": self.effective_date,
            "source_url": self.source_url,
            "combined_score": self.combined_score,
            "vector_score": self.vector_score,
            "rule_score": self.rule_score,
            "triplet_score": self.triplet_score,
            "rerank_score": self.rerank_score,
            "final_score": self.final_score,
            "matched_keywords": self.matched_keywords,
            "section_ref_match": self.section_ref_match,
            "matched_triplets": self.matched_triplets,
        }


class ToSHybridSearch:
    """Hybrid search combining Vector, Rule-based, and Triplet search.

    Combines three search methods:
    1. Vector search: Semantic similarity using embeddings
    2. Rule-based: Pattern matching for section references and keywords
    3. Triplet: Subject-Predicate-Object relationship matching
    """

    def __init__(
        self,
        vector_store: "ToSVectorStore",
        rule_matcher: ToSRuleMatcher | None = None,
        triplet_store: TripletStore | None = None,
        reranker: ToSReranker | None = None,
        config: HybridSearchConfig | None = None,
    ):
        """Initialize hybrid search.

        Args:
            vector_store: ToS vector store for embedding-based search
            rule_matcher: Rule-based matcher (created if not provided)
            triplet_store: Triplet store (created if not provided)
            config: Hybrid search configuration
        """
        self.vector_store = vector_store
        self.rule_matcher = rule_matcher or ToSRuleMatcher()
        self.triplet_store = triplet_store or TripletStore()
        self.config = config or HybridSearchConfig.from_embedding_config()
        self.reranker = None
        if self.config.rerank_enabled:
            self.reranker = reranker or ToSReranker(
                model_name=self.config.rerank_model,
                device=self.config.rerank_device,
                batch_size=self.config.rerank_batch_size,
                max_seq_length=self.config.rerank_max_seq_length,
            )

        logger.info(
            f"ToSHybridSearch initialized with weights: "
            f"vector={self.config.vector_weight:.2f}, "
            f"rule={self.config.rule_weight:.2f}, "
            f"triplet={self.config.triplet_weight:.2f}"
        )

    def search(
        self,
        query: str,
        n_results: int = 5,
        category_filter: str | None = None,
    ) -> list[HybridSearchResult]:
        """Perform hybrid search.

        Args:
            query: Search query
            n_results: Number of results to return
            category_filter: Optional category filter

        Returns:
            List of HybridSearchResult sorted by combined score
        """
        # 1. Vector search - get more candidates for re-ranking
        candidate_count = n_results * 3
        if self.config.rerank_enabled:
            candidate_count = max(candidate_count, self.config.rerank_candidates)

        vector_results = self.vector_store.search(
            query=query,
            n_results=candidate_count,  # Over-fetch for re-ranking
            category_filter=category_filter,
        )

        if not vector_results:
            logger.debug("No vector search results")
            return []

        # Convert to dicts for processing
        candidates: dict[str, dict[str, Any]] = {}
        for vr in vector_results:
            candidates[vr.id] = {
                "chunk_id": vr.id,
                "section_title": vr.section_title,
                "section_content": vr.section_content,
                "document_title": vr.document_title,
                "category": vr.category,
                "effective_date": vr.effective_date,
                "source_url": vr.source_url,
                "vector_score": vr.score,
                "rule_score": 0.0,
                "triplet_score": 0.0,
                "matched_keywords": [],
                "section_ref_match": False,
                "matched_triplets": [],
            }

        # 2. Rule-based scoring
        boosted = self.rule_matcher.boost_vector_results(
            query=query,
            vector_results=list(candidates.values()),
        )

        for item in boosted:
            chunk_id = item["chunk_id"]
            if chunk_id in candidates:
                candidates[chunk_id]["rule_score"] = item.get("rule_score", 0.0)
                candidates[chunk_id]["matched_keywords"] = item.get("matched_keywords", [])
                candidates[chunk_id]["section_ref_match"] = item.get("section_ref_match", False)

        # 3. Triplet-based scoring
        if self.triplet_store.count() > 0:
            triplet_results = self.triplet_store.search(query)

            # Map triplet scores to chunks
            chunk_triplet_scores: dict[str, float] = {}
            chunk_triplets: dict[str, list[dict[str, str]]] = {}

            for tr in triplet_results:
                chunk_id = tr.source_chunk_id
                if chunk_id in candidates:
                    # Accumulate triplet score
                    if chunk_id not in chunk_triplet_scores:
                        chunk_triplet_scores[chunk_id] = 0.0
                        chunk_triplets[chunk_id] = []

                    chunk_triplet_scores[chunk_id] += tr.score
                    chunk_triplets[chunk_id].append(
                        {
                            "subject": tr.triplet.subject,
                            "predicate": tr.triplet.predicate,
                            "object": tr.triplet.obj,
                        }
                    )

            # Normalize and apply
            max_triplet_score = max(chunk_triplet_scores.values()) if chunk_triplet_scores else 1.0
            for chunk_id, score in chunk_triplet_scores.items():
                candidates[chunk_id]["triplet_score"] = score / max_triplet_score
                candidates[chunk_id]["matched_triplets"] = chunk_triplets.get(chunk_id, [])

        # 4. Combine scores
        for data in candidates.values():
            combined_score = (
                self.config.vector_weight * data["vector_score"]
                + self.config.rule_weight * data["rule_score"]
                + self.config.triplet_weight * data["triplet_score"]
            )
            data["combined_score"] = combined_score

        sorted_candidates = sorted(
            candidates.values(),
            key=lambda x: x.get("combined_score", 0.0),
            reverse=True,
        )

        # 5. Optional Cross-Encoder reranking
        if self.reranker:
            candidate_pool = sorted_candidates[: max(self.config.rerank_candidates, n_results)]
            rerank_top_k = min(len(candidate_pool), n_results)
            if self.config.rerank_top_k is not None:
                rerank_top_k = min(rerank_top_k, self.config.rerank_top_k)
            reranked = self.reranker.rerank(
                query=query,
                candidates=candidate_pool,
                top_k=rerank_top_k,
                score_key="rerank_score",
            )
            reranked = self._apply_rerank_blend(reranked)
            reranked_ids = {item.get("chunk_id") for item in reranked}
            filled = reranked + [
                {**item, "final_score": item.get("combined_score", 0.0)}
                for item in sorted_candidates
                if item.get("chunk_id") not in reranked_ids
            ]
            return self._build_results(filled[:n_results])

        return self._build_results(sorted_candidates[:n_results])

    def _build_results(self, items: list[dict[str, Any]]) -> list[HybridSearchResult]:
        results = []
        for data in items:
            combined_score = data.get("combined_score", 0.0)
            final_score = data.get("final_score")
            if final_score is None:
                final_score = combined_score
            results.append(
                HybridSearchResult(
                    chunk_id=data["chunk_id"],
                    section_title=data["section_title"],
                    section_content=data["section_content"],
                    document_title=data["document_title"],
                    category=data["category"],
                    effective_date=data["effective_date"],
                    source_url=data["source_url"],
                    combined_score=combined_score,
                    vector_score=data.get("vector_score", 0.0),
                    rule_score=data.get("rule_score", 0.0),
                    triplet_score=data.get("triplet_score", 0.0),
                    rerank_score=data.get("rerank_score"),
                    final_score=final_score,
                    matched_keywords=data.get("matched_keywords", []),
                    section_ref_match=data.get("section_ref_match", False),
                    matched_triplets=data.get("matched_triplets", []),
                )
            )
        return results

    def _apply_rerank_blend(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        weight = self.config.rerank_weight
        rerank_scores: list[float] = []
        for item in items:
            score = item.get("rerank_score")
            if isinstance(score, (int, float)) and math.isfinite(score):
                rerank_scores.append(float(score))
        min_score = min(rerank_scores) if rerank_scores else 0.0
        max_score = max(rerank_scores) if rerank_scores else 0.0

        for item in items:
            combined = item.get("combined_score", 0.0)
            rerank_score = item.get("rerank_score")
            if rerank_score is None or not math.isfinite(rerank_score):
                item["final_score"] = combined
                continue
            if max_score > min_score:
                normalized = (rerank_score - min_score) / (max_score - min_score)
            else:
                normalized = 0.5
            final_score = (combined * (1 - weight)) + (normalized * weight)
            item["final_score"] = final_score

        return sorted(
            items,
            key=lambda x: x.get("final_score", x.get("combined_score", 0.0)),
            reverse=True,
        )

    def build_triplet_index(self) -> int:
        """Build triplet index from vector store contents.

        Returns:
            Number of triplets extracted
        """
        logger.info("Building triplet index from ToS chunks...")

        # Get all chunks from vector store
        all_results = self.vector_store.collection.get(
            include=["metadatas"],
        )

        if not all_results["ids"]:
            logger.warning("No documents in vector store to index")
            return 0

        # Prepare chunks for extraction
        chunks = []
        for i, chunk_id in enumerate(all_results["ids"]):
            metadata = all_results["metadatas"][i] if all_results["metadatas"] else {}
            chunks.append(
                {
                    "id": chunk_id,
                    "section_title": metadata.get("section_title", ""),
                    "section_content": metadata.get("section_content", ""),
                }
            )

        # Extract triplets
        extractor = TripletExtractor()
        triplets = extractor.extract_from_chunks(chunks)

        # Add to store
        self.triplet_store.add_batch(triplets)

        logger.info(f"Built triplet index with {len(triplets)} triplets")
        return len(triplets)

    def get_search_explanation(self, result: HybridSearchResult) -> str:
        """Generate human-readable explanation of search result.

        Args:
            result: Hybrid search result

        Returns:
            Explanation string
        """
        parts = []

        # Vector score
        parts.append(f"벡터 유사도: {result.vector_score:.2f}")

        # Rule score
        if result.rule_score > 0:
            rule_info = f"규칙 매칭: {result.rule_score:.2f}"
            if result.section_ref_match:
                rule_info += " (조항 일치)"
            if result.matched_keywords:
                rule_info += f" (키워드: {', '.join(result.matched_keywords)})"
            parts.append(rule_info)

        # Triplet score
        if result.triplet_score > 0:
            triplet_info = f"관계 매칭: {result.triplet_score:.2f}"
            if result.matched_triplets:
                t = result.matched_triplets[0]
                triplet_info += f" ({t['subject']}-{t['predicate']}-{t['object']})"
            parts.append(triplet_info)

        # Rerank score
        if result.rerank_score is not None:
            parts.append(f"재랭킹: {result.rerank_score:.2f}")

        if result.final_score is not None:
            parts.append(f"재랭킹 반영: {result.final_score:.2f}")

        final_score = (
            result.final_score if result.final_score is not None else result.combined_score
        )
        parts.append(f"최종 점수: {final_score:.2f}")

        return " | ".join(parts)
