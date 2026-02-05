# Search and Ranking

To ensure high retrieval accuracy for complex legal documents (ToS), we employ a multi-stage search strategy combining hybrid retrieval and re-ranking.

## 1. Hybrid Search
Implemented in `src/tos_search/hybrid_search.py`. This combines three signals:

1.  **Dense Vector Search (Semantic)**:
    *   Finds meaning-based matches (e.g., "money back" matching "refund policy").
    *   Powered by ChromaDB + E5 Embeddings.
2.  **Rule/Keyword Matcher (Lexical)**:
    *   Implemented in `src/tos_search/rule_matcher.py`.
    *   Extracts specific references like "Article 5" or "Clause 3" using Regex.
    *   Matches exact keywords found in the query against document titles/headers.
3.  **Triplet Matcher (Relation-aware)**:
    *   Implemented with `src/tos_search/triplet_store.py`.
    *   Uses subject-predicate-object triplets extracted from ToS chunks.
    *   Scores relation-level matches between query and stored triplets.

**Score Combination (Base Hybrid Score)**:
$$
\text{CombinedScore} =
\alpha \cdot \text{VectorScore} +
\beta \cdot \text{RuleScore} +
\gamma \cdot \text{TripletScore}
$$

Where:
*   `α = vector_weight` (default `0.5`)
*   `β = rule_weight` (default `0.3`)
*   `γ = triplet_weight` (default `0.2`)
*   `α + β + γ = 1` (weights are normalized in code if needed)

### Triplet Scoring Details
*   Triplet scoring is applied only when the triplet index has data (`triplet_store.count() > 0`).
*   Multiple matched triplets for the same chunk are accumulated.
*   The chunk-level triplet score is normalized by the maximum chunk triplet score among candidates:
$$
\text{TripletScore}_i =
\frac{\sum \text{MatchedTripletScore}_{i}}{\max_j \left(\sum \text{MatchedTripletScore}_{j}\right)}
$$
*   If no triplet matches are found, `triplet_score` remains `0.0`.

## 2. Re-Ranking (Cross-Encoder)
After retrieving the top candidates (e.g., Top 20) using Hybrid Search, we use a **Cross-Encoder** to re-score them.

### Bi-Encoder vs. Cross-Encoder
*   **Bi-Encoder (Retrieval)**: Fast. Encodes query and doc separately. Good for finding *candidates*.
*   **Cross-Encoder (Reranking)**: Slower but more accurate. Takes `(Query, Document)` pair as input and outputs a relevance score. It "reads" the query and document together, understanding nuances better.

### Implementation
*   **Module**: `src/tos_search/reranker.py`.
*   **Model**: `BAAI/bge-reranker-v2-m3`.
*   **Process**:
    1. Retrieve an expanded candidate pool from hybrid search (`n_results * 3`, or at least configured rerank candidates).
    2. Pass pairs `(Query, Doc Title + Content)` to the Cross-Encoder.
    3. Min-max normalize rerank scores within the candidate pool.
    4. Blend hybrid and rerank scores to produce final score:
$$
\text{FinalScore} = (1-w)\cdot \text{CombinedScore} + w \cdot \text{NormalizedRerankScore}
$$
    5. Sort by `FinalScore` and return top results.

Where:
*   `w = rerank_weight` (default `0.3`)
*   If rerank score is missing/invalid, `FinalScore = CombinedScore`.

This architecture significantly reduces "Lost in the Middle" phenomena and improves precision for specific legal queries.
