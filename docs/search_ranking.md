# Search and Ranking

To ensure high retrieval accuracy for complex legal documents (ToS), we employ a multi-stage search strategy combining hybrid retrieval and re-ranking.

## 1. Hybrid Search
Implemented in `src/tos_search/hybrid_search.py`. This combines two signals:

1.  **Dense Vector Search (Semantic)**:
    *   Finds meaning-based matches (e.g., "money back" matching "refund policy").
    *   Powered by ChromaDB + E5 Embeddings.
2.  **Rule/Keyword Matcher (Lexical)**:
    *   Implemented in `src/tos_search/rule_matcher.py`.
    *   Extracts specific references like "Article 5" or "Clause 3" using Regex.
    *   Matches exact keywords found in the query against document titles/headers.

**Combination**: The final score is a weighted average (controlled by `alpha` in config):
$$ Score = \alpha \times VectorScore + (1 - \alpha) \times RuleScore $$

## 2. Re-Ranking (Cross-Encoder)
After retrieving the top candidates (e.g., Top 20) using Hybrid Search, we use a **Cross-Encoder** to re-score them.

### Bi-Encoder vs. Cross-Encoder
*   **Bi-Encoder (Retrieval)**: Fast. Encodes query and doc separately. Good for finding *candidates*.
*   **Cross-Encoder (Reranking)**: Slower but more accurate. Takes `(Query, Document)` pair as input and outputs a relevance score. It "reads" the query and document together, understanding nuances better.

### Implementation
*   **Module**: `src/tos_search/reranker.py`.
*   **Model**: `BAAI/bge-reranker-v2-m3`.
*   **Process**:
    1. Retrieve top 20 docs via Hybrid Search.
    2. Pass pairs `(Query, Doc Title + Content)` to the Cross-Encoder.
    3. Sort by new score.
    4. Return Top 5 to the LLM.

This architecture significantly reduces "Lost in the Middle" phenomena and improves precision for specific legal queries.
