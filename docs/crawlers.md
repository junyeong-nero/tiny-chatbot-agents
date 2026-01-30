# Data Crawlers

The project includes `playwright` based crawlers to fetch fresh data for the vector stores.

## 🕷️ Components

### 1. QnA Crawler (`src/crawlers/qna_crawler.py`)
*   **Target**: FAQ pages.
*   **Method**: Selects Question/Answer pairs using CSS selectors (accordion elements, lists).
*   **Output**: JSON list of objects `{ "question": "...", "answer": "...", "category": "..." }`.

### 2. ToS Crawler (`src/crawlers/tos_crawler.py`)
*   **Target**: Terms of Service pages.
*   **Method**:
    *   Identifies document structure (Title, Articles, Clauses).
    *   Parses hierarchical text (Article 1 > Clause 2).
*   **Output**: JSON structure preserving the hierarchy.

## 🔄 Ingestion Pipeline

1.  **Crawl**: Run `python scripts/crawl.py`. This saves raw JSONs to `data/raw/`.
2.  **Process**: (Optional) `data/processed/` for intermediate cleaning.
3.  **Ingest**:
    *   `scripts/ingest_qna.py`: Reads JSON -> Embeds -> ChromaDB (QnA).
    *   `scripts/ingest_tos.py`: Reads JSON -> Chunks (by Article/Section) -> Embeds -> ChromaDB (ToS).
