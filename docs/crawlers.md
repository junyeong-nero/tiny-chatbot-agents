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

1.  **Crawl**: Use the unified CLI. This saves raw JSONs under `data/raw/`.
    ```bash
    python main.py crawl qna
    python main.py crawl tos
    python main.py crawl all
    ```
2.  **Process**: Optional cleanup can happen before ingestion if you maintain an intermediate dataset.
3.  **Ingest**:
    ```bash
    python main.py ingest-qna
    python main.py ingest-tos
    ```
4.  **Validate**: Both ingest commands support post-load search checks via `--search` and `--top-k`.

### Useful Options

*   `python main.py crawl qna --categories CARD,LOAN --visible`
*   `python main.py ingest-qna --file data/raw/qna/example.json --clear --search "비밀번호"`
*   `python main.py ingest-tos --file data/raw/tos/example.json --category 약관 --search "제1조"`
