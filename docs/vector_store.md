# Vector Search System

This project uses **ChromaDB** as the underlying vector store for both QnA and ToS data.

## 💾 ChromaDB Configuration

-   **Persistence**: Data is stored locally in `data/vectordb/`.
-   **Collection**:
    -   `qna`: Stores FAQ pairs.
    -   `tos`: Stores chunked Terms of Service documents.
-   **Distance Metric**: Cosine Similarity (`hnsw:space`: "cosine").

## 🧠 Embedding Models (Bi-Encoders)

We use **Bi-Encoders** to convert text into fixed-size vectors. The model choice is critical for performance, especially for Korean text support.

### Default Model: `intfloat/multilingual-e5-large`
*   **Type**: E5 (Embeddings from Bidirectional Encoder Representations).
*   **Dimension**: 1024.
*   **Why**: Superior performance on multilingual retrieval tasks compared to standard BERT models. It requires specific prefixing:
    *   **Queries**: `query: ` prefix (e.g., "query: 환불 규정이 어떻게 되나요?")
    *   **Documents**: `passage: ` prefix.

### Other Supported Models
Configuration in `configs/embedding_config.yaml`:
*   `multilingual-e5-base`: Faster, lower dimension (768).
*   `BAAI/bge-m3`: Excellent multilingual support, long context.
*   `text-embedding-3-small`: OpenAI's embedding model (if configured).

## 🗂️ Data Schema

### QnA Collection Metadata
| Field | Type | Description |
| :--- | :--- | :--- |
| `question` | String | The question text (Vectorized). |
| `answer` | String | The answer text (Stored in metadata). |
| `category` | String | e.g., "Payment", "Account". |
| `source` | String | "FAQ" or "Manual". |

### ToS Collection Metadata
| Field | Type | Description |
| :--- | :--- | :--- |
| `document_title` | String | Name of the ToS (e.g., "Service Usage Terms"). |
| `section_title` | String | e.g., "Article 1 (Purpose)". |
| `section_content` | String | The actual text of the clause. |
| `chunk_index` | Int | Order of the chunk in the document. |
