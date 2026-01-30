# RAG Architecture

The Request-Augmented Generation (RAG) pipeline is the core of this chatbot. It is designed to handle queries with varying levels of complexity, from simple FAQ lookups to complex interpretation of Terms of Service (ToS).

## Pipeline Strategy: QnA First, ToS Second

The pipeline (`src/pipeline/rag_pipeline.py`) follows a strict waterfall logic to ensure efficiency and accuracy.

### Flow Diagram

```mermaid
flowchart TD
    Start([User Query]) --> QnASearch[Search QnA Vector DB]
    QnASearch --> CheckQnA{Score >= QnA Threshold?}
    
    CheckQnA -- Yes --> BuildContextQnA[Use QnA Pair as Context]
    BuildContextQnA --> GenAnswer[Generate Answer]
    
    CheckQnA -- No --> ToSSearch[Search ToS Vector DB (Hybrid)]
    ToSSearch --> CheckToS{Score >= ToS Threshold?}
    
    CheckToS -- Yes --> Rerank[Rerank Results]
    Rerank --> BuildContextToS[Use Top-k ToS Sections]
    BuildContextToS --> GenAnswer
    
    CheckToS -- No --> NoContext[Answer without Context / Handoff]
    
    GenAnswer --> Verify[Verify Answer]
    Verify --> End([Final Response])
```

## Thresholds Configuration

The pipeline behavior is controlled by several confidence thresholds defined in `configs/agent_config.yaml`:

| Threshold | Default | Description |
| :--- | :--- | :--- |
| `qna.threshold` | **0.85** | Minimum similarity score to accept a QnA match. High to prevent wrong FAQ answers. |
| `tos.threshold` | **0.70** | Minimum score to consider a ToS section relevant. |
| `qna.mid_threshold` | **0.70** | "Mid-band" for QnA. If score is here, we might give a tentative answer or fallback to ToS. |
| `verifier.confidence` | **0.70** | Minimum score from the hallucination verifier to pass the answer. |

## 1. QnA Stage (High Precision)
*   **Goal**: Instantly answer common questions.
*   **Mechanism**: Dense Vector Search (Cosine Similarity).
*   **Data Source**: `data/vectordb/qna` (ChromaDB).
*   **Logic**: If the similarity between the user query and a stored question is extremely high (>0.85), we assume the stored answer is correct and use it to prompt the LLM to paraphrase the response.

## 2. ToS Stage (Deep Retrieval)
*   **Goal**: Answer complex questions based on legal text.
*   **Mechanism**: **Hybrid Search** (Vector + Keyword/Rule).
*   **Fallback**: If QnA fails, the system queries the ToS database.
*   **Refinement**: Retrieved candidates are often re-ranked (see [Search & Ranking](search_ranking.md)) to ensure the specific legal clause is at the top.

## 3. Verification Stage
*   **Goal**: Prevent hallucinations.
*   **Component**: `src/verifier/verifier.py`.
*   **Process**: A separate LLM call compares the *Generated Answer* against the *Retrieved Context* (QnA or ToS) to check for factual consistency. If the answer contains claims not supported by the context, it is flagged.
