# RAG Architecture

The Request-Augmented Generation (RAG) pipeline is the core of this chatbot. It is designed to handle queries with varying levels of complexity, from simple FAQ lookups to complex interpretation of Terms of Service (ToS).

## Pipeline Strategy: QnA First, ToS Second

The pipeline (`src/pipeline/rag_pipeline.py`) follows a strict waterfall logic to ensure efficiency and accuracy.

### Flow Diagram

```mermaid
flowchart TD
    Start([User Query]) --> QnASearch[Search QnA Vector DB]
    QnASearch --> CheckQnA{Score >= 0.80?}
    
    CheckQnA -- Yes --> BuildContextQnA[Use QnA Pair as Context]
    BuildContextQnA --> GenAnswer[Generate Answer]
    
    CheckQnA -- No --> CheckQnAMid{Score >= 0.70?}
    CheckQnAMid -- Yes --> ToSSearchMid[Search ToS (Check Better Match)]
    CheckQnAMid -- No --> ToSSearch[Search ToS Vector DB (Hybrid)]
    
    ToSSearchMid --> Compare{ToS Confidence > QnA?}
    Compare -- Yes --> ToSSearch
    Compare -- No --> BuildContextQnALimited[Use QnA (Limited Answer)]
    BuildContextQnALimited --> GenAnswer
    
    ToSSearch --> CheckToS{Score >= 0.65?}
    
    CheckToS -- Yes --> Rerank[Rerank Results]
    Rerank --> BuildContextToS[Use Top-k ToS Sections]
    BuildContextToS --> GenAnswer
    
    CheckToS -- No --> CheckToSMid{Score >= 0.55?}
    CheckToSMid -- Yes --> BuildContextToSLimited[Use ToS (Limited Answer)]
    BuildContextToSLimited --> GenAnswer
    
    CheckToSMid -- No --> CheckToSLow{Score >= 0.40?}
    CheckToSLow -- Yes --> Clarify[Ask for Clarification]
    CheckToSLow -- No --> NoContext[Answer without Context / Handoff]
    
    GenAnswer --> Verify[Verify Answer]
    Verify --> End([Final Response])
    Clarify --> End
    NoContext --> End
```

## Thresholds Configuration

The pipeline behavior is controlled by several confidence thresholds defined in `configs/agent_config.yaml`:

| Threshold | Default | Description |
| :--- | :--- | :--- |
| `qna.threshold` | **0.80** | Minimum similarity score to accept a QnA match. High to prevent wrong FAQ answers. |
| `tos.threshold` | **0.65** | Minimum score to consider a ToS section highly relevant. |
| `qna.mid_threshold` | **0.70** | "Mid-band" for QnA. If score is here, we give a tentative answer or fallback to ToS if better match found. |
| `tos.mid_threshold` | **0.55** | "Mid-band" for ToS. Used for "Limited Answer" when confidence is moderate. |
| `tos.low_threshold` | **0.40** | "Low-band" for ToS. Used to trigger a "Clarification" response asking for more info. |
| `verifier.confidence` | **0.70** | Minimum score from the hallucination verifier to pass the answer. |

## 1. QnA Stage (High Precision)
*   **Goal**: Instantly answer common questions.
*   **Mechanism**: Dense Vector Search (Cosine Similarity).
*   **Data Source**: `data/vectordb/qna` (ChromaDB).
*   **Logic**:
    *   **High Confidence (>= 0.80)**: Directly use the matched QnA to answer.
    *   **Mid Confidence (0.70 - 0.79)**: Check ToS. If ToS has a better match, use ToS. Otherwise, provide a "Limited Answer" based on the QnA, explicitly stating uncertainty.

## 2. ToS Stage (Deep Retrieval)
*   **Goal**: Answer complex questions based on legal text.
*   **Mechanism**: **Hybrid Search** (Vector + Keyword/Rule).
*   **Fallback**: If QnA fails (or is in mid-band with lower confidence), the system queries the ToS database.
*   **Logic**:
    *   **High Confidence (>= 0.65)**: Use retrieved sections to generate a definitive answer.
    *   **Mid Confidence (0.55 - 0.64)**: Generate a "Limited Answer" with a disclaimer about uncertainty.
    *   **Low Confidence (0.40 - 0.54)**: Return a "Clarification" response, asking the user for more specific details.
*   **Refinement**: Retrieved candidates are often re-ranked (see [Search & Ranking](search_ranking.md)) to ensure the specific legal clause is at the top.

## 3. Verification Stage
*   **Goal**: Prevent hallucinations.
*   **Component**: `src/verifier/verifier.py`.
*   **Process**: A separate LLM call compares the *Generated Answer* against the *Retrieved Context* (QnA or ToS) to check for factual consistency. If the answer contains claims not supported by the context, it is flagged.
