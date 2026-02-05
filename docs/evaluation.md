# Evaluation Framework

The `tiny-chatbot-agents` evaluation framework provides a robust system for measuring the performance of RAG (Retrieval-Augmented Generation) pipelines. It combines traditional NLP metrics with modern LLM-as-a-Judge techniques to provide a multi-faceted view of model quality, accuracy, and reliability.

The framework is implemented in `src/evaluation/` and executed via `scripts/run_evaluation.py`.

## 1. Scope and Execution Flow

The evaluation process follows a structured pipeline to ensure consistency across different models and configurations.

### High-level Flow
1.  **Dataset Loading**: Loads a curated set of questions, reference answers, and expected sources.
2.  **Pipeline Initialization**: (Optional) Dynamically creates a RAG pipeline for the target model being evaluated.
3.  **Inference**: For each test case, the pipeline generates an answer and retrieves relevant context.
4.  **Automatic Scoring**: Computes metrics like BLEU, semantic similarity, and faithfulness (via `AnswerVerifier`).
5.  **LLM-as-a-Judge**: (Optional) Sends the question, reference answer, and generated answer to a high-capability model (e.g., GPT-4o) for nuanced quality assessment.
6.  **Aggregation**: Collects results across all test cases, computes statistics (mean, std), and breaks down performance by category.
7.  **Reporting**: Saves detailed results to JSON and optionally generates Markdown/CSV summaries.

### Core Components
*   **Runner** (`src/evaluation/runner.py`): Orchestrates the execution, handles parallelization, and aggregates results.
*   **Evaluator** (`src/evaluation/evaluator.py`): Implementation of all automatic metrics.
*   **LLM Judge** (`src/evaluation/llm_judge.py`): Interface for LLM-based evaluation.
*   **Dataset Generator** (`src/evaluation/dataset_generator.py`): Tool for creating synthetic evaluation datasets.

## 2. Metrics

The framework employs two layers of metrics to provide both efficiency and depth.

### Automatic Metrics (Layer 1)
Computed locally without requiring additional LLM calls (except for embeddings).

*   **`answer_similarity`**: Semantic similarity between the generated and reference answer using embedding cosine similarity (0.0 to 1.0).
*   **`bleu_score`**: N-gram overlap (0.0 to 1.0). For Korean text, it utilizes `kiwipiepy` for accurate morphological tokenization, falling back to whitespace if unavailable.
*   **`verifier_faithfulness`**: A score (0.0 to 1.0) derived from the `AnswerVerifier`, measuring how well the answer is supported by the retrieved context (hallucination check).
*   **Context Metrics**:
    *   **`context_recall`**: Proportion of expected source documents that were actually retrieved.
    *   **`context_precision`**: Proportion of retrieved documents that were relevant (expected).
*   **Efficiency**: `latency_ms`, `input_tokens`, and `output_tokens`.

### LLM-as-a-Judge Metrics (Layer 2)
Optional high-fidelity metrics computed on a 1-5 scale (also provided as normalized 0-1 scores).

*   **`llm_correctness`**: Accuracy compared to the golden reference answer.
*   **`llm_helpfulness`**: How well the answer addresses the user's intent.
*   **`judge_context_faithfulness`**: LLM's assessment of whether the answer contradicts the provided context.
*   **`llm_fluency`**: Grammatical correctness and naturalness of the language.
*   **`llm_judge_score`**: Overall quality score.

> [!TIP]
> Use automatic metrics for rapid iteration and LLM-as-a-Judge for final model selection or deep-dive analysis.

## 3. Dataset Format

The evaluation runner expects a JSON file containing a list of test cases.

```json
[
  {
    "question": "What are the key terms for personal data processing?",
    "expected_answer": "The key terms include consent, purpose limitation, and data minimization...",
    "category": "Privacy Policy",
    "expected_sources": ["doc_001", "privacy_section_5"]
  }
]
```

### Key Fields
*   `question`: The input query.
*   `expected_answer`: The "golden" reference answer.
*   `category`: (Optional) Used for performance breakdown in reports.
*   `expected_sources`: (Optional) List of document IDs or titles used to compute context recall/precision.

## 4. How to Run

### Basic Usage
Evaluate a specific model using the default dataset:
```bash
python scripts/run_evaluation.py --models "llama3.1:8b"
```

### Advanced Options
Evaluate multiple models in parallel with LLM-as-a-Judge enabled:
```bash
python scripts/run_evaluation.py \
  --models "llama3.1:8b,mistral:7b" \
  --dataset data/evaluation/my_test_set.json \
  --use-llm-judge \
  --parallel \
  --max-workers 8 \
  --report
```

### Running Reference-to-Reference
To evaluate the metrics themselves or compare reference answers without running the RAG pipeline:
```bash
python scripts/run_evaluation.py --no-pipeline
```

## 5. Model Diversity and Bias Prevention

To prevent "home-field advantage" (where a model evaluates its own outputs more favorably), the framework includes a **Model Diversity** feature.

When `--auto-diverse-judge` is used, the system automatically selects a judge model from a different provider than the model that generated the dataset. For example:
*   If the dataset was generated by GPT-4, the judge will be Claude 3.5.
*   If the dataset was generated by Claude, the judge will be Gemini.

This ensures a more objective and "strict" evaluation.

## 6. Configuration

The framework is highly configurable via `configs/evaluation_config.yaml`.

*   **`judge`**: Define the provider, model, and retry logic for LLM-as-a-Judge.
*   **`embedding`**: Specify the model used for semantic similarity (default: `multilingual-e5-small`).
*   **`runner`**: Set default parallelization and progress reporting.
*   **`diversity_pairs`**: Map generator models to their respective diverse judge models.

## 7. Outputs and Reports

Results are saved to the `results/` directory:
*   **JSON**: Full trace of every test case, including generated text, retrieved context, and individual scores.
*   **Markdown Report**: A human-readable summary with tables and category breakdowns.
*   **CSV Report**: Suitable for further analysis in Excel or data tools.

---
*For more details on the RAG architecture, see [RAG Architecture](./rag_architecture.md).*
