# Evaluation Framework

To ensure the quality of the chatbot, we implement an automated evaluation system using an **LLM Judge**.

## 📏 Metrics

The evaluation (`src/evaluation/llm_judge.py`) assesses answers on three key metrics:

1.  **Relevance (1-5)**: Does the answer directly address the user's query?
2.  **Faithfulness (1-5)**: Is the answer derived *only* from the provided context (no hallucinations)?
3.  **Completeness (1-5)**: Does the answer cover all aspects of the query?
4.  **Context Overlap**: Measures retrieval quality by comparing retrieved chunks with expected sources.
    *   **Context Recall**: Proportion of expected sources found.
    *   **Context Precision**: Proportion of retrieved chunks that are relevant.

## 🤖 LLM Judge

We use a strong LLM (typically GPT-4 or Claude 3.5 Sonnet) to act as the judge.

### Judge Process
1.  **Input**:
    *   `Question`
    *   `Ground Truth Answer` (optional but recommended)
    *   `Retrieved Context`
    *   `Generated Answer`
2.  **Prompting**: The judge is prompted with specific rubrics (`src/evaluation/judge_prompts.py`) to score each metric and provide reasoning.
3.  **Output**: A JSON report containing scores and explanations.

## 🏃 Running Evaluations

1.  **Generate Dataset**: Create a synthetic dataset from your docs.
    ```bash
    python scripts/generate_dataset.py
    ```
2.  **Run Evaluation**:
    ```bash
    # Sequential execution (default)
    python scripts/run_evaluation.py --input_file data/evaluation/dataset.json

    # Parallel execution (faster)
    python scripts/run_evaluation.py --input_file data/evaluation/dataset.json --parallel --max-workers 8
    ```

## 🚀 Advanced Features

### Parallel Evaluation
To speed up the evaluation process, which can be slow due to multiple LLM calls per query, you can use the `--parallel` flag.
*   **`--parallel`**: Enables concurrent evaluation of test cases.
*   **`--max-workers N`**: Controls the number of concurrent threads (default: 4).

### Robust LLM Judge
The judge implementation includes several features to handle API instability and non-standard LLM responses:
*   **Retry Logic**: Automatically retries failed API calls with exponential backoff.
*   **Partial Parsing**: Can extract scores even if the returned JSON is malformed.
*   **Score Clamping**: Ensures all scores are strictly within the 1-5 range.

### Korean Morphological Tokenization
For accurate BLEU score calculation on Korean text, we use `kiwipiepy`.
*   Traditional space-based splitting is inaccurate for agglutinative languages like Korean.
*   We tokenize text into morphemes (e.g., "한국투자증권에서" → "한국투자증권", "에서") to ensure n-gram matching reflects true semantic overlap.
