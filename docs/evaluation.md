# Evaluation Framework

To ensure the quality of the chatbot, we implement an automated evaluation system using an **LLM Judge**.

## 📏 Metrics

The evaluation (`src/evaluation/llm_judge.py`) assesses answers on three key metrics:

1.  **Relevance (1-5)**: Does the answer directly address the user's query?
2.  **Faithfulness (1-5)**: Is the answer derived *only* from the provided context (no hallucinations)?
3.  **Completeness (1-5)**: Does the answer cover all aspects of the query?

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
    python scripts/run_evaluation.py --input_file data/evaluation/dataset.json
    ```
