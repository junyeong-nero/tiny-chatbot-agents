"""Evaluation package for LLM benchmarking.

This package provides tools for:
- Evaluating LLM-generated answers against expected answers
- Computing similarity, BLEU, and faithfulness metrics
- LLM-as-a-Judge evaluation using frontier models
- Generating evaluation datasets with golden answers
- Running batch evaluations and generating reports
"""

from .evaluator import EvaluationMetrics, LLMEvaluator
from .runner import EvaluationResult, EvaluationRunner

# LLM-as-a-Judge components
from .llm_judge import CriteriaScore, JudgeResult, LLMJudge, create_llm_judge

# Dataset generation components
from .dataset_generator import (
    DatasetGenerator,
    Difficulty,
    EvaluationDataset,
    EvaluationItem,
    create_dataset_generator,
)

# Frontier model client
from .frontier_client import (
    FrontierClient,
    FrontierModelConfig,
    FrontierProvider,
    JudgeModelSelector,
    create_frontier_client,
)

__all__ = [
    # Core evaluation
    "LLMEvaluator",
    "EvaluationMetrics",
    "EvaluationRunner",
    "EvaluationResult",
    # LLM-as-a-Judge
    "LLMJudge",
    "JudgeResult",
    "CriteriaScore",
    "create_llm_judge",
    "JudgeModelSelector",
    # Dataset generation
    "DatasetGenerator",
    "EvaluationDataset",
    "EvaluationItem",
    "Difficulty",
    "create_dataset_generator",
    # Frontier client
    "FrontierClient",
    "FrontierModelConfig",
    "FrontierProvider",
    "create_frontier_client",
]
