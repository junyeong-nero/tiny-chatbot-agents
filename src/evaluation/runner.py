"""Evaluation Runner for batch LLM evaluation.

This module provides the EvaluationRunner class for running
evaluation across multiple test cases and models.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np

from .evaluator import EvaluationMetrics, LLMEvaluator
from .schemas import validate_dataset

if TYPE_CHECKING:
    from .config import EvaluationConfig

logger = logging.getLogger(__name__)


def _get_config() -> "EvaluationConfig":
    """Get evaluation config (lazy import to avoid circular dependency)."""
    from .config import get_config

    return get_config()


def _get_runner_config() -> tuple[bool, int, bool, int]:
    """Get runner configuration from config file.

    Returns:
        Tuple of (parallel, max_workers, show_progress, progress_interval)
    """
    try:
        config = _get_config()
        return (
            config.runner.parallel,
            config.runner.max_workers,
            config.runner.show_progress,
            config.runner.progress_interval,
        )
    except Exception:
        return (False, 4, True, 10)


# Default number of parallel workers (fallback)
DEFAULT_MAX_WORKERS = 4


def _safe_mean(values: list) -> float:
    """Compute mean of values, returning 0.0 for empty lists.

    Args:
        values: List of numeric values

    Returns:
        Mean of values or 0.0 if empty
    """
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: list) -> float:
    """Compute standard deviation of values, returning 0.0 for empty lists.

    Args:
        values: List of numeric values

    Returns:
        Standard deviation of values or 0.0 if empty
    """
    return float(np.std(values)) if values else 0.0


def _normalize_score(score: float, min_val: float = 1.0, max_val: float = 5.0) -> float:
    """Normalize a score from [min_val, max_val] to [0, 1] range.

    Args:
        score: Score to normalize
        min_val: Minimum value of the original scale
        max_val: Maximum value of the original scale

    Returns:
        Normalized score in [0, 1] range
    """
    if score <= 0:
        return 0.0
    return (score - min_val) / (max_val - min_val)


@dataclass
class EvaluationResult:
    """Aggregated evaluation results for a model."""

    model_name: str
    total_cases: int = 0
    evaluated_cases: int = 0

    # Aggregated metrics
    mean_similarity: float = 0.0
    mean_bleu: float = 0.0
    mean_faithfulness: float = 0.0
    mean_context_recall: float = 0.0
    mean_context_precision: float = 0.0
    mean_latency_ms: float = 0.0

    # Statistics
    std_similarity: float = 0.0
    std_bleu: float = 0.0
    std_latency_ms: float = 0.0

    # Verification stats
    verified_count: int = 0
    verification_rate: float = 0.0

    # LLM-as-Judge aggregated metrics (1-5 scale)
    mean_llm_judge_score: float = 0.0
    mean_llm_correctness: float = 0.0
    mean_llm_helpfulness: float = 0.0
    mean_llm_faithfulness: float = 0.0
    mean_llm_fluency: float = 0.0
    llm_judge_model: str = ""

    # Normalized LLM-Judge metrics (0-1 scale for consistency with similarity/bleu)
    mean_llm_judge_normalized: float = 0.0
    mean_llm_correctness_normalized: float = 0.0
    mean_llm_helpfulness_normalized: float = 0.0
    mean_llm_faithfulness_normalized: float = 0.0
    mean_llm_fluency_normalized: float = 0.0

    # Category breakdown
    category_scores: dict[str, dict[str, float]] = field(default_factory=dict)

    # Individual results
    case_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_cases": self.total_cases,
            "evaluated_cases": self.evaluated_cases,
            "metrics": {
                "mean_similarity": self.mean_similarity,
                "mean_bleu": self.mean_bleu,
                "mean_faithfulness": self.mean_faithfulness,
                "mean_context_recall": self.mean_context_recall,
                "mean_context_precision": self.mean_context_precision,
                "mean_latency_ms": self.mean_latency_ms,
                "std_similarity": self.std_similarity,
                "std_bleu": self.std_bleu,
                "std_latency_ms": self.std_latency_ms,
            },
            "llm_judge": {
                "mean_score": self.mean_llm_judge_score,
                "mean_correctness": self.mean_llm_correctness,
                "mean_helpfulness": self.mean_llm_helpfulness,
                "mean_faithfulness": self.mean_llm_faithfulness,
                "mean_fluency": self.mean_llm_fluency,
                "judge_model": self.llm_judge_model,
                # Normalized scores (0-1 scale)
                "mean_score_normalized": self.mean_llm_judge_normalized,
                "mean_correctness_normalized": self.mean_llm_correctness_normalized,
                "mean_helpfulness_normalized": self.mean_llm_helpfulness_normalized,
                "mean_faithfulness_normalized": self.mean_llm_faithfulness_normalized,
                "mean_fluency_normalized": self.mean_llm_fluency_normalized,
            },
            "verification": {
                "verified_count": self.verified_count,
                "verification_rate": self.verification_rate,
            },
            "category_scores": self.category_scores,
            "case_results": self.case_results,
        }


class EvaluationRunner:
    """Runner for batch LLM evaluation.

    Loads evaluation dataset, runs RAG pipeline with specified LLM,
    collects metrics, and aggregates results.

    Configuration is loaded from evaluation_config.yaml by default.
    """

    def __init__(
        self,
        pipeline: Any = None,
        evaluator: LLMEvaluator | None = None,
        dataset_path: str | Path | None = None,
    ) -> None:
        """Initialize the evaluation runner.

        Args:
            pipeline: RAGPipeline instance
            evaluator: LLMEvaluator instance (created if not provided)
            dataset_path: Path to evaluation dataset JSON (uses config default if not provided)
        """
        self.pipeline = pipeline
        self.evaluator = evaluator or LLMEvaluator()

        # Get dataset path from config if not provided
        if dataset_path is None:
            try:
                config = _get_config()
                dataset_path = config.dataset.path
            except Exception:
                pass

        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.dataset: list[dict[str, Any]] = []

        # Load runner config
        self._parallel, self._max_workers, self._show_progress, self._progress_interval = (
            _get_runner_config()
        )

    def _warmup_evaluator(self) -> None:
        """Pre-initialize lazy-loaded models for thread-safe parallel execution.

        This method triggers initialization of:
        - Embedding model (sentence-transformers)
        - Korean tokenizer (kiwipiepy)

        Call this before running parallel evaluation to avoid race conditions.
        """
        if not self.evaluator:
            return

        # Initialize embedding model
        if hasattr(self.evaluator, "embeddings"):
            try:
                _ = self.evaluator.embeddings
                logger.debug("Embedding model warmed up")
            except Exception as e:
                logger.warning(f"Failed to warm up embedding model: {e}")

        # Initialize Korean tokenizer by running a dummy tokenization
        try:
            from .evaluator import _get_korean_tokenizer

            _ = _get_korean_tokenizer()
            logger.debug("Korean tokenizer warmed up")
        except Exception as e:
            logger.warning(f"Failed to warm up Korean tokenizer: {e}")

    def _extract_tokens(
        self,
        response: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[int, int]:
        """Extract input and output token counts from response and metadata.

        Args:
            response: Pipeline response dictionary
            metadata: Response metadata dictionary

        Returns:
            Tuple of (input_tokens, output_tokens)
        """
        # Extract input tokens
        input_tokens = response.get("input_tokens")
        if input_tokens is None:
            input_tokens = metadata.get("prompt_tokens", metadata.get("input_tokens", 0))
        if input_tokens is None:
            input_tokens = 0

        # Extract output tokens
        output_tokens = response.get("output_tokens")
        if output_tokens is None:
            output_tokens = response.get("completion_tokens")

        if output_tokens is None:
            completion_tokens = metadata.get("completion_tokens")
            total_tokens = metadata.get("total_tokens", metadata.get("tokens_used"))
            if completion_tokens is not None:
                output_tokens = completion_tokens
            elif total_tokens is not None:
                safe_input = int(input_tokens) if isinstance(input_tokens, (int, float)) else 0
                safe_total = int(total_tokens) if isinstance(total_tokens, (int, float)) else 0
                output_tokens = max(0, safe_total - safe_input)
            else:
                output_tokens = 0

        if output_tokens is None:
            output_tokens = 0

        # Convert to int
        input_tokens_int = int(input_tokens) if isinstance(input_tokens, (int, float)) else 0
        output_tokens_int = int(output_tokens) if isinstance(output_tokens, (int, float)) else 0

        return input_tokens_int, output_tokens_int

    def load_dataset(
        self,
        path: str | Path | None = None,
        validate: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Load evaluation dataset from JSON file.

        Args:
            path: Path to dataset JSON (overrides init path)
            validate: Whether to validate dataset schema (uses config default if not provided)

        Returns:
            List of test cases

        Raises:
            FileNotFoundError: If dataset file does not exist
            ValueError: If validation is enabled and dataset has errors
        """
        dataset_path = Path(path) if path else self.dataset_path
        if not dataset_path or not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        # Get validation setting from config if not provided
        if validate is None:
            try:
                config = _get_config()
                validate = config.dataset.validate
            except Exception:
                validate = True

        with open(dataset_path, encoding="utf-8") as f:
            raw_data = json.load(f)

        if validate:
            validated_data, errors = validate_dataset(raw_data)
            if errors:
                error_summary = "; ".join(errors[:5])  # Show first 5 errors
                if len(errors) > 5:
                    error_summary += f" (and {len(errors) - 5} more)"
                raise ValueError(f"Dataset validation failed: {error_summary}")
            self.dataset = validated_data
        else:
            self.dataset = raw_data

        logger.info(f"Loaded {len(self.dataset)} test cases from {dataset_path}")
        return self.dataset

    def run_single(
        self,
        test_case: dict[str, Any],
    ) -> EvaluationMetrics:
        """Run evaluation on a single test case.

        Args:
            test_case: Test case with question, expected_answer, category

        Returns:
            EvaluationMetrics for the test case
        """
        question = test_case.get("question", "")
        expected = test_case.get("expected_answer", "")
        category = test_case.get("category", "")
        expected_sources = test_case.get("expected_sources", [])

        # Generate answer using pipeline
        start_time = time.perf_counter()
        generated = ""
        context: list[Any] = []
        input_tokens_int = 0
        output_tokens_int = 0

        if self.pipeline:
            try:
                response = self.pipeline.query(question)
                if isinstance(response, dict):
                    generated = response.get("answer", "")
                    context = response.get("context", [])
                    raw_metadata = response.get("metadata")
                    metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
                    input_tokens_int, output_tokens_int = self._extract_tokens(response, metadata)
                else:
                    generated = response.answer
                    context = response.context
                    raw_metadata = getattr(response, "metadata", None)
                    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
                    input_tokens_int = int(metadata.get("prompt_tokens", 0))
                    output_tokens_int = int(metadata.get("completion_tokens", 0))
            except Exception as e:
                logger.warning(f"Pipeline query failed: {e}")
                generated = f"[Error: {e}]"
        else:
            # No pipeline - just evaluate without generation
            generated = expected

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Evaluate
        return self.evaluator.evaluate(
            question=question,
            expected_answer=expected,
            generated_answer=generated,
            category=category,
            context=context,
            expected_sources=expected_sources if expected_sources else None,
            latency_ms=latency_ms,
            input_tokens=input_tokens_int,
            output_tokens=output_tokens_int,
        )

    def run(
        self,
        dataset: list[dict[str, Any]] | None = None,
        limit: int | None = None,
        model_name: str = "unknown",
        parallel: bool | None = None,
        max_workers: int | None = None,
    ) -> EvaluationResult:
        """Run evaluation on the full dataset.

        Args:
            dataset: Optional dataset to use (overrides loaded dataset)
            limit: Optional limit on number of test cases
            model_name: Name of the model being evaluated
            parallel: Whether to run evaluation in parallel (uses config default if not provided)
            max_workers: Maximum number of parallel workers (uses config default if not provided)

        Returns:
            EvaluationResult with aggregated metrics
        """
        test_cases = dataset or self.dataset
        if not test_cases:
            raise ValueError("No test cases loaded. Call load_dataset() first.")

        if limit:
            test_cases = test_cases[:limit]

        # Use config defaults if not provided
        use_parallel = parallel if parallel is not None else self._parallel
        workers = max_workers if max_workers is not None else self._max_workers

        if use_parallel:
            return self._run_parallel(test_cases, model_name, workers)
        return self._run_sequential(test_cases, model_name)

    def _run_sequential(
        self,
        test_cases: list[dict[str, Any]],
        model_name: str,
    ) -> EvaluationResult:
        """Run evaluation sequentially (original implementation).

        Args:
            test_cases: List of test cases to evaluate
            model_name: Name of the model being evaluated

        Returns:
            EvaluationResult with aggregated metrics
        """
        logger.info(f"Running sequential evaluation on {len(test_cases)} test cases")

        all_metrics: list[EvaluationMetrics] = []
        progress_interval = self._progress_interval

        for i, test_case in enumerate(test_cases):
            try:
                metrics = self.run_single(test_case)
                all_metrics.append(metrics)

                if self._show_progress and (i + 1) % progress_interval == 0:
                    logger.info(f"Evaluated {i + 1}/{len(test_cases)} cases")

            except Exception as e:
                logger.error(f"Evaluation failed for case {i}: {e}")
                continue

        return self._aggregate_results(all_metrics, len(test_cases), model_name)

    def _run_parallel(
        self,
        test_cases: list[dict[str, Any]],
        model_name: str,
        max_workers: int | None = None,
    ) -> EvaluationResult:
        """Run evaluation in parallel using ThreadPoolExecutor.

        Args:
            test_cases: List of test cases to evaluate
            model_name: Name of the model being evaluated
            max_workers: Maximum number of parallel workers

        Returns:
            EvaluationResult with aggregated metrics
        """
        workers = max_workers or self._max_workers or DEFAULT_MAX_WORKERS
        progress_interval = self._progress_interval
        show_progress = self._show_progress

        logger.info(
            f"Running parallel evaluation on {len(test_cases)} test cases "
            f"with {workers} workers"
        )

        # Pre-initialize models to avoid thread-safety issues
        # This ensures all lazy-loaded models are ready before parallel access
        self._warmup_evaluator()

        all_metrics: list[EvaluationMetrics] = []
        completed_count = 0
        progress_lock = Lock()

        def evaluate_with_progress(idx: int, test_case: dict[str, Any]) -> EvaluationMetrics | None:
            """Evaluate a single test case and report progress."""
            nonlocal completed_count
            try:
                metrics = self.run_single(test_case)
                with progress_lock:
                    completed_count += 1
                    if show_progress and completed_count % progress_interval == 0:
                        logger.info(f"Evaluated {completed_count}/{len(test_cases)} cases")
                return metrics
            except Exception as e:
                logger.error(f"Evaluation failed for case {idx}: {e}")
                with progress_lock:
                    completed_count += 1
                return None

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(evaluate_with_progress, i, tc): i
                for i, tc in enumerate(test_cases)
            }

            # Collect results as they complete
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_metrics.append(result)

        logger.info(f"Parallel evaluation complete: {len(all_metrics)}/{len(test_cases)} succeeded")

        return self._aggregate_results(all_metrics, len(test_cases), model_name)

    def _aggregate_results(
        self,
        all_metrics: list[EvaluationMetrics],
        total_cases: int,
        model_name: str,
    ) -> EvaluationResult:
        """Aggregate individual metrics into an EvaluationResult.

        Args:
            all_metrics: List of individual EvaluationMetrics
            total_cases: Total number of test cases attempted
            model_name: Name of the model being evaluated

        Returns:
            EvaluationResult with aggregated metrics
        """
        # Collect individual results
        similarities = []
        bleus = []
        faiths = []
        context_recalls = []
        context_precisions = []
        latencies = []
        verified_count = 0
        category_scores: dict[str, list[dict[str, float]]] = {}
        case_results = []

        # LLM-judge scores
        llm_judge_scores = []
        llm_correctness_scores = []
        llm_helpfulness_scores = []
        llm_faithfulness_scores = []
        llm_fluency_scores = []

        for metrics in all_metrics:
            similarities.append(metrics.answer_similarity)
            bleus.append(metrics.bleu_score)
            faiths.append(metrics.faithfulness)
            # Only include context metrics if they were computed (non-zero)
            if metrics.context_recall > 0 or metrics.context_precision > 0:
                context_recalls.append(metrics.context_recall)
                context_precisions.append(metrics.context_precision)
            latencies.append(metrics.latency_ms)

            if metrics.verified:
                verified_count += 1

            # Collect LLM-judge scores if available
            if metrics.llm_judge_score > 0:
                llm_judge_scores.append(metrics.llm_judge_score)
                llm_correctness_scores.append(metrics.llm_correctness)
                llm_helpfulness_scores.append(metrics.llm_helpfulness)
                llm_faithfulness_scores.append(metrics.llm_faithfulness)
                llm_fluency_scores.append(metrics.llm_fluency)

            # Track by category
            cat = metrics.category or "unknown"
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(
                {
                    "similarity": metrics.answer_similarity,
                    "bleu": metrics.bleu_score,
                    "faithfulness": metrics.faithfulness,
                    "llm_judge_score": metrics.llm_judge_score,
                }
            )

            case_results.append(metrics.to_dict())

        # Aggregate category scores
        category_aggregated = {}
        for cat, scores in category_scores.items():
            category_aggregated[cat] = {
                "count": len(scores),
                "mean_similarity": _safe_mean([s["similarity"] for s in scores]),
                "mean_bleu": _safe_mean([s["bleu"] for s in scores]),
                "mean_faithfulness": _safe_mean([s["faithfulness"] for s in scores]),
                "mean_llm_judge": _safe_mean([s["llm_judge_score"] for s in scores]),
            }

        # Get LLM judge model name from evaluator if available
        llm_judge_model = ""
        if self.evaluator and hasattr(self.evaluator, "llm_judge") and self.evaluator.llm_judge:
            llm_judge_model = getattr(self.evaluator.llm_judge.client, "model_name", "")

        # Build result
        n = len(similarities)
        result = EvaluationResult(
            model_name=model_name,
            total_cases=total_cases,
            evaluated_cases=n,
            mean_similarity=_safe_mean(similarities),
            mean_bleu=_safe_mean(bleus),
            mean_faithfulness=_safe_mean(faiths),
            mean_context_recall=_safe_mean(context_recalls),
            mean_context_precision=_safe_mean(context_precisions),
            mean_latency_ms=_safe_mean(latencies),
            std_similarity=_safe_std(similarities),
            std_bleu=_safe_std(bleus),
            std_latency_ms=_safe_std(latencies),
            verified_count=verified_count,
            verification_rate=verified_count / n if n > 0 else 0.0,
            # LLM-judge aggregated metrics (1-5 scale)
            mean_llm_judge_score=_safe_mean(llm_judge_scores),
            mean_llm_correctness=_safe_mean(llm_correctness_scores),
            mean_llm_helpfulness=_safe_mean(llm_helpfulness_scores),
            mean_llm_faithfulness=_safe_mean(llm_faithfulness_scores),
            mean_llm_fluency=_safe_mean(llm_fluency_scores),
            llm_judge_model=llm_judge_model,
            # Normalized LLM-judge metrics (0-1 scale)
            mean_llm_judge_normalized=_normalize_score(_safe_mean(llm_judge_scores)),
            mean_llm_correctness_normalized=_normalize_score(_safe_mean(llm_correctness_scores)),
            mean_llm_helpfulness_normalized=_normalize_score(_safe_mean(llm_helpfulness_scores)),
            mean_llm_faithfulness_normalized=_normalize_score(_safe_mean(llm_faithfulness_scores)),
            mean_llm_fluency_normalized=_normalize_score(_safe_mean(llm_fluency_scores)),
            category_scores=category_aggregated,
            case_results=case_results,
        )

        logger.info(
            f"Evaluation complete: similarity={result.mean_similarity:.3f}, "
            f"bleu={result.mean_bleu:.3f}, latency={result.mean_latency_ms:.1f}ms"
        )
        if llm_judge_scores:
            logger.info(
                f"LLM-Judge: score={result.mean_llm_judge_score:.2f}/5, "
                f"correctness={result.mean_llm_correctness:.2f}, "
                f"helpfulness={result.mean_llm_helpfulness:.2f}"
            )

        return result

    def save_results(
        self,
        result: EvaluationResult,
        output_path: str | Path | None = None,
        filename: str | None = None,
    ) -> Path:
        """Save evaluation results to JSON file.

        Args:
            result: EvaluationResult to save
            output_path: Output file path (uses config default directory if not provided)
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to saved file
        """
        if output_path is None:
            # Get output directory from config
            try:
                config = _get_config()
                output_dir = Path(config.output.directory)
            except Exception:
                output_dir = Path("data/evaluation/results")

            # Generate filename if not provided
            if filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                model_slug = result.model_name.replace("/", "_").replace(" ", "_")
                filename = f"eval_{model_slug}_{timestamp}.json"

            output_path = output_dir / filename

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"Saved results to {output_path}")
        return output_path
