"""LLM Evaluation Metrics.

This module provides metrics for evaluating LLM-generated answers
against expected answers in the RAG pipeline.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics of a single QA pair."""

    question: str
    expected_answer: str
    generated_answer: str
    category: str

    # Core metrics
    answer_similarity: float = 0.0
    bleu_score: float = 0.0
    faithfulness: float = 0.0

    # Performance metrics
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    # Verification results
    verified: bool = False
    verification_issues: list[str] = field(default_factory=list)

    # LLM-as-Judge metrics
    llm_judge_score: float = 0.0  # Overall 1-5 score
    llm_judge_normalized: float = 0.0  # 0-1 normalized
    llm_correctness: float = 0.0
    llm_helpfulness: float = 0.0
    llm_faithfulness: float = 0.0
    llm_fluency: float = 0.0
    llm_judge_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "expected_answer": self.expected_answer,
            "generated_answer": self.generated_answer,
            "category": self.category,
            "answer_similarity": self.answer_similarity,
            "bleu_score": self.bleu_score,
            "faithfulness": self.faithfulness,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "verified": self.verified,
            "verification_issues": self.verification_issues,
            # LLM-as-Judge metrics
            "llm_judge_score": self.llm_judge_score,
            "llm_judge_normalized": self.llm_judge_normalized,
            "llm_correctness": self.llm_correctness,
            "llm_helpfulness": self.llm_helpfulness,
            "llm_faithfulness": self.llm_faithfulness,
            "llm_fluency": self.llm_fluency,
            "llm_judge_summary": self.llm_judge_summary,
        }


class LLMEvaluator:
    """Evaluator for computing metrics on LLM-generated answers.

    Computes:
    - Answer similarity (embedding cosine similarity)
    - BLEU score (n-gram overlap)
    - Faithfulness (using AnswerVerifier)
    - Latency and token usage
    - LLM-as-Judge metrics (optional)
    """

    def __init__(
        self,
        embedding_model: Any = None,
        verifier: Any = None,
        llm_judge: Any = None,
        use_llm_judge: bool = False,
    ) -> None:
        """Initialize the evaluator.

        Args:
            embedding_model: Optional embedding model for similarity
            verifier: Optional AnswerVerifier for faithfulness check
            llm_judge: Optional LLMJudge for LLM-as-a-Judge evaluation
            use_llm_judge: Whether to use LLM-as-Judge evaluation
        """
        self.embedding_model = embedding_model
        self.verifier = verifier
        self.llm_judge = llm_judge
        self.use_llm_judge = use_llm_judge

        # Lazy load embeddings if not provided
        self._embeddings = None

    @property
    def embeddings(self):
        """Lazy load embeddings model."""
        if self._embeddings is None and self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embeddings = SentenceTransformer(
                    "intfloat/multilingual-e5-small"
                )
                logger.info("Loaded default embedding model for evaluation")
            except ImportError:
                logger.warning("sentence-transformers not available")
        return self._embeddings or self.embedding_model

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts using embeddings.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if self.embeddings is None:
            return 0.0

        try:
            emb1 = self.embeddings.encode(text1, normalize_embeddings=True)
            emb2 = self.embeddings.encode(text2, normalize_embeddings=True)
            similarity = float(np.dot(emb1, emb2))
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.warning(f"Similarity computation failed: {e}")
            return 0.0

    def compute_bleu(self, reference: str, candidate: str, n: int = 4) -> float:
        """Compute BLEU score (simplified implementation).

        Args:
            reference: Reference text
            candidate: Candidate text
            n: Maximum n-gram size

        Returns:
            BLEU score (0.0 to 1.0)
        """
        # Tokenize (simple split for Korean/mixed text)
        ref_tokens = self._tokenize(reference)
        cand_tokens = self._tokenize(candidate)

        if not cand_tokens or not ref_tokens:
            return 0.0

        # Compute n-gram precisions
        precisions = []
        for i in range(1, n + 1):
            ref_ngrams = self._get_ngrams(ref_tokens, i)
            cand_ngrams = self._get_ngrams(cand_tokens, i)

            if not cand_ngrams:
                precisions.append(0.0)
                continue

            matches = sum(1 for ng in cand_ngrams if ng in ref_ngrams)
            precision = matches / len(cand_ngrams)
            precisions.append(precision)

        # Geometric mean of precisions
        if not precisions or all(p == 0 for p in precisions):
            return 0.0

        # Add smoothing for zero precisions
        smoothed = [max(p, 1e-10) for p in precisions]
        log_sum = sum(np.log(p) for p in smoothed) / len(smoothed)
        bleu = np.exp(log_sum)

        # Brevity penalty
        ref_len = len(ref_tokens)
        cand_len = len(cand_tokens)
        if cand_len < ref_len:
            bp = np.exp(1 - ref_len / max(cand_len, 1))
        else:
            bp = 1.0

        return float(bleu * bp)

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for Korean/mixed text."""
        # Remove special characters but keep Korean and alphanumeric
        text = re.sub(r"[^\w\s가-힣]", " ", text)
        return text.split()

    def _get_ngrams(self, tokens: list[str], n: int) -> list[tuple]:
        """Get n-grams from token list."""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def compute_faithfulness(
        self,
        question: str,
        answer: str,
        context: list[dict[str, Any]] | None = None,
    ) -> tuple[float, bool, list[str]]:
        """Compute faithfulness using the verifier.

        Args:
            question: Original question
            answer: Generated answer
            context: Optional context used for generation

        Returns:
            Tuple of (faithfulness_score, verified, issues)
        """
        if self.verifier is None:
            return 1.0, True, []

        try:
            result = self.verifier.verify(
                question=question,
                answer=answer,
                context=context or [],
            )
            return result.confidence, result.verified, result.issues
        except Exception as e:
            logger.warning(f"Verification failed: {e}")
            return 0.5, False, [str(e)]

    def evaluate(
        self,
        question: str,
        expected_answer: str,
        generated_answer: str,
        category: str = "",
        context: list[dict[str, Any]] | None = None,
        latency_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> EvaluationMetrics:
        """Evaluate a single QA pair.

        Args:
            question: Original question
            expected_answer: Expected/reference answer
            generated_answer: LLM-generated answer
            category: Question category
            context: Optional retrieval context
            latency_ms: Response latency in milliseconds
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            EvaluationMetrics with all computed metrics
        """
        # Compute similarity
        similarity = self.compute_similarity(expected_answer, generated_answer)

        # Compute BLEU
        bleu = self.compute_bleu(expected_answer, generated_answer)

        # Compute faithfulness
        faithfulness, verified, issues = self.compute_faithfulness(
            question, generated_answer, context
        )

        # Compute LLM-as-Judge metrics if enabled
        llm_scores = self._compute_llm_judge_scores(
            question, expected_answer, generated_answer, context
        )

        return EvaluationMetrics(
            question=question,
            expected_answer=expected_answer,
            generated_answer=generated_answer,
            category=category,
            answer_similarity=similarity,
            bleu_score=bleu,
            faithfulness=faithfulness,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            verified=verified,
            verification_issues=issues,
            **llm_scores,
        )

    def _compute_llm_judge_scores(
        self,
        question: str,
        expected_answer: str,
        generated_answer: str,
        context: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """Compute LLM-as-Judge scores if judge is configured.

        Args:
            question: Original question
            expected_answer: Reference answer (golden)
            generated_answer: Generated answer to evaluate
            context: Optional retrieval context

        Returns:
            Dict with LLM judge score fields
        """
        empty_scores = {
            "llm_judge_score": 0.0,
            "llm_judge_normalized": 0.0,
            "llm_correctness": 0.0,
            "llm_helpfulness": 0.0,
            "llm_faithfulness": 0.0,
            "llm_fluency": 0.0,
            "llm_judge_summary": "",
        }

        if not self.use_llm_judge or self.llm_judge is None:
            return empty_scores

        try:
            result = self.llm_judge.judge(
                question=question,
                golden_answer=expected_answer,
                generated_answer=generated_answer,
                context=context,
            )

            return {
                "llm_judge_score": result.overall_score,
                "llm_judge_normalized": result.normalized_score,
                "llm_correctness": result.get_criterion_score("correctness"),
                "llm_helpfulness": result.get_criterion_score("helpfulness"),
                "llm_faithfulness": result.get_criterion_score("faithfulness"),
                "llm_fluency": result.get_criterion_score("fluency"),
                "llm_judge_summary": result.summary,
            }
        except Exception as e:
            logger.warning(f"LLM judge evaluation failed: {e}")
            return {
                **empty_scores,
                "llm_judge_summary": f"Error: {e}",
            }
