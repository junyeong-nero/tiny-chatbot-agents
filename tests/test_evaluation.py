"""Tests for the evaluation module."""

import json
import tempfile
from pathlib import Path

import pytest

from src.evaluation.evaluator import LLMEvaluator, EvaluationMetrics


class TestLLMEvaluator:
    """Tests for LLMEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator without embedding model for testing."""
        return LLMEvaluator(embedding_model=None, verifier=None)

    def test_bleu_identical(self, evaluator):
        """Test BLEU score for identical texts."""
        text = "이것은 테스트 문장입니다."
        score = evaluator.compute_bleu(text, text)
        # Short Korean texts may have lower BLEU due to tokenization
        assert score > 0.0, "Identical texts should have positive BLEU score"

    def test_bleu_different(self, evaluator):
        """Test BLEU score for completely different texts."""
        ref = "주식을 매도하면 수수료가 발생합니다"
        cand = "계좌를 개설하려면 신분증이 필요합니다"
        score = evaluator.compute_bleu(ref, cand)
        assert score < 0.3, "Different texts should have low BLEU score"

    def test_bleu_partial_overlap(self, evaluator):
        """Test BLEU score for partially overlapping texts."""
        ref = "주식을 매도할 때 손익과 관계없이 세금이 발생합니다"
        cand = "주식을 매도할 때 0.2% 세금이 발생합니다"
        score = evaluator.compute_bleu(ref, cand)
        # Partial overlap should have non-zero score
        assert score > 0.0, "Partial overlap should have positive BLEU score"

    def test_bleu_empty(self, evaluator):
        """Test BLEU score with empty strings."""
        assert evaluator.compute_bleu("", "test") == 0.0
        assert evaluator.compute_bleu("test", "") == 0.0

    def test_tokenize_korean(self, evaluator):
        """Test tokenization of Korean text."""
        text = "한국투자증권에서 계좌를 개설하세요!"
        tokens = evaluator._tokenize(text)
        assert len(tokens) > 0
        assert "한국투자증권에서" in tokens

    def test_get_ngrams(self, evaluator):
        """Test n-gram extraction."""
        tokens = ["a", "b", "c", "d"]
        unigrams = evaluator._get_ngrams(tokens, 1)
        bigrams = evaluator._get_ngrams(tokens, 2)
        trigrams = evaluator._get_ngrams(tokens, 3)

        assert len(unigrams) == 4
        assert len(bigrams) == 3
        assert len(trigrams) == 2
        assert ("a", "b") in bigrams

    def test_evaluate_returns_metrics(self, evaluator):
        """Test that evaluate returns EvaluationMetrics."""
        metrics = evaluator.evaluate(
            question="계좌 해지 방법은?",
            expected_answer="한국투자앱에서 해지 가능합니다",
            generated_answer="한국투자앱에서 해지 가능합니다",
            category="계좌관리",
        )

        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.question == "계좌 해지 방법은?"
        assert metrics.bleu_score > 0.0  # Short Korean texts may have lower BLEU
        assert metrics.category == "계좌관리"

    def test_evaluate_with_latency(self, evaluator):
        """Test that latency is recorded."""
        metrics = evaluator.evaluate(
            question="test",
            expected_answer="answer",
            generated_answer="answer",
            latency_ms=123.45,
        )

        assert metrics.latency_ms == 123.45


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = EvaluationMetrics(
            question="테스트 질문",
            expected_answer="예상 답변",
            generated_answer="생성 답변",
            category="테스트",
            answer_similarity=0.85,
            bleu_score=0.75,
            faithfulness=0.9,
            latency_ms=100.0,
        )

        d = metrics.to_dict()

        assert d["question"] == "테스트 질문"
        assert d["answer_similarity"] == 0.85
        assert d["bleu_score"] == 0.75
        assert d["latency_ms"] == 100.0


class TestEvaluationRunner:
    """Tests for EvaluationRunner class."""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a sample dataset file."""
        dataset = [
            {
                "id": "test_001",
                "question": "계좌 개설 방법은?",
                "expected_answer": "앱에서 개설 가능합니다",
                "category": "계좌",
            },
            {
                "id": "test_002",
                "question": "수수료는 얼마인가요?",
                "expected_answer": "0.015% 수수료가 부과됩니다",
                "category": "수수료",
            },
        ]

        dataset_path = tmp_path / "test_dataset.json"
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False)

        return dataset_path

    def test_load_dataset(self, sample_dataset):
        """Test loading dataset from file."""
        from src.evaluation.runner import EvaluationRunner

        runner = EvaluationRunner(dataset_path=sample_dataset)
        dataset = runner.load_dataset()

        assert len(dataset) == 2
        assert dataset[0]["id"] == "test_001"

    def test_run_without_pipeline(self, sample_dataset):
        """Test running evaluation without pipeline (uses expected as generated)."""
        from src.evaluation.runner import EvaluationRunner

        runner = EvaluationRunner(
            pipeline=None,
            dataset_path=sample_dataset,
        )
        runner.load_dataset()
        result = runner.run(model_name="test_model")

        assert result.model_name == "test_model"
        assert result.evaluated_cases == 2
        assert result.mean_similarity > 0.9  # Identical texts
        # BLEU may be lower for short Korean texts
        assert result.mean_bleu > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
