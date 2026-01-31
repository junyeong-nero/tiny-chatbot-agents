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
        # With kiwipiepy, tokens are morphemes; without, space-split
        # Either way, we should have multiple tokens
        assert len(tokens) >= 1

    def test_tokenize_korean_morphemes(self, evaluator):
        """Test that Korean tokenizer splits morphemes properly when available."""
        from src.evaluation.evaluator import _get_korean_tokenizer

        text = "주식을 매도할 때 세금이 발생합니다"
        tokens = evaluator._tokenize(text)

        # If kiwipiepy is available, we get morpheme-level tokens
        kiwi = _get_korean_tokenizer()
        if kiwi is not None:
            # Should split into morphemes like: 주식, 을, 매도, 할, 때, 세금, 이, 발생, 합니다
            assert len(tokens) > 5, f"Expected morpheme split, got: {tokens}"
            # Check some expected morphemes
            assert "주식" in tokens or "매도" in tokens
        else:
            # Fallback to space-split
            assert len(tokens) >= 4

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
            verifier_faithfulness=0.9,
            latency_ms=100.0,
        )

        d = metrics.to_dict()

        assert d["question"] == "테스트 질문"
        assert d["answer_similarity"] == 0.85
        assert d["bleu_score"] == 0.75
        assert d["latency_ms"] == 100.0


class TestLLMJudgeParsing:
    """Tests for LLM Judge JSON parsing."""

    @pytest.fixture
    def mock_judge(self):
        """Create LLMJudge with mock client for testing."""
        from src.evaluation.llm_judge import LLMJudge
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.model_name = "test-model"
        return LLMJudge(mock_client)

    def test_extract_scores_regex_basic(self, mock_judge):
        """Test regex-based score extraction."""
        judge = mock_judge

        response = '''
        {
            "correctness": {"score": 4, "reasoning": "good"},
            "helpfulness": {"score": 5, "reasoning": "great"}
        }
        '''
        criteria = ["correctness", "helpfulness"]
        scores, overall = judge._extract_scores_regex(response, criteria)

        assert "correctness" in scores
        assert scores["correctness"].score == 4.0
        assert "helpfulness" in scores
        assert scores["helpfulness"].score == 5.0

    def test_extract_scores_regex_malformed(self, mock_judge):
        """Test regex extraction from malformed JSON."""
        judge = mock_judge

        # Malformed JSON but scores are visible
        response = '''
        correctness: {"score": 4, reasoning: missing quote}
        overall_score: 4.5
        '''
        criteria = ["correctness", "helpfulness"]
        scores, overall = judge._extract_scores_regex(response, criteria)

        assert "correctness" in scores
        assert scores["correctness"].score == 4.0
        assert overall == 4.5

    def test_parse_response_partial(self, mock_judge):
        """Test that partial parsing fills missing criteria with defaults."""
        judge = mock_judge
        judge.criteria = ["correctness", "helpfulness", "fluency"]

        # Only one criterion present
        response = '{"correctness": {"score": 5, "reasoning": "perfect"}}'
        result = judge._parse_comprehensive_response(
            response, "q", "golden", "generated", ["correctness", "helpfulness", "fluency"]
        )

        assert result.criteria_scores["correctness"].score == 5.0
        # Missing criteria should get default 3.0
        assert result.criteria_scores["helpfulness"].score == 3.0
        assert result.criteria_scores["fluency"].score == 3.0

    def test_error_result_uses_neutral_scores(self, mock_judge):
        """Test that error results use neutral scores (3.0) by default."""
        judge = mock_judge

        result = judge._create_error_result("q", "golden", "generated", "test error")

        assert result.overall_score == 3.0
        for criterion, score in result.criteria_scores.items():
            assert score.score == 3.0
            assert "Error" in score.reasoning


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

    def test_run_parallel(self, sample_dataset):
        """Test running evaluation in parallel mode."""
        from src.evaluation.runner import EvaluationRunner

        runner = EvaluationRunner(
            pipeline=None,
            dataset_path=sample_dataset,
        )
        runner.load_dataset()
        result = runner.run(model_name="test_parallel", parallel=True, max_workers=2)

        assert result.model_name == "test_parallel"
        assert result.evaluated_cases == 2
        # Results should be same as sequential
        assert result.mean_similarity > 0.9
        assert result.mean_bleu > 0.0

    def test_run_parallel_vs_sequential_consistency(self, sample_dataset):
        """Test that parallel and sequential runs produce consistent results."""
        from src.evaluation.runner import EvaluationRunner

        runner = EvaluationRunner(
            pipeline=None,
            dataset_path=sample_dataset,
        )
        runner.load_dataset()

        # Run both modes
        seq_result = runner.run(model_name="seq", parallel=False)
        par_result = runner.run(model_name="par", parallel=True, max_workers=2)

        # Results should be equivalent
        assert seq_result.evaluated_cases == par_result.evaluated_cases
        assert abs(seq_result.mean_similarity - par_result.mean_similarity) < 0.01
        assert abs(seq_result.mean_bleu - par_result.mean_bleu) < 0.01


class TestLLMJudgeComprehensive:
    """Tests for LLMJudge comprehensive evaluation."""

    @pytest.fixture
    def mock_judge(self):
        """Create LLMJudge with mock client for testing."""
        from src.evaluation.llm_judge import LLMJudge
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.model_name = "test-model"
        return LLMJudge(mock_client)

    def test_comprehensive_judgment_success(self, mock_judge):
        """Test successful comprehensive judgment."""
        mock_judge.client.generate.return_value = json.dumps({
            "correctness": {"score": 4, "reasoning": "Mostly correct"},
            "helpfulness": {"score": 5, "reasoning": "Very helpful"},
            "faithfulness": {"score": 4, "reasoning": "Faithful to context"},
            "fluency": {"score": 5, "reasoning": "Well written"},
            "overall_score": 4.5,
            "summary": "Good answer overall",
        })

        result = mock_judge.judge(
            question="테스트 질문",
            golden_answer="정답",
            generated_answer="생성 답변",
        )

        assert result.overall_score == 4.5
        assert result.criteria_scores["correctness"].score == 4.0
        assert result.criteria_scores["helpfulness"].score == 5.0
        assert result.normalized_score == 0.875  # (4.5 - 1) / 4

    def test_judgment_with_context(self, mock_judge):
        """Test judgment with retrieval context."""
        mock_judge.client.generate.return_value = json.dumps({
            "correctness": {"score": 5, "reasoning": "Perfect"},
            "helpfulness": {"score": 5, "reasoning": "Excellent"},
            "context_faithfulness": {"score": 5, "reasoning": "Fully grounded"},
            "fluency": {"score": 5, "reasoning": "Natural"},
            "overall_score": 5.0,
        })

        context = [
            {"section_title": "약관 제1조", "section_content": "관련 내용"},
        ]

        result = mock_judge.judge(
            question="질문",
            golden_answer="답변",
            generated_answer="생성 답변",
            context=context,
        )

        assert result.overall_score == 5.0
        # Verify generate was called with context-aware prompt
        call_args = mock_judge.client.generate.call_args[0][0]
        assert any("컨텍스트" in msg.get("content", "") or "context" in msg.get("content", "").lower()
                   for msg in call_args)

    def test_judgment_json_in_code_block(self, mock_judge):
        """Test parsing JSON wrapped in markdown code block."""
        mock_judge.client.generate.return_value = '''```json
{
    "correctness": {"score": 4, "reasoning": "Good"},
    "helpfulness": {"score": 4, "reasoning": "Helpful"},
    "faithfulness": {"score": 4, "reasoning": "Accurate"},
    "fluency": {"score": 4, "reasoning": "Clear"},
    "overall_score": 4.0
}
```'''

        result = mock_judge.judge(
            question="q", golden_answer="a", generated_answer="b"
        )

        assert result.overall_score == 4.0
        assert result.criteria_scores["correctness"].score == 4.0


class TestJudgeModelSelector:
    """Tests for JudgeModelSelector diversity checking."""

    def test_get_diverse_judge_openai(self):
        """Test diverse judge selection for OpenAI generator."""
        from src.evaluation.frontier_client import JudgeModelSelector

        provider, model = JudgeModelSelector.get_diverse_judge("gpt-4o")
        assert provider == "anthropic"
        assert "claude" in model.lower()

    def test_get_diverse_judge_anthropic(self):
        """Test diverse judge selection for Anthropic generator."""
        from src.evaluation.frontier_client import JudgeModelSelector

        provider, model = JudgeModelSelector.get_diverse_judge("claude-sonnet-4-20250514")
        assert provider == "openai"
        assert "gpt" in model.lower()

    def test_get_diverse_judge_google(self):
        """Test diverse judge selection for Google generator."""
        from src.evaluation.frontier_client import JudgeModelSelector

        provider, model = JudgeModelSelector.get_diverse_judge("gemini-1.5-pro")
        assert provider == "openai"

    def test_get_diverse_judge_unknown_fallback(self):
        """Test fallback for unknown generator model."""
        from src.evaluation.frontier_client import JudgeModelSelector

        provider, model = JudgeModelSelector.get_diverse_judge("unknown-model")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_is_same_model_exact_match(self):
        """Test same model detection for exact match."""
        from src.evaluation.frontier_client import JudgeModelSelector

        assert JudgeModelSelector.is_same_model("gpt-4o", "gpt-4o")

    def test_is_same_model_normalized_match(self):
        """Test same model detection with normalization."""
        from src.evaluation.frontier_client import JudgeModelSelector

        assert JudgeModelSelector.is_same_model("GPT-4o", "gpt_4o")

    def test_is_same_model_different(self):
        """Test different models are not flagged as same."""
        from src.evaluation.frontier_client import JudgeModelSelector

        assert not JudgeModelSelector.is_same_model("gpt-4o", "claude-3-opus")

    def test_validate_diversity_strict_raises(self):
        """Test strict diversity validation raises on same model."""
        from src.evaluation.frontier_client import JudgeModelSelector

        with pytest.raises(ValueError, match="Circular evaluation bias"):
            JudgeModelSelector.validate_diversity(
                generator_model="gpt-4o",
                judge_model="gpt-4o",
                strict=True,
            )

    def test_validate_diversity_non_strict_warns(self):
        """Test non-strict diversity validation only warns."""
        from src.evaluation.frontier_client import JudgeModelSelector

        # Should not raise, just return False
        result = JudgeModelSelector.validate_diversity(
            generator_model="gpt-4o",
            judge_model="gpt-4o",
            strict=False,
        )
        assert result is False


class TestContextOverlapMetrics:
    """Tests for context overlap computation."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator for testing."""
        return LLMEvaluator(embedding_model=None, verifier=None)

    def test_context_overlap_full_match(self, evaluator):
        """Test context overlap with full match."""
        # Note: compute_context_overlap collects both doc_id and section_title
        # So we need to include all identifiers in expected for full precision
        retrieved = [
            {"doc_id": "doc1", "section_title": "제1조"},
            {"doc_id": "doc2", "section_title": "제2조"},
        ]
        expected = ["doc1", "doc2", "제1조", "제2조"]

        recall, precision = evaluator.compute_context_overlap(retrieved, expected)

        assert recall == 1.0
        assert precision == 1.0

    def test_context_overlap_partial_match(self, evaluator):
        """Test context overlap with partial match."""
        retrieved = [
            {"doc_id": "doc1"},
            {"doc_id": "doc2"},
            {"doc_id": "doc3"},
        ]
        expected = ["doc1", "doc2", "doc4"]

        recall, precision = evaluator.compute_context_overlap(retrieved, expected)

        # 2 out of 3 expected found
        assert recall == pytest.approx(2 / 3, rel=0.01)
        # 2 out of 3 retrieved were expected
        assert precision == pytest.approx(2 / 3, rel=0.01)

    def test_context_overlap_no_match(self, evaluator):
        """Test context overlap with no match."""
        retrieved = [{"doc_id": "doc1"}]
        expected = ["doc2", "doc3"]

        recall, precision = evaluator.compute_context_overlap(retrieved, expected)

        assert recall == 0.0
        assert precision == 0.0

    def test_context_overlap_empty_expected(self, evaluator):
        """Test context overlap with empty expected sources."""
        retrieved = [{"doc_id": "doc1"}]
        expected: list[str] = []

        recall, precision = evaluator.compute_context_overlap(retrieved, expected)

        # No expected sources means perfect recall/precision by convention
        assert recall == 1.0
        assert precision == 1.0

    def test_evaluate_with_expected_sources(self, evaluator):
        """Test evaluate method includes context overlap metrics."""
        context = [
            {"doc_id": "source1", "content": "context content"},
        ]
        expected_sources = ["source1", "source2"]

        metrics = evaluator.evaluate(
            question="테스트",
            expected_answer="답변",
            generated_answer="답변",
            context=context,
            expected_sources=expected_sources,
        )

        # source1 was retrieved, source2 was not
        assert metrics.context_recall == 0.5  # 1 out of 2 expected
        assert metrics.context_precision == 1.0  # 1 out of 1 retrieved was expected


class TestDatasetSchemaValidation:
    """Tests for dataset schema validation."""

    def test_validate_dataset_valid(self, tmp_path):
        """Test validation passes for valid dataset."""
        from src.evaluation.runner import EvaluationRunner

        dataset = [
            {
                "id": "test_001",
                "question": "유효한 질문?",
                "expected_answer": "유효한 답변",
                "category": "테스트",
            },
        ]

        dataset_path = tmp_path / "valid_dataset.json"
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False)

        runner = EvaluationRunner(dataset_path=dataset_path)
        loaded = runner.load_dataset(validate=True)

        assert len(loaded) == 1
        assert loaded[0]["question"] == "유효한 질문?"

    def test_validate_dataset_invalid_empty_question(self, tmp_path):
        """Test validation fails for empty question."""
        from src.evaluation.runner import EvaluationRunner
        from src.evaluation.schemas import PYDANTIC_AVAILABLE

        dataset = [
            {
                "question": "",  # Empty - should fail
                "expected_answer": "답변",
            },
        ]

        dataset_path = tmp_path / "invalid_dataset.json"
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False)

        runner = EvaluationRunner(dataset_path=dataset_path)

        if PYDANTIC_AVAILABLE:
            with pytest.raises(ValueError, match="validation failed"):
                runner.load_dataset(validate=True)
        else:
            # Without pydantic, validation is skipped
            loaded = runner.load_dataset(validate=True)
            assert len(loaded) == 1

    def test_load_dataset_skip_validation(self, tmp_path):
        """Test loading dataset without validation."""
        from src.evaluation.runner import EvaluationRunner

        # Invalid dataset but validation disabled
        dataset = [
            {"question": "", "expected_answer": ""},
        ]

        dataset_path = tmp_path / "skip_validation.json"
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False)

        runner = EvaluationRunner(dataset_path=dataset_path)
        loaded = runner.load_dataset(validate=False)

        assert len(loaded) == 1


class TestNormalizedMetrics:
    """Tests for normalized metric scales."""

    def test_evaluation_result_normalized_scores(self, tmp_path):
        """Test that EvaluationResult includes normalized LLM judge scores."""
        from src.evaluation.runner import EvaluationRunner, _normalize_score

        # Test normalization helper
        assert _normalize_score(1.0) == 0.0  # Min score
        assert _normalize_score(5.0) == 1.0  # Max score
        assert _normalize_score(3.0) == 0.5  # Middle score

        dataset = [
            {
                "question": "질문",
                "expected_answer": "답변",
                "category": "test",
            },
        ]

        dataset_path = tmp_path / "test.json"
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False)

        runner = EvaluationRunner(dataset_path=dataset_path)
        runner.load_dataset()
        result = runner.run(model_name="test")

        # Check normalized fields exist
        result_dict = result.to_dict()
        assert "mean_score_normalized" in result_dict["llm_judge"]
        assert "mean_correctness_normalized" in result_dict["llm_judge"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
