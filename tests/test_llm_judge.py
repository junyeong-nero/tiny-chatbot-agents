"""Tests for LLM-as-a-Judge evaluation."""

import json
import pytest
from unittest.mock import Mock, patch

from src.evaluation.llm_judge import (
    CriteriaScore,
    JudgeResult,
    LLMJudge,
    create_llm_judge,
)
from src.evaluation.judge_prompts import DEFAULT_CRITERIA


class TestCriteriaScore:
    """Tests for CriteriaScore dataclass."""

    def test_create_criteria_score(self):
        """Test basic creation."""
        score = CriteriaScore(
            criterion="correctness",
            score=4.0,
            reasoning="Accurate information",
        )

        assert score.criterion == "correctness"
        assert score.score == 4.0
        assert score.reasoning == "Accurate information"

    def test_to_dict(self):
        """Test serialization."""
        score = CriteriaScore(
            criterion="helpfulness",
            score=5.0,
            reasoning="Very helpful response",
        )

        d = score.to_dict()
        assert d["criterion"] == "helpfulness"
        assert d["score"] == 5.0
        assert d["reasoning"] == "Very helpful response"

    def test_empty_score(self):
        """Test creating empty score."""
        score = CriteriaScore.empty("fluency")

        assert score.criterion == "fluency"
        assert score.score == 0.0
        assert score.reasoning == ""


class TestJudgeResult:
    """Tests for JudgeResult dataclass."""

    def test_create_judge_result(self):
        """Test basic creation."""
        result = JudgeResult(
            question="계좌 개설 방법은?",
            golden_answer="앱에서 개설 가능합니다",
            generated_answer="한국투자앱에서 계좌를 개설할 수 있습니다",
            overall_score=4.5,
        )

        assert result.question == "계좌 개설 방법은?"
        assert result.overall_score == 4.5

    def test_normalized_score(self):
        """Test score normalization to 0-1 range."""
        # Score 5 -> 1.0
        result = JudgeResult(
            question="q", golden_answer="a", generated_answer="b", overall_score=5.0
        )
        assert result.normalized_score == 1.0

        # Score 1 -> 0.0
        result.overall_score = 1.0
        assert result.normalized_score == 0.0

        # Score 3 -> 0.5
        result.overall_score = 3.0
        assert result.normalized_score == 0.5

        # Score 4.5 -> 0.875
        result.overall_score = 4.5
        assert result.normalized_score == pytest.approx(0.875, rel=0.01)

    def test_normalized_score_zero(self):
        """Test normalized score with zero overall score."""
        result = JudgeResult(
            question="q", golden_answer="a", generated_answer="b", overall_score=0.0
        )
        assert result.normalized_score == 0.0

    def test_get_criterion_score(self):
        """Test getting individual criterion score."""
        result = JudgeResult(
            question="q",
            golden_answer="a",
            generated_answer="b",
            criteria_scores={
                "correctness": CriteriaScore("correctness", 4.0, "test"),
                "helpfulness": CriteriaScore("helpfulness", 5.0, "test"),
            },
        )

        assert result.get_criterion_score("correctness") == 4.0
        assert result.get_criterion_score("helpfulness") == 5.0
        assert result.get_criterion_score("nonexistent") == 0.0

    def test_to_dict(self):
        """Test serialization."""
        result = JudgeResult(
            question="q",
            golden_answer="a",
            generated_answer="b",
            criteria_scores={
                "correctness": CriteriaScore("correctness", 4.0, "test"),
            },
            overall_score=4.0,
            summary="Good answer",
            judge_model="gpt-4o",
        )

        d = result.to_dict()
        assert d["question"] == "q"
        assert d["overall_score"] == 4.0
        assert d["normalized_score"] == 0.75
        assert d["judge_model"] == "gpt-4o"
        assert "correctness" in d["criteria_scores"]


class TestLLMJudge:
    """Tests for LLMJudge class."""

    @pytest.fixture
    def mock_client(self):
        """Create mock frontier client."""
        client = Mock()
        client.model_name = "gpt-4o"
        client.generate.return_value = json.dumps({
            "correctness": {"score": 4, "reasoning": "Accurate"},
            "helpfulness": {"score": 5, "reasoning": "Very helpful"},
            "faithfulness": {"score": 4, "reasoning": "Faithful"},
            "fluency": {"score": 5, "reasoning": "Natural"},
            "overall_score": 4.5,
            "summary": "Good answer",
        })
        return client

    def test_judge_returns_result(self, mock_client):
        """Test that judge returns JudgeResult."""
        judge = LLMJudge(frontier_client=mock_client)

        result = judge.judge(
            question="계좌 개설 방법은?",
            golden_answer="앱에서 개설 가능합니다",
            generated_answer="한국투자앱에서 계좌를 개설할 수 있습니다",
        )

        assert isinstance(result, JudgeResult)
        assert result.overall_score == 4.5
        assert "correctness" in result.criteria_scores

    def test_judge_with_markdown_code_block(self, mock_client):
        """Test JSON extraction from markdown code block."""
        mock_client.generate.return_value = """Here is my evaluation:
```json
{
    "correctness": {"score": 4, "reasoning": "test"},
    "helpfulness": {"score": 4, "reasoning": "test"},
    "faithfulness": {"score": 4, "reasoning": "test"},
    "fluency": {"score": 4, "reasoning": "test"},
    "overall_score": 4,
    "summary": "test"
}
```
"""
        judge = LLMJudge(frontier_client=mock_client)

        result = judge.judge("q", "a", "b")

        assert result.overall_score == 4

    def test_judge_calculates_average_if_no_overall(self, mock_client):
        """Test that overall score is calculated from criteria averages."""
        mock_client.generate.return_value = json.dumps({
            "correctness": {"score": 4, "reasoning": "test"},
            "helpfulness": {"score": 4, "reasoning": "test"},
            "faithfulness": {"score": 4, "reasoning": "test"},
            "fluency": {"score": 4, "reasoning": "test"},
            "summary": "test",
        })

        judge = LLMJudge(frontier_client=mock_client)
        result = judge.judge("q", "a", "b")

        assert result.overall_score == 4.0

    def test_judge_handles_parse_error(self, mock_client):
        """Test graceful handling of parse errors."""
        mock_client.generate.return_value = "invalid json response"

        judge = LLMJudge(frontier_client=mock_client)
        result = judge.judge("q", "a", "b")

        assert result.overall_score == 0.0
        assert "Parse error" in result.summary

    def test_judge_handles_api_error(self, mock_client):
        """Test graceful handling of API errors."""
        mock_client.generate.side_effect = Exception("API error")

        judge = LLMJudge(frontier_client=mock_client)
        result = judge.judge("q", "a", "b")

        assert result.overall_score == 0.0

    def test_judge_batch(self, mock_client):
        """Test batch judging."""
        judge = LLMJudge(frontier_client=mock_client)

        items = [
            {"question": "q1", "golden_answer": "a1", "generated_answer": "b1"},
            {"question": "q2", "golden_answer": "a2", "generated_answer": "b2"},
        ]

        results = judge.judge_batch(items, show_progress=False)

        assert len(results) == 2
        assert all(isinstance(r, JudgeResult) for r in results)

    def test_extract_json_direct(self, mock_client):
        """Test JSON extraction without code block."""
        judge = LLMJudge(frontier_client=mock_client)

        text = '{"score": 4, "reasoning": "test"}'
        extracted = judge._extract_json(text)

        assert '"score": 4' in extracted

    def test_extract_json_nested_braces(self, mock_client):
        """Test JSON extraction with nested braces."""
        judge = LLMJudge(frontier_client=mock_client)

        text = 'Some text {"outer": {"inner": 1}} more text'
        extracted = judge._extract_json(text)

        assert '{"outer": {"inner": 1}}' in extracted


class TestJudgePerCriterion:
    """Tests for per-criterion judging mode."""

    @pytest.fixture
    def mock_client(self):
        """Create mock client that returns single criterion scores."""
        client = Mock()
        client.model_name = "gpt-4o"
        client.generate.return_value = '{"score": 4, "reasoning": "test"}'
        return client

    def test_judge_per_criterion(self, mock_client):
        """Test per-criterion judging mode."""
        judge = LLMJudge(frontier_client=mock_client, use_comprehensive=False)

        result = judge.judge("q", "a", "b")

        # Should call generate for each criterion
        assert mock_client.generate.call_count == len(DEFAULT_CRITERIA)
        assert result.overall_score == 4.0


class TestCreateLLMJudge:
    """Tests for factory function."""

    @patch("src.evaluation.frontier_client.create_frontier_client")
    def test_create_llm_judge(self, mock_create_client):
        """Test factory function creates judge with correct config."""
        mock_client = Mock()
        mock_client.model_name = "gpt-4o"
        mock_create_client.return_value = mock_client

        judge = create_llm_judge(
            provider="openai",
            model="gpt-4o",
            criteria=["correctness", "helpfulness"],
        )

        assert isinstance(judge, LLMJudge)
        mock_create_client.assert_called_once_with(
            provider="openai",
            model="gpt-4o",
            temperature=0.0,
        )
