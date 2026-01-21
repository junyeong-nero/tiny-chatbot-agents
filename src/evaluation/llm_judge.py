"""LLM-as-a-Judge evaluator for RAG pipeline answers.

This module implements LLM-based evaluation of generated answers
against golden (reference) answers using various quality criteria.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .frontier_client import FrontierClient
from .judge_prompts import (
    COMPREHENSIVE_JUDGE_PROMPT,
    CRITERIA_PROMPTS,
    DEFAULT_CRITERIA,
    JUDGE_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class CriteriaScore:
    """Score for a single evaluation criterion."""

    criterion: str
    score: float  # 1-5 scale
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion,
            "score": self.score,
            "reasoning": self.reasoning,
        }

    @classmethod
    def empty(cls, criterion: str) -> "CriteriaScore":
        """Create an empty/default score."""
        return cls(criterion=criterion, score=0.0, reasoning="")


@dataclass
class JudgeResult:
    """Complete judgment result for a single QA pair."""

    question: str
    golden_answer: str
    generated_answer: str
    criteria_scores: dict[str, CriteriaScore] = field(default_factory=dict)
    overall_score: float = 0.0
    summary: str = ""
    raw_response: str = ""
    judge_model: str = ""

    @property
    def normalized_score(self) -> float:
        """Return overall score normalized to 0-1 range."""
        if self.overall_score <= 0:
            return 0.0
        return (self.overall_score - 1) / 4  # Convert 1-5 to 0-1

    def get_criterion_score(self, criterion: str) -> float:
        """Get score for a specific criterion."""
        if criterion in self.criteria_scores:
            return self.criteria_scores[criterion].score
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "golden_answer": self.golden_answer,
            "generated_answer": self.generated_answer,
            "criteria_scores": {k: v.to_dict() for k, v in self.criteria_scores.items()},
            "overall_score": self.overall_score,
            "normalized_score": self.normalized_score,
            "summary": self.summary,
            "judge_model": self.judge_model,
        }


class LLMJudge:
    """LLM-as-a-Judge evaluator for comparing generated answers to golden answers.

    Uses a frontier model (Claude, GPT, Gemini) to evaluate generated answers
    on multiple criteria: correctness, helpfulness, faithfulness, and fluency.
    """

    def __init__(
        self,
        frontier_client: FrontierClient,
        criteria: list[str] | None = None,
        use_comprehensive: bool = True,
    ) -> None:
        """Initialize LLM Judge.

        Args:
            frontier_client: Client for frontier model API
            criteria: List of criteria to evaluate (default: all)
            use_comprehensive: Use single comprehensive prompt (more efficient)
        """
        self.client = frontier_client
        self.criteria = criteria or DEFAULT_CRITERIA
        self.use_comprehensive = use_comprehensive

    def judge(
        self,
        question: str,
        golden_answer: str,
        generated_answer: str,
        context: list[dict[str, Any]] | None = None,
    ) -> JudgeResult:
        """Judge a single QA pair.

        Args:
            question: Original question
            golden_answer: Reference/expected answer
            generated_answer: Model-generated answer to evaluate
            context: Optional retrieval context (for additional info)

        Returns:
            JudgeResult with scores and reasoning
        """
        if self.use_comprehensive:
            return self._judge_comprehensive(question, golden_answer, generated_answer)
        return self._judge_per_criterion(question, golden_answer, generated_answer)

    def _judge_comprehensive(
        self,
        question: str,
        golden_answer: str,
        generated_answer: str,
    ) -> JudgeResult:
        """Judge using comprehensive single-prompt approach."""
        prompt = COMPREHENSIVE_JUDGE_PROMPT.format(
            question=question,
            golden_answer=golden_answer,
            generated_answer=generated_answer,
        )

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.client.generate(messages)
            return self._parse_comprehensive_response(
                response, question, golden_answer, generated_answer
            )
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            return self._create_error_result(
                question, golden_answer, generated_answer, str(e)
            )

    def _judge_per_criterion(
        self,
        question: str,
        golden_answer: str,
        generated_answer: str,
    ) -> JudgeResult:
        """Judge each criterion separately (more tokens, potentially more accurate)."""
        criteria_scores = {}

        for criterion in self.criteria:
            if criterion not in CRITERIA_PROMPTS:
                logger.warning(f"Unknown criterion: {criterion}")
                continue

            prompt = CRITERIA_PROMPTS[criterion].format(
                question=question,
                golden_answer=golden_answer,
                generated_answer=generated_answer,
            )

            messages = [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            try:
                response = self.client.generate(messages)
                score_data = self._parse_single_criterion(response)
                criteria_scores[criterion] = CriteriaScore(
                    criterion=criterion,
                    score=score_data.get("score", 3.0),
                    reasoning=score_data.get("reasoning", ""),
                )
            except Exception as e:
                logger.warning(f"Failed to evaluate {criterion}: {e}")
                criteria_scores[criterion] = CriteriaScore(
                    criterion=criterion,
                    score=3.0,
                    reasoning=f"Error: {e}",
                )

        # Calculate overall score as average
        if criteria_scores:
            overall = sum(cs.score for cs in criteria_scores.values()) / len(criteria_scores)
        else:
            overall = 0.0

        return JudgeResult(
            question=question,
            golden_answer=golden_answer,
            generated_answer=generated_answer,
            criteria_scores=criteria_scores,
            overall_score=overall,
            judge_model=self.client.model_name,
        )

    def _parse_comprehensive_response(
        self,
        response: str,
        question: str,
        golden_answer: str,
        generated_answer: str,
    ) -> JudgeResult:
        """Parse comprehensive judge response JSON."""
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            criteria_scores = {}
            for criterion in self.criteria:
                if criterion in data and isinstance(data[criterion], dict):
                    criteria_scores[criterion] = CriteriaScore(
                        criterion=criterion,
                        score=float(data[criterion].get("score", 3)),
                        reasoning=data[criterion].get("reasoning", ""),
                    )

            # Calculate overall if not provided
            if "overall_score" in data:
                overall = float(data["overall_score"])
            elif criteria_scores:
                overall = sum(cs.score for cs in criteria_scores.values()) / len(criteria_scores)
            else:
                overall = 3.0

            return JudgeResult(
                question=question,
                golden_answer=golden_answer,
                generated_answer=generated_answer,
                criteria_scores=criteria_scores,
                overall_score=overall,
                summary=data.get("summary", ""),
                raw_response=response,
                judge_model=self.client.model_name,
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse judge response: {e}")
            logger.debug(f"Raw response: {response}")
            return self._create_error_result(
                question, golden_answer, generated_answer, f"Parse error: {e}"
            )

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text, handling markdown code blocks."""
        text = text.strip()

        # Try to find JSON in code block
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        # Try to find JSON object directly
        if "{" in text:
            start = text.find("{")
            # Find matching closing brace
            depth = 0
            for i, char in enumerate(text[start:], start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]

        # Return as-is
        return text

    def _parse_single_criterion(self, response: str) -> dict[str, Any]:
        """Parse single criterion response."""
        try:
            json_str = self._extract_json(response)
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"score": 3, "reasoning": "Parse error"}

    def _create_error_result(
        self,
        question: str,
        golden_answer: str,
        generated_answer: str,
        error_msg: str,
    ) -> JudgeResult:
        """Create a result for error cases."""
        return JudgeResult(
            question=question,
            golden_answer=golden_answer,
            generated_answer=generated_answer,
            criteria_scores={
                criterion: CriteriaScore(
                    criterion=criterion,
                    score=0.0,
                    reasoning=error_msg,
                )
                for criterion in self.criteria
            },
            overall_score=0.0,
            summary=error_msg,
            judge_model=self.client.model_name,
        )

    def judge_batch(
        self,
        items: list[dict[str, str]],
        show_progress: bool = True,
    ) -> list[JudgeResult]:
        """Judge multiple QA pairs.

        Args:
            items: List of dicts with 'question', 'golden_answer', 'generated_answer'
            show_progress: Whether to log progress

        Returns:
            List of JudgeResult
        """
        results = []
        total = len(items)

        for i, item in enumerate(items):
            result = self.judge(
                question=item["question"],
                golden_answer=item["golden_answer"],
                generated_answer=item["generated_answer"],
                context=item.get("context"),
            )
            results.append(result)

            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"Judged {i + 1}/{total} items")

        return results


def create_llm_judge(
    provider: str = "openai",
    model: str | None = None,
    criteria: list[str] | None = None,
    use_comprehensive: bool = True,
) -> LLMJudge:
    """Factory function to create an LLM judge.

    Args:
        provider: Provider name ('openai', 'anthropic', 'google')
        model: Optional model override
        criteria: Criteria to evaluate
        use_comprehensive: Use single comprehensive prompt

    Returns:
        Configured LLMJudge
    """
    from .frontier_client import create_frontier_client

    client = create_frontier_client(
        provider=provider,
        model=model,
        temperature=0.0,  # Use deterministic output for judging
    )

    return LLMJudge(
        frontier_client=client,
        criteria=criteria,
        use_comprehensive=use_comprehensive,
    )
