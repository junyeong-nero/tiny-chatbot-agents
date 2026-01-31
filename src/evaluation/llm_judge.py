"""LLM-as-a-Judge evaluator for RAG pipeline answers.

This module implements LLM-based evaluation of generated answers
against golden (reference) answers using various quality criteria.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .frontier_client import FrontierClient
from .judge_prompts import (
    COMPREHENSIVE_JUDGE_PROMPT,
    CONTEXT_AWARE_CRITERIA,
    CONTEXT_AWARE_JUDGE_PROMPT,
    CRITERIA_PROMPTS,
    DEFAULT_CRITERIA,
    JUDGE_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# Retry configuration
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_DELAY = 1.0  # seconds
RETRY_BACKOFF_MULTIPLIER = 2.0


def _clamp_score(score: float, min_val: float = 1.0, max_val: float = 5.0) -> float:
    """Clamp score to valid range [min_val, max_val].

    Args:
        score: Score value to clamp
        min_val: Minimum allowed value (default: 1.0)
        max_val: Maximum allowed value (default: 5.0)

    Returns:
        Clamped score value
    """
    return max(min_val, min(max_val, score))


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
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
    ) -> None:
        """Initialize LLM Judge.

        Args:
            frontier_client: Client for frontier model API
            criteria: List of criteria to evaluate (default: all)
            use_comprehensive: Use single comprehensive prompt (more efficient)
            max_retries: Maximum number of retries on parse failure (default: 2)
            retry_delay: Initial delay between retries in seconds (default: 1.0)
        """
        self.client = frontier_client
        self.criteria = criteria or DEFAULT_CRITERIA
        self.use_comprehensive = use_comprehensive
        self.max_retries = max_retries
        self.retry_delay = retry_delay

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
            context: Optional retrieval context (for context-aware evaluation)

        Returns:
            JudgeResult with scores and reasoning
        """
        if self.use_comprehensive:
            return self._judge_comprehensive(question, golden_answer, generated_answer, context)
        return self._judge_per_criterion(question, golden_answer, generated_answer)

    def _judge_comprehensive(
        self,
        question: str,
        golden_answer: str,
        generated_answer: str,
        context: list[dict[str, Any]] | None = None,
    ) -> JudgeResult:
        """Judge using comprehensive single-prompt approach with retry logic."""
        if context:
            formatted_context = self._format_context_for_judge(context)
            prompt = CONTEXT_AWARE_JUDGE_PROMPT.format(
                question=question,
                golden_answer=golden_answer,
                retrieved_context=formatted_context,
                generated_answer=generated_answer,
            )
            effective_criteria = CONTEXT_AWARE_CRITERIA
        else:
            prompt = COMPREHENSIVE_JUDGE_PROMPT.format(
                question=question,
                golden_answer=golden_answer,
                generated_answer=generated_answer,
            )
            effective_criteria = self.criteria

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        last_error = None
        delay = self.retry_delay

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.generate(messages)
                result = self._parse_comprehensive_response(
                    response, question, golden_answer, generated_answer, effective_criteria
                )

                # Check if parsing produced valid scores (not all zeros from error)
                if result.overall_score > 0 or any(
                    cs.score > 0 for cs in result.criteria_scores.values()
                ):
                    return result

                # If we got a result but all scores are 0, it might be a parse issue
                # Try again if we have retries left
                if attempt < self.max_retries:
                    logger.warning(
                        f"Judge returned zero scores (attempt {attempt + 1}/{self.max_retries + 1}), "
                        "retrying..."
                    )
                    time.sleep(delay)
                    delay *= RETRY_BACKOFF_MULTIPLIER
                    continue

                return result

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(
                        f"Judge evaluation failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}, "
                        "retrying..."
                    )
                    time.sleep(delay)
                    delay *= RETRY_BACKOFF_MULTIPLIER
                else:
                    logger.error(f"Judge evaluation failed after {self.max_retries + 1} attempts: {e}")

        return self._create_error_result(
            question, golden_answer, generated_answer, str(last_error)
        )

    def _format_context_for_judge(self, context: list[dict[str, Any]]) -> str:
        """Format retrieved context for judge prompt."""
        if not context:
            return "(검색된 컨텍스트 없음)"

        formatted = []
        for i, c in enumerate(context, 1):
            title = c.get("section_title", c.get("title", f"문서 {i}"))
            content = c.get("section_content", c.get("content", c.get("answer", "")))
            source = c.get("source", c.get("doc_id", ""))

            header = f"[{title}]"
            if source:
                header += f" (출처: {source})"
            formatted.append(f"{header}\n{content}")

        return "\n\n".join(formatted)

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
        criteria: list[str] | None = None,
    ) -> JudgeResult:
        """Parse comprehensive judge response JSON with partial parsing support.

        If JSON parsing fails completely, attempts to extract scores using regex.
        If only some criteria are found, uses those and fills missing with neutral scores.
        """
        effective_criteria = criteria or self.criteria
        criteria_scores: dict[str, CriteriaScore] = {}
        overall = 0.0
        summary = ""
        parse_warnings: list[str] = []

        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            for criterion in effective_criteria:
                if criterion in data and isinstance(data[criterion], dict):
                    try:
                        score = float(data[criterion].get("score", 3))
                        # Clamp score to valid range
                        score = _clamp_score(score)
                        criteria_scores[criterion] = CriteriaScore(
                            criterion=criterion,
                            score=score,
                            reasoning=data[criterion].get("reasoning", ""),
                        )
                    except (ValueError, TypeError) as e:
                        parse_warnings.append(f"Invalid score for {criterion}: {e}")

            # Get overall score
            if "overall_score" in data:
                try:
                    overall = float(data["overall_score"])
                    overall = _clamp_score(overall)
                except (ValueError, TypeError):
                    pass

            summary = data.get("summary", "")

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed, attempting regex extraction: {e}")
            # Try regex-based score extraction as fallback
            criteria_scores, overall = self._extract_scores_regex(response, effective_criteria)
            if criteria_scores:
                parse_warnings.append("Scores extracted via regex fallback")

        except (KeyError, TypeError) as e:
            logger.warning(f"Failed to parse judge response structure: {e}")
            parse_warnings.append(str(e))

        # Fill missing criteria with neutral scores (3.0)
        for criterion in effective_criteria:
            if criterion not in criteria_scores:
                criteria_scores[criterion] = CriteriaScore(
                    criterion=criterion,
                    score=3.0,
                    reasoning="Score not parsed (using neutral default)",
                )
                parse_warnings.append(f"Missing {criterion}, using default 3.0")

        # Calculate overall if not found
        if overall == 0.0 and criteria_scores:
            overall = sum(cs.score for cs in criteria_scores.values()) / len(criteria_scores)

        # Add parse warnings to summary if any
        if parse_warnings:
            warning_text = f" [Parse warnings: {'; '.join(parse_warnings)}]"
            summary = (summary + warning_text) if summary else warning_text.strip()
            logger.debug(f"Parse warnings: {parse_warnings}")

        return JudgeResult(
            question=question,
            golden_answer=golden_answer,
            generated_answer=generated_answer,
            criteria_scores=criteria_scores,
            overall_score=overall,
            summary=summary,
            raw_response=response,
            judge_model=self.client.model_name,
        )

    def _extract_scores_regex(
        self,
        response: str,
        criteria: list[str],
    ) -> tuple[dict[str, CriteriaScore], float]:
        """Extract scores from response using regex patterns as fallback.

        Handles cases where LLM returns malformed JSON but scores are visible.
        """
        criteria_scores: dict[str, CriteriaScore] = {}
        overall = 0.0

        # Pattern variations LLMs might use
        # e.g., "correctness": {"score": 4, "reasoning": "..."}
        # e.g., "score": 4 or "score":4 or score: 4
        score_pattern = re.compile(
            r'"?(\w+)"?\s*:\s*\{[^}]*"?score"?\s*:\s*(\d+(?:\.\d+)?)',
            re.IGNORECASE | re.DOTALL,
        )

        for match in score_pattern.finditer(response):
            criterion_name = match.group(1).lower()
            try:
                score = float(match.group(2))
                score = _clamp_score(score)

                # Map to expected criteria names
                for criterion in criteria:
                    if criterion.lower() in criterion_name or criterion_name in criterion.lower():
                        criteria_scores[criterion] = CriteriaScore(
                            criterion=criterion,
                            score=score,
                            reasoning="(extracted via regex)",
                        )
                        break
            except ValueError:
                continue

        # Try to extract overall score
        overall_pattern = re.compile(
            r'"?overall_?score"?\s*:\s*(\d+(?:\.\d+)?)',
            re.IGNORECASE,
        )
        overall_match = overall_pattern.search(response)
        if overall_match:
            try:
                overall = float(overall_match.group(1))
                overall = _clamp_score(overall)
            except ValueError:
                pass

        return criteria_scores, overall

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
        use_neutral_scores: bool = True,
    ) -> JudgeResult:
        """Create a result for error cases.

        Args:
            question: Original question
            golden_answer: Reference answer
            generated_answer: Generated answer that was being evaluated
            error_msg: Error message describing what went wrong
            use_neutral_scores: If True, use 3.0 (neutral) instead of 0.0 for scores
                              This prevents error cases from unfairly penalizing results.

        Returns:
            JudgeResult with error information
        """
        # Use neutral score (3.0) to avoid unfairly penalizing when evaluation fails
        # 0.0 would be misleading as it suggests the answer was terrible
        error_score = 3.0 if use_neutral_scores else 0.0

        return JudgeResult(
            question=question,
            golden_answer=golden_answer,
            generated_answer=generated_answer,
            criteria_scores={
                criterion: CriteriaScore(
                    criterion=criterion,
                    score=error_score,
                    reasoning=f"[Error: {error_msg}]",
                )
                for criterion in self.criteria
            },
            overall_score=error_score,
            summary=f"Evaluation error: {error_msg}",
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
    generator_model: str | None = None,
    generator_provider: str | None = None,
    auto_diverse: bool = False,
    strict_diversity: bool = True,
) -> LLMJudge:
    """Factory function to create an LLM judge.

    Args:
        provider: Provider name ('openai', 'anthropic', 'google')
        model: Optional model override
        criteria: Criteria to evaluate
        use_comprehensive: Use single comprehensive prompt
        generator_model: Model used for golden answer generation (for diversity check)
        generator_provider: Provider of generator model (for diversity check)
        auto_diverse: Auto-select a different model from the generator
        strict_diversity: Raise error if generator and judge are the same model (default: True)

    Returns:
        Configured LLMJudge
    """
    from .frontier_client import JudgeModelSelector, create_frontier_client

    final_provider = provider
    final_model = model

    if auto_diverse and generator_model:
        diverse_provider, diverse_model = JudgeModelSelector.get_diverse_judge(
            generator_model=generator_model,
            generator_provider=generator_provider,
        )
        final_provider = diverse_provider
        final_model = diverse_model
        logger.info(
            f"Auto-selected diverse judge: {diverse_provider}/{diverse_model} "
            f"(generator was {generator_provider}/{generator_model})"
        )
    elif generator_model:
        resolved_model = model or _get_default_model(provider)
        JudgeModelSelector.validate_diversity(
            generator_model=generator_model,
            judge_model=resolved_model,
            generator_provider=generator_provider,
            judge_provider=provider,
            strict=strict_diversity,
        )

    client = create_frontier_client(
        provider=final_provider,
        model=final_model,
        temperature=0.0,
    )

    return LLMJudge(
        frontier_client=client,
        criteria=criteria,
        use_comprehensive=use_comprehensive,
    )


def _get_default_model(provider: str) -> str:
    """Get the default model for a provider."""
    defaults = {
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-20250514",
        "google": "gemini-1.5-pro",
    }
    return defaults.get(provider, "gpt-4o")
