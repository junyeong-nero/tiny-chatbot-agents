"""Answer Verifier for Hallucination detection.

This module provides verification of LLM-generated answers against
the source context to prevent hallucinations.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from .prompts import SYSTEM_PROMPTS, PROMPT_TEMPLATES

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result from answer verification."""

    verified: bool
    confidence: float
    issues: list[str]
    reasoning: str
    citations_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "confidence": self.confidence,
            "issues": self.issues,
            "reasoning": self.reasoning,
            "citations_valid": self.citations_valid,
        }


class AnswerVerifier:
    """Verifier for LLM-generated answers.

    Implements a 3-layer defense system:
    1. Citation check: Verifies all citations exist in context
    2. Rule-based check: Pattern matching for common hallucination signs
    3. LLM-based check: Uses LLM to verify faithfulness

    Attributes:
        llm_client: LLM client for verification
        confidence_threshold: Minimum confidence for verified answers
        require_citations: Whether citations are required
    """

    DEFAULT_CONFIDENCE_THRESHOLD = 0.7

    # Patterns that may indicate hallucination
    HALLUCINATION_PATTERNS = [
        r"일반적으로",
        r"보통",
        r"아마도",
        r"~할 수도 있습니다",
        r"제 생각에는",
        r"추측컨대",
    ]

    # Pattern for citation extraction
    CITATION_PATTERN = re.compile(r"\[참조:\s*([^\]]+)\]|\[출처:\s*([^\]]+)\]")

    def __init__(
        self,
        llm_client: Any = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        require_citations: bool = True,
        use_llm_verification: bool = True,
    ) -> None:
        """Initialize the Answer Verifier.

        Args:
            llm_client: LLM client for deep verification
            confidence_threshold: Minimum confidence score
            require_citations: Whether to require citations in answers
            use_llm_verification: Whether to use LLM for verification
        """
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold
        self.require_citations = require_citations
        self.use_llm_verification = use_llm_verification

        logger.info(
            f"Answer Verifier initialized. "
            f"Threshold: {confidence_threshold}, "
            f"Require citations: {require_citations}"
        )

    def verify(
        self,
        question: str,
        answer: str,
        context: list[dict[str, Any]],
    ) -> VerificationResult:
        """Verify if an answer is faithful to the context.

        Args:
            question: Original user question
            answer: Generated answer to verify
            context: Source context used for generation

        Returns:
            VerificationResult with verification status
        """
        issues = []
        
        # Layer 1: Citation check
        citations_valid = self._check_citations(answer, context)
        if self.require_citations and not citations_valid:
            issues.append("답변에 출처가 명시되지 않았거나 유효하지 않습니다")

        # Layer 2: Rule-based hallucination detection
        hallucination_signs = self._check_hallucination_patterns(answer)
        if hallucination_signs:
            issues.extend(hallucination_signs)

        # Layer 3: LLM-based verification (if available)
        llm_result = None
        if self.use_llm_verification and self.llm_client:
            llm_result = self._verify_with_llm(question, answer, context)
            if llm_result and not llm_result.get("verified", True):
                issues.extend(llm_result.get("issues", []))

        # Calculate overall confidence
        confidence = self._calculate_confidence(
            citations_valid=citations_valid,
            hallucination_signs=hallucination_signs,
            llm_result=llm_result,
        )

        verified = confidence >= self.confidence_threshold and len(issues) == 0

        reasoning = self._generate_reasoning(
            citations_valid, hallucination_signs, llm_result
        )

        return VerificationResult(
            verified=verified,
            confidence=confidence,
            issues=issues,
            reasoning=reasoning,
            citations_valid=citations_valid,
        )

    def _check_citations(
        self,
        answer: str,
        context: list[dict[str, Any]],
    ) -> bool:
        """Check if citations in answer are valid.

        Args:
            answer: Answer text with citations
            context: Source context

        Returns:
            True if all citations are valid
        """
        citations = self.CITATION_PATTERN.findall(answer)
        
        if not citations:
            return not self.require_citations

        # Flatten citations (pattern has 2 groups)
        cited_refs = [c[0] or c[1] for c in citations]

        # Get section titles from context
        context_sections = set()
        for c in context:
            title = c.get("section_title", "")
            if title:
                context_sections.add(title)
                # Also add "제N조" pattern
                match = re.search(r"제\s*\d+\s*조", title)
                if match:
                    context_sections.add(match.group())

        # Check if cited sections exist in context
        for ref in cited_refs:
            ref_clean = ref.strip()
            if not any(ref_clean in s or s in ref_clean for s in context_sections):
                # Check for "제N조" pattern
                match = re.search(r"제\s*\d+\s*조", ref_clean)
                if match and match.group() not in str(context_sections):
                    logger.warning(f"Citation not found in context: {ref}")
                    return False

        return True

    def _check_hallucination_patterns(self, answer: str) -> list[str]:
        """Check for patterns that may indicate hallucination.

        Args:
            answer: Answer text to check

        Returns:
            List of detected hallucination signs
        """
        signs = []
        
        for pattern in self.HALLUCINATION_PATTERNS:
            if re.search(pattern, answer):
                signs.append(f"불확실한 표현 감지: '{pattern}'")

        # Check for overly long answers without citations
        if len(answer) > 500 and not self.CITATION_PATTERN.search(answer):
            signs.append("긴 답변에 출처가 없습니다")

        return signs

    def _verify_with_llm(
        self,
        question: str,
        answer: str,
        context: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Use LLM to verify answer faithfulness.

        Args:
            question: Original question
            answer: Answer to verify
            context: Source context

        Returns:
            Verification result dict or None if failed
        """
        if not self.llm_client:
            return None

        # Format context
        context_str = "\n\n".join(
            f"[{c.get('section_title', '')}]\n{c.get('section_content', '')}"
            for c in context
        )

        # Build prompt
        prompt = PROMPT_TEMPLATES["verification"].format(
            context=context_str,
            query=question,
            answer=answer,
        )

        try:
            response = self.llm_client.generate(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS["verifier"]},
                    {"role": "user", "content": prompt},
                ]
            )

            # Handle both dict-like and object responses
            if hasattr(response, "content"):
                content = response.content
            else:
                content = response.get("content", "")

            # Parse JSON response
            result = json.loads(content)
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM verification response: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM verification failed: {e}")
            return None

    def _calculate_confidence(
        self,
        citations_valid: bool,
        hallucination_signs: list[str],
        llm_result: dict[str, Any] | None,
    ) -> float:
        """Calculate overall confidence score.

        Args:
            citations_valid: Whether citations are valid
            hallucination_signs: List of detected issues
            llm_result: LLM verification result

        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 1.0

        # Reduce confidence for citation issues
        if not citations_valid:
            confidence -= 0.3

        # Reduce confidence for each hallucination sign
        confidence -= len(hallucination_signs) * 0.15

        # Factor in LLM result
        if llm_result:
            llm_confidence = llm_result.get("confidence", 0.5)
            # Weight LLM confidence at 50%
            confidence = (confidence + llm_confidence) / 2

        return max(0.0, min(1.0, confidence))

    def _generate_reasoning(
        self,
        citations_valid: bool,
        hallucination_signs: list[str],
        llm_result: dict[str, Any] | None,
    ) -> str:
        """Generate human-readable reasoning for verification result.

        Args:
            citations_valid: Whether citations are valid
            hallucination_signs: List of detected issues
            llm_result: LLM verification result

        Returns:
            Reasoning text
        """
        parts = []

        if citations_valid:
            parts.append("✓ 출처 인용이 유효합니다")
        else:
            parts.append("✗ 출처 인용에 문제가 있습니다")

        if not hallucination_signs:
            parts.append("✓ 불확실한 표현이 감지되지 않았습니다")
        else:
            parts.append(f"✗ {len(hallucination_signs)}개의 주의 표현이 감지되었습니다")

        if llm_result:
            if llm_result.get("verified"):
                parts.append("✓ LLM 검증 통과")
            else:
                parts.append(f"✗ LLM 검증 실패: {llm_result.get('reasoning', '')}")

        return " | ".join(parts)

    def quick_verify(self, answer: str, context: list[dict[str, Any]]) -> bool:
        """Quick verification without LLM (rule-based only).

        Args:
            answer: Answer to verify
            context: Source context

        Returns:
            True if answer passes quick verification
        """
        citations_valid = self._check_citations(answer, context)
        hallucination_signs = self._check_hallucination_patterns(answer)

        return citations_valid and len(hallucination_signs) == 0
