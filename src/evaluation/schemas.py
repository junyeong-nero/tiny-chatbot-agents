"""Pydantic schemas for evaluation dataset validation.

This module provides data validation for evaluation datasets.
If pydantic is not installed, validation is skipped gracefully.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Try to import pydantic; if not available, provide fallback
try:
    from pydantic import BaseModel, Field, field_validator

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore[misc, assignment]

    def Field(*args, **kwargs):  # type: ignore[no-redef]
        return None

    def field_validator(*args, **kwargs):  # type: ignore[no-redef]
        def decorator(func):
            return func

        return decorator


if PYDANTIC_AVAILABLE:

    class EvaluationTestCase(BaseModel):
        """Schema for a single evaluation test case."""

        id: str | None = Field(default=None, description="Unique identifier for the test case")
        question: str = Field(..., min_length=1, description="The question to evaluate")
        expected_answer: str = Field(
            ..., min_length=1, description="The expected/golden answer"
        )
        category: str = Field(default="", description="Category of the question")
        expected_sources: list[str] = Field(
            default_factory=list,
            description="Expected source identifiers for context overlap evaluation",
        )

        @field_validator("question", "expected_answer")
        @classmethod
        def strip_whitespace(cls, v: str) -> str:
            """Strip leading/trailing whitespace."""
            return v.strip()

    class EvaluationDataset(BaseModel):
        """Schema for a complete evaluation dataset."""

        test_cases: list[EvaluationTestCase] = Field(
            ..., min_length=1, description="List of test cases"
        )

        @classmethod
        def from_list(cls, data: list[dict[str, Any]]) -> EvaluationDataset:
            """Create dataset from a list of dictionaries."""
            return cls(test_cases=[EvaluationTestCase(**item) for item in data])

        def to_list(self) -> list[dict[str, Any]]:
            """Convert back to list of dictionaries."""
            return [tc.model_dump() for tc in self.test_cases]

else:
    # Fallback classes when pydantic is not available
    class EvaluationTestCase:  # type: ignore[no-redef]
        """Fallback schema (no validation)."""

        pass

    class EvaluationDataset:  # type: ignore[no-redef]
        """Fallback schema (no validation)."""

        @classmethod
        def from_list(cls, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Pass through without validation."""
            return data  # type: ignore[return-value]

        def to_list(self) -> list[dict[str, Any]]:
            """Not applicable for fallback."""
            return []


def validate_dataset(data: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate evaluation dataset.

    Args:
        data: List of test case dictionaries

    Returns:
        Tuple of (validated_data, errors)
        - validated_data: List of validated test cases (original if pydantic unavailable)
        - errors: List of validation error messages (empty if valid or no validation)
    """
    if not PYDANTIC_AVAILABLE:
        logger.debug("Pydantic not available, skipping dataset validation")
        return data, []

    errors: list[str] = []
    validated: list[dict[str, Any]] = []

    for i, item in enumerate(data):
        try:
            tc = EvaluationTestCase(**item)
            validated.append(tc.model_dump())
        except Exception as e:
            errors.append(f"Test case {i}: {e}")

    if errors:
        logger.warning(f"Dataset validation found {len(errors)} error(s)")

    return validated if validated else data, errors
