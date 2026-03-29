import logging
from typing import Any

from src.graph.state import GraphState
from src.pipeline.models import ResponseSource

logger = logging.getLogger(__name__)


def make_verify_answer(verifier: Any, enable_verification: bool):
    def verify_answer(state: GraphState) -> dict[str, Any]:
        if not enable_verification or verifier is None:
            return {}

        if state.source != ResponseSource.TOS or state.response_mode not in {
            "answer",
            "limited_answer",
        }:
            return {}

        verification_context = [
            {
                "section_title": item.get("section_title", ""),
                "section_content": item.get("section_content", ""),
            }
            for item in state.context
        ]

        try:
            result = verifier.verify(
                question=state.query,
                answer=state.answer,
                context=verification_context,
            )
        except Exception as exc:
            logger.error("Verification failed with error: %s", exc)
            return {}

        metadata = dict(state.metadata)
        metadata["verification_reasoning"] = result.reasoning

        if result.verified:
            logger.info("Answer verified. Score: %.2f", result.confidence)
        else:
            logger.warning(
                "Answer verification failed. Score: %.2f, Issues: %s",
                result.confidence,
                result.issues,
            )

        return {
            "verified": result.verified,
            "verification_score": result.confidence,
            "verification_issues": result.issues,
            "metadata": metadata,
        }

    return verify_answer
