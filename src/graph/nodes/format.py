from src.graph.state import GraphState
from src.pipeline.models import PipelineResponse, ResponseSource


def format_response(state: GraphState) -> dict[str, PipelineResponse]:
    response = PipelineResponse(
        query=state.query,
        answer=state.answer,
        source=state.source or ResponseSource.NO_CONTEXT,
        confidence=state.confidence,
        response_mode=state.response_mode,
        context=state.context,
        citations=state.citations,
        metadata=state.metadata,
        verified=state.verified,
        verification_score=state.verification_score,
        verification_issues=state.verification_issues,
    )
    return {"response": response}
