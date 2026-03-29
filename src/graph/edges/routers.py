from src.graph.state import GraphState


def route_qna(
    state: GraphState,
    qna_threshold: float = 0.80,
    qna_mid_threshold: float = 0.70,
) -> str:
    if state.qna_score >= qna_threshold:
        return "generate_qna_answer"
    if state.qna_score >= qna_mid_threshold:
        return "generate_qna_limited"
    return "search_tos"


def route_tos(
    state: GraphState,
    tos_threshold: float = 0.65,
    tos_mid_threshold: float = 0.55,
    tos_low_threshold: float = 0.40,
) -> str:
    if state.tos_score >= tos_threshold:
        return "generate_tos_answer"
    if state.tos_score >= tos_mid_threshold:
        return "generate_tos_limited"
    if state.tos_score >= tos_low_threshold:
        return "generate_clarification"
    return "generate_no_context"
