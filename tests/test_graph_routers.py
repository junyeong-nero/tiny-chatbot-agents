from src.graph.edges.routers import route_qna, route_tos
from src.graph.state import GraphState


def test_route_qna_high_confidence_goes_direct_to_qna_answer():
    state = GraphState(query="test", qna_score=0.91)

    assert route_qna(state, qna_threshold=0.8, qna_mid_threshold=0.7) == "generate_qna_answer"


def test_route_qna_mid_confidence_goes_to_qna_limited():
    state = GraphState(query="test", qna_score=0.78)

    assert (
        route_qna(
            state,
            qna_threshold=0.8,
            qna_mid_threshold=0.7,
        )
        == "generate_qna_limited"
    )


def test_route_qna_low_confidence_falls_back_to_tos():
    state = GraphState(query="test", qna_score=0.4)

    assert (
        route_qna(
            state,
            qna_threshold=0.8,
            qna_mid_threshold=0.7,
        )
        == "search_tos"
    )


def test_route_tos_low_confidence_requests_clarification():
    state = GraphState(query="test", tos_score=0.45)

    assert (
        route_tos(
            state,
            tos_threshold=0.65,
            tos_mid_threshold=0.55,
            tos_low_threshold=0.4,
        )
        == "generate_clarification"
    )


def test_route_tos_mid_confidence_returns_limited_answer():
    state = GraphState(query="test", tos_score=0.6)

    assert (
        route_tos(
            state,
            tos_threshold=0.65,
            tos_mid_threshold=0.55,
            tos_low_threshold=0.4,
        )
        == "generate_tos_limited"
    )


def test_route_tos_high_confidence_returns_answer():
    state = GraphState(query="test", tos_score=0.8)

    assert (
        route_tos(
            state,
            tos_threshold=0.65,
            tos_mid_threshold=0.55,
            tos_low_threshold=0.4,
        )
        == "generate_tos_answer"
    )


def test_route_tos_below_low_returns_no_context():
    state = GraphState(query="test", tos_score=0.2)

    assert (
        route_tos(
            state,
            tos_threshold=0.65,
            tos_mid_threshold=0.55,
            tos_low_threshold=0.4,
        )
        == "generate_no_context"
    )
