from langgraph.graph import END, START, StateGraph

from src.graph.edges import route_qna, route_tos
from src.graph.nodes import (
    format_response,
    make_generate_clarification,
    make_generate_no_context,
    make_generate_qna_answer,
    make_generate_qna_limited,
    make_generate_tos_answer,
    make_generate_tos_limited,
    make_search_qna,
    make_search_tos,
    make_verify_answer,
)
from src.graph.state import GraphState


def build_graph(
    qna_store,
    tos_store,
    llm,
    verifier,
    *,
    qna_threshold: float,
    qna_mid_threshold: float,
    tos_threshold: float,
    tos_mid_threshold: float,
    tos_low_threshold: float,
    enable_verification: bool,
    enable_hybrid_tos_search: bool,
):
    graph = StateGraph(GraphState)

    graph.add_node("search_qna", make_search_qna(qna_store))
    graph.add_node(
        "search_tos",
        make_search_tos(tos_store, enable_hybrid_tos_search),
    )
    graph.add_node("generate_qna_answer", make_generate_qna_answer(llm, qna_threshold))
    graph.add_node(
        "generate_qna_limited",
        make_generate_qna_limited(llm, qna_threshold, qna_mid_threshold),
    )
    graph.add_node("generate_tos_answer", make_generate_tos_answer(llm))
    graph.add_node(
        "generate_tos_limited",
        make_generate_tos_limited(llm, tos_threshold, tos_mid_threshold),
    )
    graph.add_node(
        "generate_clarification",
        make_generate_clarification(llm, tos_threshold, tos_mid_threshold, tos_low_threshold),
    )
    graph.add_node("generate_no_context", make_generate_no_context(llm))
    graph.add_node(
        "verify_answer",
        make_verify_answer(verifier, enable_verification),
    )
    graph.add_node("format_response", format_response)

    graph.add_edge(START, "search_qna")
    graph.add_conditional_edges(
        "search_qna",
        lambda state: route_qna(state, qna_threshold, qna_mid_threshold),
        {
            "generate_qna_answer": "generate_qna_answer",
            "generate_qna_limited": "generate_qna_limited",
            "search_tos": "search_tos",
        },
    )
    graph.add_conditional_edges(
        "search_tos",
        lambda state: route_tos(
            state,
            tos_threshold,
            tos_mid_threshold,
            tos_low_threshold,
        ),
        {
            "generate_tos_answer": "generate_tos_answer",
            "generate_tos_limited": "generate_tos_limited",
            "generate_clarification": "generate_clarification",
            "generate_no_context": "generate_no_context",
        },
    )

    for node in [
        "generate_qna_answer",
        "generate_qna_limited",
        "generate_tos_answer",
        "generate_tos_limited",
        "generate_clarification",
        "generate_no_context",
    ]:
        graph.add_edge(node, "verify_answer")

    graph.add_edge("verify_answer", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()
