from langgraph.graph import END, START, StateGraph

from state.graph_state import AgentState


def test_graph_state_preserves_resolved_paper_id_from_direct_analyzer():
    graph = StateGraph(AgentState)
    graph.add_node("direct", lambda _state: {"resolved_paper_id": "mmc"})
    graph.add_edge(START, "direct")
    graph.add_edge("direct", END)

    result = graph.compile().invoke({})

    assert result["resolved_paper_id"] == "mmc"
