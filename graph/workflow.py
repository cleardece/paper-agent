"""
LangGraph Workflow - Supervisor多Agent协作状态图

流程：
START → Supervisor → Fetcher → END
        Supervisor → Retriever → Analyzer → Critic → Presenter → Reflector → END
        Supervisor → DirectAnalyzer → Presenter → END
"""

from langgraph.graph import StateGraph, START, END
from state.graph_state import AgentState
from agents.supervisor import SupervisorAgent
from agents.fetcher import FetcherAgent
from agents.retriever import RetrieverAgent
from agents.analyzer import AnalyzerAgent
from agents.critic import CriticAgent
from agents.presenter import PresenterAgent
from agents.reflector import ReflectorAgent
from agents.direct_analyzer import DirectAnalyzerAgent


# 路由函数

def supervisor_route(state: AgentState) -> str:
    next_agent = state.get("next_agent", "END")
    return next_agent


def fetcher_route(state: AgentState) -> str:
    if state.get("answer"):
        return "presenter"
    return END


def critic_route(state: AgentState) -> str:
    # 如果有错误，强制终止循环
    if state.get("error"):
        return "presenter"
    return state.get("next_agent", "END")


def build_workflow(
    supervisor: SupervisorAgent,
    fetcher: FetcherAgent,
    retriever: RetrieverAgent,
    analyzer: AnalyzerAgent,
    critic: CriticAgent,
    presenter: PresenterAgent,
    reflector: ReflectorAgent = None,
    direct_analyzer: DirectAnalyzerAgent = None,
) -> StateGraph:
    graph = StateGraph(AgentState)

    # 节点
    graph.add_node("supervisor", supervisor.invoke)
    graph.add_node("fetcher", fetcher.invoke)
    graph.add_node("retriever", retriever.invoke)
    graph.add_node("analyzer", analyzer.invoke)
    graph.add_node("critic", critic.invoke)
    graph.add_node("presenter", presenter.invoke)

    # 如果有 reflector，添加节点
    if reflector:
        graph.add_node("reflector", reflector.invoke)

    # 如果有 direct_analyzer，添加节点
    if direct_analyzer:
        graph.add_node("direct", direct_analyzer.invoke)

    # 边
    graph.add_edge(START, "supervisor")

    # 构建 supervisor 路由映射
    supervisor_targets = {"fetcher": "fetcher", "retriever": "retriever", "END": "presenter"}
    if direct_analyzer:
        supervisor_targets["direct"] = "direct"

    graph.add_conditional_edges(
        "supervisor",
        supervisor_route,
        supervisor_targets,
    )

    graph.add_conditional_edges("fetcher", fetcher_route, {"presenter": "presenter", END: END})

    graph.add_edge("retriever", "analyzer")
    graph.add_edge("analyzer", "critic")

    graph.add_conditional_edges(
        "critic",
        critic_route,
        {"presenter": "presenter", "retriever": "retriever", "END": END},
    )

    # Direct → Presenter（直接路由到格式化输出）
    if direct_analyzer:
        graph.add_edge("direct", "presenter")

    # Presenter → Reflector → END（如果有 reflector）
    if reflector:
        graph.add_edge("presenter", "reflector")
        graph.add_edge("reflector", END)
    else:
        graph.add_edge("presenter", END)

    return graph.compile()
