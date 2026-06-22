"""
LangGraph Workflow - Supervisor多Agent协作状态图

流程：
START → Supervisor → Translator → Fetcher → END
        Supervisor → Retriever → Analyzer → Critic → Presenter → END
"""

from langgraph.graph import StateGraph, START, END
from state.graph_state import AgentState
from agents.supervisor import SupervisorAgent
from agents.translator import TranslatorAgent
from agents.fetcher import FetcherAgent
from agents.retriever import RetrieverAgent
from agents.analyzer import AnalyzerAgent
from agents.critic import CriticAgent
from agents.presenter import PresenterAgent
from agents.reflector import ReflectorAgent


# 路由函数

def supervisor_route(state: AgentState) -> str:
    next_agent = state.get("next_agent", "END")
    # 如果是 fetcher，先经过翻译
    if next_agent == "fetcher":
        return "translator"
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


def reflector_route(state: AgentState) -> str:
    """Reflector 完成后进入 Presenter"""
    return "presenter"


def build_workflow(
    supervisor: SupervisorAgent,
    translator: TranslatorAgent,
    fetcher: FetcherAgent,
    retriever: RetrieverAgent,
    analyzer: AnalyzerAgent,
    critic: CriticAgent,
    presenter: PresenterAgent,
    reflector: ReflectorAgent = None,
) -> StateGraph:
    graph = StateGraph(AgentState)

    # 节点
    graph.add_node("supervisor", supervisor.invoke)
    graph.add_node("translator", translator.invoke)
    graph.add_node("fetcher", fetcher.invoke)
    graph.add_node("retriever", retriever.invoke)
    graph.add_node("analyzer", analyzer.invoke)
    graph.add_node("critic", critic.invoke)
    graph.add_node("presenter", presenter.invoke)

    # 如果有 reflector，添加节点
    if reflector:
        graph.add_node("reflector", reflector.invoke)

    # 边
    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        supervisor_route,
        {"translator": "translator", "retriever": "retriever", "END": "presenter"},
    )

    graph.add_edge("translator", "fetcher")
    graph.add_conditional_edges("fetcher", fetcher_route, {"presenter": "presenter", END: END})

    graph.add_edge("retriever", "analyzer")
    graph.add_edge("analyzer", "critic")

    graph.add_conditional_edges(
        "critic",
        critic_route,
        {"presenter": "presenter", "retriever": "retriever", "END": END},
    )

    graph.add_edge("presenter", END)

    return graph.compile()
