from types import SimpleNamespace

from agents.supervisor import SupervisorAgent
from core.paper_context import PaperContextResolver, PaperFocusState
from core.search_policy import SearchRequest, guarded_arxiv_search
from core.session_state import SessionStateReducer
from core.turn_context import TurnContextBuilder
from web.app import paper_context_for_result


class FakePaperRepository:
    def __init__(self):
        self.papers = [
            {
                "arxiv_id": "P001",
                "title": "Physics Informed Neural Networks",
                "abstract": "",
            },
            {
                "arxiv_id": "P002",
                "title": "Paper Two",
                "abstract": "",
            },
        ]

    def get_paper(self, paper_id):
        return next(
            (paper for paper in self.papers if paper["arxiv_id"] == paper_id),
            None,
        )

    def list_papers(self, **_kwargs):
        return list(self.papers)


class RecordingArxiv:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = []

    def search(self, query, max_results):
        self.calls.append((query, max_results))
        return self.results[:max_results]


def run_turn(
    query,
    initial_primary=None,
    explicit_target=None,
    search_results=None,
):
    repository = FakePaperRepository()
    focus = PaperFocusState(
        primary_paper_id=initial_primary,
        active_paper_ids=[initial_primary] if initial_primary else [],
    )
    paper_context = PaperContextResolver(repository).resolve(
        query=query,
        explicit_target_paper_id=explicit_target,
        paper_focus=focus,
        recent_paper_contexts=[],
    )
    supervisor_output = SupervisorAgent(llm=None).invoke({
        "user_query": query,
        "paper_context": paper_context.to_dict(),
    })
    turn_output = TurnContextBuilder(repository).invoke({
        "user_query": query,
        "intent": supervisor_output["intent"],
        "paper_context": paper_context.to_dict(),
    })
    turn_context = turn_output["turn_context"]
    arxiv = RecordingArxiv(search_results)
    agent_result = {
        "error": turn_output.get("error"),
        "primary_paper_id": (
            paper_context.primary_paper_id
            if supervisor_output["intent"] == "analyze" else None
        ),
        "resolved_paper_ids": (
            list(paper_context.paper_ids)
            if supervisor_output["intent"] == "analyze" else []
        ),
    }

    request_data = turn_context.get("search_request")
    if request_data:
        request = SearchRequest.from_dict(request_data)
        papers = guarded_arxiv_search(
            arxiv,
            turn_context,
            request,
            max_results=1 if request.mode in {"arxiv_id", "title"} else 5,
        )
        paper_ids = [paper["arxiv_id"] for paper in papers]
        agent_result = {
            "error": None,
            "primary_paper_id": paper_ids[0] if len(paper_ids) == 1 else None,
            "resolved_paper_ids": paper_ids,
        }

    final_context = paper_context_for_result(
        paper_context.to_dict(),
        agent_result,
        turn_context,
        focus.primary_paper_id,
    )
    final_focus = SessionStateReducer().reduce(
        focus,
        agent_result,
        final_context,
    )
    return SimpleNamespace(
        paper_context=paper_context,
        intent=supervisor_output["intent"],
        turn_output=turn_output,
        turn_context=turn_context,
        agent_result=agent_result,
        final_focus=final_focus,
        final_context=final_context,
        arxiv_calls=arxiv.calls,
    )


def test_case_1_structured_analysis_inherits_p001_without_arxiv():
    result = run_turn(
        "请从方法、实验设置、消融和局限四方面给出结构化分析",
        initial_primary="P001",
    )

    assert result.paper_context.primary_paper_id == "P001"
    assert result.intent == "analyze"
    assert result.arxiv_calls == []


def test_case_2_summarize_experiments_inherits_p001_without_arxiv():
    result = run_turn("总结实验结果", initial_primary="P001")

    assert result.final_focus.primary_paper_id == "P001"
    assert result.arxiv_calls == []


def test_case_3_find_similar_papers_admits_external_search():
    result = run_turn(
        "帮我找几篇类似论文",
        initial_primary="P001",
        search_results=[
            {"arxiv_id": "S001"},
            {"arxiv_id": "S002"},
        ],
    )

    assert result.intent == "search"
    assert result.turn_context["allow_external_search"] is True
    assert len(result.arxiv_calls) == 1
    assert result.final_focus.primary_paper_id == "P001"


def test_case_4_explicit_p002_switches_focus():
    result = run_turn(
        "分析这篇论文",
        initial_primary="P001",
        explicit_target="P002",
    )

    assert result.paper_context.status == "switch"
    assert result.paper_context.switched_from_paper_id == "P001"
    assert result.final_focus.primary_paper_id == "P002"


def test_case_5_analysis_without_focus_requires_context_and_never_searches():
    result = run_turn("分析它的实验方法")

    assert result.intent == "analyze"
    assert result.turn_output["error"] == "NEED_PAPER_CONTEXT"
    assert result.arxiv_calls == []


def test_case_6_searching_p003_updates_session_primary():
    result = run_turn(
        "搜索 arXiv:2401.00003 并分析",
        search_results=[{"arxiv_id": "2401.00003"}],
    )

    assert result.arxiv_calls == [("2401.00003", 1)]
    assert result.final_focus.primary_paper_id == "2401.00003"
    assert result.final_context["source"] == "arxiv_id"


def test_case_7_long_analysis_request_never_becomes_arxiv_query():
    query = "帮我找 PINN 求解 Navier-Stokes 的论文，并分析以下八个问题：方法、实验、局限"

    result = run_turn(query)

    assert result.arxiv_calls == [("PINN Navier-Stokes", 5)]
    assert result.arxiv_calls[0][0] != query
