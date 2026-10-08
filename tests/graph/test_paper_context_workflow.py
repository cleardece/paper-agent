from graph.workflow import build_workflow


class RecordingNode:
    def __init__(self, name, events, output):
        self.name = name
        self.events = events
        self.output = output

    def invoke(self, _state):
        self.events.append(self.name)
        return dict(self.output)


def make_workflow(primary_id):
    events = []
    context = {
        "primary_paper_id": primary_id,
        "paper_ids": [primary_id] if primary_id else [],
        "status": "resolved" if primary_id else "unresolved",
        "source": "session_focus" if primary_id else "none",
        "confidence": 0.98 if primary_id else 0.0,
        "inherited": bool(primary_id),
    }
    turn_error = None if primary_id else "NEED_PAPER_CONTEXT"
    resolver = RecordingNode("resolver", events, {"paper_context": context})
    supervisor = RecordingNode("supervisor", events, {"intent": "analyze"})
    turn = RecordingNode("turn", events, {
        "turn_context": {
            "intent": "analyze",
            "primary_paper_id": primary_id,
            "paper_ids": context["paper_ids"],
            "allow_external_search": False,
        },
        "next_agent": "direct" if primary_id else "END",
        "error": turn_error,
        "answer": "请先选择论文。" if turn_error else None,
    })
    direct = RecordingNode("direct", events, {
        "answer": "analysis",
        "error": None,
        "primary_paper_id": primary_id,
        "resolved_paper_ids": [primary_id] if primary_id else [],
    })
    presenter = RecordingNode("presenter", events, {"answer": "done"})
    noop = RecordingNode("noop", events, {})
    workflow = build_workflow(
        paper_context_resolver=resolver,
        supervisor=supervisor,
        turn_context_builder=turn,
        fetcher=noop,
        retriever=noop,
        analyzer=noop,
        critic=RecordingNode("critic", events, {"next_agent": "presenter"}),
        presenter=presenter,
        direct_analyzer=direct,
    )
    return workflow, events


def test_resolver_runs_before_supervisor_and_turn_policy_before_direct():
    workflow, events = make_workflow("P001")

    result = workflow.invoke({"user_query": "总结实验结果"})

    assert events == ["resolver", "supervisor", "turn", "direct", "presenter"]
    assert result["primary_paper_id"] == "P001"


def test_unresolved_analysis_never_enters_direct_analyzer():
    workflow, events = make_workflow(None)

    result = workflow.invoke({"user_query": "分析它的实验方法"})

    assert "direct" not in events
    assert events == ["resolver", "supervisor", "turn", "presenter"]
    assert result["error"] == "NEED_PAPER_CONTEXT"
