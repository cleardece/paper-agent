from core.turn_context import TurnContextBuilder, build_turn_context


def resolved_context(paper_id: str) -> dict:
    return {
        "primary_paper_id": paper_id,
        "paper_ids": [paper_id],
        "status": "resolved",
        "source": "session_focus",
        "confidence": 0.98,
        "inherited": True,
        "switched_from_paper_id": None,
    }


def unresolved_context() -> dict:
    return {
        "primary_paper_id": None,
        "paper_ids": [],
        "status": "unresolved",
        "source": "none",
        "confidence": 0.0,
        "inherited": False,
        "switched_from_paper_id": None,
    }


class FakePaperRepository:
    def get_paper(self, paper_id):
        if paper_id == "P001":
            return {
                "arxiv_id": "P001",
                "title": "Physics Informed Neural Networks",
            }
        return None


def test_analyze_without_primary_is_stopped_before_direct_analyzer():
    result = build_turn_context(
        "分析它的实验方法",
        "analyze",
        unresolved_context(),
    )

    assert result["error"] == "NEED_PAPER_CONTEXT"
    assert result["next_agent"] == "END"
    assert result["turn_context"]["allow_external_search"] is False


def test_analyze_with_primary_routes_to_direct_without_search_permission():
    result = build_turn_context(
        "总结实验结果",
        "analyze",
        resolved_context("P001"),
    )

    assert result["error"] is None
    assert result["next_agent"] == "direct"
    assert result["target_paper_id"] == "P001"
    assert result["turn_context"]["primary_paper_id"] == "P001"
    assert result["turn_context"]["allow_external_search"] is False


def test_only_search_intent_admits_external_search_and_builds_request():
    builder = TurnContextBuilder(FakePaperRepository())

    result = builder.invoke({
        "user_query": "帮我找几篇类似论文",
        "intent": "search",
        "paper_context": resolved_context("P001"),
    })

    turn = result["turn_context"]
    assert result["next_agent"] == "fetcher"
    assert turn["allow_external_search"] is True
    assert turn["search_request"] == {
        "mode": "keywords",
        "value": "Physics Informed Neural Networks",
    }


def test_compare_routes_to_retriever_with_all_resolved_ids():
    context = resolved_context("P001")
    context["paper_ids"] = ["P001", "P002"]

    result = build_turn_context("对比", "compare", context)

    assert result["next_agent"] == "retriever"
    assert result["turn_context"]["paper_ids"] == ["P001", "P002"]
    assert result["turn_context"]["allow_external_search"] is False
