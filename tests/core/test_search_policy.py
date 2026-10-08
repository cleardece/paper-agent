import pytest

from core.search_policy import (
    ExternalSearchNotAllowed,
    SearchQueryBuilder,
    SearchRequest,
    guarded_arxiv_search,
)


class RecordingArxiv:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = []

    def search(self, query, max_results):
        self.calls.append((query, max_results))
        return self.results[:max_results]


def test_denied_turn_cannot_call_arxiv():
    arxiv = RecordingArxiv()
    turn = {"intent": "analyze", "allow_external_search": False}

    with pytest.raises(ExternalSearchNotAllowed):
        guarded_arxiv_search(
            arxiv,
            turn,
            SearchRequest(mode="keywords", value="PINN"),
            max_results=5,
        )

    assert arxiv.calls == []


def test_builder_strips_analysis_requirements_from_search_query():
    raw = "帮我找 PINN 求解 Navier-Stokes 的论文，并分析以下八个问题：方法、实验、局限"

    request = SearchQueryBuilder().build(raw)

    assert request == SearchRequest(
        mode="keywords",
        value="PINN Navier-Stokes",
    )
    assert request.value != raw


def test_builder_canonicalizes_arxiv_id():
    request = SearchQueryBuilder().build("搜索 arXiv:2401.01234v2")

    assert request == SearchRequest(
        mode="arxiv_id",
        value="2401.01234",
    )


def test_builder_uses_quoted_title():
    request = SearchQueryBuilder().build(
        "帮我找论文“Attention Is All You Need”"
    )

    assert request == SearchRequest(
        mode="title",
        value="Attention Is All You Need",
    )


def test_similar_search_uses_current_paper_title_fallback():
    request = SearchQueryBuilder().build(
        "帮我找几篇类似论文",
        fallback_title="Physics Informed Neural Networks",
    )

    assert request == SearchRequest(
        mode="keywords",
        value="Physics Informed Neural Networks",
    )


def test_admitted_search_uses_validated_request_value_only():
    arxiv = RecordingArxiv([{"arxiv_id": "P001"}])
    turn = {"intent": "search", "allow_external_search": True}
    request = SearchRequest(mode="keywords", value="PINN Navier-Stokes")

    result = guarded_arxiv_search(arxiv, turn, request, max_results=5)

    assert result == [{"arxiv_id": "P001"}]
    assert arxiv.calls == [("PINN Navier-Stokes", 5)]
