"""Build the normalized execution contract after intent routing."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from core.search_policy import InvalidSearchRequest, SearchQueryBuilder


Intent = Literal["analyze", "rag", "compare", "search", "general"]


class TurnContext(TypedDict):
    query: str
    intent: Intent
    primary_paper_id: str | None
    paper_ids: list[str]
    paper_resolution_source: str
    paper_resolution_confidence: float
    allow_external_search: bool
    search_request: dict[str, str] | None


ROUTES = {
    "analyze": "direct",
    "rag": "retriever",
    "compare": "retriever",
    "search": "fetcher",
    "general": "END",
}


def build_turn_context(
    query: str,
    intent: str,
    paper_context: dict[str, Any],
) -> dict[str, Any]:
    normalized_intent = intent if intent in ROUTES else "general"
    primary_id = paper_context.get("primary_paper_id")
    error = None
    if paper_context.get("status") == "ambiguous":
        error = "PAPER_CONTEXT_AMBIGUOUS"
    elif normalized_intent == "analyze" and not primary_id:
        error = "NEED_PAPER_CONTEXT"

    next_agent = "END" if error else ROUTES[normalized_intent]
    turn_context = {
        "query": query,
        "intent": normalized_intent,
        "primary_paper_id": primary_id,
        "paper_ids": list(paper_context.get("paper_ids") or []),
        "paper_resolution_source": paper_context.get("source", "none"),
        "paper_resolution_confidence": float(
            paper_context.get("confidence") or 0.0
        ),
        "allow_external_search": normalized_intent == "search" and not error,
        "search_request": None,
    }
    return {
        "intent": normalized_intent,
        "turn_context": turn_context,
        "next_agent": next_agent,
        "target_paper_id": primary_id,
        "error": error,
        "answer": _business_answer(error),
    }


def _business_answer(error: str | None) -> str | None:
    if error == "NEED_PAPER_CONTEXT":
        return "请先选择或明确指定要分析的论文。"
    if error == "PAPER_CONTEXT_AMBIGUOUS":
        return "当前消息可能指向多篇论文，请明确选择其中一篇。"
    if error == "INVALID_SEARCH_REQUEST":
        return "没有提取到有效的论文搜索条件，请补充标题、arXiv ID 或关键词。"
    return None


class TurnContextBuilder:
    def __init__(self, paper_repository=None, query_builder=None):
        self.repository = paper_repository
        self.query_builder = query_builder or SearchQueryBuilder()

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        result = build_turn_context(
            state["user_query"],
            state.get("intent", "general"),
            state.get("paper_context") or {},
        )
        if result["intent"] != "search" or result.get("error"):
            return result

        fallback_title = self._primary_title(
            result["turn_context"].get("primary_paper_id")
        )
        try:
            request = self.query_builder.build(
                state["user_query"],
                fallback_title=fallback_title,
            )
        except InvalidSearchRequest:
            result["error"] = "INVALID_SEARCH_REQUEST"
            result["answer"] = _business_answer(result["error"])
            result["next_agent"] = "END"
            result["turn_context"]["allow_external_search"] = False
            return result
        result["turn_context"]["search_request"] = request.to_dict()
        result["search_query"] = request.value
        return result

    def _primary_title(self, paper_id: str | None) -> str | None:
        if not paper_id or self.repository is None:
            return None
        try:
            paper = self.repository.get_paper(paper_id)
        except Exception:
            return None
        return str(paper.get("title", "")).strip() if paper else None
