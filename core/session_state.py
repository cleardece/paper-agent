"""Normalize agent paper results and reduce them into session focus."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TypedDict

from core.paper_context import PaperContext, PaperFocusState


class AgentResult(TypedDict, total=False):
    answer: str | None
    error: str | None
    primary_paper_id: str | None
    resolved_paper_ids: list[str]


def normalize_agent_result(
    result: dict[str, Any] | None,
) -> AgentResult:
    result = result or {}
    primary_id = result.get("primary_paper_id") or result.get(
        "resolved_paper_id"
    )
    resolved_ids = list(result.get("resolved_paper_ids") or [])
    if primary_id and primary_id not in resolved_ids:
        resolved_ids.insert(0, primary_id)
    return {
        "answer": result.get("answer"),
        "error": result.get("error"),
        "primary_paper_id": primary_id,
        "resolved_paper_ids": list(dict.fromkeys(resolved_ids)),
    }


class SessionStateReducer:
    def reduce(
        self,
        focus: PaperFocusState,
        agent_result: dict[str, Any] | None,
        paper_context: PaperContext | dict[str, Any],
    ) -> PaperFocusState:
        normalized = normalize_agent_result(agent_result)
        context = (
            paper_context
            if isinstance(paper_context, PaperContext)
            else PaperContext.from_dict(paper_context)
        )
        if normalized["error"]:
            return PaperFocusState.from_dict(focus.to_dict())

        primary_id = normalized["primary_paper_id"]
        resolved_ids = normalized["resolved_paper_ids"]
        if not primary_id and context.status in {"resolved", "switch"}:
            primary_id = context.primary_paper_id
        if not resolved_ids and context.status in {"resolved", "switch"}:
            resolved_ids = list(context.paper_ids)
        if not primary_id:
            return PaperFocusState.from_dict(focus.to_dict())

        active_ids = list(dict.fromkeys([primary_id, *resolved_ids]))
        return PaperFocusState(
            primary_paper_id=primary_id,
            active_paper_ids=active_ids,
            source=context.source,
            confidence=context.confidence,
            last_resolved_at=datetime.now(timezone.utc),
        )
