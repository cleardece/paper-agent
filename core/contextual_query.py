"""Rewrite contextual research follow-ups into standalone retrieval queries."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from core.llm_utils import invoke_json_with_retry

logger = logging.getLogger("paper-agent")

RESEARCH_INTENTS = {"analyze", "rag", "compare"}

SYSTEM_PROMPT = """你是学术检索查询改写器，只负责把对话中的省略问题改写成独立、明确的检索问题。

严格遵守：
1. 不回答问题，不补充论文事实，不生成引用。
2. 不改变给定论文范围，不添加或猜测论文。
3. 历史对话只用于消解代词、省略、章节、回答结构和比较维度。
4. 助手历史回答不是论文事实，只能用于识别用户正在追问的对象。
5. 保留当前问题的语言和真实意图。
6. 只输出 JSON：{"retrieval_query":"独立检索问题"}
"""


def _clean(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


class ContextualQueryRewriter:
    """Best-effort query rewrite that never changes paper identity or evidence."""

    def __init__(self, llm, paper_repository=None):
        self.llm = llm
        self.repository = paper_repository

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        original = _clean(state.get("user_query"), 1000)
        fallback = {
            "retrieval_query": original,
            "contextual_query_rewritten": False,
        }
        turn_context = state.get("turn_context") or {}
        intent = turn_context.get("intent") or state.get("intent")
        context = self._context(state, turn_context)
        if not original or intent not in RESEARCH_INTENTS:
            return fallback
        if self.llm is None or not self._has_conversation_context(context):
            return fallback

        payload = {
            "current_question": original,
            "resolved_papers": self._paper_labels(turn_context.get("paper_ids") or []),
            "active_section": context["active_section"],
            "active_task": context["active_task"],
            "recent_user_messages": context["recent_user_messages"],
            "recent_dialogue": context["recent_dialogue"],
            "rolling_summary": context["rolling_summary"],
        }
        prompt = (
            "以下上下文只用于理解意图，不能作为论文证据，也不能写入回答。\n"
            "助手回答只能用于识别追问对象或上一轮回答结构，不能视为论文事实。\n"
            "请将 current_question 改写为脱离对话也能用于论文数据库检索的问题。\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        result = invoke_json_with_retry(
            self.llm,
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)],
            max_retries=0,
        )
        rewritten = _clean((result or {}).get("retrieval_query"), 1000)
        if not rewritten:
            logger.info("[ContextualQueryRewriter] 改写失败，使用原始查询")
            return fallback
        logger.info(
            "[ContextualQueryRewriter] %s",
            "已生成独立检索查询" if rewritten != original else "查询无需改写",
        )
        return {
            "retrieval_query": rewritten,
            "contextual_query_rewritten": rewritten != original,
        }

    @staticmethod
    def _context(state: dict[str, Any], turn_context: dict[str, Any]) -> dict[str, Any]:
        recent = [
            _clean(message, 500)
            for message in list(state.get("recent_user_messages") or [])[-6:]
            if _clean(message, 500)
        ]
        return {
            "active_section": _clean(state.get("active_section"), 200),
            "active_task": _clean(state.get("active_task"), 500),
            "recent_user_messages": recent,
            "recent_dialogue": _clean(state.get("conversation_context"), 6000),
            "rolling_summary": _clean(state.get("conversation_summary"), 800),
        }

    @staticmethod
    def _has_conversation_context(context: dict[str, Any]) -> bool:
        return any((
            context["active_section"],
            context["active_task"],
            context["recent_user_messages"],
            context["recent_dialogue"],
            context["rolling_summary"],
        ))

    def _paper_labels(self, paper_ids: list[str]) -> list[dict[str, str]]:
        labels = []
        for paper_id in paper_ids[:8]:
            title = ""
            if self.repository is not None:
                try:
                    paper = self.repository.get_paper(paper_id)
                    title = _clean((paper or {}).get("title"), 300)
                except Exception:
                    title = ""
            labels.append({"paper_id": str(paper_id), "title": title})
        return labels
