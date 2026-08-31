"""Intent-only routing for Paper Agent."""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from core.llm_utils import invoke_json_with_retry
from core.paper_context import is_explicit_search_request
from state.graph_state import AgentState

logger = logging.getLogger("paper-agent")


SUPERVISOR_PROMPT = """你是论文助手的意图路由器。论文身份已经由上游解析完成；你不得提取、猜测或修改论文 ID。

只判断当前动作：
- analyze：针对一篇明确或被指代论文的分析、总结、解释、复现
- rag：对本地论文库内容的一般问答或综合检索
- compare：比较两篇或多篇论文
- search：明确要求搜索、查找、下载或推荐外部论文
- general：闲聊或与论文无关

严格输出 JSON：
{"intent": "analyze|rag|compare|search|general", "reason": "简短依据"}
"""


class SupervisorAgent:
    """Classify user intent without resolving paper identity."""

    SEARCH_MARKERS = (
        "搜索", "查找", "找论文", "找几篇", "类似论文", "相似论文",
        "相关论文", "检索文献", "下载论文", "search", "find papers",
        "look for papers",
    )
    COMPARE_MARKERS = ("对比", "比较", "区别", "compare", "versus", " vs ")
    ANALYZE_MARKERS = (
        "分析", "总结", "解释", "实验", "方法", "贡献", "局限", "复现",
        "结果", "消融", "评估", "介绍", "analyze", "summarize", "method",
        "experiment", "result", "contribution", "limitation", "reproduce",
    )
    RAG_MARKERS = ("什么", "怎么", "为什么", "哪些", "?", "？", "问", "研究")
    GREETINGS = {"你好", "您好", "嗨", "hi", "hello", "hey"}
    VALID_INTENTS = {"analyze", "rag", "compare", "search", "general"}

    def __init__(self, llm, mongodb_client=None):
        self.llm = llm
        self.mongo = mongodb_client

    def invoke(self, state: AgentState) -> dict:
        query = state["user_query"].strip()
        paper_context = state.get("paper_context") or {}
        intent = self._deterministic_intent(query, paper_context)
        reason = "deterministic routing"

        if intent is None and self.llm is not None:
            decision = invoke_json_with_retry(
                self.llm,
                [
                    SystemMessage(content=SUPERVISOR_PROMPT),
                    HumanMessage(content=(
                        f"论文上下文：{json.dumps(paper_context, ensure_ascii=False)}\n"
                        f"当前消息：{query}"
                    )),
                ],
                max_retries=2,
            )
            candidate = decision.get("intent") if decision else None
            if candidate in self.VALID_INTENTS:
                intent = candidate
                reason = str(decision.get("reason", "LLM routing"))

        if intent is None:
            intent = "analyze" if paper_context.get("primary_paper_id") else "general"
            reason = "safe fallback"

        logger.info("[Supervisor] intent=%s", intent)
        return {"intent": intent, "routing_reason": reason, "error": None}

    def _deterministic_intent(self, query: str, paper_context: dict) -> str | None:
        lowered = f" {query.casefold()} "
        if is_explicit_search_request(query) or any(
            marker in lowered for marker in self.SEARCH_MARKERS
        ):
            return "search"
        if any(marker in lowered for marker in self.COMPARE_MARKERS):
            return "compare"
        if query.casefold() in self.GREETINGS:
            return "general"
        if any(marker in lowered for marker in self.ANALYZE_MARKERS):
            return "analyze"
        if paper_context.get("primary_paper_id"):
            return "analyze"
        if any(marker in lowered for marker in self.RAG_MARKERS):
            return "rag"
        return None
