"""
Paper Agent - Supervisor Agent
路由中枢，根据用户意图分派任务到 fetcher 或 retriever
"""

import logging
from langchain_core.messages import SystemMessage, HumanMessage
from state.graph_state import AgentState

logger = logging.getLogger("paper-agent")


SUPERVISOR_PROMPT = SUPERVISOR_PROMPT = """你是论文助手的路由调度器。根据用户输入判断意图，选择执行Agent。

## 判断规则

**→ fetcher**（满足任一）：
- 要求搜索/查找/下载/抓取论文
- 提供了arXiv链接或论文标题要求入库
- 关键词：搜索、找论文、下载、入库、fetch、search

**→ retriever**（满足任一）：
- 对已有论文内容提问（是什么、为什么、怎么、对比、总结）
- 要求解释/分析/比较论文中的概念
- 关键词：什么、怎么、为什么、解释、分析、对比、区别

**→ END**：
- 闲聊、打招呼、与论文无关的问题
- 无法判断意图

## 输出格式（严格JSON）
{"next_agent": "fetcher" | "retriever" | "END", "search_query": "提取的搜索关键词（仅fetcher需要）", "reason": "判断依据"}

## 注意
- 如果判断为fetcher，必须从用户输入中提取核心搜索关键词（英文最佳），去掉"帮我找"、"搜索"等修饰词
- 比如"帮我找几篇关于RAG的最新论文" → search_query应为"RAG latest papers"

## 示例
用户："帮我找几篇关于RAG的最新论文" → fetcher, search_query="RAG latest papers"
用户："搜索流体力学PINN求解的论文" → fetcher, search_query="fluid dynamics PINN solving"
用户："Transformer和RNN的区别是什么" → retriever
用户："你好" → END
"""


class SupervisorAgent:
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, state: AgentState) -> dict:
        query = state["user_query"]
        logger.info(f"[Supervisor] 收到查询: {query[:50]}...")
        messages = [
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=f"用户输入：{query}"),
        ]
        logger.info("[Supervisor] 正在调用 LLM 判断意图...")
        response = self.llm.invoke(messages)
        content = response.content.strip()
        logger.info(f"[Supervisor] LLM 返回: {content[:100]}...")

        import json, re
        json_match = re.search(r'\{[^}]+\}', content)
        next_agent = "END"
        search_query = None

        if json_match:
            try:
                decision = json.loads(json_match.group())
                next_agent = decision.get("next_agent", "END")
                search_query = decision.get("search_query")
            except json.JSONDecodeError:
                next_agent = "END"
        else:
            lower = query.lower()
            if any(kw in lower for kw in ["搜索", "抓取", "下载", "找论文", "入库", "search", "fetch"]):
                next_agent = "fetcher"
            elif any(kw in lower for kw in ["?", "？", "什么", "怎么", "为什么", "分析", "对比"]):
                next_agent = "retriever"
            else:
                next_agent = "END"

        # 如果是 fetcher 但没有提取到 search_query，使用用户原始输入
        if next_agent == "fetcher" and not search_query:
            search_query = query

        logger.info(f"[Supervisor] 路由: {next_agent}, 搜索词: {search_query}")

        result = {"next_agent": next_agent, "search_query": search_query, "error": None}
        return result