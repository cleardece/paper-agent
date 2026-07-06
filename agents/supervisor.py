"""
Paper Agent - Supervisor Agent
路由中枢，根据用户意图分派任务到 fetcher 或 retriever
"""

import logging
from langchain_core.messages import SystemMessage, HumanMessage
from state.graph_state import AgentState

logger = logging.getLogger("paper-agent")


SUPERVISOR_PROMPT = """你是论文助手的路由调度器。根据用户输入判断意图，选择执行Agent，并生成英文搜索关键词。

## 判断规则

**→ direct**（单篇论文分析，满足任一）：
- 用户指定了**一篇**具体论文（标题、arXiv链接、或"这篇论文"指代）
- 要求分析/总结/解释**一篇**论文的内容
- 关键词：分析、总结、解释、介绍、这篇论文、它的

**→ fetcher**（多篇论文入库，满足任一）：
- 要求搜索/查找/下载多篇论文
- 关键词：搜索、找论文、下载、几篇、多篇

**→ retriever**（知识库问答，满足任一）：
- 对**多篇已有论文**提问、对比
- 关键词：对比、区别、比较、哪些论文

**→ END**：
- 闲聊、打招呼、与论文无关的问题
- 无法判断意图

## 输出格式（严格JSON）
{"next_agent": "direct" | "fetcher" | "retriever" | "END", "search_query": "英文搜索关键词（fetcher和retriever需要）", "reason": "判断依据"}

## 重要规则
- **单篇分析 → direct**，不要走 fetcher 或 retriever
- **多篇搜索 → fetcher**
- **多篇对比/知识库问答 → retriever**
- 如果判断为fetcher或retriever，**必须生成英文搜索关键词**（直接翻译，不要解释）
- 英文关键词：3-8个词，空格分隔，学术术语准确

## 示例
用户："帮我分析这篇论文" → {"next_agent": "direct", "search_query": "", "reason": "单篇分析"}
用户："它的实验怎么设计的" → {"next_agent": "direct", "search_query": "", "reason": "跟随意图，单篇"}
用户："搜索流体力学PINN求解的论文" → {"next_agent": "fetcher", "search_query": "fluid dynamics PINN solving", "reason": "多篇搜索"}
用户："对比RAG和GraphRAG" → {"next_agent": "retriever", "search_query": "RAG GraphRAG comparison", "reason": "多篇对比"}
用户："你好" → {"next_agent": "END", "search_query": "", "reason": "闲聊"}
"""


class SupervisorAgent:
    def __init__(self, llm, mongodb_client=None):
        self.llm = llm
        self.mongo = mongodb_client

    def _is_followup_query(self, query: str) -> bool:
        """检测是否为跟随意图（指代之前讨论的论文）"""
        followup_patterns = [
            "这篇论文", "该论文", "上一篇", "刚才的", "之前",
            "它的", "这篇的", "该文章", "这篇文章",
        ]
        return any(p in query for p in followup_patterns)

    def _extract_paper_from_context(self, query: str, context: str) -> str:
        """从对话上下文中提取最近讨论的论文标题"""
        import re
        if not context:
            # 没有上下文，从知识库取最近的论文
            if self.mongo:
                try:
                    papers = self.mongo.list_papers(limit=3)
                    if papers:
                        return papers[-1].get("title", "")
                except Exception:
                    pass
            return None

        # 从上下文中提取英文标题（大写开头的连续词组）
        titles = re.findall(r'[A-Z][a-zA-Z\s\-:]{5,80}', context)
        if titles:
            return titles[-1].strip()
        return None

    def _check_paper_exists(self, query: str) -> bool:
        """检查知识库中是否有与查询相关的论文"""
        if not self.mongo:
            return False
        try:
            papers = self.mongo.list_papers(limit=50)
            if not papers:
                return False
            # 简单关键词匹配：检查论文标题是否包含查询中的关键词
            query_lower = query.lower()
            # 提取英文关键词（过滤掉常见中文停用词）
            import re
            keywords = re.findall(r'[a-zA-Z]{3,}', query_lower)
            if not keywords:
                return False
            for paper in papers:
                title = paper.get("title", "").lower()
                abstract = paper.get("abstract", "").lower()
                text = f"{title} {abstract}"
                # 至少匹配 2 个关键词
                matched = sum(1 for kw in keywords if kw in text)
                if matched >= 2:
                    return True
            return False
        except Exception as e:
            logger.warning(f"[Supervisor] 检查知识库失败: {e}")
            return False

    def invoke(self, state: AgentState) -> dict:
        query = state["user_query"]
        logger.info(f"[Supervisor] 收到查询: {query[:50]}...")

        # 构建带对话上下文的输入
        context = state.get("conversation_context", "")
        if context:
            user_input = f"## 对话上下文\n{context}\n\n## 当前用户输入\n{query}"
        else:
            user_input = f"用户输入：{query}"

        messages = [
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=user_input),
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

        # direct 模式不需要 search_query，用原始查询即可
        if next_agent == "direct" and not search_query:
            search_query = query

        # 提取目标论文（跟随意图时）
        target_paper = None
        is_followup = self._is_followup_query(query)
        logger.info(f"[Supervisor] is_followup={is_followup}, query={query[:30]}")

        # 检查查询中是否包含具体论文标题（英文关键词 >= 5 个）
        import re
        english_keywords = [kw for kw in re.findall(r'[a-zA-Z]{3,}', query) if len(kw) > 3]
        has_explicit_title = len(english_keywords) >= 5

        if is_followup:
            context = state.get("conversation_context", "")
            logger.info(f"[Supervisor] context={repr(context[:50] if context else '')}, next_agent={next_agent}")
            target_paper = self._extract_paper_from_context(query, context)
            if target_paper:
                logger.info(f"[Supervisor] 识别到目标论文: {target_paper[:50]}")
            elif not context and not has_explicit_title:
                # 新对话 + 跟随意图 + 没有具体标题 → 走 retriever
                if next_agent == "direct":
                    logger.info("[Supervisor] 新对话无上下文且无具体标题，跟随意图改走 retriever")
                    next_agent = "retriever"
            elif not context and has_explicit_title:
                # 新对话 + 有具体标题 → 走 direct，用标题作为 target_paper
                target_paper = " ".join(english_keywords[:8])
                logger.info(f"[Supervisor] 有具体标题，提取为: {target_paper[:50]}")

        logger.info(f"[Supervisor] 路由: {next_agent}, 搜索词: {search_query}")

        result = {"next_agent": next_agent, "search_query": search_query, "error": None, "target_paper": target_paper}
        return result