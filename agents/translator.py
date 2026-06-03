"""
Paper Agent - Translator Agent
将中文查询翻译为英文搜索关键词
"""

import logging
from langchain_core.messages import SystemMessage, HumanMessage
from state.graph_state import AgentState

logger = logging.getLogger("paper-agent")


class TranslatorAgent:
    """查询翻译 Agent"""

    def __init__(self, llm):
        self.llm = llm

    def invoke(self, state: AgentState) -> dict:
        """
        翻译用户查询

        Args:
            state: 包含 user_query 的状态

        Returns:
            包含 search_query 的状态更新
        """
        query = state["user_query"]
        search_query = self.translate_query(query)
        logger.info(f"[Translator] {query[:30]}... → {search_query[:30]}...")
        return {"search_query": search_query, "error": None}

    def translate_query(self, query: str) -> str:
        """
        将中文查询翻译为英文搜索关键词

        Args:
            query: 用户原始查询（可能是中文）

        Returns:
            英文搜索关键词
        """
        # 如果已经是英文（主要是 ASCII 字符），直接返回
        if self._is_english(query):
            logger.info(f"[Translator] 检测到英文查询，跳过翻译: {query[:50]}")
            return query

        logger.info(f"[Translator] 正在翻译查询: {query[:50]}...")

        messages = [
            SystemMessage(content="""你是一个学术搜索助手。将用户的中文查询翻译成简短的英文搜索关键词，用于在 arXiv 上搜索论文。

规则：
1. 只输出英文关键词，不要解释
2. 保持学术术语的准确性
3. 关键词之间用空格分隔
4. 保持简洁，通常 3-8 个词

示例：
- "流体力学PINN求解" → "fluid dynamics PINN solving"
- " transformer 注意力机制" → "transformer attention mechanism"
- "深度学习图像分割" → "deep learning image segmentation"
- "自然语言处理大模型" → "natural language processing large language model"
"""),
            HumanMessage(content=query),
        ]

        try:
            response = self.llm.invoke(messages)
            translated = response.content.strip()
            logger.info(f"[Translator] 翻译结果: {translated}")
            return translated
        except Exception as e:
            logger.error(f"[Translator] 翻译失败: {e}")
            # 翻译失败时返回原始查询
            return query

    def _is_english(self, text: str) -> bool:
        """检测文本是否主要是英文"""
        if not text:
            return True

        # 统计非 ASCII 字符的比例
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        total_count = len(text)

        if total_count == 0:
            return True

        # 如果非 ASCII 字符少于 30%，认为是英文
        return (non_ascii_count / total_count) < 0.3
