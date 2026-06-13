"""
Paper Agent - Analyzer Agent
ReAct模式，综合多篇论文分析问题
"""

import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from state.graph_state import AgentState

logger = logging.getLogger("paper-agent")


ANALYZER_PROMPT = """你是一个学术论文分析专家。根据检索到的论文片段，综合分析用户的问题。

要求：
1. 基于提供的论文内容回答，不要编造
2. 明确引用来源（论文标题）
3. 指出不同论文之间的共识和分歧
4. 如果用户要求对比多篇论文，按以下结构输出：
   - 共同点
   - 差异点（方法、结果、创新点）
   - 各论文的优缺点
   - 综合评价
5. 如果信息不足，明确说明

可用工具：
- query_paper: 进一步查询某篇论文的更多内容
- synthesize: 将多个观点整合为结构化分析

输出格式：
- 共识点
- 分歧点
- 各论文的贡献
- 综合结论"""


def _build_analyzer_tools(mongo_client):
    """构建Analyzer可用的工具"""
    from langchain_core.tools import tool

    @tool
    def query_paper(arxiv_id: str, question: str) -> str:
        """查询某篇论文的详细内容"""
        paper = mongo_client.get_paper(arxiv_id)
        if not paper:
            return f"未找到论文 {arxiv_id}"

        chunks = mongo_client.get_chunks_by_paper(arxiv_id)
        content = "\n".join(c["content"] for c in chunks)

        return f"论文：{paper['title']}\n\n内容：\n{content[:3000]}"

    @tool
    def synthesize观点(points: list[str]) -> str:
        """将多个观点整合为结构化分析"""
        return "\n".join(f"- {p}" for p in points)

    return [query_paper, synthesize观点]


class AnalyzerAgent:
    """ReAct分析Agent"""

    def __init__(self, llm, mongodb_client):
        self.llm = llm
        self.mongo = mongodb_client
        self.tools = _build_analyzer_tools(mongodb_client)

    def invoke(self, state: AgentState) -> dict:
        """
        ReAct流程：
        1. 将检索到的chunks组装为上下文
        2. 创建ReAct Agent（带工具）
        3. 运行直到得出结论
        """
        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            logger.info("[Analyzer] 无可用的论文内容进行分析")
            return {"analysis": "无可用的论文内容进行分析。"}

        logger.info(f"[Analyzer] 收到 {len(chunks)} 个分块，开始分析...")

        # 上下文窗口限制：MiMo 32K token ≈ 128K 字符
        # Prompt 约占 2K token，留 30K token 给内容 ≈ 120K 字符
        # 安全起见用 50K 字符（约 12K token）
        MAX_CONTEXT_CHARS = 50000
        context = ""
        total_chars = 0
        for c in chunks:
            chunk_text = f"【{c['paper_title']}】(chunk {c['chunk_index']}, score: {c['score']:.3f})\n{c['content']}"
            if total_chars + len(chunk_text) > MAX_CONTEXT_CHARS:
                logger.info(f"[Analyzer] 上下文已满，截断到 {len(context)} 字符，保留 {len(context.split(chr(10)))} 行")
                break
            context += chunk_text + "\n\n---\n\n"
            total_chars += len(chunk_text)

        messages = [
            SystemMessage(content=ANALYZER_PROMPT),
            HumanMessage(content=f"检索到的论文内容：\n\n{context}\n\n用户问题：{state['user_query']}"),
        ]

        # ReAct Agent执行
        logger.info("[Analyzer] 正在调用 LLM 进行分析...")
        agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=ANALYZER_PROMPT,
        )

        result = agent.invoke({"messages": messages})

        # 提取最终回答
        answer = result["messages"][-1].content
        logger.info(f"[Analyzer] 分析完成，回答长度: {len(answer)}")

        return {"analysis": answer, "error": None}
