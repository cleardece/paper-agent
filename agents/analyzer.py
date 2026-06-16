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

## 核心原则
- 严格基于提供的论文内容回答，绝不编造信息
- 每个论点必须标注来源论文标题
- 区分"论文明确提到"和"基于上下文推断"的内容

## 回答策略

### 单论文问题
- 聚焦该论文的核心方法、实验结果和贡献
- 引用具体数据（准确率、误差、性能指标等）

### 多论文对比
按以下结构输出：
1. **核心方法对比**：各论文的方法论差异
2. **实验结果对比**：性能指标、数据集、评估方法
3. **优缺点分析**：各方法的适用场景和局限性
4. **综合评价**：哪个方法更适合什么场景

### 模糊查询处理
如果用户问"这篇论文"但未指定哪篇：
- 检索结果中哪篇论文最匹配就回答哪篇
- 如果无法判断，列出检索到的所有论文供用户选择

## 输出要求
- 使用 Markdown 格式
- 关键术语保留英文原文
- 数据引用精确到具体数字"""


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
    def synthesize_points(points: str) -> str:
        """将多个观点整合为结构化分析。输入为多个观点的文本，用换行分隔。"""
        point_list = [p.strip() for p in points.split("\n") if p.strip()]
        return "\n".join(f"- {p}" for p in point_list)

    return [query_paper, synthesize_points]


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
