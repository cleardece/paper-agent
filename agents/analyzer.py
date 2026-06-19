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

## 核心原则（最重要）
1. **严格基于检索内容**：只使用下方【检索到的论文内容】中的信息回答
2. **绝不编造**：如果检索内容中没有相关信息，明确说"根据现有检索结果，未找到相关信息"
3. **每个论点必须引用来源**：格式为 [论文标题]
4. **不使用外部知识**：即使你知道答案，也不能用，只能用检索到的内容

## 回答策略

### 信息充分时
- 直接回答问题，引用具体数据和论文来源
- 多篇论文有分歧时，对比呈现

### 信息不足时
- 明确说明哪些部分有答案，哪些部分缺失
- 建议用户补充关键词重新搜索
- 不要猜测或编造

### 模糊查询（如"这篇论文"）
- 列出检索到的所有论文，让用户确认
- 不要假设用户指的是哪篇

## 输出格式
- 先用一句话直接回答
- 再展开分析
- 最后列出参考来源"""


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
