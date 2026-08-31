"""
Paper Agent - Analyzer Agent
ReAct模式，综合多篇论文分析问题
"""

import logging
import re
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from state.graph_state import AgentState

logger = logging.getLogger("paper-agent")


# 推理残留模式：Analyzer 输出中常见的推理提示词
_REASONING_PATTERNS = [
    # 开头推理
    r'^(根据(以上|上述|检索(结果|内容)|分析|这些论文)|基于(以上|上述|检索)|通过(分析|对比|阅读)|经过(分析|对比))[，,。：:\s]*',
    r'^(由此(可知|可见|得出)|综上(所述|所述)|总的(来说|来看)|综合(来看|分析|以上))[，,。：:\s]*',
    r'^(我(认为|推断|觉得|发现)|从(中|内容)可以看出|可以看出|不难发现)[，,。：:\s]*',
    # 中间过渡推理
    r'\n(综上|因此|所以|由此可见|由此可知|综上所述)[，,。：:\s]*',
    # 结尾推理
    r'(以上(是|为|就是).*?(分析|总结|结论|回答)|如有(疑问|问题)请(提问|咨询|指出))[。.\s]*$',
]


def _strip_reasoning_traces(text: str) -> str:
    """清洗 Analyzer 输出中的推理残留，只保留结论和证据

    示例：
        "根据以上分析，论文A提出了X方法..." → "论文A提出了X方法..."
        "综上所述，Y指标提升了15%..." → "Y指标提升了15%..."
    """
    cleaned = text
    for pattern in _REASONING_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
    # 清理多余空行
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


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

## 输出格式（严格遵循）
你的输出必须分为两个明确部分，用分隔符隔开：

### 结论
（用 1-3 句话直接回答用户问题，不要写推理过程，只写最终结论）

### 分析
（展开分析：每个论点必须引用 [论文标题] 作为来源。多篇论文有分歧时对比呈现。信息不足时明确说明。）

## 重要
- **结论部分不要包含推理过程**，只写最终判断/答案
- **分析部分不要重复结论**，只写支撑证据和逻辑
- 不要写"我认为""我推断""根据以上分析"等推理提示词"""


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

        # 提取中间推理步骤（ReAct 的思考链），写入日志
        intermediate_steps = []
        for msg in result["messages"][1:-1]:  # 排除 system 和最终回答
            if hasattr(msg, 'content') and msg.content:
                intermediate_steps.append(msg.content[:200])
        if intermediate_steps:
            logger.info(f"[Analyzer] 推理过程 ({len(intermediate_steps)} 步):\n" +
                       "\n---\n".join(intermediate_steps))

        # 提取最终回答（结论 + 分析，不含推理过程）
        raw_answer = result["messages"][-1].content

        # 清洗推理残留（"根据以上分析""综上所述"等），实现对话隔离
        answer = _strip_reasoning_traces(raw_answer)
        if len(answer) < len(raw_answer):
            logger.info(f"[Analyzer] 清洗推理残留: {len(raw_answer)} → {len(answer)} 字符")

        logger.info(f"[Analyzer] 分析完成，输出长度: {len(answer)}")

        paper_ids = list(dict.fromkeys(
            chunk.get("paper_arxiv_id")
            for chunk in chunks
            if chunk.get("paper_arxiv_id")
        ))
        turn_context = state.get("turn_context") or {}
        primary_id = turn_context.get("primary_paper_id")
        if not primary_id and len(paper_ids) == 1:
            primary_id = paper_ids[0]
        return {
            "analysis": answer,
            "error": None,
            "primary_paper_id": primary_id,
            "resolved_paper_ids": paper_ids,
        }
