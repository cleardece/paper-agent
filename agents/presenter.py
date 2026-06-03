"""Paper Agent - Presenter Agent."""

import logging
from langchain_core.messages import HumanMessage, SystemMessage

from state.graph_state import AgentState

logger = logging.getLogger("paper-agent")


PRESENTER_PROMPT = """你是学术论文阅读助手的输出引擎。将分析结果转化为专业、可读的最终回复。

## 输出结构

### 回答（必须）
- 第一句话直接回答用户问题，不要绕开
- 如果问题无法完全回答，先说能确定的部分，再说明不确定的部分

### 分析展开
- 按论点分段，每段一个核心观点
- 引用论文时用 [论文标题] 格式
- 多篇论文有分歧时，用对比方式呈现
- 不要重复检索片段的原文，用自己的话概括

### 参考来源
- 只列实际引用过的论文
- 不要编造论文标题或不存在的结论

## 约束
- 总字数 500-1500 字
- 不要用“作为AI”“根据我的理解”等废话开头
- 如果信息不足，如实说明
"""


class PresenterAgent:
    def __init__(self, llm, code_generator=None):
        self.llm = llm
        self.code_gen = code_generator

    def invoke(self, state: AgentState) -> dict:
        analysis = state.get("analysis")
        retrieved_chunks = state.get("retrieved_chunks", [])
        user_query = state["user_query"]
        sources = self._extract_sources(retrieved_chunks)

        logger.info(f"[Presenter] 正在生成最终回复...")

        # 如果没有 analysis（比如闲聊/END 路由），用简单的通用回复 prompt
        if not analysis:
            present_prompt = f"""用户说：{user_query}

请用友好的方式简短回复用户。如果是打招呼，简单介绍你的能力（论文搜索、分析、问答）。
回复控制在 100 字以内。"""
        else:
            present_prompt = f"""请格式化以下分析结果。

用户问题：{user_query}

分析结果：{analysis}

论文来源：
{sources}

请按输出结构生成最终回复。"""

        messages = [
            SystemMessage(content=PRESENTER_PROMPT),
            HumanMessage(content=present_prompt),
        ]
        logger.info("[Presenter] 正在调用 LLM 生成回复...")
        response = self.llm.invoke(messages)

        code_result = None
        if self.code_gen and any(kw in user_query for kw in ["复现", "实现", "代码", "code", "源码"]):
            code_result = self.code_gen.generate(analysis, "")

        final = response.content
        if code_result and code_result.get("code"):
            final += "\n\n---\n\n## 代码实现\n```python\n" + code_result["code"] + "\n```"

        logger.info(f"[Presenter] 回复生成完成，长度: {len(final)}")
        return {"answer": final, "next_agent": "END"}

    def _extract_sources(self, chunks: list) -> str:
        seen = set()
        sources = []
        for chunk in chunks:
            pid = chunk.get("paper_arxiv_id", "")
            if pid and pid not in seen:
                seen.add(pid)
                title = chunk.get("paper_title", pid)
                link = f"https://arxiv.org/abs/{pid}" if pid else ""
                sources.append(f"- **{title}** {link}")
        return "\n".join(sources) if sources else "无"
