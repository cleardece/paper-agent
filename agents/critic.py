"""
Paper Agent - Critic Agent
评估Analyzer输出的答案质量，决定是否需要重试
"""

import logging
from langchain_core.messages import SystemMessage, HumanMessage
from state.graph_state import AgentState

logger = logging.getLogger("paper-agent")


CRITIC_PROMPT = CRITIC_PROMPT = """你是学术论文问答的质量审核员。逐条检查以下回答，输出JSON评分。

## 检查清单
1. **幻觉检测**：回答中的每个事实性断言，是否能在检索片段中找到依据？列出所有无依据的断言。
2. **切题度**：回答是否直接回应了用户问题的核心？是否有跑题内容？
3. **完整性**：用户问题涉及的方面，回答覆盖了几个？遗漏了什么？
4. **引用准确性**：提到的论文标题、作者、结论是否与检索片段一致？
5. **逻辑连贯性**：推理链是否通顺？有无自相矛盾？

## 输出格式（严格JSON）
{
  "faithfulness": 0-10,
  "relevancy": 0-10,
  "completeness": 0-10,
  "citation_accuracy": 0-10,
  "coherence": 0-10,
  "hallucinations": ["无依据断言1", "无依据断言2"],
  "missing_aspects": ["遗漏方面1"],
  "suggestions": ["改进建议1"],
  "score": 0-100,
  "verdict": "pass" | "revise"
}

## 评分规则（重要：宽松标准，避免不必要的重试）
- score >= 60 且无严重幻觉 → pass
- 只有出现严重幻觉（编造数据、错误引用）时才 revise
- 完整性不足、措辞不完美不算 revise 理由
- 默认倾向 pass，除非回答有明显错误
"""


class CriticAgent:
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, state: AgentState) -> dict:
        analysis = state.get("analysis", "")
        retrieved_chunks = state.get("retrieved_chunks", [])
        user_query = state["user_query"]
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 3)

        logger.info(f"[Critic] 正在评估回答质量（第 {iteration + 1}/{max_iterations} 次）...")

        if not analysis:
            logger.info("[Critic] 无分析结果，要求重新检索")
            return {
                "critic_score": {"score": 0, "verdict": "revise", "suggestions": ["无分析结果"]},
                "next_agent": "retriever",
                "iteration": iteration + 1,
            }

        context = "\n".join(
            f"[{c['paper_title']}] {c['content'][:300]}"
            for c in retrieved_chunks[:5]
        )

        eval_prompt = f"""请评估以下学术问答质量。

用户问题：{user_query}

检索到的论文片段：
{context}

AI生成的回答：
{analysis}

请输出JSON评估结果。"""

        messages = [
            SystemMessage(content=CRITIC_PROMPT),
            HumanMessage(content=eval_prompt),
        ]

        logger.info("[Critic] 正在调用 LLM 进行评估...")
        response = self.llm.invoke(messages)
        evaluation = self._parse(response.content)
        verdict = evaluation.get("verdict", "revise")
        logger.info(f"[Critic] 评估完成: 分数={evaluation.get('score', 'N/A')}, 判定={verdict}")

        # 使用 max_iterations 而不是硬编码
        if verdict == "pass" or iteration >= max_iterations - 1:
            next_agent = "presenter"
        else:
            next_agent = "retriever"

        return {
            "critic_score": evaluation,
            "next_agent": next_agent,
            "iteration": iteration + 1,
        }

    def _parse(self, raw: str) -> dict:
        import json, re
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        score_match = re.search(r'"?score"?\s*[:=]\s*(\d+)', raw)
        verdict_match = re.search(r'"?verdict"?\s*[:=]\s*"(pass|revise)"', raw)
        return {
            "score": int(score_match.group(1)) if score_match else 50,
            "verdict": verdict_match.group(1) if verdict_match else "revise",
            "suggestions": ["解析失败，使用默认值"],
        }

