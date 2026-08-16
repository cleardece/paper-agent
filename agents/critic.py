"""
Paper Agent - Critic Agent
评估Analyzer输出的答案质量，决定是否需要重试
"""

import json
import logging
import re

from langchain_core.messages import SystemMessage, HumanMessage
from core.evidence import validate_answer_evidence
from core.llm_utils import invoke_json_with_retry
from state.graph_state import AgentState

logger = logging.getLogger("paper-agent")


CRITIC_PROMPT = """你是学术论文问答的独立质量审核员。

## 你的角色
你是一个**第三方评审**，独立评估 AI 生成的回答质量。
- 你不知道 AI 是怎么得出这个结论的（没有看到推理过程）
- 你只能基于**检索到的原始论文片段**来判断回答是否正确
- 你的唯一任务是：**回答中的每个事实，能在检索片段中找到依据吗？**

## 检查清单（按重要性排序）

### 1. 幻觉检测（最重要）
- 逐句检查回答中的事实性断言（数据、方法、结论、引用）
- 每个断言必须在检索片段中有**明确文字依据**
- 找不到依据的 = 幻觉，必须列出

### 2. 引用准确性
- 回答中提到的论文标题、作者、数据集、指标是否与检索片段一致？
- 是否存在张冠李戴（把 A 论文的结论归到 B 论文）？

### 3. 切题度
- 回答是否直接回应了用户问题的核心？
- 是否有跑题或无关内容？

### 4. 完整性
- 用户问题涉及的方面，回答覆盖了几个？
- 检索片段中有明显相关信息但回答未提及 = 遗漏

### 5. 逻辑连贯性
- 回答内部是否存在自相矛盾？
- 多个论点之间是否一致？

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
- **默认倾向 pass**，除非回答有明显错误
- 检索片段本身不完整导致的遗漏 ≠ 幻觉，不扣 faithfulness 分
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

        evidence_report = validate_answer_evidence(analysis, retrieved_chunks)
        if evidence_report["status"] == "retry":
            logger.warning(
                "[Critic] 证据规则校验未通过: %s",
                evidence_report["reason"],
            )
            if iteration < max_iterations - 1:
                return {
                    "critic_score": {
                        "score": 0,
                        "verdict": "revise",
                        "suggestions": [evidence_report["reason"]],
                    },
                    "evidence_report": evidence_report,
                    "next_agent": "retriever",
                    "iteration": iteration + 1,
                }

        context = "\n".join(
            f"[{c['paper_title']}] {c['content'][:300]}"
            for c in retrieved_chunks[:5]
        )

        eval_prompt = f"""请独立评估以下学术问答质量。

## 你的任务
你是第三方评审，**没有看到 AI 的推理过程**，只能基于检索到的原始论文片段来判断回答是否正确。

## 评估方法
1. 仔细阅读用户问题
2. 阅读检索到的论文片段（这是你能看到的全部证据）
3. 阅读 AI 生成的回答
4. 逐句检查：回答中的每个事实性断言，在检索片段中能找到明确依据吗？

## 注意
- 回答中可能包含"结论"和"分析"两部分，都需要检查
- 如果检索片段本身不包含某个信息，回答却写了 → 幻觉
- 如果检索片段有信息但回答没提 → 遗漏（扣 completeness 分，不算幻觉）

---

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
        evaluation = invoke_json_with_retry(
            self.llm, messages, max_retries=2,
            fallback={"score": 60, "verdict": "pass", "suggestions": ["LLM 评估失败，默认通过"]},
        )
        verdict = evaluation.get("verdict", "pass")
        logger.info(f"[Critic] 评估完成: 分数={evaluation.get('score', 'N/A')}, 判定={verdict}")

        # 使用 max_iterations 而不是硬编码
        if verdict == "pass" or iteration >= max_iterations - 1:
            next_agent = "presenter"
        else:
            next_agent = "retriever"

        return {
            "critic_score": evaluation,
            "evidence_report": evidence_report,
            "next_agent": next_agent,
            "iteration": iteration + 1,
        }

    def _parse(self, raw: str) -> dict:
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

