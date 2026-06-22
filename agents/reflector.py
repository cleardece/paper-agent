"""
Paper Agent - Reflection Agent
分析完论文后自动生成洞察、问题、未来方向
"""

import logging
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger("paper-agent")


REFLECTOR_PROMPT = """你是一个学术研究助手。分析完一篇论文后，自动生成反思记忆。

## 输出格式（严格JSON）

{
  "insights": ["洞察1", "洞察2", ...],
  "unanswered_questions": ["未解决的问题1", ...],
  "future_directions": ["未来研究方向1", ...],
  "connections": ["与XX论文的关联1", ...]
}

## 要求

1. **insights**: 从论文中提取 3-5 个核心洞察（不是摘要，是深层理解）
2. **unanswered_questions**: 提出 2-3 个论文未回答的问题
3. **future_directions**: 基于论文提出 2-3 个未来研究方向
4. **connections**: 如果检索到其他论文，指出关联

## 示例

论文：Attention Is All You Need

{
  "insights": [
    "Self-attention 比 RNN 更适合捕捉长距离依赖",
    "Multi-head attention 允许模型同时关注不同位置的不同表示子空间",
    "位置编码是 Transformer 处理序列信息的关键"
  ],
  "unanswered_questions": [
    "为什么 LayerNorm 放在残差连接之后而不是之前？",
    "Transformer 在极长序列上的表现如何？"
  ],
  "future_directions": [
    "探索更高效的位置编码方法",
    "研究 Transformer 在不同领域的迁移能力"
  ],
  "connections": [
    "BERT 使用了 Encoder-only 架构",
    "GPT 使用了 Decoder-only 架构"
  ]
}"""


class ReflectorAgent:
    """反思 Agent - 自动生成论文洞察"""

    def __init__(self, llm):
        self.llm = llm

    def invoke(self, paper_title: str, analysis: str, retrieved_chunks: list) -> dict:
        """
        生成反思记忆

        Args:
            paper_title: 论文标题
            analysis: 分析结果
            retrieved_chunks: 检索到的 chunks（用于发现关联）

        Returns:
            反思记忆字典
        """
        # 构建上下文
        chunk_summary = "\n".join([
            f"- [{c.get('paper_title', '')}] {c.get('content', '')[:100]}..."
            for c in retrieved_chunks[:5]
        ])

        prompt = f"""论文标题：{paper_title}

分析结果：
{analysis}

检索到的其他论文片段：
{chunk_summary if chunk_summary else "无"}

请生成反思记忆（严格JSON格式）："""

        try:
            messages = [
                SystemMessage(content=REFLECTOR_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            content = response.content.strip()

            # 提取 JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                import json
                reflection = json.loads(json_match.group())
                logger.info(f"[Reflector] 生成反思: {len(reflection.get('insights', []))} 个洞察")
                return reflection

        except Exception as e:
            logger.error(f"[Reflector] 生成反思失败: {e}")

        # 默认返回
        return {
            "insights": [],
            "unanswered_questions": [],
            "future_directions": [],
            "connections": [],
        }
