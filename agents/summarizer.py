"""
Paper Agent - Summarizer Agent
Slow Path: Section Summary + Paper Summary
"""

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger("paper-agent")


SECTION_SUMMARY_PROMPT = """你是一个学术论文分析专家。请为以下论文章节生成简洁摘要。

## 输出格式（严格JSON）
{
    "summary": "2-3句话的摘要",
    "key_points": ["要点1", "要点2", ...],
    "concepts": ["概念1", "概念2", ...]
}

## 要求
1. summary: 概括章节核心内容，不超过100字
2. key_points: 提取3-5个关键点
3. concepts: 提取专业概念（用于Concept Memory）"""

PAPER_SUMMARY_PROMPT = """你是一个学术论文分析专家。请根据各章节摘要，生成论文的整体总结。

## 各章节摘要
{section_summaries}

## 输出格式（严格JSON）
{
    "contributions": ["贡献1", "贡献2", ...],
    "limitations": ["局限性1", ...],
    "datasets": ["数据集1", ...],
    "metrics": ["指标1", ...],
    "keywords": ["关键词1", ...],
    "abstract_summary": "论文摘要（50字内）"
}

## 要求
1. contributions: 提取2-3个核心贡献
2. limitations: 提取1-2个局限性
3. datasets: 提到的数据集
4. metrics: 实验指标
5. keywords: 3-5个关键词"""


class SummarizerAgent:
    """总结 Agent - Section Summary + Paper Summary"""

    def __init__(self, llm):
        self.llm = llm

    def summarize_section(self, heading: str, content: str) -> dict:
        """为单个 Section 生成摘要"""
        prompt = f"""章节标题: {heading}

章节内容:
{content[:3000]}

请生成摘要（严格JSON格式）:"""

        try:
            messages = [
                SystemMessage(content=SECTION_SUMMARY_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            content = response.content.strip()

            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())

        except Exception as e:
            logger.error(f"[Summarizer] Section 摘要失败: {e}")

        return {"summary": "", "key_points": [], "concepts": []}

    def summarize_paper(self, paper_title: str, section_summaries: list[dict]) -> dict:
        """为整篇论文生成摘要"""
        # 构建章节摘要文本
        summaries_text = "\n".join([
            f"### {s.get('heading', 'Unknown')}\n{s.get('summary', '')}"
            for s in section_summaries
        ])

        prompt = f"""论文标题: {paper_title}

{summaries_text}

请生成论文总结（严格JSON格式）:"""

        try:
            messages = [
                SystemMessage(content=PAPER_SUMMARY_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            content = response.content.strip()

            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())

        except Exception as e:
            logger.error(f"[Summarizer] Paper 摘要失败: {e}")

        return {
            "contributions": [],
            "limitations": [],
            "datasets": [],
            "metrics": [],
            "keywords": [],
            "abstract_summary": "",
        }


import re
