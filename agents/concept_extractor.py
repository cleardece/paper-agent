"""
Paper Agent - Concept Extractor
Slow Path: 从论文中提取概念，用于 Concept Memory
"""

import json
import re
import logging
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger("paper-agent")


CONCEPT_EXTRACTION_PROMPT = """你是一个学术概念提取专家。从论文内容中提取核心概念。

## 输出格式（严格JSON）
{
    "concepts": [
        {
            "name": "概念名称",
            "definition": "简短定义（一句话）",
            "aliases": ["别名1", "别名2"],
            "category": "method/metric/architecture/technique"
        }
    ]
}

## 要求
1. 只提取核心概念，不要提取普通词汇
2. name 使用英文标准名称
3. definition 不超过30字
4. aliases 包含常见别名
5. category 分类：method/metric/architecture/technique

## 示例
输入: "We use multi-head attention to jointly attend to information from different representation subspaces."
输出: {
    "concepts": [{
        "name": "Multi-head Attention",
        "definition": "同时关注不同表示子空间信息的注意力机制",
        "aliases": ["MHA", "多头注意力"],
        "category": "technique"
    }]
}"""


class ConceptExtractorAgent:
    """概念提取 Agent"""

    def __init__(self, llm):
        self.llm = llm

    def extract(self, heading: str, content: str) -> list[dict]:
        """从章节内容中提取概念"""
        prompt = f"""章节标题: {heading}

内容:
{content[:2000]}

请提取核心概念（严格JSON格式）:"""

        try:
            messages = [
                SystemMessage(content=CONCEPT_EXTRACTION_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            response_content = response.content.strip()

            json_match = re.search(r'\{[\s\S]*\}', response_content)
            if json_match:
                result = json.loads(json_match.group())
                concepts = result.get("concepts", [])
                logger.info(f"[ConceptExtractor] 提取 {len(concepts)} 个概念")
                return concepts

        except Exception as e:
            logger.error(f"[ConceptExtractor] 提取失败: {e}")

        return []

    def extract_from_paper(self, sections: list[dict]) -> list[dict]:
        """从整篇论文提取概念"""
        all_concepts = []
        seen_names = set()

        for section in sections:
            heading = section.get("heading", "")
            content = section.get("content", "")[:2000]

            concepts = self.extract(heading, content)
            for c in concepts:
                name = c.get("name", "")
                if name and name not in seen_names:
                    seen_names.add(name)
                    all_concepts.append(c)

        logger.info(f"[ConceptExtractor] 论文共提取 {len(all_concepts)} 个唯一概念")
        return all_concepts
