"""
Paper Agent - Importance Evaluator
Fast Path: 规则评估概念重要性，不调用LLM
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger("paper-agent")


class ImportanceEvaluator:
    """重要性评估器 - 纯规则，无LLM"""

    # 权重配置
    WEIGHTS = {
        "frequency": 0.4,      # 出现频率
        "citation_count": 0.3, # 引用次数
        "access_count": 0.2,   # 访问次数
        "recency": 0.1,        # 时效性
    }

    def evaluate(self, concept: dict, context: dict = None) -> float:
        """
        评估概念重要性

        Args:
            concept: 概念信息
            context: 上下文（论文信息等）

        Returns:
            importance_score: 0-1
        """
        context = context or {}

        # 1. 频率分（0-1）
        frequency = min(concept.get("frequency", 1) / 10, 1.0)

        # 2. 引用分（0-1）
        citations = concept.get("citation_count", 0)
        citation_score = min(citations / 100, 1.0) if citations > 0 else 0.1

        # 3. 访问分（0-1）
        access_count = concept.get("access_count", 0)
        access_score = min(access_count / 20, 1.0)

        # 4. 时效分（0-1）
        created_at = concept.get("created_at")
        if created_at and isinstance(created_at, datetime):
            days_old = (datetime.now(timezone.utc) - created_at).days
            recency_score = max(0, 1 - days_old / 365)  # 一年内衰减
        else:
            recency_score = 0.5

        # 加权计算
        score = (
            self.WEIGHTS["frequency"] * frequency +
            self.WEIGHTS["citation_count"] * citation_score +
            self.WEIGHTS["access_count"] * access_score +
            self.WEIGHTS["recency"] * recency_score
        )

        return round(min(score, 1.0), 3)

    def should_store(self, score: float, threshold: float = 0.3) -> bool:
        """判断是否值得存储"""
        return score >= threshold

    def get_weight(self, score: float) -> float:
        """将重要性分数转换为记忆权重（1-10）"""
        return max(1.0, min(score * 10, 10.0))
