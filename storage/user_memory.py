"""
Paper Agent - User Memory System
动态用户记忆：LLM 智能提取 + 形成 → 强化 → 合并 → 衰减 → 遗忘
"""

import math
import json
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Optional
from pymongo import MongoClient, ASCENDING, DESCENDING

logger = logging.getLogger("paper-agent")


class UserMemory:
    """动态用户记忆 - LLM 智能提取"""

    DECAY_RATE = 0.05
    FORGET_THRESHOLD = 0.1
    MERGE_THRESHOLD = 0.8

    def __init__(self, db, llm=None):
        self.db = db
        self.collection = db["user_memory"]
        self.llm = llm  # LLM 用于智能提取
        self._ensure_indexes()

    def _ensure_indexes(self):
        self.collection.create_index([("user_id", ASCENDING)])
        self.collection.create_index([("user_id", ASCENDING), ("type", ASCENDING)])
        self.collection.create_index([("user_id", ASCENDING), ("weight", DESCENDING)])

    # ==================== LLM 智能提取 ====================

    def extract_with_llm(self, query: str, context: str = None) -> dict:
        """用 LLM 从用户输入中提取记忆"""
        if not self.llm:
            return self._extract_with_rules(query)

        prompt = f"""分析用户查询，提取关键信息用于记忆。

用户查询: {query}
对话上下文: {context or "无"}

输出JSON格式（不要其他内容）:
{{
    "interests": ["研究兴趣1", "研究兴趣2"],  // 学术领域、方法、概念
    "focus": "当前关注点",  // 方法/实验/对比/理论 等
    "pattern": "问题模式",  // definition/reasoning/comparison/evaluation
    "importance": 0-10,  // 查询的重要性（深度、专业性）
    "entities": ["实体1", "实体2"],  // 提到的论文、模型、数据集
    "intent": "用户意图"  // 想了解什么、想做什么
}}

示例:
查询: "Transformer的Self-Attention和RNN的LSTM有什么区别"
输出: {{"interests": ["transformer", "attention mechanism", "sequence modeling"], "focus": "comparison", "pattern": "comparison", "importance": 7, "entities": ["Transformer", "LSTM", "Self-Attention"], "intent": "比较两种序列建模方法的差异"}}"""

        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()

            # 提取 JSON
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())

        except Exception as e:
            logger.warning(f"[UserMemory] LLM 提取失败，降级到规则: {e}")

        return self._extract_with_rules(query)

    def _extract_with_rules(self, query: str) -> dict:
        """规则降级提取"""
        interests = []
        interest_map = {
            "PINN": "physics-informed neural network",
            "CFD": "computational fluid dynamics",
            "Transformer": "transformer",
            "BERT": "BERT",
            "GPT": "GPT",
            "attention": "attention mechanism",
            "CNN": "CNN",
            "RNN": "RNN",
            "LSTM": "LSTM",
        }
        for k, v in interest_map.items():
            if k.lower() in query.lower():
                interests.append(v)

        focus = None
        for kw, f in {"方法": "methodology", "实验": "experiment", "对比": "comparison"}.items():
            if kw in query:
                focus = f
                break

        return {
            "interests": interests,
            "focus": focus,
            "pattern": None,
            "importance": 5,
            "entities": [],
            "intent": query,
        }

    # ==================== 形成（Formation） ====================

    def form_from_query(self, user_id: str, query: str, context: str = None) -> list[dict]:
        """从用户查询中提取记忆"""
        # LLM 智能提取
        extracted = self.extract_with_llm(query, context)

        memories = []

        # 提取兴趣
        for interest in extracted.get("interests", []):
            mem = self._create_or_reinforce(user_id, "interest", interest,
                                           importance=extracted.get("importance", 5))
            memories.append(mem)

        # 提取关注点
        focus = extracted.get("focus")
        if focus:
            mem = self._create_or_reinforce(user_id, "focus", focus,
                                           importance=extracted.get("importance", 5))
            memories.append(mem)

        # 提取问题模式
        pattern = extracted.get("pattern")
        if pattern:
            mem = self._create_or_reinforce(user_id, "pattern", pattern,
                                           importance=extracted.get("importance", 5))
            memories.append(mem)

        # 提取实体
        for entity in extracted.get("entities", []):
            mem = self._create_or_reinforce(user_id, "entity", entity,
                                           importance=extracted.get("importance", 5))
            memories.append(mem)

        # 提取意图
        intent = extracted.get("intent")
        if intent:
            mem = self._create_or_reinforce(user_id, "intent", intent,
                                           importance=extracted.get("importance", 5))
            memories.append(mem)

        logger.info(f"[UserMemory] LLM 提取: {len(memories)} 条记忆")
        return memories

    # ==================== 强化（Reinforcement） ====================

    def _create_or_reinforce(self, user_id: str, mem_type: str, content: str,
                            importance: float = 5.0) -> dict:
        """创建新记忆或强化已有记忆"""
        existing = self.collection.find_one({
            "user_id": user_id,
            "type": mem_type,
            "content": content,
        })

        # 根据重要性调整强化幅度
        reinforce_amount = importance / 5.0  # 重要性越高，强化越多

        if existing:
            new_weight = min(existing["weight"] + reinforce_amount, 10.0)
            self.collection.update_one(
                {"_id": existing["_id"]},
                {"$set": {
                    "weight": new_weight,
                    "last_accessed": datetime.now(timezone.utc),
                    "access_count": existing.get("access_count", 0) + 1,
                }}
            )
            return {**existing, "weight": new_weight}
        else:
            doc = {
                "user_id": user_id,
                "type": mem_type,
                "content": content,
                "weight": importance / 5.0,  # 重要性作为初始权重
                "created_at": datetime.now(timezone.utc),
                "last_accessed": datetime.now(timezone.utc),
                "access_count": 1,
            }
            result = self.collection.insert_one(doc)
            doc["_id"] = result.inserted_id
            return doc

    # ==================== 合并（Merging） ====================

    def merge_similar(self, user_id: str):
        """合并相似记忆"""
        memories = list(self.collection.find({"user_id": user_id}))

        by_type = {}
        for mem in memories:
            t = mem["type"]
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(mem)

        merged_count = 0
        for mem_type, group in by_type.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    m1, m2 = group[i], group[j]
                    similarity = self._calculate_similarity(m1["content"], m2["content"])

                    if similarity >= self.MERGE_THRESHOLD:
                        if m1["weight"] >= m2["weight"]:
                            keep, delete = m1, m2
                        else:
                            keep, delete = m2, m1

                        self.collection.update_one(
                            {"_id": keep["_id"]},
                            {"$set": {"weight": keep["weight"] + 0.5}}
                        )
                        self.collection.delete_one({"_id": delete["_id"]})
                        merged_count += 1

        return merged_count

    def _calculate_similarity(self, a: str, b: str) -> float:
        """简单相似度计算"""
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        if not a_words or not b_words:
            return 0.0
        intersection = a_words & b_words
        union = a_words | b_words
        return len(intersection) / len(union)

    # ==================== 衰减（Decay） ====================

    def decay_all(self, user_id: str):
        """对所有记忆进行衰减"""
        now = datetime.now(timezone.utc)
        memories = list(self.collection.find({"user_id": user_id}))

        for mem in memories:
            last_accessed = mem.get("last_accessed", now)
            if isinstance(last_accessed, datetime):
                # 确保时区一致
                if last_accessed.tzinfo is None:
                    last_accessed = last_accessed.replace(tzinfo=timezone.utc)
                days_elapsed = (now - last_accessed).total_seconds() / 86400
            else:
                days_elapsed = 0

            decay_factor = (1 - self.DECAY_RATE) ** days_elapsed
            new_weight = mem["weight"] * decay_factor

            self.collection.update_one(
                {"_id": mem["_id"]},
                {"$set": {"weight": round(new_weight, 3)}}
            )

    # ==================== 遗忘（Forgetting） ====================

    def forget_low_weight(self, user_id: str) -> int:
        """删除低权重记忆"""
        result = self.collection.delete_many({
            "user_id": user_id,
            "weight": {"$lt": self.FORGET_THRESHOLD}
        })
        return result.deleted_count

    # ==================== 查询 ====================

    def get_interests(self, user_id: str, top_k: int = 10) -> list[str]:
        """获取用户最感兴趣的主题"""
        memories = self.collection.find({
            "user_id": user_id,
            "type": "interest",
        }).sort("weight", DESCENDING).limit(top_k)
        return [m["content"] for m in memories]

    def get_focus(self, user_id: str) -> Optional[str]:
        """获取用户当前关注点"""
        mem = self.collection.find_one(
            {"user_id": user_id, "type": "focus"},
            sort=[("weight", DESCENDING)]
        )
        return mem["content"] if mem else None

    def get_entities(self, user_id: str, top_k: int = 20) -> list[str]:
        """获取用户关注的实体"""
        memories = self.collection.find({
            "user_id": user_id,
            "type": "entity",
        }).sort("weight", DESCENDING).limit(top_k)
        return [m["content"] for m in memories]

    def get_all(self, user_id: str) -> list[dict]:
        """获取用户所有记忆"""
        return list(self.collection.find({"user_id": user_id}).sort("weight", DESCENDING))

    def get_stats(self, user_id: str) -> dict:
        """获取记忆统计"""
        memories = list(self.collection.find({"user_id": user_id}))
        total_weight = sum(m["weight"] for m in memories)
        return {
            "total_memories": len(memories),
            "total_weight": round(total_weight, 2),
            "avg_weight": round(total_weight / len(memories), 2) if memories else 0,
        }

    # ==================== 完整生命周期 ====================

    def process_interaction(self, user_id: str, query: str, context: str = None):
        """处理一次用户交互的完整生命周期"""
        # 1. 形成（LLM 提取）
        self.form_from_query(user_id, query, context)

        # 2. 衰减
        self.decay_all(user_id)

        # 3. 合并
        self.merge_similar(user_id)

        # 4. 遗忘
        self.forget_low_weight(user_id)
