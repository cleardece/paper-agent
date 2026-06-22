"""
Paper Agent - Memory System
六层记忆架构
"""

from datetime import datetime, timezone
from typing import Optional
from pymongo import MongoClient, ASCENDING, DESCENDING
import logging

logger = logging.getLogger("paper-agent")


class PaperMemory:
    """论文记忆 - 每篇论文一个知识对象"""

    def __init__(self, db):
        self.db = db
        self.collection = db["paper_memory"]
        self._ensure_indexes()

    def _ensure_indexes(self):
        self.collection.create_index([("arxiv_id", ASCENDING)], unique=True)
        self.collection.create_index([("keywords", ASCENDING)])
        self.collection.create_index([("year", DESCENDING)])

    def create(self, paper: dict) -> str:
        """创建论文记忆"""
        doc = {
            "arxiv_id": paper["arxiv_id"],
            "title": paper.get("title", ""),
            "authors": paper.get("authors", []),
            "year": paper.get("year"),
            "abstract_summary": paper.get("abstract", ""),

            # 结构化 sections
            "sections": paper.get("sections", []),  # [{heading, summary, embedding}]

            # 知识提取
            "contributions": paper.get("contributions", []),
            "limitations": paper.get("limitations", []),
            "datasets": paper.get("datasets", []),
            "metrics": paper.get("metrics", []),
            "keywords": paper.get("keywords", []),
            "references": paper.get("references", []),

            # 元数据
            "status": "parsed",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        self.collection.update_one(
            {"arxiv_id": doc["arxiv_id"]},
            {"$set": doc},
            upsert=True,
        )
        return doc["arxiv_id"]

    def get(self, arxiv_id: str) -> Optional[dict]:
        """获取论文记忆"""
        return self.collection.find_one({"arxiv_id": arxiv_id})

    def update(self, arxiv_id: str, updates: dict):
        """更新论文记忆"""
        updates["updated_at"] = datetime.now(timezone.utc)
        self.collection.update_one(
            {"arxiv_id": arxiv_id},
            {"$set": updates},
        )

    def list_all(self, limit: int = 100) -> list[dict]:
        """列出所有论文记忆"""
        return list(self.collection.find().limit(limit))

    def search_by_keywords(self, keywords: list[str]) -> list[dict]:
        """按关键词搜索"""
        return list(self.collection.find({"keywords": {"$in": keywords}}))

    def delete(self, arxiv_id: str):
        """删除论文记忆"""
        self.collection.delete_one({"arxiv_id": arxiv_id})


class SectionMemory:
    """Section 记忆 - 层级检索的中间层"""

    def __init__(self, db):
        self.db = db
        self.collection = db["section_memory"]
        self._ensure_indexes()

    def _ensure_indexes(self):
        self.collection.create_index([("paper_arxiv_id", ASCENDING)])
        self.collection.create_index([("paper_arxiv_id", ASCENDING), ("section_index", ASCENDING)])

    def create(self, paper_arxiv_id: str, sections: list[dict]):
        """为论文创建 section 记忆"""
        docs = []
        for i, section in enumerate(sections):
            docs.append({
                "paper_arxiv_id": paper_arxiv_id,
                "section_index": i,
                "heading": section.get("heading", ""),
                "summary": section.get("summary", ""),  # LLM 生成的摘要
                "content_preview": section.get("content", "")[:500],  # 前500字符
                "chunk_count": section.get("chunk_count", 0),
            })
        # 删除旧的 sections
        self.collection.delete_many({"paper_arxiv_id": paper_arxiv_id})
        if docs:
            self.collection.insert_many(docs)

    def get_by_paper(self, paper_arxiv_id: str) -> list[dict]:
        """获取论文的所有 sections"""
        return list(self.collection.find(
            {"paper_arxiv_id": paper_arxiv_id},
            sort=[("section_index", ASCENDING)]
        ))

    def delete(self, paper_arxiv_id: str):
        """删除论文的所有 sections"""
        self.collection.delete_many({"paper_arxiv_id": paper_arxiv_id})


class ConceptMemory:
    """概念记忆 - 跨论文知识融合"""

    def __init__(self, db):
        self.db = db
        self.collection = db["concept_memory"]
        self._ensure_indexes()

    def _ensure_indexes(self):
        self.collection.create_index([("name", ASCENDING)], unique=True)
        self.collection.create_index([("aliases", ASCENDING)])

    def create(self, concept: dict) -> str:
        """创建或更新概念"""
        name = concept["name"]
        self.collection.update_one(
            {"name": name},
            {"$set": {
                "name": name,
                "aliases": concept.get("aliases", []),
                "definition": concept.get("definition", ""),
                "formulas": concept.get("formulas", []),
                "source_papers": concept.get("source_papers", []),
            }},
            upsert=True,
        )
        return name

    def get(self, name: str) -> Optional[dict]:
        """获取概念"""
        return self.collection.find_one({"name": name})

    def search(self, query: str) -> list[dict]:
        """搜索概念"""
        return list(self.collection.find({
            "$or": [
                {"name": {"$regex": query, "$options": "i"}},
                {"aliases": {"$regex": query, "$options": "i"}},
            ]
        }))

    def add_source_paper(self, name: str, paper_arxiv_id: str):
        """添加概念的来源论文"""
        self.collection.update_one(
            {"name": name},
            {"$addToSet": {"source_papers": paper_arxiv_id}},
        )


class ReflectionMemory:
    """反思记忆 - Agent 自动生成的洞察"""

    def __init__(self, db):
        self.db = db
        self.collection = db["reflection_memory"]
        self._ensure_indexes()

    def _ensure_indexes(self):
        self.collection.create_index([("paper_arxiv_id", ASCENDING)])
        self.collection.create_index([("created_at", DESCENDING)])

    def create(self, paper_arxiv_id: str, reflection: dict):
        """创建反思记忆"""
        doc = {
            "paper_arxiv_id": paper_arxiv_id,
            "insights": reflection.get("insights", []),
            "unanswered_questions": reflection.get("unanswered_questions", []),
            "future_directions": reflection.get("future_directions", []),
            "connections": reflection.get("connections", []),  # 与其他论文的关联
            "created_at": datetime.now(timezone.utc),
        }
        self.collection.insert_one(doc)

    def get_by_paper(self, paper_arxiv_id: str) -> list[dict]:
        """获取论文的反思记录"""
        return list(self.collection.find(
            {"paper_arxiv_id": paper_arxiv_id},
            sort=[("created_at", DESCENDING)]
        ))

    def get_recent(self, limit: int = 10) -> list[dict]:
        """获取最近的反思"""
        return list(self.collection.find().sort("created_at", DESCENDING).limit(limit))


class MemoryManager:
    """记忆管理器 - 统一管理所有记忆层"""

    def __init__(self, mongodb_client, llm=None, knowledge_graph=None):
        self.db = mongodb_client.db
        self.paper = PaperMemory(self.db)
        self.section = SectionMemory(self.db)
        self.concept = ConceptMemory(self.db)
        self.reflection = ReflectionMemory(self.db)
        self.kg = knowledge_graph
        self.llm = llm

    def process_paper_slow_path(self, arxiv_id: str, paper_data: dict, chunks: list[dict]):
        """
        Slow Path: 论文入库后的慢处理流程
        调用 LLM 生成 Section Summary → Paper Summary → Concept Extraction
        """
        if not self.llm:
            logger.warning("[Memory] 无 LLM，跳过 Slow Path")
            return

        from agents.summarizer import SummarizerAgent
        from agents.concept_extractor import ConceptExtractorAgent
        from agents.importance_evaluator import ImportanceEvaluator

        summarizer = SummarizerAgent(self.llm)
        concept_extractor = ConceptExtractorAgent(self.llm)
        evaluator = ImportanceEvaluator()

        title = paper_data.get("title", "Unknown")
        logger.info(f"[Memory] Slow Path 开始: {title[:50]}...")

        # 1. Section Summary（每个 section 一次 LLM）
        sections = paper_data.get("sections", [])
        section_summaries = []

        for section in sections:
            heading = section.get("heading", "")
            content = section.get("content", "")

            # 生成 section 摘要
            summary = summarizer.summarize_section(heading, content)
            summary["heading"] = heading
            section_summaries.append(summary)

            # 提取概念
            concepts = concept_extractor.extract(heading, content)

            # 评估重要性并存储
            for concept in concepts:
                importance = evaluator.evaluate(concept)
                if evaluator.should_store(importance):
                    self.concept.create({
                        "name": concept["name"],
                        "definition": concept.get("definition", ""),
                        "aliases": concept.get("aliases", []),
                        "source_papers": [arxiv_id],
                    })

                    # 添加到 Knowledge Graph
                    if self.kg:
                        self.kg.add_node(concept["name"], "Concept", {
                            "definition": concept.get("definition", ""),
                        })
                        self.kg.add_edge(arxiv_id, concept["name"], "proposes")

        # 2. Paper Summary（一次 LLM）
        paper_summary = summarizer.summarize_paper(title, section_summaries)

        # 3. 创建 Paper Memory
        self.paper.create({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": paper_data.get("authors", []),
            "year": paper_data.get("year"),
            "abstract": paper_data.get("abstract", ""),
            "sections": section_summaries,
            "contributions": paper_summary.get("contributions", []),
            "limitations": paper_summary.get("limitations", []),
            "datasets": paper_summary.get("datasets", []),
            "metrics": paper_summary.get("metrics", []),
            "keywords": paper_summary.get("keywords", []),
            "references": paper_data.get("references", []),
        })

        # 4. 创建 Section Memory
        self.section.create(arxiv_id, [
            {"heading": s["heading"], "summary": s.get("summary", ""), "chunk_count": 0}
            for s in section_summaries
        ])

        logger.info(f"[Memory] Slow Path 完成: {len(section_summaries)} sections, {len(paper_summary.get('keywords', []))} keywords")

    def init_paper_memory(self, arxiv_id: str, paper_data: dict):
        """初始化论文的完整记忆（Fast Path）"""
        # 1. 创建 Paper Memory（基础信息）
        self.paper.create({
            "arxiv_id": arxiv_id,
            "title": paper_data.get("title", ""),
            "authors": paper_data.get("authors", []),
            "year": paper_data.get("year"),
            "abstract": paper_data.get("abstract", ""),
            "sections": [],
            "contributions": [],
            "limitations": [],
            "datasets": [],
            "metrics": [],
            "keywords": [],
            "references": [],
        })

        # 2. 创建 Section Memory
        sections = paper_data.get("sections", [])
        if sections:
            self.section.create(arxiv_id, sections)

        # 3. 添加到 Knowledge Graph
        if self.kg:
            self.kg.add_node(arxiv_id, "Paper", {
                "title": paper_data.get("title", ""),
            })

        logger.info(f"[Memory] Fast Path 初始化: {arxiv_id}")

    def delete_paper_memory(self, arxiv_id: str):
        """删除论文的所有记忆"""
        self.paper.delete(arxiv_id)
        self.section.delete(arxiv_id)
        self.concept.db.delete_many({"source_papers": arxiv_id})
        self.reflection.db.delete_many({"paper_arxiv_id": arxiv_id})

        # 从 Knowledge Graph 删除
        if self.kg:
            # 删除论文节点和相关边
            neighbors = self.kg.get_neighbors(arxiv_id)
            for n in neighbors:
                # 删除 concepts 的边
                pass
            # 简单处理：删除节点
            if self.kg.use_neo4j:
                from neo4j import GraphDatabase
                with self.kg.driver.session() as session:
                    session.run("MATCH (n {id: $id}) DETACH DELETE n", id=arxiv_id)
            else:
                self.kg.nodes.pop(arxiv_id, None)
                self.kg.edges = [(f, t, r) for f, t, r in self.kg.edges if f != arxiv_id and t != arxiv_id]

        logger.info(f"[Memory] 删除论文记忆: {arxiv_id}")
