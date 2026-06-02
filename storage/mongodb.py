"""
Paper Agent - MongoDB存储层
存储论文元数据、解析状态、对话记录
"""

from datetime import datetime, timezone
from typing import Optional
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database


class MongoDBClient:
    """MongoDB客户端 - 管理论文元数据"""

    def __init__(self, uri: str = "mongodb://localhost:27017", db_name: str = "paper_agent"):
        self.client = MongoClient(uri)
        self.db: Database = self.client[db_name]
        self._ensure_indexes()

    def _ensure_indexes(self):
        """创建索引，幂等操作"""
        self.papers.create_index([("arxiv_id", ASCENDING)], unique=True)
        self.papers.create_index([("title", ASCENDING)])
        self.papers.create_index([("status", ASCENDING)])
        self.papers.create_index([("created_at", DESCENDING)])

        self.chunks.create_index([("paper_arxiv_id", ASCENDING)])
        self.chunks.create_index([("chunk_index", ASCENDING)])

        self.conversations.create_index([("session_id", ASCENDING)])
        self.conversations.create_index([("created_at", DESCENDING)])

    @property
    def papers(self) -> Collection:
        return self.db["papers"]

    @property
    def chunks(self) -> Collection:
        return self.db["chunks"]

    @property
    def conversations(self) -> Collection:
        return self.db["conversations"]

    # ==================== 论文操作 ====================

    def upsert_paper(self, paper: dict) -> str:
        now = datetime.now(timezone.utc)
        paper.setdefault("status", "pending")
        paper.setdefault("created_at", now)
        paper["updated_at"] = now
        self.papers.update_one(
            {"arxiv_id": paper["arxiv_id"]},
            {"$set": paper},
            upsert=True,
        )
        return paper["arxiv_id"]

    def update_paper_status(self, arxiv_id: str, status: str, **extra_fields):
        update = {"$set": {"status": status, "updated_at": datetime.now(timezone.utc)}}

        if extra_fields:
            update["$set"].update(extra_fields)
            self.papers.update_one({"arxiv_id": arxiv_id}, update)

    def get_paper(self, arxiv_id: str) -> Optional[dict]:
        return self.papers.find_one({"arxiv_id": arxiv_id})

    def get_papers_by_status(self, status: str, limit: int = 100) -> list[dict]:
        return list(self.papers.find({"status": status}).limit(limit))

    def list_papers(self, limit: int = 50, skip: int = 0) -> list[dict]:
        return list(self.papers.find().sort("created_at", DESCENDING).skip(skip).limit(limit))

    def delete_paper(self, arxiv_id: str) -> bool:
        paper_result = self.papers.delete_one({"arxiv_id": arxiv_id})
        self.chunks.delete_many({"paper_arxiv_id": arxiv_id})
        return paper_result.deleted_count > 0

    def paper_exists(self, arxiv_id: str) -> bool:
        return self.papers.count_documents({"arxiv_id": arxiv_id}, limit=1) > 0

    def count_papers(self, status: Optional[str] = None) -> int:
        query = {"status": status} if status else {}
        return self.papers.count_documents(query)

    # ==================== 分块操作 ====================

    def insert_chunks(self, chunks: list[dict]) -> int:
        if not chunks:
            return 0
        result = self.chunks.insert_many(chunks, ordered=False)
        return len(result.inserted_ids)

    def get_chunks_by_paper(self, arxiv_id: str) -> list[dict]:
        return list(self.chunks.find({"paper_arxiv_id": arxiv_id}).sort("chunk_index", ASCENDING))

    def get_all_chunks(self, limit: int = 10000) -> list[dict]:
        return list(self.chunks.find().limit(limit))

    def count_chunks(self, arxiv_id: Optional[str] = None) -> int:
        query = {"paper_arxiv_id": arxiv_id} if arxiv_id else {}
        return self.chunks.count_documents(query)

    # ==================== 对话操作 ====================

    def save_message(self, session_id: str, role: str, content: str, metadata: dict = None):
        doc = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": datetime.now(timezone.utc),
        }
        if metadata:
            doc["metadata"] = metadata
        self.conversations.insert_one(doc)

    def get_conversation(self, session_id: str, limit: int = 50) -> list[dict]:
        return list(
            self.conversations.find({"session_id": session_id})
            .sort("created_at", ASCENDING)
            .limit(limit)
        )

        # ==================== 统计 ====================

    def get_stats(self) -> dict:
        return {
            "total_papers": self.count_papers(),
            "papers_by_status": {
                status: self.count_papers(status)
                for status in ["pending", "parsed", "chunked", "embedded", "indexed"]
            },
            "total_chunks": self.count_chunks(),
        }

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

