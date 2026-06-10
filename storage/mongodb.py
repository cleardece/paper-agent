"""
Paper Agent - MongoDB存储层
存储论文元数据、解析状态、对话记录
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database

logger = logging.getLogger("paper-agent")


class MongoDBClient:
    """MongoDB客户端 - 管理论文元数据"""

    def __init__(self, uri: str = "mongodb://localhost:27017", db_name: str = "paper_agent"):
        logger.info(f"[MongoDB] 正在连接 {uri}...")
        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # 测试连接
        self.client.admin.command('ping')
        logger.info("[MongoDB] 连接成功")
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

    # ==================== Session 持久化 ====================

    def save_session(self, session_id: str, title: str, messages: list[dict] = None, updated_at: float = None):
        """保存 Session 到 MongoDB"""
        doc = {
            "session_id": session_id,
            "title": title,
            "updated_at": updated_at or datetime.now(timezone.utc).timestamp(),
            "created_at": datetime.now(timezone.utc),
        }
        if messages:
            doc["messages"] = messages
        self.db["sessions"].update_one(
            {"session_id": session_id},
            {"$set": doc},
            upsert=True,
        )

    def get_session(self, session_id: str) -> Optional[dict]:
        """从 MongoDB 获取 Session"""
        return self.db["sessions"].find_one({"session_id": session_id})

    def list_sessions(self, limit: int = 50) -> list[dict]:
        """列出所有 Session"""
        return list(
            self.db["sessions"].find()
            .sort("updated_at", DESCENDING)
            .limit(limit)
        )

    def delete_session(self, session_id: str) -> bool:
        """删除 Session 及其消息"""
        result = self.db["sessions"].delete_one({"session_id": session_id})
        self.conversations.delete_many({"session_id": session_id})
        return result.deleted_count > 0

    def delete_all_sessions(self) -> int:
        """删除所有 Session"""
        result = self.db["sessions"].delete_many({})
        self.conversations.delete_many({})
        return result.deleted_count

    # ==================== 收藏操作 ====================

    def add_favorite(self, paper_arxiv_id: str, tags: list[str] = None, notes: str = ""):
        """收藏论文"""
        doc = {
            "paper_arxiv_id": paper_arxiv_id,
            "tags": tags or [],
            "notes": notes,
            "created_at": datetime.now(timezone.utc),
        }
        self.db["favorites"].update_one(
            {"paper_arxiv_id": paper_arxiv_id},
            {"$set": doc},
            upsert=True,
        )

    def remove_favorite(self, paper_arxiv_id: str) -> bool:
        """取消收藏"""
        result = self.db["favorites"].delete_one({"paper_arxiv_id": paper_arxiv_id})
        return result.deleted_count > 0

    def is_favorite(self, paper_arxiv_id: str) -> bool:
        """是否已收藏"""
        return self.db["favorites"].count_documents({"paper_arxiv_id": paper_arxiv_id}, limit=1) > 0

    def list_favorites(self, tag: str = None) -> list[dict]:
        """列出收藏"""
        query = {"tags": tag} if tag else {}
        return list(self.db["favorites"].find(query).sort("created_at", DESCENDING))

    def update_favorite(self, paper_arxiv_id: str, tags: list[str] = None, notes: str = None):
        """更新收藏"""
        update = {}
        if tags is not None:
            update["tags"] = tags
        if notes is not None:
            update["notes"] = notes
        if update:
            self.db["favorites"].update_one(
                {"paper_arxiv_id": paper_arxiv_id},
                {"$set": update},
            )

    # ==================== 统计 ====================

    def get_stats(self) -> dict:
        return {
            "total_papers": self.count_papers(),
            "total_favorites": self.db["favorites"].count_documents({}),
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

