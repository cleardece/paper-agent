"""MongoDB-backed persistent queue for local PDF upload batches."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from pymongo import ASCENDING, DESCENDING, ReturnDocument


TERMINAL_STATUSES = {"completed", "failed", "skipped"}
PROCESSING_STATUSES = {"parsing", "chunking", "indexing"}


class UploadQueueRepository:
    """Persist batch jobs and atomically claim only one job at a time."""

    def __init__(self, db: Any):
        self.batches = db["upload_batches"]
        self.jobs = db["upload_jobs"]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.batches.create_index([("batch_id", ASCENDING)], unique=True)
        self.jobs.create_index([("job_id", ASCENDING)], unique=True)
        self.jobs.create_index([("batch_id", ASCENDING), ("sequence", ASCENDING)])
        self.jobs.create_index([("status", ASCENDING), ("created_at", ASCENDING), ("sequence", ASCENDING)])
        self.jobs.create_index([("arxiv_id", ASCENDING), ("status", ASCENDING)])
        self.jobs.create_index([("finished_at", DESCENDING)])

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def create_batch(self, batch_id: str, total_count: int) -> dict[str, Any]:
        now = self._now()
        batch = {
            "batch_id": batch_id,
            "total_count": total_count,
            "created_at": now,
            "updated_at": now,
        }
        self.batches.insert_one(batch)
        return batch

    def create_jobs(self, batch_id: str, jobs: list[dict[str, Any]]) -> None:
        now = self._now()
        docs = []
        for job in jobs:
            doc = dict(job)
            doc.setdefault("batch_id", batch_id)
            doc.setdefault("status", "queued")
            doc.setdefault("stage_detail", "等待处理")
            doc.setdefault("chunk_count", 0)
            doc.setdefault("parse_source", None)
            doc.setdefault("attempt_count", 0)
            doc.setdefault("max_attempts", 2)
            doc.setdefault("error", None)
            doc.setdefault("created_at", now)
            doc.setdefault("updated_at", now)
            doc.setdefault("finished_at", None)
            docs.append(doc)
        if docs:
            self.jobs.insert_many(docs)

    def count_pending(self) -> int:
        return self.jobs.count_documents({"status": {"$in": ["queued", *PROCESSING_STATUSES]}})

    def has_nonterminal_arxiv_id(self, arxiv_id: str) -> bool:
        return self.jobs.count_documents({
            "arxiv_id": arxiv_id,
            "status": {"$in": ["queued", *PROCESSING_STATUSES]},
        }) > 0

    def claim_next_job(self, batch_id: str | None = None) -> dict[str, Any] | None:
        now = self._now()
        query = {"status": "queued"}
        if batch_id:
            query["batch_id"] = batch_id
        return self.jobs.find_one_and_update(
            query,
            {
                "$set": {
                    "status": "parsing",
                    "stage_detail": "正在解析",
                    "updated_at": now,
                },
                "$inc": {"attempt_count": 1},
            },
            sort=[("created_at", ASCENDING), ("sequence", ASCENDING)],
            return_document=ReturnDocument.AFTER,
        )

    def mark_retrying(self, job_id: str, detail: str) -> None:
        self.jobs.update_one(
            {"job_id": job_id},
            {
                "$set": {"status": "parsing", "stage_detail": detail, "updated_at": self._now()},
                "$inc": {"attempt_count": 1},
            },
        )

    def update_job(self, job_id: str, status: str, *, detail: str = "", **fields: Any) -> None:
        now = self._now()
        values = {"status": status, "updated_at": now, **fields}
        if detail:
            values["stage_detail"] = detail
        if status in TERMINAL_STATUSES:
            values.setdefault("finished_at", now)
        self.jobs.update_one({"job_id": job_id}, {"$set": values})

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self.jobs.find_one({"job_id": job_id})

    def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        batch = self.batches.find_one({"batch_id": batch_id})
        if not batch:
            return None
        batch["jobs"] = list(self.jobs.find({"batch_id": batch_id}).sort([("sequence", ASCENDING)]))
        return batch

    def get_next_job_in_batch(self, batch_id: str) -> dict[str, Any] | None:
        return self.jobs.find_one(
            {"batch_id": batch_id, "status": "queued"},
            sort=[("sequence", ASCENDING)],
        )

    def is_multi_file_batch(self, batch_id: str) -> bool:
        batch = self.batches.find_one({"batch_id": batch_id})
        return bool(batch and batch.get("total_count", 0) > 1)

    def requeue_interrupted_jobs(self) -> int:
        result = self.jobs.update_many(
            {"status": {"$in": list(PROCESSING_STATUSES)}},
            {
                "$set": {
                    "status": "queued",
                    "stage_detail": "服务重启后等待恢复",
                    "updated_at": self._now(),
                }
            },
        )
        return result.modified_count

    def cleanup_terminal_jobs(self, retention_days: int) -> int:
        cutoff = self._now() - timedelta(days=retention_days)
        result = self.jobs.delete_many({
            "status": {"$in": list(TERMINAL_STATUSES)},
            "finished_at": {"$lt": cutoff},
        })
        return result.deleted_count
