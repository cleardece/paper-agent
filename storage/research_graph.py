"""MongoDB-backed lifecycle for the evidence-backed research graph."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from pymongo import ASCENDING, DESCENDING, ReturnDocument

GRAPH_VERSION = "evidence-graph-v2"
RELATIONS = {"proposes", "uses", "compares_with"}
ENTITY_TYPES = {"method", "dataset", "metric"}
REVIEW_STATUSES = {"auto", "confirmed", "rejected"}
RUNNABLE_STATUSES = {"pending", "retry_wait"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


class ResearchGraphRepository:
    """Persist graph data and recoverable, versioned extraction jobs."""

    def __init__(self, db: Any):
        self.nodes = db["research_graph_nodes"]
        self.edges = db["research_graph_edges"]
        self.jobs = db["research_graph_jobs"]
        self.papers = db["papers"]
        self.control = db["research_graph_control"]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.nodes.create_index([("node_type", ASCENDING), ("normalized_name", ASCENDING)], unique=True)
        self.edges.create_index([("source_paper_id", ASCENDING), ("review_status", ASCENDING)])
        self.edges.create_index([("target_node_id", ASCENDING), ("relation", ASCENDING)])
        self.edges.create_index([("updated_at", DESCENDING)])
        self.jobs.create_index([("paper_id", ASCENDING)], unique=True)
        self.jobs.create_index([
            ("status", ASCENDING), ("priority", DESCENDING),
            ("next_attempt_at", ASCENDING), ("created_at", ASCENDING),
        ])

    @staticmethod
    def _priority(source: str) -> int:
        return {"manual": 30, "new": 20, "backfill": 10}.get(source, 10)

    def enqueue(self, paper_id: str, *, source: str = "new",
                graph_version: str = GRAPH_VERSION, force: bool = False) -> dict[str, Any]:
        now = _now()
        existing = self.jobs.find_one({"paper_id": paper_id})
        if existing and not force and existing.get("graph_version") == graph_version:
            return existing
        run_number = int((existing or {}).get("run_number", 0)) + 1
        values = {
            "paper_id": paper_id, "graph_version": graph_version,
            "status": "pending", "source": source, "priority": self._priority(source),
            "run_number": run_number, "attempt_count": 0, "max_attempts": 2,
            "next_attempt_at": now, "error": None, "error_kind": None,
            "edge_count": 0, "diagnostics": {}, "updated_at": now, "finished_at": None,
        }
        update: dict[str, Any] = {
            "$set": values,
            "$setOnInsert": {
                "job_id": uuid4().hex, "created_at": now,
            },
            "$unset": {
                "worker_id": "", "lease_expires_at": "",
                "heartbeat_at": "", "started_at": "",
            },
        }
        # 版本升级时不能把旧任务的失败原因和诊断信息直接抹掉。保留在同一
        # 任务的历史中，既便于排障，也能让用户区分“旧版本失败”和“新版本重跑”。
        if existing and force:
            update["$push"] = {"attempt_history": {
                "run_number": int(existing.get("run_number", 1)),
                "attempt": int(existing.get("attempt_count", 0)),
                "status": "superseded_by_graph_version",
                "graph_version": existing.get("graph_version"),
                "previous_status": existing.get("status"),
                "edge_count": int(existing.get("edge_count", 0)),
                "error": existing.get("error"),
                "error_kind": existing.get("error_kind"),
                "diagnostics": existing.get("diagnostics", {}),
                "finished_at": now,
            }}
        else:
            update["$setOnInsert"]["attempt_history"] = []
        self.jobs.update_one(
            {"paper_id": paper_id},
            update,
            upsert=True,
        )
        self.papers.update_one(
            {"arxiv_id": paper_id},
            {"$set": {
                "graph_status": "pending", "graph_version": graph_version,
                "graph_error": None,
            }},
        )
        return self.jobs.find_one({"paper_id": paper_id})

    def reconcile_indexed_papers(self, papers: list[dict[str, Any]],
                                 graph_version: str = GRAPH_VERSION) -> dict[str, int]:
        result = {"created": 0, "upgraded": 0, "unchanged": 0}
        for paper in papers:
            paper_id = paper.get("arxiv_id")
            if not paper_id:
                continue
            job = self.jobs.find_one({"paper_id": paper_id})
            if not job:
                self.enqueue(paper_id, source="backfill", graph_version=graph_version)
                result["created"] += 1
            elif job.get("graph_version") != graph_version:
                self.enqueue(
                    paper_id, source="backfill", graph_version=graph_version, force=True
                )
                result["upgraded"] += 1
            else:
                result["unchanged"] += 1
        return result

    def enqueue_missing_indexed_papers(self, papers: list[dict[str, Any]]) -> int:
        result = self.reconcile_indexed_papers(papers)
        return result["created"] + result["upgraded"]

    def recover_expired_leases(self) -> dict[str, int]:
        now = _now()
        query = {
            "status": "extracting",
            "$or": [
                {"lease_expires_at": {"$lte": now}},
                {"lease_expires_at": {"$exists": False}},
            ],
        }
        recovered = failed = 0
        for job in list(self.jobs.find(query)):
            retrying = int(job.get("attempt_count", 0)) < int(job.get("max_attempts", 2))
            status = "retry_wait" if retrying else "failed"
            history = {
                "run_number": int(job.get("run_number", 1)),
                "attempt": int(job.get("attempt_count", 0)),
                "status": "lease_expired",
                "error": "工作者租约过期，任务已被回收",
                "error_kind": "lease_expired", "finished_at": now,
            }
            self.jobs.update_one(
                {"_id": job["_id"], "status": "extracting"},
                {
                    "$set": {
                        "status": status, "error": history["error"],
                        "error_kind": "lease_expired",
                        "next_attempt_at": now if retrying else None,
                        "updated_at": now, "finished_at": None if retrying else now,
                    },
                    "$push": {"attempt_history": history},
                    "$unset": {
                        "worker_id": "", "lease_expires_at": "", "heartbeat_at": "",
                    },
                },
            )
            self.papers.update_one(
                {"arxiv_id": job["paper_id"]},
                {"$set": {
                    "graph_status": "pending" if retrying else "failed",
                    "graph_error": history["error"],
                }},
            )
            recovered += int(retrying)
            failed += int(not retrying)
        return {"recovered": recovered, "failed": failed}

    def claim_next_job(self, worker_id: str, lease_seconds: int) -> dict[str, Any] | None:
        now = _now()
        return self.jobs.find_one_and_update(
            {
                "status": {"$in": list(RUNNABLE_STATUSES)},
                "next_attempt_at": {"$lte": now},
            },
            {
                "$set": {
                    "status": "extracting", "worker_id": worker_id,
                    "lease_expires_at": now + timedelta(seconds=lease_seconds),
                    "heartbeat_at": now, "started_at": now,
                    "error": None, "error_kind": None, "updated_at": now,
                },
                "$inc": {"attempt_count": 1},
            },
            sort=[
                ("priority", DESCENDING), ("next_attempt_at", ASCENDING),
                ("created_at", ASCENDING),
            ],
            return_document=ReturnDocument.AFTER,
        )

    def heartbeat(self, paper_id: str, worker_id: str, lease_seconds: int) -> bool:
        now = _now()
        result = self.jobs.update_one(
            {"paper_id": paper_id, "status": "extracting", "worker_id": worker_id},
            {"$set": {
                "heartbeat_at": now,
                "lease_expires_at": now + timedelta(seconds=lease_seconds),
                "updated_at": now,
            }},
        )
        return result.matched_count == 1

    def complete_job(self, paper_id: str, worker_id: str, edge_count: int,
                     diagnostics: dict[str, Any]) -> bool:
        now = _now()
        status = "completed_with_edges" if edge_count else "completed_empty"
        job = self.jobs.find_one({
            "paper_id": paper_id, "status": "extracting", "worker_id": worker_id,
        })
        if not job:
            return False
        history = {
            "run_number": int(job.get("run_number", 1)),
            "attempt": int(job.get("attempt_count", 1)), "status": status,
            "edge_count": edge_count, "diagnostics": diagnostics, "finished_at": now,
        }
        self.jobs.update_one(
            {"_id": job["_id"], "status": "extracting", "worker_id": worker_id},
            {
                "$set": {
                    "status": status, "edge_count": edge_count,
                    "diagnostics": diagnostics, "error": None, "error_kind": None,
                    "finished_at": now, "updated_at": now,
                },
                "$push": {"attempt_history": history},
                "$unset": {
                    "worker_id": "", "lease_expires_at": "", "heartbeat_at": "",
                },
            },
        )
        self.papers.update_one(
            {"arxiv_id": paper_id},
            {"$set": {
                "graph_status": "ready" if edge_count else "empty",
                "graph_edge_count": edge_count,
                "graph_version": job.get("graph_version", GRAPH_VERSION),
                "graph_error": None, "graph_diagnostics": diagnostics,
            }},
        )
        return True

    def fail_attempt(self, paper_id: str, worker_id: str, error: str,
                     error_kind: str, retry_delay_seconds: int) -> str:
        now = _now()
        job = self.jobs.find_one({
            "paper_id": paper_id, "status": "extracting", "worker_id": worker_id,
        })
        if not job:
            return "ignored"
        retrying = int(job.get("attempt_count", 0)) < int(job.get("max_attempts", 2))
        status = "retry_wait" if retrying else "failed"
        history = {
            "run_number": int(job.get("run_number", 1)),
            "attempt": int(job.get("attempt_count", 1)), "status": status,
            "error": error, "error_kind": error_kind, "finished_at": now,
        }
        self.jobs.update_one(
            {"_id": job["_id"], "status": "extracting", "worker_id": worker_id},
            {
                "$set": {
                    "status": status, "error": error, "error_kind": error_kind,
                    "next_attempt_at": (
                        now + timedelta(seconds=retry_delay_seconds)
                        if retrying else None
                    ),
                    "finished_at": None if retrying else now, "updated_at": now,
                },
                "$push": {"attempt_history": history},
                "$unset": {
                    "worker_id": "", "lease_expires_at": "", "heartbeat_at": "",
                },
            },
        )
        self.papers.update_one(
            {"arxiv_id": paper_id},
            {"$set": {
                "graph_status": "pending" if retrying else "failed",
                "graph_error": error,
            }},
        )
        return status

    def manual_retry(self, paper_id: str) -> dict[str, Any] | None:
        if not self.jobs.find_one({"paper_id": paper_id}):
            if not self.papers.find_one({"arxiv_id": paper_id, "status": "indexed"}):
                return None
        return self.enqueue(paper_id, source="manual", force=True)

    def get_job(self, paper_id: str) -> dict[str, Any] | None:
        return self.jobs.find_one({"paper_id": paper_id})

    def list_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        return list(self.jobs.find().sort([
            ("priority", DESCENDING), ("updated_at", DESCENDING),
        ]).limit(limit))

    def record_infrastructure_failure(self, error: str, *, threshold: int,
                                      pause_seconds: int) -> dict[str, Any]:
        now = _now()
        state = self.control.find_one_and_update(
            {"_id": "scheduler"},
            {
                "$inc": {"consecutive_failures": 1},
                "$set": {"last_error": error, "updated_at": now},
                "$setOnInsert": {"paused_until": None},
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        if int(state.get("consecutive_failures", 0)) >= threshold:
            paused_until = now + timedelta(seconds=pause_seconds)
            self.control.update_one(
                {"_id": "scheduler"},
                {"$set": {"paused_until": paused_until, "updated_at": now}},
            )
            state["paused_until"] = paused_until
        return state

    def record_infrastructure_success(self) -> None:
        self.control.update_one(
            {"_id": "scheduler"},
            {"$set": {
                "consecutive_failures": 0, "paused_until": None,
                "last_error": None, "updated_at": _now(),
            }},
            upsert=True,
        )

    def circuit_state(self) -> dict[str, Any]:
        state = self.control.find_one({"_id": "scheduler"}) or {}
        paused_until = _as_utc(state.get("paused_until"))
        return {
            "open": bool(paused_until and paused_until > _now()),
            "paused_until": paused_until,
            "consecutive_failures": int(state.get("consecutive_failures", 0)),
            "last_error": state.get("last_error"),
        }

    def _paper_node(self, paper: dict[str, Any]) -> str:
        paper_id = str(paper["arxiv_id"])
        node_id = f"paper:{paper_id}"
        now = _now()
        self.nodes.update_one(
            {"_id": node_id},
            {"$set": {
                "node_type": "paper", "name": paper.get("title", paper_id),
                "normalized_name": paper_id.lower(), "paper_id": paper_id,
                "updated_at": now,
            }, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        return node_id

    def _entity_node(self, entity_type: str, name: str) -> str:
        normalized = _normalized(name)
        node_id = f"{entity_type}:{hashlib.sha256(normalized.encode()).hexdigest()[:20]}"
        now = _now()
        self.nodes.update_one(
            {"_id": node_id},
            {"$set": {
                "node_type": entity_type, "name": str(name).strip(),
                "normalized_name": normalized, "updated_at": now,
            }, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        return node_id

    @staticmethod
    def _chunk_map(chunks: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        return {int(chunk.get("chunk_index", -1)): chunk for chunk in chunks}

    @staticmethod
    def _has_evidence_overlap(evidence: str, content: str) -> bool:
        evidence_tokens = set(re.findall(r"[\w-]{4,}", _normalized(evidence)))
        content_tokens = set(re.findall(r"[\w-]{4,}", _normalized(content)))
        return len(evidence_tokens & content_tokens) >= 2

    def _validated_relation(self, relation: dict[str, Any],
                            chunks: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
        relation_name = str(relation.get("relation", "")).strip()
        entity_type = str(relation.get("target_type", "")).strip()
        target_name = str(relation.get("target_name", "")).strip()
        try:
            chunk_index = int(relation.get("evidence_chunk_index"))
            confidence = float(relation.get("confidence", 0))
        except (TypeError, ValueError):
            return None
        evidence = str(relation.get("evidence", "")).strip()
        chunk = chunks.get(chunk_index)
        if (
            relation_name not in RELATIONS or entity_type not in ENTITY_TYPES
            or len(target_name) < 2 or not 0 <= confidence <= 1
            or not chunk or len(evidence) < 12
            or not self._has_evidence_overlap(evidence, chunk.get("content", ""))
        ):
            return None
        return {
            "relation": relation_name, "target_type": entity_type,
            "target_name": target_name, "evidence_chunk_index": chunk_index,
            "evidence": evidence, "confidence": confidence, "chunk": chunk,
        }

    def upsert_relations(self, paper: dict[str, Any], chunks: list[dict[str, Any]],
                         relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        paper_id = str(paper["arxiv_id"])
        chunk_map = self._chunk_map(chunks)
        valid = [self._validated_relation(item, chunk_map) for item in relations]
        valid = [item for item in valid if item]
        self._paper_node(paper)
        self.edges.delete_many({"source_paper_id": paper_id, "review_status": "auto"})
        now = _now()
        saved: list[dict[str, Any]] = []
        for item in valid:
            meta = item["chunk"].get("metadata", {})
            target_node_id = self._entity_node(item["target_type"], item["target_name"])
            seed = (
                f"{paper_id}|{item['relation']}|{target_node_id}|"
                f"{item['evidence_chunk_index']}"
            )
            edge_id = hashlib.sha256(seed.encode()).hexdigest()
            document = {
                "_id": edge_id, "source_paper_id": paper_id,
                "source_node_id": f"paper:{paper_id}", "target_node_id": target_node_id,
                "relation": item["relation"], "target_type": item["target_type"],
                "target_name": item["target_name"],
                "evidence_chunk_index": item["evidence_chunk_index"],
                "evidence_section": meta.get("section") or meta.get("heading") or "",
                "evidence_page": meta.get("page", 0),
                "evidence_content_hash": hashlib.sha256(
                    item["chunk"].get("content", "").encode()
                ).hexdigest(),
                "evidence": item["evidence"], "confidence": item["confidence"],
                "extractor_version": GRAPH_VERSION, "review_status": "auto",
                "updated_at": now,
            }
            self.edges.update_one(
                {"_id": edge_id},
                {"$set": document, "$setOnInsert": {"created_at": now}},
                upsert=True,
            )
            saved.append(document)
        return saved

    def get_edge(self, edge_id: str) -> dict[str, Any] | None:
        return self.edges.find_one({"_id": edge_id})

    def review_edge(self, edge_id: str, review_status: str) -> bool:
        if review_status not in {"confirmed", "rejected"}:
            return False
        return self.edges.update_one(
            {"_id": edge_id},
            {"$set": {"review_status": review_status, "updated_at": _now()}},
        ).matched_count == 1

    def delete_auto_data_for_paper(self, paper_id: str) -> None:
        self.edges.delete_many({"source_paper_id": paper_id, "review_status": "auto"})
        self.jobs.delete_many({"paper_id": paper_id})
        self.nodes.delete_many({"_id": f"paper:{paper_id}"})

    def search(self, query: str = "", entity_type: str | None = None,
               relation: str | None = None, review_status: str | None = None,
               limit: int = 100) -> list[dict[str, Any]]:
        filters: dict[str, Any] = {}
        if entity_type in ENTITY_TYPES:
            filters["target_type"] = entity_type
        if relation in RELATIONS:
            filters["relation"] = relation
        if review_status in REVIEW_STATUSES:
            filters["review_status"] = review_status
        if query.strip():
            escaped = re.escape(query.strip())
            filters["$or"] = [
                {"target_name": {"$regex": escaped, "$options": "i"}},
                {"source_paper_id": {"$regex": escaped, "$options": "i"}},
            ]
        return list(self.edges.find(filters).sort("updated_at", DESCENDING).limit(limit))

    def find_related_paper_ids(self, query: str, limit: int = 12) -> list[str]:
        escaped = re.escape(query.strip())
        if not escaped:
            return []
        edges = self.edges.find(
            {
                "review_status": {"$ne": "rejected"},
                "$or": [
                    {"target_name": {"$regex": escaped, "$options": "i"}},
                    {"evidence": {"$regex": escaped, "$options": "i"}},
                ],
            },
            {"source_paper_id": 1},
        ).limit(limit)
        return list(dict.fromkeys(edge["source_paper_id"] for edge in edges))

    def status_summary(self) -> dict[str, Any]:
        counts = {
            status: self.jobs.count_documents({"status": status})
            for status in [
                "pending", "retry_wait", "extracting",
                "completed_with_edges", "completed_empty", "failed",
            ]
        }
        counts["completed"] = (
            counts["completed_with_edges"] + counts["completed_empty"]
        )
        counts["edges"] = self.edges.count_documents({})
        counts["circuit"] = self.circuit_state()
        return counts
