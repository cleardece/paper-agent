"""MongoDB-backed lifecycle for the evidence-backed research graph."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from pymongo import ASCENDING, DESCENDING, ReturnDocument

from knowledge_graph.graph.repository import CanonicalGraphRepository
from knowledge_graph.schema.entity_types import ENTITY_TYPES
from knowledge_graph.schema.predicates import LEGACY_RELATIONS

GRAPH_VERSION = "evidence-graph-v4"
RELATIONS = {
    *LEGACY_RELATIONS,
}
REVIEW_STATUSES = {"auto_verified", "needs_review", "confirmed", "rejected"}
USABLE_REVIEW_STATUSES = {"auto_verified", "confirmed"}
SYSTEM_REVIEW_STATUSES = {"auto", "auto_verified", "needs_review"}
RUNNABLE_STATUSES = {"pending", "retry_wait"}
NONRETRYABLE_QUOTA_PATTERN = re.compile(
    r"insufficient[_\\]?quota|allocated quota exceeded|increase your quota limit|"
    r"exceeded your current quota",
    re.IGNORECASE,
)


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
        self.canonical = CanonicalGraphRepository(db)
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
            "batch_total": 0, "completed_batches": [],
            "staged_relations": [], "batch_diagnostics": {},
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

    def finalize_nonretryable_retries(self) -> int:
        """将旧版本误排队的配额错误转为终止失败，不消耗下一次尝试。"""
        now = _now()
        finalized = 0
        jobs = list(self.jobs.find({
            "status": "retry_wait",
            "error": {"$regex": NONRETRYABLE_QUOTA_PATTERN},
        }))
        for job in jobs:
            result = self.jobs.update_one(
                {"_id": job["_id"], "status": "retry_wait"},
                {"$set": {
                    "status": "failed",
                    "error_kind": "llm_quota_exhausted",
                    "next_attempt_at": None,
                    "finished_at": now,
                    "updated_at": now,
                }},
            )
            if result.modified_count != 1:
                continue
            self.papers.update_one(
                {"arxiv_id": job["paper_id"]},
                {"$set": {
                    "graph_status": "failed",
                    "graph_error": job.get("error"),
                }},
            )
            finalized += 1
        return finalized

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

    def set_batch_total(self, paper_id: str, worker_id: str, batch_total: int) -> bool:
        """在首次模型调用前保存批次数，确保早期失败也有真实进度。"""
        result = self.jobs.update_one(
            {
                "paper_id": paper_id,
                "status": "extracting",
                "worker_id": worker_id,
            },
            {"$set": {
                "batch_total": max(0, int(batch_total)),
                "updated_at": _now(),
            }},
        )
        return result.matched_count == 1

    def save_batch_result(self, paper_id: str, worker_id: str, batch_index: int,
                          batch_total: int, relations: list[dict[str, Any]],
                          diagnostics: dict[str, Any]) -> bool:
        job = self.jobs.find_one({
            "paper_id": paper_id, "status": "extracting", "worker_id": worker_id,
        })
        if not job:
            return False
        if batch_index in job.get("completed_batches", []):
            return True
        now = _now()
        result = self.jobs.update_one(
            {
                "_id": job["_id"], "status": "extracting", "worker_id": worker_id,
                "completed_batches": {"$ne": batch_index},
            },
            {
                "$set": {
                    "batch_total": batch_total,
                    f"batch_diagnostics.{batch_index}": diagnostics,
                    "updated_at": now,
                },
                "$addToSet": {"completed_batches": batch_index},
                "$push": {"staged_relations": {"$each": relations}},
            },
        )
        return result.modified_count == 1

    def staged_relations(self, paper_id: str, worker_id: str) -> list[dict[str, Any]]:
        job = self.jobs.find_one({
            "paper_id": paper_id, "status": "extracting", "worker_id": worker_id,
        }) or {}
        return list(job.get("staged_relations", []))

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
                    "staged_relations": "",
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
                     error_kind: str, retry_delay_seconds: int, *,
                     retryable: bool = True) -> str:
        now = _now()
        job = self.jobs.find_one({
            "paper_id": paper_id, "status": "extracting", "worker_id": worker_id,
        })
        if not job:
            return "ignored"
        retrying = (
            retryable
            and int(job.get("attempt_count", 0)) < int(job.get("max_attempts", 2))
        )
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

    def _validated_relation(self, relation: dict[str, Any],
                            chunks: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
        extracted_relation = str(relation.get("relation", "")).strip()
        relation_name = str(
            relation.get("validated_relation") or extracted_relation
        ).strip()
        entity_type = str(relation.get("target_type", "")).strip()
        target_name = str(relation.get("target_name", "")).strip()
        try:
            chunk_index = int(relation.get("evidence_chunk_index"))
        except (TypeError, ValueError):
            return None
        evidence = str(relation.get("evidence", "")).strip()
        chunk = chunks.get(chunk_index)
        verdict = str(relation.get("validation_verdict", "uncertain"))
        if (
            relation_name not in RELATIONS or entity_type not in ENTITY_TYPES
            or extracted_relation not in RELATIONS or len(target_name) < 2
            or not chunk or len(evidence) < 12
            or _normalized(evidence) not in _normalized(chunk.get("content", ""))
            or verdict == "rejected"
        ):
            return None
        review_status = (
            "auto_verified"
            if verdict == "supported" and relation_name == extracted_relation
            else "needs_review"
        )
        return {
            "relation": relation_name, "target_type": entity_type,
            "target_name": target_name, "evidence_chunk_index": chunk_index,
            "evidence": evidence, "chunk": chunk,
            "review_status": review_status,
            "extracted_relation": extracted_relation,
            "validation_verdict": verdict,
            "validation_reason": str(relation.get("validation_reason", "")),
        }

    def upsert_relations(self, paper: dict[str, Any], chunks: list[dict[str, Any]],
                         relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        paper_id = str(paper["arxiv_id"])
        chunk_map = self._chunk_map(chunks)
        valid = [self._validated_relation(item, chunk_map) for item in relations]
        valid = [item for item in valid if item]
        self._paper_node(paper)
        self.edges.delete_many({
            "source_paper_id": paper_id,
            "review_status": {"$in": list(SYSTEM_REVIEW_STATUSES)},
        })
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
                "evidence": item["evidence"],
                "evidence_context": item["chunk"].get("content", ""),
                "extracted_relation": item["extracted_relation"],
                "validation_verdict": item["validation_verdict"],
                "validation_reason": item["validation_reason"],
                "extractor_version": GRAPH_VERSION,
                "updated_at": now,
            }
            existing = self.edges.find_one({"_id": edge_id}, {"review_status": 1})
            review_status = (
                existing.get("review_status")
                if existing and existing.get("review_status") in {"confirmed", "rejected"}
                else item["review_status"]
            )
            self.edges.update_one(
                {"_id": edge_id},
                {"$set": {**document, "review_status": review_status},
                 "$setOnInsert": {"created_at": now}},
                upsert=True,
            )
            saved.append({**document, "review_status": review_status})
        return saved

    def upsert_canonical_claims(
        self, paper: dict[str, Any], claims: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Materialize canonical claims into the legacy edge contract."""
        paper_id = str(paper["arxiv_id"])
        self._paper_node(paper)
        now = _now()
        saved: list[dict[str, Any]] = []
        new_edge_ids: list[str] = []
        for claim in claims:
            normalized_target = _normalized(claim["object_canonical_name"])
            existing_node = self.nodes.find_one({
                "node_type": claim["target_type"],
                "normalized_name": normalized_target,
            })
            target_node_id = (
                existing_node["_id"] if existing_node
                else f"entity:{claim['object_entity_id']}"
            )
            self.nodes.update_one(
                {"_id": target_node_id},
                {"$set": {
                    "node_type": claim["target_type"],
                    "name": claim["object_canonical_name"],
                    "normalized_name": normalized_target,
                    "canonical_entity_id": claim["object_entity_id"],
                    "aliases": claim.get("object_aliases", []),
                    "updated_at": now,
                }, "$setOnInsert": {"created_at": now}}, upsert=True,
            )
            seed = (
                f"{paper_id}|{claim['relation']}|{target_node_id}|"
                f"{claim['chunk_index']}|{claim['claim_id']}"
            )
            edge_id = hashlib.sha256(seed.encode()).hexdigest()
            previous = self.edges.find_one({
                "source_paper_id": paper_id,
                "relation": claim["relation"],
                "evidence_chunk_index": claim["chunk_index"],
                "target_name": {"$regex": f"^{re.escape(claim['object_name'])}$", "$options": "i"},
                "review_status": {"$in": ["confirmed", "rejected"]},
            })
            if previous:
                edge_id = previous["_id"]
            review_status = previous.get("review_status") if previous else claim["review_status"]
            if previous:
                self.canonical.review_claim(claim["claim_id"], review_status)
            document = {
                "_id": edge_id, "source_paper_id": paper_id,
                "source_node_id": f"paper:{paper_id}", "target_node_id": target_node_id,
                "relation": claim["relation"], "target_type": claim["target_type"],
                "target_name": claim["object_canonical_name"],
                "evidence_chunk_index": claim["chunk_index"],
                "evidence_section": claim.get("section", ""),
                "evidence_page": claim.get("page", 0),
                "evidence_content_hash": claim["evidence_content_hash"],
                "evidence": claim["evidence"],
                "evidence_context": claim["evidence_context"],
                "extracted_relation": claim.get("predicate_raw", ""),
                "validation_verdict": claim["validation_verdict"],
                "validation_reason": claim["validation_reason"],
                "extractor_version": GRAPH_VERSION, "review_status": review_status,
                "claim_id": claim["claim_id"], "fact_id": claim["fact_id"],
                "canonical_predicate": claim["predicate"],
                "subject_entity_id": claim["subject_entity_id"],
                "subject_name": claim["subject_canonical_name"],
                "object_entity_id": claim["object_entity_id"],
                "object_name": claim["object_canonical_name"],
                "qualifiers": claim.get("qualifiers", {}),
                "stance": claim.get("stance", "support"),
                "provenance": {
                    "paper_id": paper_id, "chunk_id": claim["chunk_id"],
                    "chunk_index": claim["chunk_index"], "section": claim.get("section", ""),
                    "page": claim.get("page", 0), "evidence": claim["evidence"],
                    "confidence": claim.get("confidence", 0.5),
                    "graph_version": GRAPH_VERSION,
                },
                "updated_at": now,
            }
            self.edges.update_one(
                {"_id": edge_id},
                {"$set": document, "$setOnInsert": {"created_at": now}}, upsert=True,
            )
            saved.append(document)
            new_edge_ids.append(edge_id)
        stale_query: dict[str, Any] = {
            "source_paper_id": paper_id,
            "review_status": {"$in": list(SYSTEM_REVIEW_STATUSES)},
        }
        if new_edge_ids:
            stale_query["_id"] = {"$nin": new_edge_ids}
        self.edges.delete_many(stale_query)
        return saved

    def get_edge(self, edge_id: str) -> dict[str, Any] | None:
        return self.edges.find_one({"_id": edge_id})

    def review_edge(self, edge_id: str, review_status: str) -> bool:
        if review_status not in {"confirmed", "rejected"}:
            return False
        edge = self.edges.find_one({"_id": edge_id})
        matched = self.edges.update_one(
            {"_id": edge_id},
            {"$set": {"review_status": review_status, "updated_at": _now()}},
        ).matched_count == 1
        if matched and edge and edge.get("claim_id"):
            self.canonical.review_claim(edge["claim_id"], review_status)
        return matched

    def delete_auto_data_for_paper(self, paper_id: str) -> None:
        self.edges.delete_many({
            "source_paper_id": paper_id,
            "review_status": {"$in": list(SYSTEM_REVIEW_STATUSES)},
        })
        self.jobs.delete_many({"paper_id": paper_id})
        self.nodes.delete_many({"_id": f"paper:{paper_id}"})
        self.canonical.delete_paper(paper_id)

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
        elif review_status is None:
            filters["review_status"] = {"$in": list(USABLE_REVIEW_STATUSES)}
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
                "review_status": {"$in": list(USABLE_REVIEW_STATUSES)},
                "$or": [
                    {"target_name": {"$regex": escaped, "$options": "i"}},
                    {"evidence": {"$regex": escaped, "$options": "i"}},
                ],
            },
            {"source_paper_id": 1},
        ).limit(limit)
        return list(dict.fromkeys(edge["source_paper_id"] for edge in edges))

    def paper_links(self, limit: int = 100) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for edge in self.edges.find({
            "review_status": {"$in": list(USABLE_REVIEW_STATUSES)},
        }).limit(5000):
            grouped.setdefault(edge["target_node_id"], []).append(edge)

        links: dict[tuple[str, str], dict[str, Any]] = {}
        for edges in grouped.values():
            by_paper = {edge["source_paper_id"]: edge for edge in edges}
            paper_ids = sorted(by_paper)
            for left_index, left_id in enumerate(paper_ids):
                for right_id in paper_ids[left_index + 1:]:
                    left = by_paper[left_id]
                    right = by_paper[right_id]
                    key = (left_id, right_id)
                    item = links.setdefault(key, {
                        "paper_a_id": left_id, "paper_b_id": right_id,
                        "reasons": [],
                    })
                    item["reasons"].append({
                        "entity_name": left["target_name"],
                        "entity_type": left["target_type"],
                        "paper_a_relation": left["relation"],
                        "paper_b_relation": right["relation"],
                        "paper_a_evidence": left.get("evidence", ""),
                        "paper_b_evidence": right.get("evidence", ""),
                    })

        paper_ids = {paper_id for key in links for paper_id in key}
        titles = {
            paper["arxiv_id"]: paper.get("title", paper["arxiv_id"])
            for paper in self.papers.find(
                {"arxiv_id": {"$in": list(paper_ids)}}, {"arxiv_id": 1, "title": 1}
            )
        }
        result = []
        for item in sorted(
            links.values(), key=lambda value: len(value["reasons"]), reverse=True
        )[:limit]:
            item["paper_a_title"] = titles.get(item["paper_a_id"], item["paper_a_id"])
            item["paper_b_title"] = titles.get(item["paper_b_id"], item["paper_b_id"])
            item["shared_evidence_count"] = len(item["reasons"])
            result.append(item)
        return result

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
        counts["usable_edges"] = self.edges.count_documents({
            "review_status": {"$in": list(USABLE_REVIEW_STATUSES)},
        })
        counts["needs_review"] = self.edges.count_documents({
            "review_status": "needs_review",
        })
        counts["canonical"] = self.canonical.evaluation()
        counts["circuit"] = self.circuit_state()
        return counts
