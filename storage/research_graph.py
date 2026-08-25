"""Persistent evidence-backed research graph stored in MongoDB."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING, DESCENDING, ReturnDocument

RELATIONS = {"proposes", "uses", "compares_with"}
ENTITY_TYPES = {"method", "dataset", "metric"}
REVIEW_STATUSES = {"auto", "confirmed", "rejected"}
JOB_TERMINAL = {"completed", "failed"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


class ResearchGraphRepository:
    """Owns graph collections; every automatic edge retains its source evidence."""

    def __init__(self, db: Any):
        self.nodes = db["research_graph_nodes"]
        self.edges = db["research_graph_edges"]
        self.jobs = db["research_graph_jobs"]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.nodes.create_index([("node_type", ASCENDING), ("normalized_name", ASCENDING)], unique=True)
        self.edges.create_index([("source_paper_id", ASCENDING), ("review_status", ASCENDING)])
        self.edges.create_index([("target_node_id", ASCENDING), ("relation", ASCENDING)])
        self.edges.create_index([("updated_at", DESCENDING)])
        self.jobs.create_index([("paper_id", ASCENDING)], unique=True)
        self.jobs.create_index([("status", ASCENDING), ("created_at", ASCENDING)])

    def enqueue(self, paper_id: str) -> dict[str, Any]:
        now = _now()
        return self.jobs.find_one_and_update(
            {"paper_id": paper_id},
            {
                "$setOnInsert": {
                    "paper_id": paper_id,
                    "status": "pending",
                    "attempt_count": 0,
                    "max_attempts": 2,
                    "error": None,
                    "edge_count": 0,
                    "created_at": now,
                },
                "$set": {"updated_at": now},
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

    def enqueue_missing_indexed_papers(self, papers: list[dict[str, Any]]) -> int:
        created = 0
        for paper in papers:
            if not paper.get("arxiv_id"):
                continue
            existing = self.jobs.find_one({"paper_id": paper["arxiv_id"]}, {"_id": 1})
            if not existing:
                self.enqueue(paper["arxiv_id"])
                created += 1
        return created

    def claim_next_job(self) -> dict[str, Any] | None:
        now = _now()
        return self.jobs.find_one_and_update(
            {"status": "pending"},
            {
                "$set": {"status": "extracting", "error": None, "updated_at": now, "started_at": now},
                "$inc": {"attempt_count": 1},
            },
            sort=[("created_at", ASCENDING)],
            return_document=ReturnDocument.AFTER,
        )

    def complete_job(self, paper_id: str, edge_count: int) -> None:
        self.jobs.update_one(
            {"paper_id": paper_id},
            {"$set": {"status": "completed", "edge_count": edge_count, "error": None,
                      "finished_at": _now(), "updated_at": _now()}},
        )

    def fail_or_retry_job(self, paper_id: str, error: str) -> bool:
        job = self.jobs.find_one({"paper_id": paper_id}) or {}
        if int(job.get("attempt_count", 0)) < int(job.get("max_attempts", 2)):
            self.jobs.update_one(
                {"paper_id": paper_id},
                {"$set": {"status": "pending", "error": error, "updated_at": _now()}},
            )
            return True
        self.jobs.update_one(
            {"paper_id": paper_id},
            {"$set": {"status": "failed", "error": error, "finished_at": _now(), "updated_at": _now()}},
        )
        return False

    def get_job(self, paper_id: str) -> dict[str, Any] | None:
        return self.jobs.find_one({"paper_id": paper_id})

    def _paper_node(self, paper: dict[str, Any]) -> str:
        paper_id = str(paper["arxiv_id"])
        node_id = f"paper:{paper_id}"
        now = _now()
        self.nodes.update_one(
            {"_id": node_id},
            {"$set": {"node_type": "paper", "name": paper.get("title", paper_id),
                      "normalized_name": paper_id.lower(), "paper_id": paper_id, "updated_at": now},
             "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        return node_id

    def _entity_node(self, entity_type: str, name: str) -> str:
        normalized = _normalized(name)
        node_id = f"{entity_type}:{hashlib.sha256(normalized.encode()).hexdigest()[:20]}"
        now = _now()
        self.nodes.update_one(
            {"_id": node_id},
            {"$set": {"node_type": entity_type, "name": str(name).strip(),
                      "normalized_name": normalized, "updated_at": now},
             "$setOnInsert": {"created_at": now}},
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

    def _validated_relation(self, relation: dict[str, Any], chunks: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
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
        if (relation_name not in RELATIONS or entity_type not in ENTITY_TYPES
                or len(target_name) < 2 or not 0 <= confidence <= 1
                or not chunk or len(evidence) < 12
                or not self._has_evidence_overlap(evidence, chunk.get("content", ""))):
            return None
        return {
            "relation": relation_name, "target_type": entity_type, "target_name": target_name,
            "evidence_chunk_index": chunk_index, "evidence": evidence, "confidence": confidence, "chunk": chunk,
        }

    def upsert_relations(self, paper: dict[str, Any], chunks: list[dict[str, Any]],
                         relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        paper_id = str(paper["arxiv_id"])
        chunk_map = self._chunk_map(chunks)
        valid = [self._validated_relation(item, chunk_map) for item in relations]
        valid = [item for item in valid if item]
        if not valid:
            return []
        self._paper_node(paper)
        self.edges.delete_many({"source_paper_id": paper_id, "review_status": "auto"})
        now = _now()
        saved: list[dict[str, Any]] = []
        for item in valid:
            meta = item["chunk"].get("metadata", {})
            target_node_id = self._entity_node(item["target_type"], item["target_name"])
            edge_seed = f"{paper_id}|{item['relation']}|{target_node_id}|{item['evidence_chunk_index']}"
            edge_id = hashlib.sha256(edge_seed.encode()).hexdigest()
            document = {
                "_id": edge_id,
                "source_paper_id": paper_id,
                "source_node_id": f"paper:{paper_id}",
                "target_node_id": target_node_id,
                "relation": item["relation"],
                "target_type": item["target_type"],
                "target_name": item["target_name"],
                "evidence_chunk_index": item["evidence_chunk_index"],
                "evidence_section": meta.get("section") or meta.get("heading") or "",
                "evidence_page": meta.get("page", 0),
                "evidence_content_hash": hashlib.sha256(item["chunk"].get("content", "").encode()).hexdigest(),
                "evidence": item["evidence"],
                "confidence": item["confidence"],
                "extractor_version": "evidence-graph-v1",
                "review_status": "auto",
                "updated_at": now,
            }
            self.edges.update_one({"_id": edge_id}, {"$set": document, "$setOnInsert": {"created_at": now}}, upsert=True)
            saved.append(document)
        return saved

    def get_edge(self, edge_id: str) -> dict[str, Any] | None:
        return self.edges.find_one({"_id": edge_id})

    def review_edge(self, edge_id: str, review_status: str) -> bool:
        if review_status not in {"confirmed", "rejected"}:
            return False
        return self.edges.update_one(
            {"_id": edge_id}, {"$set": {"review_status": review_status, "updated_at": _now()}}
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
            {"review_status": {"$ne": "rejected"},
             "$or": [{"target_name": {"$regex": escaped, "$options": "i"}},
                     {"evidence": {"$regex": escaped, "$options": "i"}}]},
            {"source_paper_id": 1},
        ).limit(limit)
        return list(dict.fromkeys(edge["source_paper_id"] for edge in edges))

    def status_summary(self) -> dict[str, int]:
        return {
            "pending": self.jobs.count_documents({"status": "pending"}),
            "extracting": self.jobs.count_documents({"status": "extracting"}),
            "completed": self.jobs.count_documents({"status": "completed"}),
            "failed": self.jobs.count_documents({"status": "failed"}),
            "edges": self.edges.count_documents({}),
        }

