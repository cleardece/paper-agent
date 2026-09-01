"""MongoDB repositories for Canonical Entity, Claim, Fact, alias and cache."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING

from knowledge_graph.entity_resolution.normalizer import blocking_keys, normalize_name


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}{hashlib.sha256(value.encode('utf-8')).hexdigest()[:24]}"


class CanonicalGraphRepository:
    """Focused persistence API used by the resolution pipeline."""

    SYSTEM_STATUSES = {"auto_verified", "needs_review", "auto"}
    USABLE_STATUSES = {"auto_verified", "confirmed"}

    def __init__(self, db: Any):
        self.entities = db["research_graph_entities"]
        self.aliases = db["research_graph_aliases"]
        self.claims = db["research_graph_claims"]
        self.facts = db["research_graph_facts"]
        self.resolution_cache = db["research_graph_resolution_cache"]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.entities.create_index([("type", ASCENDING), ("canonical_normalized", ASCENDING)])
        self.entities.create_index([("type", ASCENDING), ("blocking_keys", ASCENDING)])
        self.aliases.create_index(
            [("type", ASCENDING), ("normalized_alias", ASCENDING)], unique=True
        )
        self.claims.create_index([("paper_id", ASCENDING), ("review_status", ASCENDING)])
        self.claims.create_index([("fact_id", ASCENDING), ("stance", ASCENDING)])
        self.facts.create_index([("signature", ASCENDING)], unique=True)
        self.resolution_cache.create_index([("kind", ASCENDING), ("key", ASCENDING)], unique=True)

    def find_entity_by_alias(self, entity_type: str, name: str) -> dict[str, Any] | None:
        alias = self.aliases.find_one({
            "type": entity_type, "normalized_alias": normalize_name(name),
        })
        return self.entities.find_one({"_id": alias["entity_id"]}) if alias else None

    def find_entity_candidates(
        self, entity_type: str, keys: list[str], limit: int = 10
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"type": entity_type}
        if keys:
            query["blocking_keys"] = {"$in": keys}
        return list(self.entities.find(query).sort("updated_at", -1).limit(limit))

    def get_entities(self, entity_ids: list[str]) -> list[dict[str, Any]]:
        if not entity_ids:
            return []
        return list(self.entities.find({"_id": {"$in": list(dict.fromkeys(entity_ids))}}))

    def create_entity(
        self,
        mention: dict[str, Any],
        embedding: list[float] | None,
        *,
        resolution_status: str,
        resolver_version: str,
    ) -> dict[str, Any]:
        entity_type = str(mention["type"])
        name = str(mention["name"]).strip()
        normalized = normalize_name(name)
        entity_id = _stable_id("E", f"{entity_type}|{normalized}")
        now = _now()
        document = {
            "_id": entity_id, "entity_id": entity_id,
            "canonical_name": name, "canonical_normalized": normalized,
            "type": entity_type, "aliases": [name],
            "normalized_aliases": [normalized], "blocking_keys": blocking_keys(name),
            "domain": str(mention.get("domain") or ""),
            "context": str(mention.get("context") or "")[:1000],
            "embedding": embedding, "resolution_status": resolution_status,
            "resolver_version": resolver_version, "updated_at": now,
        }
        self.entities.update_one(
            {"_id": entity_id},
            {"$set": document, "$setOnInsert": {"created_at": now}}, upsert=True,
        )
        self.aliases.update_one(
            {"type": entity_type, "normalized_alias": normalized},
            {"$setOnInsert": {
                "entity_id": entity_id, "alias": name, "created_at": now,
            }},
            upsert=True,
        )
        return self.entities.find_one({"_id": entity_id}) or document

    def add_alias(
        self, entity_id: str, name: str, *, resolver_version: str
    ) -> dict[str, Any] | None:
        entity = self.entities.find_one({"_id": entity_id})
        if not entity:
            return None
        normalized = normalize_name(name)
        existing = self.aliases.find_one({
            "type": entity["type"], "normalized_alias": normalized,
        })
        if existing and existing.get("entity_id") != entity_id:
            return entity
        now = _now()
        self.aliases.update_one(
            {"type": entity["type"], "normalized_alias": normalized},
            {"$set": {"entity_id": entity_id, "alias": name, "updated_at": now},
             "$setOnInsert": {"created_at": now}}, upsert=True,
        )
        self.entities.update_one(
            {"_id": entity_id},
            {
                "$addToSet": {
                    "aliases": name, "normalized_aliases": normalized,
                    "blocking_keys": {"$each": blocking_keys(name)},
                },
                "$set": {"resolver_version": resolver_version, "updated_at": now},
            },
        )
        return self.entities.find_one({"_id": entity_id})

    def find_fact_by_signature(self, signature: str) -> dict[str, Any] | None:
        return self.facts.find_one({"signature": signature})

    def get_facts(self, fact_ids: list[str]) -> list[dict[str, Any]]:
        if not fact_ids:
            return []
        return list(self.facts.find({"_id": {"$in": list(dict.fromkeys(fact_ids))}}))

    def find_fact_candidates(self, claim: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
        return list(self.facts.find({
            "$or": [
                {"subject_entity_id": claim["subject_entity_id"]},
                {"object_entity_id": claim["object_entity_id"]},
            ]
        }).sort("updated_at", -1).limit(limit))

    def create_fact(
        self, claim: dict[str, Any], *, resolution_status: str,
        resolver_version: str, embedding: list[float] | None,
    ) -> dict[str, Any]:
        signature = str(claim["fact_signature"])
        fact_id = _stable_id("F", signature)
        now = _now()
        document = {
            "_id": fact_id, "fact_id": fact_id, "signature": signature,
            "subject_entity_id": claim["subject_entity_id"],
            "predicate": claim["predicate"],
            "object_entity_id": claim["object_entity_id"],
            "embedding": embedding, "resolution_status": resolution_status,
            "resolver_version": resolver_version, "updated_at": now,
        }
        self.facts.update_one(
            {"signature": signature},
            {"$set": document, "$setOnInsert": {
                "created_at": now, "support_count": 0, "contradict_count": 0,
                "needs_review_count": 0,
            }}, upsert=True,
        )
        return self.facts.find_one({"signature": signature}) or document

    def replace_paper_claims(self, paper_id: str, claims: list[dict[str, Any]]) -> None:
        old_fact_ids = {
            item.get("fact_id") for item in self.claims.find(
                {"paper_id": paper_id}, {"fact_id": 1}
            ) if item.get("fact_id")
        }
        now = _now()
        new_fact_ids: set[str] = set()
        new_claim_ids: list[str] = []
        for claim in claims:
            existing = self.claims.find_one({"_id": claim["_id"]}, {"review_status": 1})
            status = (
                existing["review_status"]
                if existing and existing.get("review_status") in {"confirmed", "rejected"}
                else claim["review_status"]
            )
            self.claims.update_one(
                {"_id": claim["_id"]},
                {"$set": {**claim, "review_status": status, "updated_at": now},
                 "$setOnInsert": {"created_at": now}}, upsert=True,
            )
            new_fact_ids.add(claim["fact_id"])
            new_claim_ids.append(claim["_id"])
        stale_query: dict[str, Any] = {
            "paper_id": paper_id,
            "review_status": {"$in": list(self.SYSTEM_STATUSES)},
        }
        if new_claim_ids:
            stale_query["_id"] = {"$nin": new_claim_ids}
        self.claims.delete_many(stale_query)
        self.recompute_fact_counts(old_fact_ids | new_fact_ids)

    def recompute_fact_counts(self, fact_ids: set[str]) -> None:
        for fact_id in fact_ids:
            support = self.claims.count_documents({
                "fact_id": fact_id, "stance": "support",
                "review_status": {"$in": list(self.USABLE_STATUSES)},
            })
            contradict = self.claims.count_documents({
                "fact_id": fact_id, "stance": "contradict",
                "review_status": {"$in": list(self.USABLE_STATUSES)},
            })
            needs_review = self.claims.count_documents({
                "fact_id": fact_id, "review_status": "needs_review",
            })
            self.facts.update_one(
                {"_id": fact_id},
                {"$set": {
                    "support_count": support, "contradict_count": contradict,
                    "needs_review_count": needs_review, "updated_at": _now(),
                }},
            )

    def review_claim(self, claim_id: str, review_status: str) -> bool:
        claim = self.claims.find_one({"_id": claim_id})
        if not claim:
            return False
        matched = self.claims.update_one(
            {"_id": claim_id},
            {"$set": {"review_status": review_status, "updated_at": _now()}},
        ).matched_count == 1
        if matched:
            self.recompute_fact_counts({claim["fact_id"]})
        return matched

    def delete_paper(self, paper_id: str) -> None:
        fact_ids = {
            item.get("fact_id") for item in self.claims.find(
                {"paper_id": paper_id}, {"fact_id": 1}
            ) if item.get("fact_id")
        }
        self.claims.delete_many({"paper_id": paper_id})
        self.recompute_fact_counts(fact_ids)

    def evaluation(self) -> dict[str, Any]:
        claim_count = self.claims.count_documents({})
        complete = self.claims.count_documents({
            "paper_id": {"$exists": True, "$nin": [None, ""]},
            "chunk_id": {"$exists": True, "$nin": [None, ""]},
            "evidence": {"$exists": True, "$nin": [None, ""]},
            "evidence_content_hash": {"$exists": True, "$nin": [None, ""]},
        })
        unresolved_entities = self.entities.count_documents({
            "resolution_status": "ambiguous_unresolved"
        })
        unresolved_facts = self.facts.count_documents({
            "resolution_status": "ambiguous_unresolved"
        })
        return {
            "entity_count": self.entities.count_documents({}),
            "fact_count": self.facts.count_documents({}),
            "claim_count": claim_count,
            "unresolved_entity_count": unresolved_entities,
            "unresolved_fact_count": unresolved_facts,
            "duplicate_entity_rate": 0.0,
            "duplicate_fact_rate": 0.0,
            "incorrect_merge_rate": None,
            "provenance_completeness": (complete / claim_count) if claim_count else 1.0,
        }
