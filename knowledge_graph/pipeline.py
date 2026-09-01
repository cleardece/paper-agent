"""Raw Claim -> Canonical Entity -> Canonical Fact -> Provenance pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from knowledge_graph.entity_resolution.resolver import EntityResolver
from knowledge_graph.fact_resolution.resolver import FactResolver
from knowledge_graph.schema.entity_types import ENTITY_TYPES
from knowledge_graph.schema.predicates import PREDICATES, to_legacy_relation


def _evidence_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class KnowledgeGraphPipeline:
    VERSION = "evidence-graph-v4"

    def __init__(self, repository: Any, embedder: Any, vector_index: Any):
        self.repository = repository
        self.entity_resolver = EntityResolver(repository, embedder, vector_index)
        self.fact_resolver = FactResolver(repository, embedder, vector_index)

    @staticmethod
    def _chunk_map(chunks: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        return {int(chunk.get("chunk_index", index)): chunk for index, chunk in enumerate(chunks)}

    async def process(
        self,
        paper: dict[str, Any],
        chunks: list[dict[str, Any]],
        verified_claims: list[dict[str, Any]],
        slow_path: Any = None,
    ) -> dict[str, Any]:
        chunk_map = self._chunk_map(chunks)
        accepted: list[tuple[dict[str, Any], dict[str, Any]]] = []
        rejected_count = 0
        for raw in verified_claims:
            try:
                chunk_index = int(raw.get("evidence_chunk_index"))
            except (TypeError, ValueError):
                rejected_count += 1
                continue
            chunk = chunk_map.get(chunk_index)
            evidence = str(raw.get("evidence") or "").strip()
            if (
                not chunk or len(evidence) < 12
                or _evidence_text(evidence) not in _evidence_text(chunk.get("content"))
                or raw.get("validation_verdict") == "rejected"
                or not raw.get("valid", True)
                or raw.get("subject_type") not in ENTITY_TYPES
                or raw.get("object_type") not in ENTITY_TYPES
                or raw.get("predicate") not in PREDICATES
                or raw.get("predicate") == "UNKNOWN"
            ):
                rejected_count += 1
                continue
            accepted.append((raw, chunk))

        mentions: list[dict[str, Any]] = []
        for raw, chunk in accepted:
            context = str(chunk.get("content") or "")[:1000]
            mentions.extend([
                {
                    "name": raw["subject_name"], "type": raw["subject_type"],
                    "context": context, "domain": raw.get("domain", ""),
                    "relation_context": raw["predicate"],
                },
                {
                    "name": raw["object_name"], "type": raw["object_type"],
                    "context": context, "domain": raw.get("domain", ""),
                    "relation_context": raw["predicate"],
                },
            ])
        entities, entity_diagnostics = await self.entity_resolver.resolve(mentions, slow_path)

        canonical_claims: list[dict[str, Any]] = []
        for item_index, (raw, chunk) in enumerate(accepted):
            subject_entity = entities[item_index * 2]
            object_entity = entities[item_index * 2 + 1]
            canonical_claims.append({
                **raw,
                "subject_entity_id": subject_entity["entity_id"],
                "subject_canonical_name": subject_entity["canonical_name"],
                "object_entity_id": object_entity["entity_id"],
                "object_canonical_name": object_entity["canonical_name"],
                "relation": to_legacy_relation(raw["predicate"]),
                "target_type": object_entity["type"],
                "target_name": object_entity["canonical_name"],
                "_source_chunk": chunk,
            })

        facts, fact_diagnostics = await self.fact_resolver.resolve(canonical_claims, slow_path)
        claim_documents: list[dict[str, Any]] = []
        paper_id = str(paper["arxiv_id"])
        for claim, fact in zip(canonical_claims, facts):
            chunk = claim.pop("_source_chunk")
            metadata = dict(chunk.get("metadata") or {})
            chunk_index = int(claim["evidence_chunk_index"])
            evidence = str(claim["evidence"])
            seed = "|".join([
                paper_id, str(chunk_index), fact["fact_id"],
                _stable_hash(_evidence_text(evidence)),
                json.dumps(claim.get("qualifiers", {}), ensure_ascii=False, sort_keys=True),
                str(claim.get("stance") or "support"),
            ])
            claim_id = f"C{_stable_hash(seed)[:24]}"
            review_status = (
                "auto_verified"
                if claim.get("validation_verdict") == "supported" else "needs_review"
            )
            claim_documents.append({
                **claim,
                "_id": claim_id, "claim_id": claim_id, "fact_id": fact["fact_id"],
                "paper_id": paper_id, "chunk_id": f"{paper_id}:{chunk_index}",
                "chunk_index": chunk_index,
                "section": metadata.get("section") or metadata.get("heading") or "",
                "page": metadata.get("page", 0), "evidence": evidence,
                "evidence_context": str(chunk.get("content") or ""),
                "evidence_content_hash": _stable_hash(str(chunk.get("content") or "")),
                "review_status": review_status,
                "extractor_version": self.VERSION, "verifier_version": self.VERSION,
                "entity_resolver_version": self.entity_resolver.VERSION,
                "fact_resolver_version": self.fact_resolver.VERSION,
                "graph_version": self.VERSION,
            })

        self.repository.replace_paper_claims(paper_id, claim_documents)
        return {
            "claims": claim_documents,
            "diagnostics": {
                "input_claim_count": len(verified_claims),
                "canonical_claim_count": len(claim_documents),
                "provenance_rejected_count": rejected_count,
                "entity_resolution": entity_diagnostics,
                "fact_resolution": fact_diagnostics,
            },
        }
