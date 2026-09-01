"""Fact Resolution after Entity Resolution."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from knowledge_graph.fact_resolution.scorer import score_fact_candidate
from knowledge_graph.fact_resolution.signature import build_fact_signature

SlowPath = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class FactResolver:
    VERSION = "fact-resolution-v1"

    def __init__(self, repository: Any, embedder: Any, vector_index: Any):
        self.repository = repository
        self.embedder = embedder
        self.vector_index = vector_index

    @staticmethod
    def _embedding_text(claim: dict[str, Any]) -> str:
        return " | ".join([
            str(claim.get("subject_canonical_name") or ""),
            str(claim.get("predicate") or ""),
            str(claim.get("object_canonical_name") or ""),
        ])

    async def resolve(
        self, claims: list[dict[str, Any]], slow_path: SlowPath | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        for claim in claims:
            claim["fact_signature"] = build_fact_signature(
                claim["subject_entity_id"], claim["predicate"],
                claim["object_entity_id"], claim.get("qualifiers"),
            )
        embeddings = self.embedder.embed_texts([
            self._embedding_text(claim) for claim in claims
        ]) if claims and self.embedder else [[] for _ in claims]
        resolved: list[dict[str, Any] | None] = [None] * len(claims)
        routes = {"exact_signature": 0, "rule_merge": 0, "new": 0, "ambiguous": 0}
        ambiguous: list[dict[str, Any]] = []
        pending: dict[int, tuple[int, dict[str, Any], list[float]]] = {}

        for claim_index, (claim, embedding) in enumerate(zip(claims, embeddings)):
            exact = self.repository.find_fact_by_signature(claim["fact_signature"])
            if exact:
                resolved[claim_index] = exact
                routes["exact_signature"] += 1
                continue
            hits = self.vector_index.search_facts(
                embedding, claim["predicate"], top_k=8
            ) if self.vector_index else []
            scores_by_id = {hit["fact_id"]: float(hit.get("score", 0)) for hit in hits}
            candidates = self.repository.get_facts(list(scores_by_id))
            if not candidates:
                candidates = self.repository.find_fact_candidates(claim, limit=8)
            ranked = sorted([
                (
                    score_fact_candidate(
                        claim, candidate, scores_by_id.get(candidate["fact_id"], 0.0)
                    ), candidate,
                ) for candidate in candidates
            ], key=lambda item: item[0], reverse=True)
            best_score, best = ranked[0] if ranked else (0.0, None)
            if best is not None and best_score >= 0.90:
                resolved[claim_index] = best
                routes["rule_merge"] += 1
            elif best is None or best_score < 0.60:
                fact = self.repository.create_fact(
                    claim, resolution_status="new", resolver_version=self.VERSION,
                    embedding=embedding,
                )
                if self.vector_index:
                    self.vector_index.upsert_fact(fact["fact_id"], fact["predicate"], embedding)
                resolved[claim_index] = fact
                routes["new"] += 1
            else:
                fact_index = len(ambiguous)
                ambiguous.append({
                    "fact_index": fact_index,
                    "claim": {
                        "subject_entity_id": claim["subject_entity_id"],
                        "predicate": claim["predicate"],
                        "object_entity_id": claim["object_entity_id"],
                        "qualifiers": claim.get("qualifiers", {}),
                        "context": claim.get("evidence", ""),
                    },
                    "candidates": [
                        {"fact_id": candidate["fact_id"], "signature": candidate["signature"],
                         "rule_score": score}
                        for score, candidate in ranked[:8]
                    ],
                })
                pending[fact_index] = (claim_index, claim, embedding)

        decision_map: dict[int, dict[str, Any]] = {}
        llm_diagnostics = {"llm_called": False}
        if ambiguous and slow_path:
            result = await slow_path("resolve_facts", {"items": ambiguous})
            llm_diagnostics = result.get("diagnostics", {"llm_called": True})
            for decision in result.get("decisions", []):
                try:
                    decision_map[int(decision["fact_index"])] = decision
                except (KeyError, TypeError, ValueError):
                    continue
        for fact_index, (claim_index, claim, embedding) in pending.items():
            decision = decision_map.get(fact_index, {})
            fact = None
            if decision.get("decision") == "merge":
                candidates = self.repository.get_facts([str(decision.get("fact_id"))])
                fact = candidates[0] if candidates else None
            if fact is None:
                status = "new" if decision.get("decision") == "new" else "ambiguous_unresolved"
                fact = self.repository.create_fact(
                    claim, resolution_status=status, resolver_version=self.VERSION,
                    embedding=embedding,
                )
                if self.vector_index:
                    self.vector_index.upsert_fact(fact["fact_id"], fact["predicate"], embedding)
            resolved[claim_index] = fact
            routes["ambiguous"] += 1
        return [item for item in resolved if item is not None], {
            "route_counts": routes, "ambiguous_batch_count": int(bool(ambiguous)),
            "ambiguous_item_count": len(ambiguous), "llm": llm_diagnostics,
        }
