"""Incremental Entity Resolution with deterministic and batched slow paths."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from knowledge_graph.entity_resolution.normalizer import blocking_keys, normalize_name
from knowledge_graph.entity_resolution.scorer import score_entity_candidate

SlowPath = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class EntityResolver:
    VERSION = "entity-resolution-v1"

    def __init__(self, repository: Any, embedder: Any, vector_index: Any):
        self.repository = repository
        self.embedder = embedder
        self.vector_index = vector_index

    @staticmethod
    def _embedding_text(mention: dict[str, Any]) -> str:
        return " | ".join(filter(None, [
            str(mention.get("name") or ""), str(mention.get("type") or ""),
            str(mention.get("domain") or ""), str(mention.get("context") or "")[:500],
            str(mention.get("relation_context") or ""),
        ]))

    async def resolve(
        self, mentions: list[dict[str, Any]], slow_path: SlowPath | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        index_to_key: list[tuple[str, str]] = []
        for mention in mentions:
            key = (str(mention["type"]), normalize_name(mention["name"]))
            index_to_key.append(key)
            if key not in grouped:
                grouped[key] = dict(mention)
            elif mention.get("context"):
                grouped[key]["context"] = " ".join(filter(None, [
                    str(grouped[key].get("context") or ""), str(mention["context"]),
                ]))[:1000]

        resolved: dict[tuple[str, str], dict[str, Any]] = {}
        routes = {"exact_alias": 0, "rule_merge": 0, "new": 0, "ambiguous": 0}
        unmatched: list[tuple[tuple[str, str], dict[str, Any]]] = []
        for key, mention in grouped.items():
            entity = self.repository.find_entity_by_alias(mention["type"], mention["name"])
            if entity:
                resolved[key] = entity
                routes["exact_alias"] += 1
            else:
                unmatched.append((key, mention))

        embeddings = self.embedder.embed_texts([
            self._embedding_text(mention) for _, mention in unmatched
        ]) if unmatched and self.embedder else [[] for _ in unmatched]
        ambiguous: list[dict[str, Any]] = []
        pending: dict[int, tuple[tuple[str, str], dict[str, Any], list[float]]] = {}

        for item_index, ((key, mention), embedding) in enumerate(zip(unmatched, embeddings)):
            hits = self.vector_index.search_entities(
                embedding, mention["type"], top_k=8
            ) if self.vector_index else []
            scores_by_id = {hit["entity_id"]: float(hit.get("score", 0)) for hit in hits}
            candidates = self.repository.get_entities(list(scores_by_id))
            blocked = self.repository.find_entity_candidates(
                mention["type"], blocking_keys(mention["name"]), limit=8
            )
            candidates = list({
                candidate["entity_id"]: candidate for candidate in [*candidates, *blocked]
            }.values())
            ranked = sorted([
                (
                    score_entity_candidate(
                        mention, candidate,
                        scores_by_id.get(candidate["entity_id"], 0.0),
                    ),
                    candidate,
                )
                for candidate in candidates
            ], key=lambda item: item[0], reverse=True)
            best_score, best = ranked[0] if ranked else (0.0, None)
            if best is not None and best_score >= 0.88:
                entity = self.repository.add_alias(
                    best["entity_id"], mention["name"], resolver_version=self.VERSION
                ) or best
                resolved[key] = entity
                routes["rule_merge"] += 1
            elif best is None or best_score < 0.55:
                entity = self.repository.create_entity(
                    mention, embedding, resolution_status="new",
                    resolver_version=self.VERSION,
                )
                if self.vector_index:
                    self.vector_index.upsert_entity(
                        entity["entity_id"], entity["type"], embedding
                    )
                resolved[key] = entity
                routes["new"] += 1
            else:
                mention_index = len(ambiguous)
                payload = {
                    "mention_index": mention_index,
                    "mention": mention,
                    "candidates": [
                        {
                            "entity_id": candidate["entity_id"],
                            "canonical_name": candidate["canonical_name"],
                            "type": candidate["type"],
                            "aliases": candidate.get("aliases", []),
                            "context": candidate.get("context", ""),
                            "rule_score": score,
                        }
                        for score, candidate in ranked[:8]
                    ],
                }
                ambiguous.append(payload)
                pending[mention_index] = (key, mention, embedding)

        decision_map: dict[int, dict[str, Any]] = {}
        llm_diagnostics = {"llm_called": False}
        if ambiguous and slow_path:
            result = await slow_path("resolve_entities", {"items": ambiguous})
            llm_diagnostics = result.get("diagnostics", {"llm_called": True})
            for decision in result.get("decisions", []):
                try:
                    decision_map[int(decision["mention_index"])] = decision
                except (KeyError, TypeError, ValueError):
                    continue

        for mention_index, (key, mention, embedding) in pending.items():
            decision = decision_map.get(mention_index, {})
            entity = None
            if decision.get("decision") == "merge":
                entity = self.repository.add_alias(
                    str(decision.get("entity_id")), mention["name"],
                    resolver_version=self.VERSION,
                )
            if entity is None:
                status = "new" if decision.get("decision") == "new" else "ambiguous_unresolved"
                entity = self.repository.create_entity(
                    mention, embedding, resolution_status=status,
                    resolver_version=self.VERSION,
                )
                if self.vector_index:
                    self.vector_index.upsert_entity(
                        entity["entity_id"], entity["type"], embedding
                    )
            resolved[key] = entity
            routes["ambiguous"] += 1

        return [resolved[key] for key in index_to_key], {
            "route_counts": routes, "unique_mention_count": len(grouped),
            "ambiguous_batch_count": int(bool(ambiguous)),
            "ambiguous_item_count": len(ambiguous), "llm": llm_diagnostics,
        }
