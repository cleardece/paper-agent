"""Isolated Milvus collections used only for KG candidate retrieval."""

from __future__ import annotations

import logging
from typing import Any

from pymilvus import DataType

logger = logging.getLogger("paper-agent")

ENTITY_COLLECTION = "kg_entity_embeddings"
FACT_COLLECTION = "kg_fact_embeddings"
VECTOR_DIM = 1024


class KnowledgeGraphVectorIndex:
    def __init__(self, milvus: Any):
        self.client = getattr(milvus, "client", milvus)
        self.enabled = self.client is not None
        if self.enabled:
            try:
                self._ensure_collections()
            except Exception as exc:
                logger.warning("[KnowledgeGraph] KG 向量索引不可用，使用有界 Mongo 候选: %s", exc)
                self.enabled = False

    def _ensure_collection(self, name: str, id_field: str, filter_field: str) -> None:
        if self.client.has_collection(name):
            self.client.load_collection(name)
            return
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field(id_field, DataType.VARCHAR, max_length=64)
        schema.add_field(filter_field, DataType.VARCHAR, max_length=64)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
        self.client.create_collection(collection_name=name, schema=schema)
        params = self.client.prepare_index_params()
        params.add_index(
            field_name="embedding", index_type="HNSW", metric_type="COSINE",
            params={"M": 16, "efConstruction": 128},
        )
        self.client.create_index(collection_name=name, index_params=params)
        self.client.load_collection(name)

    def _ensure_collections(self) -> None:
        self._ensure_collection(ENTITY_COLLECTION, "entity_id", "entity_type")
        self._ensure_collection(FACT_COLLECTION, "fact_id", "predicate")

    @staticmethod
    def _quoted(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def upsert_entity(self, entity_id: str, entity_type: str, embedding: list[float]) -> None:
        if not self.enabled or not embedding:
            return
        self.client.delete(
            collection_name=ENTITY_COLLECTION,
            filter=f'entity_id == "{self._quoted(entity_id)}"',
        )
        self.client.insert(collection_name=ENTITY_COLLECTION, data=[{
            "entity_id": entity_id, "entity_type": entity_type, "embedding": embedding,
        }])

    def search_entities(
        self, embedding: list[float], entity_type: str, top_k: int = 8
    ) -> list[dict[str, Any]]:
        if not self.enabled or not embedding:
            return []
        result = self.client.search(
            collection_name=ENTITY_COLLECTION, data=[embedding], limit=top_k,
            filter=f'entity_type == "{self._quoted(entity_type)}"',
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            output_fields=["entity_id", "entity_type"],
        )
        return [
            {"entity_id": hit["entity"]["entity_id"], "score": hit["distance"]}
            for hit in (result[0] if result else [])
        ]
    def upsert_fact(self, fact_id: str, predicate: str, embedding: list[float]) -> None:
        if not self.enabled or not embedding:
            return
        self.client.delete(
            collection_name=FACT_COLLECTION,
            filter=f'fact_id == "{self._quoted(fact_id)}"',
        )
        self.client.insert(collection_name=FACT_COLLECTION, data=[{
            "fact_id": fact_id, "predicate": predicate, "embedding": embedding,
        }])

    def search_facts(
        self, embedding: list[float], predicate: str, top_k: int = 8
    ) -> list[dict[str, Any]]:
        if not self.enabled or not embedding:
            return []
        result = self.client.search(
            collection_name=FACT_COLLECTION, data=[embedding], limit=top_k,
            filter=f'predicate == "{self._quoted(predicate)}"',
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            output_fields=["fact_id", "predicate"],
        )
        return [
            {"fact_id": hit["entity"]["fact_id"], "score": hit["distance"]}
            for hit in (result[0] if result else [])
        ]
