"""
Paper Agent - Milvus向量存储层
"""

import logging
from typing import Optional
from pymilvus import DataType
from pymilvus import MilvusClient as PyMilvusClient

logger = logging.getLogger("paper-agent")

COLLECTION_NAME = "paper_chunks"
VECTOR_DIM = 1024  # BGE-M3


class MilvusClient:
    def __init__(self, uri: str = "http://localhost:19530"):
        logger.info(f"[Milvus] 正在连接 {uri}...")
        self.client = PyMilvusClient(uri=uri, timeout=10)
        logger.info("[Milvus] 连接成功")
        self._ensure_collection()

    def _ensure_collection(self):
        if self.client.has_collection(COLLECTION_NAME):
            return

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("paper_arxiv_id", DataType.VARCHAR, max_length=64)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("content", DataType.VARCHAR, max_length=8192)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
        schema.add_field("metadata_json", DataType.VARCHAR, max_length=4096)

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        self.client.create_index(
            collection_name=COLLECTION_NAME,
            index_params=index_params,
        )

    def insert(self, records: list[dict]) -> int:
        """批量插入向量"""
        if not records:
            return 0
        result = self.client.insert(collection_name=COLLECTION_NAME, data=records)
        return result["insert_count"]

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        paper_ids: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
    ) -> list[dict]:
        """语义检索"""
        if output_fields is None:
            output_fields = ["paper_arxiv_id", "chunk_index", "content", "metadata_json"]

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}

        expr = None
        if paper_ids:
            id_list = ", ".join(f'"{pid}"' for pid in paper_ids)
            expr = f"paper_arxiv_id in [{id_list}]"

        results = self.client.search(
            collection_name=COLLECTION_NAME,
            data=[query_embedding],
            limit=top_k,
            search_params=search_params,
            expr=expr,
            output_fields=output_fields,
        )

        hits = []
        if results and len(results) > 0:
            for hit in results[0]:
                record = hit["entity"]
                record["id"] = hit["id"]
                record["score"] = hit["distance"]
                hits.append(record)
        return hits

    def delete_by_paper(self, arxiv_id: str):
        self.client.delete(
            collection_name=COLLECTION_NAME,
            filter=f'paper_arxiv_id == "{arxiv_id}"',
        )

    def delete_all(self):
        self.client.drop_collection(COLLECTION_NAME)
        self._ensure_collection()

    def count(self, arxiv_id: Optional[str] = None) -> int:
        if arxiv_id:
            results = self.client.query(
                collection_name=COLLECTION_NAME,
                filter=f'paper_arxiv_id == "{arxiv_id}"',
                output_fields=["id"],
            )
            return len(results)
        stats = self.client.get_collection_stats(COLLECTION_NAME)
        return int(stats["row_count"])

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
