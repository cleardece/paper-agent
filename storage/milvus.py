"""
Paper Agent - Milvus向量存储层
两个 collection：
  - paper_chunks: chunk 级向量（用于 Stage 2 精细检索）
  - paper_embeddings: 论文级向量（标题+摘要，用于 Stage 1 论文排序）
"""

import logging
from typing import Optional
from pymilvus import DataType
from pymilvus import MilvusClient as PyMilvusClient

logger = logging.getLogger("paper-agent")

CHUNK_COLLECTION = "paper_chunks"
PAPER_COLLECTION = "paper_embeddings"
VECTOR_DIM = 1024  # BGE-M3
_CHUNK_VARCHAR_LIMITS = {
    "paper_arxiv_id": 64,
    "content": 8192,
    "section": 128,
    "heading": 256,
}


def _truncate_utf8(value: object, max_bytes: int) -> str:
    """Fit a value into a Milvus VARCHAR field without splitting UTF-8 text."""
    text = str(value)
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


class MilvusClient:
    def __init__(self, uri: str = "http://localhost:19530"):
        logger.info(f"[Milvus] 正在连接 {uri}...")
        self.client = PyMilvusClient(uri=uri, timeout=10)
        logger.info("[Milvus] 连接成功")
        self._ensure_collections()

    def _ensure_collections(self):
        """确保两个 collection 存在：paper_chunks + paper_embeddings"""
        from config import MILVUS_HNSW_M, MILVUS_HNSW_EF_CONSTRUCTION

        # ===== paper_chunks（chunk 级向量）=====
        if self.client.has_collection(CHUNK_COLLECTION):
            schema = self.client.describe_collection(CHUNK_COLLECTION)
            fields = [f["name"] for f in schema["fields"]]
            if "metadata_json" in fields and "section" not in fields:
                logger.info("[Milvus] 检测到旧 schema，重建 chunk collection...")
                self.client.drop_collection(CHUNK_COLLECTION)
            else:
                self.client.load_collection(CHUNK_COLLECTION)
        else:
            schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("paper_arxiv_id", DataType.VARCHAR, max_length=64)
            schema.add_field("chunk_index", DataType.INT64)
            schema.add_field("content", DataType.VARCHAR, max_length=8192)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
            schema.add_field("section", DataType.VARCHAR, max_length=128)
            schema.add_field("page", DataType.INT64)
            schema.add_field("heading", DataType.VARCHAR, max_length=256)

            self.client.create_collection(collection_name=CHUNK_COLLECTION, schema=schema)

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": MILVUS_HNSW_M, "efConstruction": MILVUS_HNSW_EF_CONSTRUCTION},
            )
            self.client.create_index(collection_name=CHUNK_COLLECTION, index_params=index_params)
            self.client.load_collection(CHUNK_COLLECTION)

        # ===== paper_embeddings（论文级向量，用于 Stage 1 排序）=====
        if self.client.has_collection(PAPER_COLLECTION):
            self.client.load_collection(PAPER_COLLECTION)
        else:
            schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("paper_arxiv_id", DataType.VARCHAR, max_length=64)
            schema.add_field("title", DataType.VARCHAR, max_length=512)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

            self.client.create_collection(collection_name=PAPER_COLLECTION, schema=schema)

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": MILVUS_HNSW_M, "efConstruction": MILVUS_HNSW_EF_CONSTRUCTION},
            )
            self.client.create_index(collection_name=PAPER_COLLECTION, index_params=index_params)
            self.client.load_collection(PAPER_COLLECTION)

        logger.info(f"[Milvus] Collection 就绪: {CHUNK_COLLECTION}, {PAPER_COLLECTION}")

    def insert(self, records: list[dict]) -> int:
        """批量插入 chunk 向量"""
        if not records:
            return 0
        sanitized_records = []
        for record in records:
            sanitized = dict(record)
            for field_name, max_bytes in _CHUNK_VARCHAR_LIMITS.items():
                if field_name not in sanitized or sanitized[field_name] is None:
                    continue
                original = str(sanitized[field_name])
                value = _truncate_utf8(original, max_bytes)
                if value != original:
                    logger.warning(
                        "[Milvus] 截断 %s 字段以满足 %s 字节 VARCHAR 限制",
                        field_name,
                        max_bytes,
                    )
                sanitized[field_name] = value
            sanitized_records.append(sanitized)

        result = self.client.insert(
            collection_name=CHUNK_COLLECTION,
            data=sanitized_records,
        )
        return result["insert_count"]

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        paper_ids: Optional[list[str]] = None,
        sections: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
    ) -> list[dict]:
        """语义检索"""
        if output_fields is None:
            output_fields = ["paper_arxiv_id", "chunk_index", "content", "section", "page", "heading"]

        # HNSW 搜索参数
        from config import MILVUS_SEARCH_EF
        search_params = {"metric_type": "COSINE", "params": {"ef": MILVUS_SEARCH_EF}}

        kwargs = {
            "collection_name": CHUNK_COLLECTION,
            "data": [query_embedding],
            "limit": top_k,
            "search_params": search_params,
            "output_fields": output_fields,
        }

        # 构建过滤表达式（使用独立字段，不再解析 JSON）
        filters = []
        if paper_ids:
            id_list = ", ".join(f'"{pid}"' for pid in paper_ids)
            filters.append(f"paper_arxiv_id in [{id_list}]")

        if sections:
            section_list = ", ".join(f'"{sec}"' for sec in sections)
            filters.append(f"section in [{section_list}]")

        if filters:
            kwargs["filter"] = " and ".join(filters)

        results = self.client.search(**kwargs)

        hits = []
        if results and len(results) > 0:
            for hit in results[0]:
                record = hit["entity"]
                record["id"] = hit["id"]
                record["score"] = hit["distance"]
                hits.append(record)
        return hits

    def delete_by_paper(self, arxiv_id: str):
        """删除论文的 chunk 向量 + 论文级向量"""
        self.client.delete(
            collection_name=CHUNK_COLLECTION,
            filter=f'paper_arxiv_id == "{arxiv_id}"',
        )
        self.client.delete(
            collection_name=PAPER_COLLECTION,
            filter=f'paper_arxiv_id == "{arxiv_id}"',
        )

    def delete_all(self):
        self.client.drop_collection(CHUNK_COLLECTION)
        self.client.drop_collection(PAPER_COLLECTION)
        self._ensure_collections()

    def count(self, arxiv_id: Optional[str] = None) -> int:
        if arxiv_id:
            results = self.client.query(
                collection_name=CHUNK_COLLECTION,
                filter=f'paper_arxiv_id == "{arxiv_id}"',
                output_fields=["id"],
            )
            return len(results)
        stats = self.client.get_collection_stats(CHUNK_COLLECTION)
        return int(stats["row_count"])

    # ==================== 论文级向量操作 ====================

    def insert_paper_embedding(self, arxiv_id: str, title: str, embedding: list[float]):
        """插入论文级向量（标题+摘要的 embedding）"""
        self.client.insert(
            collection_name=PAPER_COLLECTION,
            data=[{
                "paper_arxiv_id": arxiv_id,
                "title": title,
                "embedding": embedding,
            }],
        )

    def search_papers(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """论文级语义检索：返回最相关的论文列表"""
        from config import MILVUS_SEARCH_EF
        search_params = {"metric_type": "COSINE", "params": {"ef": MILVUS_SEARCH_EF}}

        results = self.client.search(
            collection_name=PAPER_COLLECTION,
            data=[query_embedding],
            limit=top_k,
            search_params=search_params,
            output_fields=["paper_arxiv_id", "title"],
        )

        hits = []
        if results and len(results) > 0:
            for hit in results[0]:
                hits.append({
                    "arxiv_id": hit["entity"]["paper_arxiv_id"],
                    "title": hit["entity"]["title"],
                    "score": hit["distance"],
                })
        return hits

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
