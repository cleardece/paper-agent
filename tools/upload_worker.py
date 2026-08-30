"""Single-worker PDF upload queue with checkpointed status and cleanup."""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any

from tools.pdf_parser import MinerUParseError


logger = logging.getLogger("paper-agent")


class UploadQueueWorker:
    def __init__(self, container: Any, repository: Any, wakeup: asyncio.Event,
                 graph_wakeup: asyncio.Event | None = None):
        self.container = container
        self.repository = repository
        self.wakeup = wakeup
        self.graph_wakeup = graph_wakeup
        self.stopped = False

    async def process_job(self, job: dict[str, Any]) -> None:
        while True:
            try:
                await self._process_once(job)
                return
            except Exception as exc:
                logger.error("[UploadQueue] %s 处理失败: %s", job["filename"], exc, exc_info=True)
                self._cleanup_partial_paper(job["arxiv_id"])
                attempts = int(job.get("attempt_count", 1))
                if attempts < int(job.get("max_attempts", 2)):
                    self.repository.mark_retrying(job["job_id"], "失败后自动重试（第 2 次）")
                    job["attempt_count"] = attempts + 1
                    continue
                self.repository.update_job(
                    job["job_id"],
                    "failed",
                    detail="处理失败，已清理半成品并继续下一篇",
                    error=str(exc),
                    finished_at=datetime.now(timezone.utc),
                )
                return

    async def _process_once(self, job: dict[str, Any]) -> None:
        loop = asyncio.get_running_loop()
        job_started = time.monotonic()
        job_id = job["job_id"]
        arxiv_id = job["arxiv_id"]
        filename = job["filename"]

        self.repository.update_job(job_id, "parsing", detail="正在使用 MinerU 解析")
        result = await loop.run_in_executor(None, self.container.pdf_parser.parse, job["pdf_path"])
        parse_source = str(result.get("source", "unknown"))

        self.repository.update_job(
            job_id, "chunking", detail="正在生成论文片段", parse_source=parse_source
        )
        chunks = self.container.pdf_parser.chunk(result["sections"])

        self.container.mongodb.upsert_paper({
            "arxiv_id": arxiv_id,
            "title": result.get("title", filename),
            "abstract": "",
            "authors": [],
            "pdf_url": f"local://{filename}",
            "status": "chunked",
        })
        self.container.mongodb.insert_chunks([
            {
                "paper_arxiv_id": arxiv_id,
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "metadata": chunk.get("metadata", {}),
            }
            for chunk in chunks
        ])

        self.repository.update_job(
            job_id, "indexing", detail="正在生成向量并入库", chunk_count=len(chunks)
        )
        texts = [chunk["content"] for chunk in chunks]
        logger.info("[UploadQueue] %s：开始生成 %d 个向量", filename, len(texts))
        embedding_started = time.monotonic()
        vectors = await loop.run_in_executor(None, self.container.embedder.embed_texts, texts)
        logger.info(
            "[UploadQueue] %s：向量生成完成，耗时 %.2fs",
            filename,
            time.monotonic() - embedding_started,
        )
        records = [
            {
                "paper_arxiv_id": arxiv_id,
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "embedding": vectors[index],
                "section": chunk.get("metadata", {}).get("section", ""),
                "page": chunk.get("metadata", {}).get("page", 0),
                "heading": chunk.get("metadata", {}).get("heading", ""),
            }
            for index, chunk in enumerate(chunks)
        ]
        logger.info("[UploadQueue] %s：开始写入 Milvus（%d 条）", filename, len(records))
        milvus_started = time.monotonic()
        self.container.milvus.insert(records)
        logger.info(
            "[UploadQueue] %s：Milvus 写入完成，耗时 %.2fs",
            filename,
            time.monotonic() - milvus_started,
        )

        title = result.get("title", filename)
        logger.info("[UploadQueue] %s：开始写入论文级向量", filename)
        paper_embedding_started = time.monotonic()
        paper_embedding = await loop.run_in_executor(
            None, lambda: self.container.embedder.embed_texts([title])[0]
        )
        self.container.milvus.insert_paper_embedding(arxiv_id, title, paper_embedding)
        logger.info(
            "[UploadQueue] %s：论文级向量写入完成，耗时 %.2fs",
            filename,
            time.monotonic() - paper_embedding_started,
        )
        self.container.mongodb.update_paper_status(arxiv_id, "indexed", title_embedding=paper_embedding)
        # 图谱是低优先级的后处理：论文此刻已经可被正常 RAG 使用。
        if getattr(self.container, "research_graph", None):
            self.container.research_graph.enqueue(arxiv_id, source="new")
            self.container.mongodb.papers.update_one(
                {"arxiv_id": arxiv_id}, {"$set": {"graph_status": "pending"}}
            )
            if self.graph_wakeup:
                self.graph_wakeup.set()

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("[UploadQueue] %s：任务完成，总耗时 %.2fs", filename, time.monotonic() - job_started)
        self.repository.update_job(
            job_id,
            "completed",
            detail="已完成",
            chunk_count=len(chunks),
            parse_source=parse_source,
            finished_at=datetime.now(timezone.utc),
        )

    def _cleanup_partial_paper(self, arxiv_id: str) -> None:
        try:
            if getattr(self.container, "research_graph", None):
                self.container.research_graph.delete_auto_data_for_paper(arxiv_id)
        except Exception as exc:
            logger.warning("[UploadQueue] 清理图谱半成品失败 %s: %s", arxiv_id, exc)
        try:
            self.container.mongodb.delete_paper(arxiv_id)
        except Exception as exc:
            logger.warning("[UploadQueue] 清理 Mongo 半成品失败 %s: %s", arxiv_id, exc)
        try:
            self.container.milvus.delete_by_paper(arxiv_id)
        except Exception as exc:
            logger.warning("[UploadQueue] 清理 Milvus 半成品失败 %s: %s", arxiv_id, exc)

    async def run(self) -> None:
        while not self.stopped:
            job = self.repository.claim_next_job()
            if not job:
                await self.wakeup.wait()
                self.wakeup.clear()
                continue
            if self.repository.is_multi_file_batch(job["batch_id"]):
                manager = getattr(self.container.pdf_parser, "mineru_manager", None)
                with (manager.batch_lease() if manager else nullcontext()):
                    await self._process_batch(job)
            else:
                await self.process_job(job)

    async def _process_batch(self, first_job: dict[str, Any]) -> None:
        await self.process_job(first_job)
        while not self.stopped:
            next_job = self.repository.claim_next_job(first_job["batch_id"])
            if not next_job:
                return
            await self.process_job(next_job)

    def stop(self) -> None:
        self.stopped = True
        self.wakeup.set()
