"""Low-priority worker for evidence-backed research graph jobs."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agents.research_graph_extractor import ResearchGraphExtractor

logger = logging.getLogger("paper-agent")


class ResearchGraphWorker:
    """Runs one LLM extraction only when no PDF job is queued or processing."""

    def __init__(self, container: Any, graph_repository: Any, upload_queue: Any,
                 wakeup: asyncio.Event):
        self.container = container
        self.graph_repository = graph_repository
        self.upload_queue = upload_queue
        self.wakeup = wakeup
        self.stopped = False
        self.extractor = ResearchGraphExtractor(container.llm)

    def _set_status(self, paper_id: str, graph_status: str, **fields: Any) -> None:
        self.container.mongodb.papers.update_one(
            {"arxiv_id": paper_id}, {"$set": {"graph_status": graph_status, **fields}}
        )

    async def process_once(self) -> bool:
        if self.upload_queue.count_pending() > 0:
            return False
        job = self.graph_repository.claim_next_job()
        if not job:
            return False
        paper_id = job["paper_id"]
        self._set_status(paper_id, "extracting")
        try:
            paper = self.container.mongodb.get_paper(paper_id)
            if not paper or paper.get("status") != "indexed":
                self.graph_repository.complete_job(paper_id, 0)
                return True
            chunks = self.container.mongodb.get_chunks_by_paper(paper_id)
            loop = asyncio.get_running_loop()
            candidates = await loop.run_in_executor(None, self.extractor.extract, paper, chunks)
            edges = self.graph_repository.upsert_relations(paper, chunks, candidates)
            self.graph_repository.complete_job(paper_id, len(edges))
            self._set_status(paper_id, "ready", graph_edge_count=len(edges), graph_error=None)
            logger.info("[ResearchGraph] %s 提取完成，关系数 %d", paper_id, len(edges))
        except Exception as exc:
            retrying = self.graph_repository.fail_or_retry_job(paper_id, str(exc))
            self._set_status(paper_id, "pending" if retrying else "failed",
                             graph_error=str(exc))
            logger.warning("[ResearchGraph] %s 提取失败%s: %s", paper_id,
                           "，将重试一次" if retrying else "，已停止重试", exc)
        return True

    async def run(self) -> None:
        while not self.stopped:
            processed = await self.process_once()
            if processed:
                continue
            try:
                await asyncio.wait_for(self.wakeup.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            self.wakeup.clear()

    def stop(self) -> None:
        self.stopped = True
        self.wakeup.set()

