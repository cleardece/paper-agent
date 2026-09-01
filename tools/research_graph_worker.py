"""Recoverable low-priority worker for research-graph extraction."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from agents.research_graph_extractor import ResearchGraphExtractor
from knowledge_graph.graph.repository import CanonicalGraphRepository
from knowledge_graph.pipeline import KnowledgeGraphPipeline
from knowledge_graph.vector_index import KnowledgeGraphVectorIndex
from config import (
    GRAPH_CIRCUIT_FAILURE_THRESHOLD,
    GRAPH_CIRCUIT_PAUSE_SECONDS,
    GRAPH_EXTRACTION_TIMEOUT_SECONDS,
    GRAPH_JOB_HEARTBEAT_SECONDS,
    GRAPH_JOB_LEASE_SECONDS,
    GRAPH_RETRY_DELAY_SECONDS,
)

logger = logging.getLogger("paper-agent")


class GraphExtractionTimeout(TimeoutError):
    pass


class GraphProcessError(RuntimeError):
    pass


class GraphSubprocessRunner:
    """Run one LLM attempt in a process that can be forcibly terminated."""

    def __init__(self, timeout_seconds: int, heartbeat_seconds: int):
        self.timeout_seconds = timeout_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.current_process: asyncio.subprocess.Process | None = None

    @staticmethod
    def _command() -> tuple[str, ...]:
        return (sys.executable, "-m", "tools.research_graph_process")

    async def run(self, payload: dict[str, Any],
                  heartbeat: Callable[[], None]) -> dict[str, Any]:
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = await asyncio.create_subprocess_exec(
            *self._command(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=environment,
            creationflags=creationflags,
        )
        self.current_process = process
        communicate_task = asyncio.create_task(
            process.communicate(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout_seconds
        try:
            while not communicate_task.done():
                remaining = deadline - loop.time()
                if remaining <= 0:
                    process.kill()
                    await process.wait()
                    communicate_task.cancel()
                    await asyncio.gather(communicate_task, return_exceptions=True)
                    raise GraphExtractionTimeout(
                        f"图谱提取超过 {self.timeout_seconds} 秒，子进程已终止"
                    )
                try:
                    await asyncio.wait_for(
                        asyncio.shield(communicate_task),
                        timeout=min(self.heartbeat_seconds, remaining),
                    )
                except asyncio.TimeoutError:
                    heartbeat()
            stdout, stderr = await communicate_task
            try:
                result = json.loads(stdout.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                detail = stderr.decode("utf-8", errors="replace")[-1000:]
                raise GraphProcessError(f"提取子进程返回无效 JSON: {detail}") from exc
            if process.returncode != 0 or not result.get("ok"):
                message = result.get("error") or stderr.decode(
                    "utf-8", errors="replace"
                )[-1000:]
                raise GraphProcessError(message or "提取子进程异常退出")
            return result
        except asyncio.CancelledError:
            if process.returncode is None:
                process.kill()
                await process.wait()
            communicate_task.cancel()
            await asyncio.gather(communicate_task, return_exceptions=True)
            raise
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
            self.current_process = None


class ResearchGraphWorker:
    """Claim one leased job at a time only while the PDF queue is idle."""

    def __init__(self, container: Any, graph_repository: Any, upload_queue: Any,
                 wakeup: asyncio.Event):
        self.container = container
        self.graph_repository = graph_repository
        self.upload_queue = upload_queue
        self.wakeup = wakeup
        self.stopped = False
        self._legacy_failures_normalized = False
        self.worker_id = f"graph-{uuid4().hex[:12]}"
        self.lease_seconds = max(
            GRAPH_JOB_LEASE_SECONDS,
            GRAPH_EXTRACTION_TIMEOUT_SECONDS + 2 * GRAPH_JOB_HEARTBEAT_SECONDS,
        )
        self.runner = GraphSubprocessRunner(
            GRAPH_EXTRACTION_TIMEOUT_SECONDS,
            GRAPH_JOB_HEARTBEAT_SECONDS,
        )
        canonical = getattr(graph_repository, "canonical", None)
        if isinstance(canonical, CanonicalGraphRepository):
            vector_index = KnowledgeGraphVectorIndex(getattr(container, "milvus", None))
            self.kg_pipeline = KnowledgeGraphPipeline(
                canonical, getattr(container, "embedder", None), vector_index,
            )
        else:
            # Compatibility for focused worker tests and older repository adapters.
            self.kg_pipeline = None

    @staticmethod
    def _safe_paper(paper: dict[str, Any]) -> dict[str, Any]:
        return {
            "arxiv_id": str(paper.get("arxiv_id", "")),
            "title": str(paper.get("title", "")),
        }

    @staticmethod
    def _safe_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "chunk_index": int(chunk.get("chunk_index", index)),
                "content": str(chunk.get("content", "")),
                "metadata": {
                    key: value
                    for key, value in dict(chunk.get("metadata", {})).items()
                    if key in {"heading", "section", "page", "is_appendix", "level", "merged"}
                },
            }
            for index, chunk in enumerate(chunks)
        ]

    def _heartbeat(self, paper_id: str) -> None:
        if not self.graph_repository.heartbeat(
            paper_id, self.worker_id, self.lease_seconds
        ):
            logger.warning("[ResearchGraph] %s 的任务租约已丢失", paper_id)

    @staticmethod
    def _split_batch(batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split near half the character volume without changing segment content."""
        if len(batch) < 2:
            return batch, []
        target = sum(len(str(item.get("content", ""))) for item in batch) / 2
        current = 0
        split_at = 1
        for index, item in enumerate(batch[:-1], start=1):
            current += len(str(item.get("content", "")))
            split_at = index
            if current >= target:
                break
        return batch[:split_at], batch[split_at:]

    @staticmethod
    def _json_protocol_error(exc: GraphProcessError) -> bool:
        message = str(exc)
        return "不是可恢复 JSON" in message or "不是有效 JSON" in message

    @staticmethod
    def _empty_response_error(exc: GraphProcessError) -> bool:
        return "response_length=0" in str(exc).lower().replace(" ", "")

    @staticmethod
    def _quota_exhausted(exc: GraphProcessError) -> bool:
        message = str(exc).lower().replace("\\_", "_")
        return any(marker in message for marker in (
            "insufficient_quota",
            "allocated quota exceeded",
            "increase your quota limit",
            "exceeded your current quota",
        ))

    @staticmethod
    def _rate_limited(exc: GraphProcessError) -> bool:
        message = str(exc).lower().replace("\\_", "_")
        return any(marker in message for marker in (
            "inference tpm exhausted",
            "429001",
            "rate_limit",
            "rate limit",
            "too many requests",
        ))

    async def _extract_and_validate_batch(
        self,
        paper_id: str,
        paper: dict[str, Any],
        batch: list[dict[str, Any]],
        *,
        split_depth: int = 0,
    ) -> dict[str, Any]:
        try:
            extracted = await self.runner.run(
                {"mode": "extract", "paper": paper, "batch": batch},
                lambda: self._heartbeat(paper_id),
            )
        except GraphProcessError as exc:
            left, right = self._split_batch(batch)
            can_split = (
                self._json_protocol_error(exc)
                and not self._empty_response_error(exc)
                and right
                and split_depth < 2
            )
            if can_split:
                logger.warning(
                    "[ResearchGraph] %s 抽取 JSON 不完整，将当前批次拆为 %d/%d 个片段重试",
                    paper_id, len(left), len(right),
                )
                left_result = await self._extract_and_validate_batch(
                    paper_id, paper, left, split_depth=split_depth + 1,
                )
                right_result = await self._extract_and_validate_batch(
                    paper_id, paper, right, split_depth=split_depth + 1,
                )
                return {
                    "relations": [*left_result["relations"], *right_result["relations"]],
                    "diagnostics": {
                        "recovery": "adaptive_json_split",
                        "split_depth": split_depth + 1,
                        "children": [left_result["diagnostics"], right_result["diagnostics"]],
                    },
                }
            raise

        candidates = extracted.get("candidates", [])
        if candidates:
            validated = await self.runner.run(
                {
                    "mode": "validate", "paper": paper,
                    "batch": batch, "candidates": candidates,
                },
                lambda: self._heartbeat(paper_id),
            )
        else:
            validated = {"relations": [], "diagnostics": {
                "validated_count": 0, "supported_count": 0,
                "uncertain_count": 0, "rejected_count": 0,
            }}
        return {
            "relations": validated.get("relations", []),
            "diagnostics": {
                "extract": extracted.get("diagnostics", {}),
                "validate": validated.get("diagnostics", {}),
            },
        }

    async def process_once(self) -> bool:
        if not self._legacy_failures_normalized:
            finalized = self.graph_repository.finalize_nonretryable_retries()
            if finalized:
                logger.warning(
                    "[ResearchGraph] 已终止 %d 个旧版误排队的配额失败任务",
                    finalized,
                )
            self._legacy_failures_normalized = True
        if self.upload_queue.count_pending() > 0:
            return False
        self.graph_repository.recover_expired_leases()
        circuit = self.graph_repository.circuit_state()
        if circuit["open"]:
            return False
        job = self.graph_repository.claim_next_job(
            self.worker_id, self.lease_seconds
        )
        if not job:
            return False

        paper_id = job["paper_id"]
        paper = self.container.mongodb.get_paper(paper_id)
        if not paper or paper.get("status") != "indexed":
            diagnostics = {
                "result_reason": "paper_not_indexed",
                "selected_chunk_count": 0,
                "model_candidate_count": 0,
                "extractor_rejected_count": 0,
                "evidence_rejected_count": 0,
            }
            self.graph_repository.complete_job(
                paper_id, self.worker_id, 0, diagnostics
            )
            return True

        chunks = self.container.mongodb.get_chunks_by_paper(paper_id)
        safe_paper = self._safe_paper(paper)
        safe_chunks = self._safe_chunks(chunks)
        batches = ResearchGraphExtractor(None).build_batches(safe_chunks)
        logger.info(
            "[ResearchGraph] %s 开始第 %d/%d 次提取（版本 %s）",
            paper_id, job.get("attempt_count", 1), job.get("max_attempts", 2),
            job.get("graph_version"),
        )
        try:
            if not self.graph_repository.set_batch_total(
                paper_id, self.worker_id, len(batches)
            ):
                raise RuntimeError("保存图谱全文批次数失败，任务租约可能已丢失")
            completed_batches = set(job.get("completed_batches", []))
            for batch_index, batch in enumerate(batches):
                if batch_index in completed_batches:
                    continue
                logger.info(
                    "[ResearchGraph] %s 处理批次 %d/%d",
                    paper_id, batch_index + 1, len(batches),
                )
                batch_result = await self._extract_and_validate_batch(
                    paper_id, safe_paper, batch,
                )
                batch_diagnostics = batch_result["diagnostics"]
                if not self.graph_repository.save_batch_result(
                    paper_id, self.worker_id, batch_index, len(batches),
                    batch_result["relations"], batch_diagnostics,
                ):
                    raise RuntimeError("保存图谱批次检查点失败，任务租约可能已丢失")
                self._heartbeat(paper_id)

            staged = self.graph_repository.staged_relations(
                paper_id, self.worker_id
            )
            resolution_diagnostics: dict[str, Any] = {}
            if self.kg_pipeline is not None:
                async def resolution_slow_path(
                    mode: str, payload: dict[str, Any]
                ) -> dict[str, Any]:
                    return await self.runner.run(
                        {"mode": mode, **payload}, lambda: self._heartbeat(paper_id)
                    )

                canonical = await self.kg_pipeline.process(
                    safe_paper, safe_chunks, staged, resolution_slow_path,
                )
                resolution_diagnostics = canonical.get("diagnostics", {})
                edges = self.graph_repository.upsert_canonical_claims(
                    safe_paper, canonical.get("claims", []),
                )
            else:
                edges = self.graph_repository.upsert_relations(
                    safe_paper, safe_chunks, staged
                )
            diagnostics = {
                "result_reason": (
                    "relations_ready" if edges else "no_verified_relations"
                ),
                "batch_total": len(batches),
                "batch_completed": len(batches),
                "candidate_count": len(staged),
                "auto_verified_count": sum(
                    edge.get("review_status") == "auto_verified" for edge in edges
                ),
                "needs_review_count": sum(
                    edge.get("review_status") == "needs_review" for edge in edges
                ),
                "evidence_rejected_count": max(0, len(staged) - len(edges)),
                "resolution": resolution_diagnostics,
            }
            self.graph_repository.complete_job(
                paper_id, self.worker_id, len(edges), diagnostics
            )
            self.graph_repository.record_infrastructure_success()
            logger.info(
                "[ResearchGraph] %s 提取完成，关系数 %d，原因 %s",
                paper_id, len(edges), diagnostics.get("result_reason"),
            )
        except GraphExtractionTimeout as exc:
            status = self.graph_repository.fail_attempt(
                paper_id, self.worker_id, str(exc), "timeout",
                GRAPH_RETRY_DELAY_SECONDS,
            )
            self.graph_repository.record_infrastructure_failure(
                str(exc), threshold=GRAPH_CIRCUIT_FAILURE_THRESHOLD,
                pause_seconds=GRAPH_CIRCUIT_PAUSE_SECONDS,
            )
            logger.warning("[ResearchGraph] %s 超时，状态 %s", paper_id, status)
        except GraphProcessError as exc:
            quota_exhausted = self._quota_exhausted(exc)
            rate_limited = not quota_exhausted and self._rate_limited(exc)
            empty_response = self._empty_response_error(exc)
            error_kind = (
                "llm_quota_exhausted" if quota_exhausted
                else "llm_rate_limited" if rate_limited
                else "llm_empty_response" if empty_response
                else "process_or_llm_error"
            )
            status = self.graph_repository.fail_attempt(
                paper_id, self.worker_id, str(exc), error_kind,
                GRAPH_RETRY_DELAY_SECONDS, retryable=not quota_exhausted,
            )
            self.graph_repository.record_infrastructure_failure(
                str(exc), threshold=(
                    1 if quota_exhausted else GRAPH_CIRCUIT_FAILURE_THRESHOLD
                ),
                pause_seconds=GRAPH_CIRCUIT_PAUSE_SECONDS,
            )
            logger.warning("[ResearchGraph] %s 子进程失败，状态 %s: %s", paper_id, status, exc)
        except Exception as exc:
            status = self.graph_repository.fail_attempt(
                paper_id, self.worker_id, str(exc), "worker_error",
                GRAPH_RETRY_DELAY_SECONDS,
            )
            logger.exception("[ResearchGraph] %s 工作者失败，状态 %s", paper_id, status)
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
