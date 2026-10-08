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
from urllib.parse import urlparse
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
    GRAPH_LLM_MAX_OUTPUT_TOKENS,
    GRAPH_LLM_BASE_URL,
    GRAPH_LLM_PROMPT_OVERHEAD_TOKENS,
    GRAPH_LLM_TPM_LIMIT,
    GRAPH_RATE_LIMIT_BASE_DELAY_SECONDS,
    GRAPH_RATE_LIMIT_MAX_DELAY_SECONDS,
    GRAPH_RETRY_DELAY_SECONDS,
)

logger = logging.getLogger("paper-agent")
GRAPH_EXTRACT_CALL_MAX_CHARS = max(1, int(os.getenv("GRAPH_EXTRACT_CALL_MAX_CHARS", "5000")))
GRAPH_DECISION_CALL_MAX_ITEMS = max(1, int(os.getenv("GRAPH_DECISION_CALL_MAX_ITEMS", "4")))


class GraphExtractionTimeout(TimeoutError):
    pass


class GraphProcessError(RuntimeError):
    def __init__(self, message: str, *, error_kind: str | None = None,
                 diagnostics: dict[str, Any] | None = None,
                 mode: str | None = None, elapsed_seconds: float | None = None):
        super().__init__(message)
        self.error_kind = error_kind
        self.diagnostics = dict(diagnostics or {})
        self.mode = mode
        self.elapsed_seconds = elapsed_seconds


class GraphRateLimitPause(RuntimeError):
    def __init__(self, retry_after_seconds: int):
        self.retry_after_seconds = max(1, int(retry_after_seconds))
        super().__init__(
            f"图谱队列处于 TPM 限流等待，剩余 {self.retry_after_seconds} 秒"
        )


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
                raise GraphProcessError(
                    f"提取子进程返回无效 JSON: {detail}",
                    error_kind="subprocess_invalid_json",
                    mode=str(payload.get("mode", "extract")),
                ) from exc
            if not isinstance(result, dict):
                raise GraphProcessError(
                    "提取子进程返回的不是 JSON 对象",
                    error_kind="subprocess_invalid_json",
                    mode=str(payload.get("mode", "extract")),
                )
            if process.returncode != 0 or not result.get("ok"):
                message = result.get("error") or stderr.decode(
                    "utf-8", errors="replace"
                )[-1000:]
                raise GraphProcessError(
                    message or "提取子进程异常退出",
                    error_kind=result.get("error_kind"),
                    diagnostics=result.get("diagnostics"),
                    mode=result.get("mode"),
                    elapsed_seconds=result.get("elapsed_seconds"),
                )
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
    def _estimated_tokens(payload: dict[str, Any]) -> int:
        encoded = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        return max(
            1,
            (len(encoded) + 3) // 4
            + GRAPH_LLM_MAX_OUTPUT_TOKENS
            + GRAPH_LLM_PROMPT_OVERHEAD_TOKENS,
        )

    def _effective_tpm_limit(self) -> int:
        resolver = getattr(
            self.graph_repository, "effective_llm_tpm_limit", None
        )
        if callable(resolver):
            value = resolver(GRAPH_LLM_TPM_LIMIT)
            if isinstance(value, (int, float)):
                return max(0, int(value))
        return max(0, GRAPH_LLM_TPM_LIMIT)

    def _payload_over_budget(self, payload: dict[str, Any]) -> bool:
        limit = self._effective_tpm_limit()
        return bool(limit and self._estimated_tokens(payload) > limit)

    @classmethod
    def _incomplete_output(cls, diagnostics: Any) -> bool:
        if not isinstance(diagnostics, dict):
            return False
        llm = diagnostics.get("llm") or {}
        if diagnostics.get("parse_status") == "recovered_truncated":
            return True
        if isinstance(llm, dict) and llm.get("finish_reason") in {"length", "max_tokens"}:
            return True
        return any(
            cls._incomplete_output(diagnostics.get(key))
            for key in ("extract", "validate")
        ) or any(
            cls._incomplete_output(child)
            for child in (diagnostics.get("children") or [])
        )

    async def _wait_for_capacity(self, paper_id: str, wait_seconds: float) -> None:
        remaining = max(0.0, float(wait_seconds))
        while True:
            circuit = self.graph_repository.circuit_state()
            if (
                circuit.get("open")
                and circuit.get("pause_kind") == "llm_rate_limited"
            ):
                raise GraphRateLimitPause(
                    int(circuit.get("retry_after_seconds", 1))
                )
            if remaining <= 0:
                return
            self._heartbeat(paper_id)
            interval = min(
                remaining, max(0.05, float(GRAPH_JOB_HEARTBEAT_SECONDS))
            )
            started = asyncio.get_running_loop().time()
            await asyncio.sleep(interval)
            remaining -= max(interval, asyncio.get_running_loop().time() - started)

    async def _run_graph_call(
        self, paper_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        estimated_tokens = self._estimated_tokens(payload)
        reservation = self.graph_repository.reserve_llm_capacity(
            estimated_tokens, configured_tpm_limit=GRAPH_LLM_TPM_LIMIT
        )
        wait_seconds = (
            float(reservation.get("wait_seconds", 0.0))
            if isinstance(reservation, dict) else 0.0
        )
        await self._wait_for_capacity(paper_id, wait_seconds)
        result = await self.runner.run(
            payload, lambda: self._heartbeat(paper_id)
        )
        if self._incomplete_output(result.get("diagnostics")):
            raise GraphProcessError(
                "图谱模型输出不完整，不能保存部分结果",
                error_kind="llm_output_truncated",
                mode=str(payload.get("mode", "extract")),
                elapsed_seconds=result.get("elapsed_seconds"),
            )
        llm = (result.get("diagnostics") or {}).get("llm") or {}
        logger.info(
            "[ResearchGraph] %s %s 完成 %.1f 秒，输出 %s tokens，结束原因 %s",
            paper_id, payload.get("mode"),
            float(result.get("elapsed_seconds") or 0),
            llm.get("output_tokens"), llm.get("finish_reason"),
        )
        self.graph_repository.record_llm_success(
            configured_tpm_limit=GRAPH_LLM_TPM_LIMIT,
            recovery_success_threshold=20,
        )
        return result

    @staticmethod
    def _split_batch(batch: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split near half the character volume without changing segment content."""
        if not batch:
            return [], []
        if len(batch) == 1:
            item = dict(batch[0])
            content = str(item.get("content", ""))
            if len(content) < 2:
                return batch, []
            midpoint = len(content) // 2
            left_edge = content.rfind(" ", 0, midpoint)
            right_edge = content.find(" ", midpoint)
            if left_edge <= 0 and right_edge < 0:
                split_at = midpoint
            elif left_edge <= 0:
                split_at = right_edge
            elif right_edge < 0:
                split_at = left_edge
            else:
                split_at = (
                    left_edge
                    if midpoint - left_edge <= right_edge - midpoint
                    else right_edge
                )
            left_item = {**item, "content": content[:split_at].strip()}
            right_item = {**item, "content": content[split_at:].strip()}
            return [left_item], [right_item]
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
        if exc.error_kind in {
            "llm_output_truncated", "llm_invalid_json", "llm_incomplete_response",
        }:
            return True
        message = str(exc)
        return "不是可恢复 JSON" in message or "不是有效 JSON" in message

    @staticmethod
    def _empty_response_error(exc: GraphProcessError) -> bool:
        if exc.error_kind == "llm_empty_response":
            return True
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

    async def _extract_batch(
        self,
        paper_id: str,
        paper: dict[str, Any],
        batch: list[dict[str, Any]],
        *,
        split_depth: int = 0,
        protocol_depth: int = 0,
    ) -> dict[str, Any]:
        payload = {"mode": "extract", "paper": paper, "batch": batch}
        left, right = self._split_batch(batch)
        over_tpm_budget = self._payload_over_budget(payload)
        over_input_budget = (
            sum(len(str(item.get("content", ""))) for item in batch)
            > GRAPH_EXTRACT_CALL_MAX_CHARS
        )
        if (over_tpm_budget or over_input_budget) and right and split_depth < 8:
            left_result = await self._extract_batch(
                paper_id, paper, left, split_depth=split_depth + 1,
                protocol_depth=protocol_depth,
            )
            right_result = await self._extract_batch(
                paper_id, paper, right, split_depth=split_depth + 1,
                protocol_depth=protocol_depth,
            )
            return {
                "candidates": [
                    *left_result.get("candidates", []),
                    *right_result.get("candidates", []),
                ],
                "diagnostics": {
                    "recovery": "tpm_budget_split" if over_tpm_budget else "input_size_split",
                    "split_depth": split_depth + 1,
                    "children": [
                        left_result.get("diagnostics", {}),
                        right_result.get("diagnostics", {}),
                    ],
                },
            }
        try:
            return await self._run_graph_call(paper_id, payload)
        except GraphProcessError as exc:
            can_split = (
                self._json_protocol_error(exc)
                and not self._empty_response_error(exc)
                and right
                and protocol_depth < 2
            )
            if not can_split:
                raise
            logger.warning(
                "[ResearchGraph] %s 抽取 JSON 不完整，将当前批次拆为 %d/%d 个片段重试",
                paper_id, len(left), len(right),
            )
            left_result = await self._extract_batch(
                paper_id, paper, left, split_depth=split_depth + 1,
                protocol_depth=protocol_depth + 1,
            )
            right_result = await self._extract_batch(
                paper_id, paper, right, split_depth=split_depth + 1,
                protocol_depth=protocol_depth + 1,
            )
            return {
                "candidates": [
                    *left_result.get("candidates", []),
                    *right_result.get("candidates", []),
                ],
                "diagnostics": {
                    "recovery": "adaptive_json_split",
                    "split_depth": split_depth + 1,
                    "children": [
                        left_result.get("diagnostics", {}),
                        right_result.get("diagnostics", {}),
                    ],
                },
            }

    @staticmethod
    def _validation_batch(
        batch: list[dict[str, Any]], candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        chunk_ids: set[int] = set()
        for candidate in candidates:
            try:
                chunk_ids.add(int(candidate.get("evidence_chunk_index")))
            except (TypeError, ValueError):
                continue
        selected = [
            dict(item) for item in batch
            if int(item.get("chunk_index", -1)) in chunk_ids
        ]
        return selected or batch

    @staticmethod
    def _trim_validation_batch(
        batch: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        max_chars: int,
    ) -> list[dict[str, Any]]:
        evidence_by_chunk: dict[int, list[str]] = {}
        for candidate in candidates:
            try:
                chunk_index = int(candidate.get("evidence_chunk_index"))
            except (TypeError, ValueError):
                continue
            evidence = str(candidate.get("evidence") or "").strip()
            if evidence:
                evidence_by_chunk.setdefault(chunk_index, []).append(evidence)
        trimmed: list[dict[str, Any]] = []
        width = max(400, int(max_chars))
        for item in batch:
            value = dict(item)
            content = str(value.get("content", ""))
            if len(content) <= width:
                trimmed.append(value)
                continue
            chunk_index = int(value.get("chunk_index", -1))
            positions = [
                content.find(evidence)
                for evidence in evidence_by_chunk.get(chunk_index, [])
                if content.find(evidence) >= 0
            ]
            center = positions[0] if positions else len(content) // 2
            start = max(0, min(len(content) - width, center - width // 3))
            value["content"] = content[start:start + width]
            trimmed.append(value)
        return trimmed

    @staticmethod
    def _merge_validation_results(
        left: dict[str, Any], right: dict[str, Any],
        recovery: str = "tpm_budget_split",
    ) -> dict[str, Any]:
        numeric_keys = {
            "validated_count", "supported_count", "uncertain_count",
            "rejected_count", "missing_decision_count",
        }
        left_diagnostics = dict(left.get("diagnostics", {}))
        right_diagnostics = dict(right.get("diagnostics", {}))
        diagnostics = {
            key: int(left_diagnostics.get(key, 0)) + int(right_diagnostics.get(key, 0))
            for key in numeric_keys
        }
        diagnostics.update({
            "recovery": recovery,
            "children": [left_diagnostics, right_diagnostics],
        })
        return {
            "relations": [
                *left.get("relations", []), *right.get("relations", [])
            ],
            "diagnostics": diagnostics,
        }

    async def _validate_candidates(
        self,
        paper_id: str,
        paper: dict[str, Any],
        batch: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        *,
        protocol_depth: int = 0,
    ) -> dict[str, Any]:
        if not candidates:
            return {"relations": [], "diagnostics": {
                "validated_count": 0, "supported_count": 0,
                "uncertain_count": 0, "rejected_count": 0,
            }}
        validation_batch = self._validation_batch(batch, candidates)
        payload = {
            "mode": "validate", "paper": paper,
            "batch": validation_batch, "candidates": candidates,
        }
        over_tpm_budget = self._payload_over_budget(payload)
        over_item_budget = len(candidates) > GRAPH_DECISION_CALL_MAX_ITEMS
        if (over_tpm_budget or over_item_budget) and len(candidates) > 1:
            midpoint = max(1, len(candidates) // 2)
            left = await self._validate_candidates(
                paper_id, paper, batch, candidates[:midpoint],
                protocol_depth=protocol_depth,
            )
            right = await self._validate_candidates(
                paper_id, paper, batch, candidates[midpoint:],
                protocol_depth=protocol_depth,
            )
            return self._merge_validation_results(
                left, right,
                "tpm_budget_split" if over_tpm_budget else "decision_size_split",
            )
        if over_tpm_budget:
            for max_chars in (2000, 1000, 400):
                validation_batch = self._trim_validation_batch(
                    validation_batch, candidates, max_chars
                )
                payload = {
                    "mode": "validate", "paper": paper,
                    "batch": validation_batch, "candidates": candidates,
                }
                if not self._payload_over_budget(payload):
                    break
        try:
            return await self._run_graph_call(paper_id, payload)
        except GraphProcessError as exc:
            if (not self._json_protocol_error(exc) or len(candidates) < 2
                    or protocol_depth >= 2):
                raise
            midpoint = len(candidates) // 2
            left = await self._validate_candidates(
                paper_id, paper, batch, candidates[:midpoint],
                protocol_depth=protocol_depth + 1,
            )
            right = await self._validate_candidates(
                paper_id, paper, batch, candidates[midpoint:],
                protocol_depth=protocol_depth + 1,
            )
            return self._merge_validation_results(
                left, right, "adaptive_protocol_split",
            )

    async def _extract_and_validate_batch(
        self,
        paper_id: str,
        paper: dict[str, Any],
        batch: list[dict[str, Any]],
        *,
        batch_index: int | None = None,
    ) -> dict[str, Any]:
        checkpoint = None
        if batch_index is not None:
            checkpoint = self.graph_repository.get_extraction_checkpoint(
                paper_id, self.worker_id, batch_index
            )
        if checkpoint and self._incomplete_output(checkpoint.get("diagnostics")):
            checkpoint = None
        if checkpoint:
            extracted = {
                "candidates": list(checkpoint.get("candidates", [])),
                "diagnostics": dict(checkpoint.get("diagnostics", {})),
            }
        else:
            extracted = await self._extract_batch(paper_id, paper, batch)
            if batch_index is not None and not self.graph_repository.save_extraction_checkpoint(
                paper_id,
                self.worker_id,
                batch_index,
                list(extracted.get("candidates", [])),
                dict(extracted.get("diagnostics", {})),
            ):
                raise RuntimeError("保存图谱抽取阶段检查点失败，任务租约可能已丢失")

        validated = await self._validate_candidates(
            paper_id, paper, batch, list(extracted.get("candidates", []))
        )
        extract_diagnostics = dict(extracted.get("diagnostics", {}))
        diagnostics = {
            **extract_diagnostics,
            "extract": extract_diagnostics,
            "validate": validated.get("diagnostics", {}),
        }
        return {
            "relations": validated.get("relations", []),
            "diagnostics": diagnostics,
        }

    @staticmethod
    def _merge_resolution_results(
        left: dict[str, Any], right: dict[str, Any], recovery: str,
    ) -> dict[str, Any]:
        left_diagnostics = dict(left.get("diagnostics", {}))
        right_diagnostics = dict(right.get("diagnostics", {}))
        diagnostics = {
            key: int(left_diagnostics.get(key, 0))
            + int(right_diagnostics.get(key, 0))
            for key in ("requested_count", "returned_count", "accepted_count")
        }
        diagnostics.update({
            "llm_called": bool(
                left_diagnostics.get("llm_called")
                or right_diagnostics.get("llm_called")
            ),
            "recovery": recovery,
            "children": [left_diagnostics, right_diagnostics],
        })
        return {
            "decisions": [*left.get("decisions", []), *right.get("decisions", [])],
            "diagnostics": diagnostics,
        }

    async def _run_resolution_call(
        self, paper_id: str, mode: str, payload: dict[str, Any], *,
        protocol_depth: int = 0,
    ) -> dict[str, Any]:
        items = list(payload.get("items", []))
        request = {"mode": mode, **payload, "items": items}
        over_tpm_budget = self._payload_over_budget(request)
        over_item_budget = len(items) > GRAPH_DECISION_CALL_MAX_ITEMS
        if (over_tpm_budget or over_item_budget) and len(items) > 1:
            midpoint = max(1, len(items) // 2)
            left = await self._run_resolution_call(
                paper_id, mode, {**payload, "items": items[:midpoint]},
                protocol_depth=protocol_depth,
            )
            right = await self._run_resolution_call(
                paper_id, mode, {**payload, "items": items[midpoint:]},
                protocol_depth=protocol_depth,
            )
            return self._merge_resolution_results(
                left, right,
                "tpm_budget_split" if over_tpm_budget else "decision_size_split",
            )
        if over_tpm_budget and len(items) == 1:
            compacted = items
            for max_chars in (800, 400, 200):
                compacted = [self._compact_resolution_value(items[0], max_chars)]
                request = {"mode": mode, **payload, "items": compacted}
                if not self._payload_over_budget(request):
                    break
        try:
            return await self._run_graph_call(paper_id, request)
        except GraphProcessError as exc:
            if (not self._json_protocol_error(exc) or len(items) < 2
                    or protocol_depth >= 2):
                raise
            midpoint = len(items) // 2
            left = await self._run_resolution_call(
                paper_id, mode, {**payload, "items": items[:midpoint]},
                protocol_depth=protocol_depth + 1,
            )
            right = await self._run_resolution_call(
                paper_id, mode, {**payload, "items": items[midpoint:]},
                protocol_depth=protocol_depth + 1,
            )
            return self._merge_resolution_results(
                left, right, "adaptive_protocol_split",
            )

    @classmethod
    def _compact_resolution_value(cls, value: Any, max_chars: int) -> Any:
        if isinstance(value, str):
            return value if len(value) <= max_chars else value[:max_chars]
        if isinstance(value, list):
            return [cls._compact_resolution_value(item, max_chars) for item in value]
        if isinstance(value, dict):
            return {
                key: cls._compact_resolution_value(item, max_chars)
                for key, item in value.items()
            }
        return value

    async def process_once(self) -> bool:
        if not self._legacy_failures_normalized:
            finalized = self.graph_repository.finalize_nonretryable_retries()
            if finalized:
                logger.warning(
                    "[ResearchGraph] 已终止 %d 个旧版误排队的配额失败任务",
                    finalized,
                )
            revived = self.graph_repository.revive_rate_limited_jobs()
            if revived:
                logger.warning(
                    "[ResearchGraph] 已恢复 %d 个旧版 TPM 限流失败任务",
                    revived,
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
            old_diagnostics = job.get("batch_diagnostics") or {}
            if any(
                not old_diagnostics.get(str(index))
                or self._incomplete_output(old_diagnostics.get(str(index)))
                for index in completed_batches
            ):
                if not self.graph_repository.reset_incomplete_progress(
                    paper_id, self.worker_id
                ):
                    raise RuntimeError("清理旧版不完整图谱批次失败，任务租约可能已丢失")
                completed_batches.clear()
                logger.warning("[ResearchGraph] %s 已重新抽取旧版不完整批次", paper_id)
            for batch_index, batch in enumerate(batches):
                if batch_index in completed_batches:
                    continue
                logger.info(
                    "[ResearchGraph] %s 处理批次 %d/%d",
                    paper_id, batch_index + 1, len(batches),
                )
                batch_result = await self._extract_and_validate_batch(
                    paper_id, safe_paper, batch, batch_index=batch_index,
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
                    return await self._run_resolution_call(
                        paper_id, mode, payload
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
        except GraphRateLimitPause as exc:
            status = self.graph_repository.defer_rate_limited(
                paper_id,
                self.worker_id,
                str(exc),
                retry_delay_seconds=exc.retry_after_seconds,
            )
            logger.warning(
                "[ResearchGraph] %s 在等待调用时检测到全局 TPM 暂停，"
                "任务状态 %s且不消耗业务重试",
                paper_id, status,
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
            local_graph = urlparse(GRAPH_LLM_BASE_URL).hostname in {
                "127.0.0.1", "localhost", "::1",
            }
            if local_graph and exc.error_kind == "llm_connection_error":
                status = self.graph_repository.defer_local_unavailable(
                    paper_id, self.worker_id, str(exc),
                    retry_delay_seconds=GRAPH_RETRY_DELAY_SECONDS,
                )
                self.graph_repository.record_infrastructure_failure(
                    str(exc), threshold=1,
                    pause_seconds=GRAPH_RETRY_DELAY_SECONDS,
                )
                logger.warning(
                    "[ResearchGraph] %s 本地图谱模型不可连接，任务状态 %s，"
                    "已暂停队列且不消耗业务重试",
                    paper_id, status,
                )
                return True
            if rate_limited:
                congestion = self.graph_repository.record_rate_limit(
                    str(exc),
                    base_delay_seconds=GRAPH_RATE_LIMIT_BASE_DELAY_SECONDS,
                    max_delay_seconds=GRAPH_RATE_LIMIT_MAX_DELAY_SECONDS,
                    configured_tpm_limit=GRAPH_LLM_TPM_LIMIT,
                    minimum_tpm_limit=(
                        GRAPH_LLM_MAX_OUTPUT_TOKENS
                        + GRAPH_LLM_PROMPT_OVERHEAD_TOKENS
                        + 1000
                    ),
                )
                retry_delay = int(congestion.get(
                    "retry_delay_seconds", GRAPH_RATE_LIMIT_BASE_DELAY_SECONDS
                ))
                status = self.graph_repository.defer_rate_limited(
                    paper_id,
                    self.worker_id,
                    str(exc),
                    retry_delay_seconds=retry_delay,
                )
                logger.warning(
                    "[ResearchGraph] %s 遇到临时 TPM 限流，已暂停图谱队列 %d 秒，"
                    "任务状态 %s且不消耗业务重试: %s",
                    paper_id, retry_delay, status, exc,
                )
                return True
            error_kind = (
                "llm_quota_exhausted" if quota_exhausted
                else "llm_empty_response" if empty_response
                else exc.error_kind or "process_or_llm_error"
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
            logger.warning(
                "[ResearchGraph] %s %s 失败，类型 %s，耗时 %s 秒，状态 %s: %s",
                paper_id, exc.mode or "graph", error_kind,
                exc.elapsed_seconds, status, exc,
            )
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
