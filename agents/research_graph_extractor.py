"""Full-paper batched extraction and independent relation adjudication."""

from __future__ import annotations

import json
from typing import Any

from knowledge_graph.llm_protocol import (
    GraphResponseError, index_decisions, invoke_graph_llm,
    load_json_rows, response_text,
)
from knowledge_graph.models import apply_verification
from knowledge_graph.schema.entity_types import ENTITY_TYPES
from knowledge_graph.schema.predicates import LEGACY_RELATIONS, PREDICATES

# Compatibility exports for older callers. Canonical predicates live in
# knowledge_graph.schema.predicates; the UI-facing relation vocabulary remains
# accepted while V3 jobs are being upgraded.
RELATIONS = LEGACY_RELATIONS


class ResearchGraphExtractor:
    """Prepare full-text batches and run one LLM phase at a time."""

    REFERENCE_HEADINGS = ("references", "bibliography", "参考文献")
    VERDICTS = {"supported", "uncertain", "rejected"}

    def __init__(self, llm: Any, max_chunks: int | None = None,
                 batch_chars: int = 12000, segment_chars: int = 4000):
        self.llm = llm
        self.max_chunks = max_chunks
        self.batch_chars = batch_chars
        self.segment_chars = segment_chars
        self._last_llm_diagnostics: dict[str, Any] = {}

    @staticmethod
    def _heading(chunk: dict[str, Any]) -> str:
        metadata = chunk.get("metadata", {})
        return str(metadata.get("heading") or metadata.get("section") or "")

    def _is_reference(self, chunk: dict[str, Any]) -> bool:
        heading = self._heading(chunk).strip().lower()
        return any(marker in heading for marker in self.REFERENCE_HEADINGS)

    def _split_content(self, content: str) -> list[str]:
        """Split long chunks near sentence boundaries without dropping text."""
        parts: list[str] = []
        start = 0
        while start < len(content):
            end = min(start + self.segment_chars, len(content))
            if end < len(content):
                search_start = start + self.segment_chars // 2
                candidates = [
                    content.rfind(marker, search_start, end)
                    for marker in ("\n", ". ", "。", "！", "？")
                ]
                boundary = max(candidates)
                if boundary >= search_start:
                    end = boundary + (1 if content[boundary] != "." else 2)
            parts.append(content[start:end])
            start = end
        return parts

    def build_segments(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        for fallback_index, chunk in enumerate(chunks):
            if self._is_reference(chunk):
                continue
            content = str(chunk.get("content", "")).strip()
            if not content:
                continue
            metadata = dict(chunk.get("metadata", {}))
            chunk_index = int(chunk.get("chunk_index", fallback_index))
            for segment_index, segment_content in enumerate(self._split_content(content)):
                segments.append({
                    "chunk_index": chunk_index,
                    "segment_index": segment_index,
                    "heading": metadata.get("heading") or metadata.get("section") or "",
                    "page": metadata.get("page", 0),
                    "content": segment_content,
                })
        return segments

    def build_batches(self, chunks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        batches: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_chars = 0
        for segment in self.build_segments(chunks):
            size = len(segment["content"])
            if current and current_chars + size > self.batch_chars:
                batches.append(current)
                current = []
                current_chars = 0
            current.append(segment)
            current_chars += size
        if current:
            batches.append(current)
        return batches

    def select_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compatibility adapter for older callers and local tests."""
        selected = [chunk for chunk in chunks if not self._is_reference(chunk)]
        return selected[:self.max_chunks] if self.max_chunks else selected

    @staticmethod
    def _load_json(content: str) -> tuple[list[dict[str, Any]], str]:
        return load_json_rows(content, allow_single_object=True)

    @staticmethod
    def _response_text(response: Any) -> str:
        return response_text(response)

    def _invoke(self, prompt: str) -> str:
        content, self._last_llm_diagnostics = invoke_graph_llm(self.llm, prompt)
        return content

    @staticmethod
    def _candidate_valid(candidate: dict[str, Any], chunk_ids: set[int]) -> bool:
        try:
            chunk_index = int(candidate.get("evidence_chunk_index"))
        except (TypeError, ValueError):
            return False
        raw_predicate = (
            candidate.get("predicate_raw") or candidate.get("predicate")
            or candidate.get("relation")
        )
        object_name = candidate.get("object_name") or candidate.get("object") or candidate.get("target_name")
        subject_name = candidate.get("subject_name") or candidate.get("subject")
        legacy = candidate.get("relation") in LEGACY_RELATIONS
        return (
            bool(raw_predicate)
            and len(str(object_name or "").strip()) >= 2
            and (legacy or len(str(subject_name or "").strip()) >= 2)
            and len(str(candidate.get("evidence", "")).strip()) >= 12
            and chunk_index in chunk_ids
        )

    def extract_batch(self, paper: dict[str, Any],
                      batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch:
            return {"candidates": [], "diagnostics": {
                "result_reason": "no_selected_chunks", "selected_chunk_count": 0,
                "model_candidate_count": 0, "extractor_rejected_count": 0,
            }}
        prompt = (
            "你是严谨的学术事实抽取器。只依据给定正文抽取实体之间明确陈述的事实，"
            "不要把相关工作、引用论文或单纯提及误判为当前论文结论。\n"
            "请逐句检查全部正文，尽可能完整地抽取所有符合条件的事实。"
            "同一主体存在多个不同的 predicate 或 object 时必须分别输出，"
            "不要只返回最显著的一条；只有全文没有符合条件的事实时才返回空数组 []。\n"
            f"当前论文：{paper.get('title', '')}\n"
            f"正文片段：{json.dumps(batch, ensure_ascii=False)}\n\n"
            "仅输出单行 JSON 数组，不要 Markdown、解释或代码围栏。不要自行归一化"
            "实体类型、qualifier、stance 或 predicate；这些由下一阶段统一完成。"
            "若同一事实重复出现只保留证据最完整的一项。"
            "每项格式为："
            '{"subject_name":"实体原名","predicate_raw":"原文关系短语",'
            '"object_name":"实体原名",'
            '"evidence_chunk_index":整数,"evidence":"原文中连续的1到3个完整句子"}。'
            "evidence 必须逐字来自给定正文；证据不足时不要输出。"
        )
        content = self._invoke(prompt)
        candidates, parse_status = self._load_json(content)
        chunk_ids = {int(item["chunk_index"]) for item in batch}
        valid = [item for item in candidates if self._candidate_valid(item, chunk_ids)]
        if not candidates:
            reason = "model_returned_no_relations"
        elif not valid:
            raise ValueError("图谱抽取器返回的候选均不符合字段或证据约束")
        else:
            reason = "candidates_ready_for_validation"
        return {"candidates": valid, "diagnostics": {
            "result_reason": reason,
            "selected_chunk_count": len(batch),
            "model_candidate_count": len(candidates),
            "extractor_rejected_count": len(candidates) - len(valid),
            "parse_status": parse_status,
            "llm": dict(self._last_llm_diagnostics),
            "response_excerpt": content[:800],
        }}

    def validate_batch(self, paper: dict[str, Any], batch: list[dict[str, Any]],
                       candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if not candidates:
            return {"relations": [], "diagnostics": {
                "validated_count": 0, "supported_count": 0,
                "uncertain_count": 0, "rejected_count": 0,
            }}
        numbered = [{"candidate_index": index, **candidate}
                    for index, candidate in enumerate(candidates)]
        prompt = (
            "你是独立的学术事实核验和 schema 归一化器。请逐项判断证据是否支持候选"
            "事实，并把实体类型、predicate、qualifier、stance 和 confidence 一次完成"
            "归一化。不要增加第三次 normalization 调用。特别区分本文结论、相关工作、"
            "引用、附录复述和单纯提及。\n"
            f"当前论文：{paper.get('title', '')}\n"
            f"原文片段：{json.dumps(batch, ensure_ascii=False)}\n"
            f"候选关系：{json.dumps(numbered, ensure_ascii=False)}\n\n"
            f"实体类型只能是：{','.join(sorted(ENTITY_TYPES))}。"
            f"predicate 只能是：{','.join(sorted(PREDICATES))}；无法判断必须返回 UNKNOWN。"
            "仅输出 JSON 数组，每个候选返回一项："
            '{"candidate_index":整数,"verdict":"supported|uncertain|rejected",'
            '"valid":布尔值,"subject_name":"标准原名","subject_type":"固定类型",'
            '"predicate":"固定 predicate","object_name":"标准原名",'
            '"object_type":"固定类型","qualifiers":{},"stance":"support|contradict",'
            '"confidence":0到1,"reason":"简短原因"}。'
            "只有原文明确定义当前论文主体和关系时才 supported；需要跨片段推断、主体"
            "不清或关系类型可能不同则 uncertain；证据不支持则 rejected。"
        )
        content = self._invoke(prompt)
        decisions, parse_status = self._load_json(content)
        invalid_verdicts = sum(
            decision.get("verdict") not in self.VERDICTS for decision in decisions
        )
        if invalid_verdicts:
            raise GraphResponseError(
                "图谱核验器返回无效 verdict",
                error_kind="llm_incomplete_response",
                diagnostics={"invalid_verdict_count": invalid_verdicts,
                             **self._last_llm_diagnostics},
            )
        decision_map = index_decisions(
            decisions, "candidate_index", set(range(len(candidates))),
            self._last_llm_diagnostics,
        )
        relations = [
            apply_verification(candidate, decision_map[index], paper)
            for index, candidate in enumerate(candidates)
        ]
        return {"relations": relations, "diagnostics": {
            "validated_count": len(relations),
            "supported_count": sum(item["validation_verdict"] == "supported" for item in relations),
            "uncertain_count": sum(item["validation_verdict"] == "uncertain" for item in relations),
            "rejected_count": sum(item["validation_verdict"] == "rejected" for item in relations),
            "returned_decision_count": len(decisions),
            "missing_or_invalid_decision_count": 0,
            "invalid_decision_count": 0,
            "duplicate_decision_count": 0,
            "parse_status": parse_status,
            "llm": dict(self._last_llm_diagnostics),
            "response_excerpt": content[:800],
        }}

    def extract_with_diagnostics(self, paper: dict[str, Any],
                                 chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Compatibility path: extract once from selected raw chunks."""
        selected = self.select_chunks(chunks)
        batch = [{
            "chunk_index": int(chunk.get("chunk_index", index)),
            "segment_index": 0,
            "heading": self._heading(chunk),
            "page": chunk.get("metadata", {}).get("page", 0),
            "content": str(chunk.get("content", ""))[:self.segment_chars],
        } for index, chunk in enumerate(selected)]
        try:
            result = self.extract_batch(paper, batch)
        except ValueError:
            return {"candidates": [], "diagnostics": {
                "result_reason": "invalid_model_response",
                "selected_chunk_count": len(batch),
                "model_candidate_count": 0,
                "extractor_rejected_count": 0,
            }}
        if result["diagnostics"].get("result_reason") == "candidates_ready_for_validation":
            result["diagnostics"]["result_reason"] = "candidates_ready_for_evidence_validation"
        return result

    def extract(self, paper: dict[str, Any],
                chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.extract_with_diagnostics(paper, chunks)["candidates"]
