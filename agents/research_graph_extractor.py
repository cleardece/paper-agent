"""Full-paper batched extraction and independent relation adjudication."""

from __future__ import annotations

import json
from typing import Any

from storage.research_graph import ENTITY_TYPES, RELATIONS


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
        start = content.find("[")
        end = content.rfind("]")
        if start < 0 or end < start:
            return [], "invalid_json"
        try:
            value = json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            return [], "invalid_json"
        if not isinstance(value, list):
            return [], "invalid_json"
        return value, "empty_array" if not value else "parsed"

    @staticmethod
    def _response_text(response: Any) -> str:
        return str(getattr(response, "content", response) or "").strip()

    def _invoke(self, prompt: str) -> str:
        client = self.llm.bind(temperature=0) if hasattr(self.llm, "bind") else self.llm
        return self._response_text(client.invoke(prompt))

    @staticmethod
    def _candidate_valid(candidate: dict[str, Any], chunk_ids: set[int]) -> bool:
        try:
            chunk_index = int(candidate.get("evidence_chunk_index"))
        except (TypeError, ValueError):
            return False
        return (
            candidate.get("relation") in RELATIONS
            and candidate.get("target_type") in ENTITY_TYPES
            and len(str(candidate.get("target_name", "")).strip()) >= 2
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
            "你是严谨的学术关系抽取器。主语始终是当前论文。只依据给定正文，"
            "不要把相关工作、引用论文或单纯提及误判为当前论文的贡献。\n"
            f"当前论文：{paper.get('title', '')}\n"
            f"正文片段：{json.dumps(batch, ensure_ascii=False)}\n\n"
            "仅输出 JSON 数组。关系可选：proposes、uses、improves、compares_with、"
            "evaluates_on、measures_with、studies；实体类型可选：method、dataset、"
            "metric、task。每项格式为："
            '{"relation":"...","target_type":"...","target_name":"实体原名",'
            '"evidence_chunk_index":整数,"evidence":"原文中连续的1到3个完整句子"}。'
            "evidence 必须逐字来自给定正文；证据不足时不要输出。"
        )
        content = self._invoke(prompt)
        candidates, parse_status = self._load_json(content)
        if parse_status == "invalid_json":
            raise ValueError("图谱抽取器返回的不是有效 JSON 数组")
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
            "你是独立的学术关系核验器。请结合当前论文、原文上下文和候选关系判断"
            "证据是否真的支持‘当前论文’与实体之间的关系。特别区分：本文提出、本文"
            "使用、相关工作提出、引用、附录复述和单纯提及。\n"
            f"当前论文：{paper.get('title', '')}\n"
            f"原文片段：{json.dumps(batch, ensure_ascii=False)}\n"
            f"候选关系：{json.dumps(numbered, ensure_ascii=False)}\n\n"
            "仅输出 JSON 数组，每个候选必须返回一项："
            '{"candidate_index":整数,"verdict":"supported|uncertain|rejected",'
            '"relation":"核验后的关系类型","reason":"简短原因"}。'
            "只有原文明确定义当前论文主体和关系时才 supported；需要跨片段推断、主体"
            "不清或关系类型可能不同则 uncertain；证据不支持则 rejected。"
        )
        content = self._invoke(prompt)
        decisions, parse_status = self._load_json(content)
        if parse_status == "invalid_json":
            raise ValueError("图谱核验器返回的不是有效 JSON 数组")
        decision_map: dict[int, dict[str, Any]] = {}
        for decision in decisions:
            try:
                index = int(decision.get("candidate_index"))
            except (TypeError, ValueError):
                continue
            verdict = str(decision.get("verdict", ""))
            relation = str(decision.get("relation", ""))
            if verdict == "rejected" and relation not in RELATIONS:
                relation = str(candidates[index].get("relation", "")) \
                    if 0 <= index < len(candidates) else ""
                decision = {**decision, "relation": relation}
            if (
                0 <= index < len(candidates)
                and verdict in self.VERDICTS and relation in RELATIONS
            ):
                decision_map[index] = decision
        if len(decision_map) != len(candidates):
            raise ValueError("图谱核验器没有完整返回每个候选的判断")
        relations = []
        for index, candidate in enumerate(candidates):
            decision = decision_map[index]
            verdict = decision["verdict"]
            validated_relation = decision["relation"]
            relations.append({
                **candidate,
                "validation_verdict": verdict,
                "validated_relation": validated_relation,
                "validation_reason": str(decision.get("reason") or "未提供核验说明"),
            })
        return {"relations": relations, "diagnostics": {
            "validated_count": len(relations),
            "supported_count": sum(item["validation_verdict"] == "supported" for item in relations),
            "uncertain_count": sum(item["validation_verdict"] == "uncertain" for item in relations),
            "rejected_count": sum(item["validation_verdict"] == "rejected" for item in relations),
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
