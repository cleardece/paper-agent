"""Full-paper batched extraction and independent relation adjudication."""

from __future__ import annotations

import json
import hashlib
import re
from typing import Any

from knowledge_graph.models import apply_verification, claim_from_candidate
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
        """Parse strict JSON while salvaging only independently valid objects."""
        text = str(content or "").strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()

        def normalized(value: Any) -> tuple[list[dict[str, Any]], str] | None:
            if isinstance(value, list):
                rows = [item for item in value if isinstance(item, dict)]
                return rows, "empty_array" if not value else "parsed"
            if isinstance(value, dict):
                for key in ("candidates", "claims", "relations", "items", "decisions"):
                    if isinstance(value.get(key), list):
                        rows = [item for item in value[key] if isinstance(item, dict)]
                        return rows, "empty_array" if not value[key] else "parsed"
                return [value], "parsed_single_object"
            return None

        try:
            direct = normalized(json.loads(text))
        except json.JSONDecodeError:
            direct = None
        if direct is not None:
            return direct

        decoder = json.JSONDecoder()
        for start, marker in enumerate(text):
            if marker != "[":
                continue
            try:
                value, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            parsed = normalized(value)
            if parsed is not None:
                return parsed

        # A response cut off at the output-token boundary can still contain
        # complete leading objects. Decode each object independently and never
        # synthesize the incomplete tail.
        start = text.find("[")
        if start >= 0:
            cursor = start + 1
            recovered: list[dict[str, Any]] = []
            while cursor < len(text):
                while cursor < len(text) and text[cursor] in " \r\n\t,":
                    cursor += 1
                if cursor >= len(text) or text[cursor] == "]":
                    break
                try:
                    value, consumed = decoder.raw_decode(text[cursor:])
                except json.JSONDecodeError:
                    break
                if not isinstance(value, dict):
                    break
                recovered.append(value)
                cursor += consumed
            if recovered:
                return recovered, "recovered_truncated"
        return [], "invalid_json"

    @staticmethod
    def _response_text(response: Any) -> str:
        content = getattr(response, "content", response)

        def text_parts(value: Any) -> list[str]:
            if isinstance(value, str):
                return [value]
            if isinstance(value, dict):
                if isinstance(value.get("text"), str):
                    return [value["text"]]
                if "content" in value:
                    return text_parts(value["content"])
                return []
            if isinstance(value, list):
                parts: list[str] = []
                for item in value:
                    parts.extend(text_parts(item))
                return parts
            return [str(value)] if value is not None else []

        return "\n".join(text_parts(content)).strip()

    def _invoke(self, prompt: str) -> str:
        client = self.llm.bind(temperature=0) if hasattr(self.llm, "bind") else self.llm
        return self._response_text(client.invoke(prompt))

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
        if parse_status == "invalid_json":
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
            raise ValueError(
                "图谱抽取器返回的不是可恢复 JSON "
                f"(response_length={len(content)}, response_digest={digest})"
            )
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
        if parse_status == "invalid_json":
            raise ValueError("图谱核验器返回的不是有效 JSON 数组")
        decision_map: dict[int, dict[str, Any]] = {}
        invalid_decision_count = 0
        duplicate_decision_count = 0
        for decision in decisions:
            try:
                index = int(decision.get("candidate_index"))
            except (TypeError, ValueError):
                invalid_decision_count += 1
                continue
            verdict = str(decision.get("verdict", ""))
            if not 0 <= index < len(candidates) or verdict not in self.VERDICTS:
                invalid_decision_count += 1
                continue
            if index in decision_map:
                duplicate_decision_count += 1
                # Conflicting duplicates are unsafe. Leave the candidate for the
                # conservative synthesized uncertain decision below.
                if decision_map[index] != decision:
                    decision_map.pop(index, None)
                continue
            decision_map[index] = decision
        relations = []
        for index, candidate in enumerate(candidates):
            decision = decision_map.get(index)
            if decision is None:
                base = claim_from_candidate(candidate, paper)
                decision = {
                    "verdict": "uncertain",
                    "valid": base.get("predicate") != "UNKNOWN",
                    "subject_name": base.get("subject_name"),
                    "subject_type": base.get("subject_type"),
                    "predicate": base.get("predicate"),
                    "object_name": base.get("object_name"),
                    "object_type": base.get("object_type"),
                    "qualifiers": base.get("qualifiers", {}),
                    "stance": base.get("stance", "support"),
                    "confidence": min(float(base.get("confidence", 0.5)), 0.49),
                    "reason": "missing_or_invalid_decision",
                }
            relations.append(apply_verification(candidate, decision, paper))
        missing_count = len(candidates) - len(decision_map)
        return {"relations": relations, "diagnostics": {
            "validated_count": len(relations),
            "supported_count": sum(item["validation_verdict"] == "supported" for item in relations),
            "uncertain_count": sum(item["validation_verdict"] == "uncertain" for item in relations),
            "rejected_count": sum(item["validation_verdict"] == "rejected" for item in relations),
            "returned_decision_count": len(decisions),
            "missing_or_invalid_decision_count": missing_count,
            "invalid_decision_count": invalid_decision_count,
            "duplicate_decision_count": duplicate_decision_count,
            "parse_status": parse_status,
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
