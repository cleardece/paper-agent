"""LLM extraction for evidence-backed paper relations."""

from __future__ import annotations

import json
from typing import Any

from storage.research_graph import ENTITY_TYPES, RELATIONS


class ResearchGraphExtractor:
    SIGNAL_HEADINGS = (
        "abstract", "introduction", "method", "approach", "experiment", "result",
        "evaluation", "conclusion", "limitation", "摘要", "方法", "实验", "结果", "结论",
    )

    def __init__(self, llm: Any, max_chunks: int = 12):
        self.llm = llm
        self.max_chunks = max_chunks

    def select_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        scored: list[tuple[int, dict[str, Any]]] = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            heading = " ".join(str(metadata.get(key, "")) for key in ("heading", "section")).lower()
            if "reference" in heading or metadata.get("is_appendix"):
                continue
            score = sum(1 for keyword in self.SIGNAL_HEADINGS if keyword in heading)
            scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], int(item[1].get("chunk_index", 0))))
        return [chunk for _, chunk in scored[:self.max_chunks]]

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
    def _is_candidate_valid(candidate: dict[str, Any], chunk_ids: set[int]) -> bool:
        try:
            confidence = float(candidate.get("confidence", -1))
            chunk_index = int(candidate.get("evidence_chunk_index"))
        except (TypeError, ValueError):
            return False
        return (
            candidate.get("relation") in RELATIONS
            and candidate.get("target_type") in ENTITY_TYPES
            and len(str(candidate.get("target_name", "")).strip()) >= 2
            and len(str(candidate.get("evidence", "")).strip()) >= 12
            and chunk_index in chunk_ids
            and 0 <= confidence <= 1
        )

    def extract_with_diagnostics(self, paper: dict[str, Any],
                                 chunks: list[dict[str, Any]]) -> dict[str, Any]:
        selected = self.select_chunks(chunks)
        if not selected:
            return {
                "candidates": [],
                "diagnostics": {
                    "result_reason": "no_selected_chunks",
                    "selected_chunk_count": 0,
                    "model_candidate_count": 0,
                    "extractor_rejected_count": 0,
                },
            }
        source = []
        for chunk in selected:
            metadata = chunk.get("metadata", {})
            source.append({
                "chunk_index": chunk.get("chunk_index"),
                "heading": metadata.get("heading") or metadata.get("section") or "",
                "page": metadata.get("page", 0),
                "content": str(chunk.get("content", ""))[:1800],
            })
        prompt = (
            "你是严谨的学术信息抽取器。只从给出的论文片段提取明确陈述的关系，"
            "不要根据常识补全，也不要提取参考文献中的关系。\n"
            f"论文标题：{paper.get('title', '')}\n"
            f"片段：{json.dumps(source, ensure_ascii=False)}\n\n"
            "仅输出 JSON 数组；每项必须是 "
            '{"relation":"proposes|uses|compares_with","target_type":"method|dataset|metric",'
            '"target_name":"实体原名","evidence_chunk_index":整数,"evidence":"能在该片段中找到的原文短句","confidence":0到1之间的数值}。'
        )
        client = self.llm.bind(temperature=0) if hasattr(self.llm, "bind") else self.llm
        response = client.invoke(prompt)
        content = getattr(response, "content", str(response)).strip()
        candidates, parse_status = self._load_json(content)
        valid_indices = {int(chunk.get("chunk_index", -1)) for chunk in selected}
        valid = [
            item for item in candidates
            if self._is_candidate_valid(item, valid_indices)
        ]
        if parse_status == "invalid_json":
            reason = "invalid_model_response"
        elif not candidates:
            reason = "model_returned_no_relations"
        elif not valid:
            reason = "all_candidates_rejected_by_extractor"
        else:
            reason = "candidates_ready_for_evidence_validation"
        return {
            "candidates": valid,
            "diagnostics": {
                "result_reason": reason,
                "selected_chunk_count": len(selected),
                "model_candidate_count": len(candidates),
                "extractor_rejected_count": len(candidates) - len(valid),
                "response_excerpt": content[:800],
            },
        }

    def extract(self, paper: dict[str, Any],
                chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.extract_with_diagnostics(paper, chunks)["candidates"]
