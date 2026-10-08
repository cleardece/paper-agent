"""Complete, provider-independent responses for the four graph LLM protocols."""

from __future__ import annotations

import json
import re
from typing import Any


class GraphResponseError(ValueError):
    """A graph response failed its contract; diagnostics contain no model text."""

    def __init__(self, message: str, *, error_kind: str,
                 diagnostics: dict[str, Any] | None = None):
        self.error_kind = error_kind
        self.diagnostics = dict(diagnostics or {})
        super().__init__(message)


def response_text(response: Any) -> str:
    def parts(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            # Reasoning blocks are never part of the graph JSON protocol.
            if value.get("type") in {"reasoning", "thinking", "reasoning_content"}:
                return []
            if isinstance(value.get("text"), str):
                return [value["text"]]
            return parts(value.get("content"))
        if isinstance(value, list):
            return [text for item in value for text in parts(item)]
        return []

    return "\n".join(parts(getattr(response, "content", response))).strip()


def invoke_graph_llm(llm: Any, prompt: str) -> tuple[str, dict[str, Any]]:
    client = llm.bind(temperature=0) if hasattr(llm, "bind") else llm
    response = client.invoke(prompt)
    content = response_text(response)
    metadata = getattr(response, "response_metadata", None) or {}
    usage = getattr(response, "usage_metadata", None) or {}
    extra = getattr(response, "additional_kwargs", None) or {}
    raw_usage = metadata.get("token_usage") or {}
    details = raw_usage.get("completion_tokens_details") or {}
    output_details = usage.get("output_token_details") or {}
    diagnostics = {
        "finish_reason": metadata.get("finish_reason", "unknown"),
        "input_tokens": usage.get("input_tokens", raw_usage.get("prompt_tokens")),
        "output_tokens": usage.get("output_tokens", raw_usage.get("completion_tokens")),
        "reasoning_tokens": output_details.get("reasoning", details.get("reasoning_tokens")),
        "response_length": len(content),
        "has_reasoning_content": bool(extra.get("reasoning_content")),
        "has_refusal": bool(extra.get("refusal")),
        "has_tool_calls": bool(getattr(response, "tool_calls", None)),
    }
    if not content:
        raise GraphResponseError(
            "图谱模型返回空正文 (response_length=0)",
            error_kind="llm_empty_response", diagnostics=diagnostics,
        )
    if diagnostics["finish_reason"] in {"length", "max_tokens"}:
        raise GraphResponseError(
            "图谱模型输出达到 token 上限，当前结果不完整",
            error_kind="llm_output_truncated", diagnostics=diagnostics,
        )
    return content, diagnostics


def load_json_rows(content: str, *, allow_single_object: bool = False,
                   diagnostics: dict[str, Any] | None = None,
                   ) -> tuple[list[dict[str, Any]], str]:
    """Require a complete document; never turn a valid prefix into success."""
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    detail = {**(diagnostics or {}), "response_length": len(content)}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        truncated = ((text.startswith("[") and not text.endswith("]"))
                     or (text.startswith("{") and not text.endswith("}")))
        raise GraphResponseError(
            "图谱模型返回的不是完整有效 JSON",
            error_kind="llm_output_truncated" if truncated else "llm_invalid_json",
            diagnostics={**detail, "json_error_position": exc.pos},
        ) from exc
    status = "parsed"
    if isinstance(value, dict):
        for key in ("candidates", "claims", "relations", "items", "decisions"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
        else:
            if allow_single_object:
                value, status = [value], "parsed_single_object"
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise GraphResponseError(
            "图谱模型 JSON 必须包含对象数组",
            error_kind="llm_invalid_json", diagnostics=detail,
        )
    return value, status if value else "empty_array"


def index_decisions(rows: list[dict[str, Any]], index_key: str,
                    expected: set[int], diagnostics: dict[str, Any],
                    ) -> dict[int, dict[str, Any]]:
    decisions: dict[int, dict[str, Any]] = {}
    invalid_count = duplicate_count = 0
    for row in rows:
        index = row.get(index_key)
        if not isinstance(index, int) or isinstance(index, bool) or index not in expected:
            invalid_count += 1
        elif index in decisions:
            duplicate_count += 1
        else:
            decisions[index] = row
    missing_count = len(expected - decisions.keys())
    if missing_count or invalid_count or duplicate_count:
        raise GraphResponseError(
            "图谱模型决策索引缺失、重复或无效",
            error_kind="llm_incomplete_response",
            diagnostics={**diagnostics, "requested_count": len(expected),
                         "returned_count": len(rows), "missing_decision_count": missing_count,
                         "invalid_decision_count": invalid_count,
                         "duplicate_decision_count": duplicate_count},
        )
    return decisions


def resolution_decisions(rows: list[dict[str, Any]], items: list[dict[str, Any]],
                         index_key: str, id_key: str, diagnostics: dict[str, Any],
                         ) -> list[dict[str, Any]]:
    allowed = {int(item[index_key]): item for item in items}
    indexed = index_decisions(rows, index_key, set(allowed), diagnostics)
    for index, decision in indexed.items():
        action = decision.get("decision")
        candidate_ids = {
            item[id_key] for item in allowed[index].get("candidates", [])
            if id_key in item
        }
        if action not in {"new", "merge"} or (
            action == "merge" and decision.get(id_key) not in candidate_ids
        ):
            raise GraphResponseError(
                "图谱消歧决策无效或选择了候选以外的 ID",
                error_kind="llm_incomplete_response",
                diagnostics={**diagnostics, "requested_count": len(items),
                             "returned_count": len(rows)},
            )
    return [indexed[int(item[index_key])] for item in items]
