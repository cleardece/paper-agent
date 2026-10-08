"""One batched LLM call for ambiguous Entity Resolution items."""

from __future__ import annotations

import json
from typing import Any

from knowledge_graph.llm_protocol import (
    invoke_graph_llm, load_json_rows, resolution_decisions,
)


def resolve_entity_batch(llm: Any, items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"decisions": [], "diagnostics": {"llm_called": False}}
    prompt = (
        "你是实体消歧器。每个 mention 只能在给出的候选中选择 merge，或返回 new。"
        "判断名称、缩写、类型、领域、上下文和关系上下文；不要跨类型合并。\n"
        f"歧义项：{json.dumps(items, ensure_ascii=False)}\n"
        "仅输出 JSON 数组，每项格式："
        '{"mention_index":整数,"decision":"merge|new","entity_id":"merge时的候选ID",'
        '"alias":"可加入的别名","reason":"简短原因"}。'
    )
    content, llm_diagnostics = invoke_graph_llm(llm, prompt)
    raw, parse_status = load_json_rows(content, diagnostics=llm_diagnostics)
    decisions = resolution_decisions(
        raw, items, "mention_index", "entity_id", llm_diagnostics,
    )
    return {"decisions": decisions, "diagnostics": {
        "llm_called": True, "requested_count": len(items),
        "returned_count": len(raw), "accepted_count": len(decisions),
        "parse_status": parse_status, "llm": llm_diagnostics,
        "response_excerpt": content[:800],
    }}
