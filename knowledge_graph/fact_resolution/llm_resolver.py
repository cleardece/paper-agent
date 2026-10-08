"""One batched LLM call for ambiguous Fact Resolution items."""

from __future__ import annotations

import json
from typing import Any

from knowledge_graph.llm_protocol import (
    invoke_graph_llm, load_json_rows, resolution_decisions,
)


def resolve_fact_batch(llm: Any, items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"decisions": [], "diagnostics": {"llm_called": False}}
    prompt = (
        "你是事实消歧器。实体已经完成 canonical resolution。优先比较 subject、固定"
        "predicate、object、qualifier 和 context；句子 embedding 不能单独决定合并。\n"
        f"歧义项：{json.dumps(items, ensure_ascii=False)}\n"
        "仅输出 JSON 数组，每项格式："
        '{"fact_index":整数,"decision":"merge|new","fact_id":"merge时的候选ID",'
        '"reason":"简短原因"}。'
    )
    content, llm_diagnostics = invoke_graph_llm(llm, prompt)
    raw, parse_status = load_json_rows(content, diagnostics=llm_diagnostics)
    decisions = resolution_decisions(
        raw, items, "fact_index", "fact_id", llm_diagnostics,
    )
    return {"decisions": decisions, "diagnostics": {
        "llm_called": True, "requested_count": len(items),
        "returned_count": len(raw), "accepted_count": len(decisions),
        "parse_status": parse_status, "llm": llm_diagnostics,
        "response_excerpt": content[:800],
    }}
