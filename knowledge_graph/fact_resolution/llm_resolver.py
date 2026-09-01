"""One batched LLM call for ambiguous Fact Resolution items."""

from __future__ import annotations

import json
from typing import Any


def _load_array(content: str) -> list[dict[str, Any]]:
    start, end = content.find("["), content.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        value = json.loads(content[start:end + 1])
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


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
    client = llm.bind(temperature=0) if hasattr(llm, "bind") else llm
    response = client.invoke(prompt)
    content = str(getattr(response, "content", response) or "").strip()
    raw = _load_array(content)
    allowed = {int(item["fact_index"]): item for item in items}
    decisions = []
    for decision in raw:
        try:
            index = int(decision.get("fact_index"))
        except (TypeError, ValueError):
            continue
        source = allowed.get(index)
        action = str(decision.get("decision", ""))
        fact_id = str(decision.get("fact_id") or "")
        candidate_ids = {item["fact_id"] for item in source.get("candidates", [])} if source else set()
        if source and (action == "new" or (action == "merge" and fact_id in candidate_ids)):
            decisions.append({**decision, "fact_index": index})
    return {"decisions": decisions, "diagnostics": {
        "llm_called": True, "requested_count": len(items),
        "returned_count": len(raw), "accepted_count": len(decisions),
        "response_excerpt": content[:800],
    }}
