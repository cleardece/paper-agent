"""One batched LLM call for ambiguous Entity Resolution items."""

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
    client = llm.bind(temperature=0) if hasattr(llm, "bind") else llm
    response = client.invoke(prompt)
    content = str(getattr(response, "content", response) or "").strip()
    raw = _load_array(content)
    allowed = {int(item["mention_index"]): item for item in items}
    decisions = []
    for decision in raw:
        try:
            index = int(decision.get("mention_index"))
        except (TypeError, ValueError):
            continue
        source = allowed.get(index)
        action = str(decision.get("decision", ""))
        entity_id = str(decision.get("entity_id") or "")
        candidate_ids = {item["entity_id"] for item in source.get("candidates", [])} if source else set()
        if source and (
            action == "new" or (action == "merge" and entity_id in candidate_ids)
        ):
            decisions.append({**decision, "mention_index": index})
    return {"decisions": decisions, "diagnostics": {
        "llm_called": True, "requested_count": len(items),
        "returned_count": len(raw), "accepted_count": len(decisions),
        "response_excerpt": content[:800],
    }}
