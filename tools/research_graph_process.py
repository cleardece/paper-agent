"""One isolated research-graph extraction attempt.

The parent worker owns the deadline. Killing this process also kills a hung LLM
network request, which a thread-based timeout cannot guarantee.
"""

from __future__ import annotations

import json
import sys
import traceback

from agents.research_graph_extractor import ResearchGraphExtractor
from config import get_graph_llm
from knowledge_graph.entity_resolution.llm_resolver import resolve_entity_batch
from knowledge_graph.fact_resolution.llm_resolver import resolve_fact_batch


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        llm = get_graph_llm()
        extractor = ResearchGraphExtractor(llm)
        mode = payload.get("mode", "extract")
        if mode == "extract":
            result = extractor.extract_batch(
                payload["paper"], payload.get("batch", []),
            )
        elif mode == "validate":
            result = extractor.validate_batch(
                payload["paper"], payload.get("batch", []),
                payload.get("candidates", []),
            )
        elif mode == "resolve_entities":
            result = resolve_entity_batch(llm, payload.get("items", []))
        elif mode == "resolve_facts":
            result = resolve_fact_batch(llm, payload.get("items", []))
        else:
            raise ValueError(f"未知图谱子进程模式: {mode}")
        json.dump({"ok": True, **result}, sys.stdout, ensure_ascii=False)
        return 0
    except BaseException as exc:
        json.dump(
            {
                "ok": False,
                "error": str(exc),
                "error_kind": "llm_or_extractor_error",
                "traceback": traceback.format_exc(limit=8),
            },
            sys.stdout,
            ensure_ascii=False,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
