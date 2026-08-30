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


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        extractor = ResearchGraphExtractor(get_graph_llm())
        result = extractor.extract_with_diagnostics(
            payload["paper"], payload.get("chunks", []),
        )
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
