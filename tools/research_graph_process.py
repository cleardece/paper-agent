"""One isolated research-graph extraction attempt.

The parent worker owns the deadline. Killing this process also kills a hung LLM
network request, which a thread-based timeout cannot guarantee.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import json
import sys
import time
import traceback


def _execute(payload: dict) -> dict:
    # Import inside the stdout redirect, so dependency diagnostics cannot corrupt
    # the single JSON document expected by the parent process.
    from agents.research_graph_extractor import ResearchGraphExtractor
    from config import get_graph_llm
    from knowledge_graph.entity_resolution.llm_resolver import resolve_entity_batch
    from knowledge_graph.fact_resolution.llm_resolver import resolve_fact_batch

    llm = get_graph_llm()
    extractor = ResearchGraphExtractor(llm)
    mode = payload.get("mode", "extract")
    if mode == "extract":
        return extractor.extract_batch(payload["paper"], payload.get("batch", []))
    if mode == "validate":
        return extractor.validate_batch(
            payload["paper"], payload.get("batch", []), payload.get("candidates", []),
        )
    if mode == "resolve_entities":
        return resolve_entity_batch(llm, payload.get("items", []))
    if mode == "resolve_facts":
        return resolve_fact_batch(llm, payload.get("items", []))
    raise ValueError(f"未知图谱子进程模式: {mode}")


def main() -> int:
    # subprocess.run(encoding="utf-8") only configures the parent, not this child.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    output = sys.stdout
    started = time.monotonic()
    mode = "extract"
    try:
        with redirect_stdout(sys.stderr):
            payload = json.load(sys.stdin)
            mode = payload.get("mode", "extract")
            result = _execute(payload)
            serialized = json.dumps(
                {"ok": True, **result, "mode": mode,
                 "elapsed_seconds": round(time.monotonic() - started, 3)},
                ensure_ascii=True,
            )
        exit_code = 0
    except BaseException as exc:
        error_kind = getattr(exc, "error_kind", None) or {
            "APITimeoutError": "llm_request_timeout",
            "APIConnectionError": "llm_connection_error",
        }.get(type(exc).__name__, "llm_or_extractor_error")
        serialized = json.dumps(
            {
                "ok": False,
                "error": str(exc),
                "error_kind": error_kind,
                "diagnostics": getattr(exc, "diagnostics", {}),
                "mode": mode,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "traceback": traceback.format_exc(limit=8),
            },
            ensure_ascii=True,
        )
        exit_code = 1
    # Serialize first: an invalid result must not leave a partial success prefix
    # followed by an error document in the same pipe.
    output.write(serialized)
    output.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
