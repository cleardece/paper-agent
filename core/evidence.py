"""Deterministic checks that bind research answers to retrieved evidence."""

from __future__ import annotations

import re
from typing import Any


_CITATION_RE = re.compile(r"\*\*\[([^\]]+)\]\*\*|(?<!\*)\[([^\]]+)\](?!\*)")
_ABSTENTION_MARKERS = (
    "证据不足",
    "无法根据现有论文证据",
    "未找到相关信息",
)


def extract_citations(answer: str) -> list[str]:
    """Return unique citation titles in their order of appearance."""
    citations = (
        (bold_title or plain_title).strip()
        for bold_title, plain_title in _CITATION_RE.findall(answer or "")
    )
    return list(dict.fromkeys(citation for citation in citations if citation))


def validate_answer_evidence(answer: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Check whether citations can be resolved to this retrieval result set.

    This deliberately validates provenance rather than factual entailment. The
    CriticAgent still performs the latter semantic review with the LLM.
    """
    citations = extract_citations(answer)
    retrieved_titles = {
        str(chunk.get("paper_title", "")).strip().casefold()
        for chunk in chunks
        if chunk.get("paper_title")
    }
    missing = [
        citation for citation in citations
        if citation.casefold() not in retrieved_titles
    ]
    abstains = any(marker in (answer or "") for marker in _ABSTENTION_MARKERS)

    if not chunks:
        status, reason = (
            ("pass", "no_evidence_abstention")
            if abstains
            else ("retry", "no_chunks_without_abstention")
        )
    elif missing:
        status, reason = "retry", "citation_not_retrieved"
    elif citations:
        status, reason = "pass", "all_citations_retrieved"
    else:
        status, reason = "retry", "answer_has_no_citations"

    return {
        "status": status,
        "reason": reason,
        "citations": citations,
        "matched_citations": [citation for citation in citations if citation not in missing],
        "missing_citations": missing,
        "source_count": len(chunks),
    }
