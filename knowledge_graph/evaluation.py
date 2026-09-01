"""Offline KG resolution and provenance evaluation helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def resolution_accuracy(cases: Iterable[dict[str, Any]]) -> float | None:
    labeled = [case for case in cases if case.get("expected_id") is not None]
    if not labeled:
        return None
    return sum(case.get("predicted_id") == case.get("expected_id") for case in labeled) / len(labeled)


def duplicate_rate(records: Iterable[dict[str, Any]], key: str) -> float:
    values = [record.get(key) for record in records if record.get(key) is not None]
    if not values:
        return 0.0
    duplicates = sum(count - 1 for count in Counter(values).values() if count > 1)
    return duplicates / len(values)


def incorrect_merge_rate(cases: Iterable[dict[str, Any]]) -> float | None:
    labeled = [case for case in cases if case.get("expected_same") is not None]
    if not labeled:
        return None
    incorrect_merges = sum(
        bool(case.get("predicted_same")) and not bool(case.get("expected_same"))
        for case in labeled
    )
    return incorrect_merges / len(labeled)


def provenance_completeness(claims: Iterable[dict[str, Any]]) -> float:
    rows = list(claims)
    if not rows:
        return 1.0
    required = ("paper_id", "chunk_id", "evidence", "evidence_content_hash")
    complete = sum(all(claim.get(field) not in (None, "") for field in required) for claim in rows)
    return complete / len(rows)
