"""Stable canonical Fact signatures."""

from __future__ import annotations

from typing import Any

from knowledge_graph.schema.predicates import normalize_predicate


def build_fact_signature(
    subject_entity_id: str,
    predicate: str,
    object_entity_id: str,
    qualifiers: dict[str, Any] | None = None,
) -> str:
    del qualifiers  # Qualifiers deliberately remain Claim-level in V4.
    return f"{subject_entity_id}|{normalize_predicate(predicate)}|{object_entity_id}"
