"""Small normalization helpers for raw and verified claim dictionaries."""

from __future__ import annotations

from typing import Any

from knowledge_graph.schema.entity_types import normalize_entity_type
from knowledge_graph.schema.predicates import (
    STANCES,
    normalize_predicate,
    predicate_for_legacy_relation,
    to_legacy_relation,
)


def bounded_confidence(value: object, default: float = 0.5) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def claim_from_candidate(candidate: dict[str, Any], paper: dict[str, Any]) -> dict[str, Any]:
    """Adapt both V4 raw claims and the V3 paper-to-target shape."""
    object_name = str(
        candidate.get("object_name") or candidate.get("object")
        or candidate.get("target_name") or ""
    ).strip()
    object_type = normalize_entity_type(
        candidate.get("object_type") or candidate.get("object_type_raw")
        or candidate.get("target_type")
    )
    subject_name = str(
        candidate.get("subject_name") or candidate.get("subject")
        or paper.get("title") or paper.get("arxiv_id") or ""
    ).strip()
    subject_type = normalize_entity_type(
        candidate.get("subject_type") or candidate.get("subject_type_raw")
    ) or "domain"
    raw_predicate = (
        candidate.get("predicate_raw") or candidate.get("predicate")
        or candidate.get("relation") or ""
    )
    predicate = (
        predicate_for_legacy_relation(candidate.get("relation"), object_type)
        if candidate.get("relation") else normalize_predicate(raw_predicate)
    )
    return {
        **candidate,
        "subject_name": subject_name,
        "subject_type": subject_type,
        "predicate_raw": str(raw_predicate),
        "predicate": predicate,
        "object_name": object_name,
        "object_type": object_type,
        "qualifiers": dict(candidate.get("qualifiers") or {}),
        "stance": (
            str(candidate.get("stance")).lower()
            if str(candidate.get("stance")).lower() in STANCES else "support"
        ),
        "confidence": bounded_confidence(candidate.get("confidence")),
        "relation": to_legacy_relation(predicate),
        "target_name": object_name,
        "target_type": object_type,
    }


def apply_verification(
    candidate: dict[str, Any], decision: dict[str, Any], paper: dict[str, Any]
) -> dict[str, Any]:
    claim = claim_from_candidate(candidate, paper)
    subject_type = normalize_entity_type(
        decision.get("subject_type") or claim.get("subject_type")
    )
    object_type = normalize_entity_type(
        decision.get("object_type") or claim.get("object_type")
    )
    raw_predicate = decision.get("predicate") or decision.get("relation") or claim["predicate"]
    predicate = (
        predicate_for_legacy_relation(raw_predicate, object_type)
        if str(raw_predicate).lower() in {
            "proposes", "uses", "improves", "compares_with", "evaluates_on",
            "measures_with", "studies",
        }
        else normalize_predicate(raw_predicate)
    )
    stance = str(decision.get("stance") or claim["stance"]).lower()
    verdict = str(decision.get("verdict") or "uncertain").lower()
    valid = bool(decision.get("valid", verdict != "rejected"))
    return {
        **claim,
        "subject_name": str(decision.get("subject_name") or claim["subject_name"]).strip(),
        "subject_type": subject_type,
        "predicate": predicate,
        "object_name": str(decision.get("object_name") or claim["object_name"]).strip(),
        "object_type": object_type,
        "qualifiers": dict(decision.get("qualifiers") or claim["qualifiers"]),
        "stance": stance if stance in STANCES else "support",
        "confidence": bounded_confidence(decision.get("confidence"), claim["confidence"]),
        "valid": valid and bool(subject_type and object_type) and predicate != "UNKNOWN",
        "validation_verdict": verdict if verdict in {"supported", "uncertain", "rejected"} else "uncertain",
        "validated_relation": to_legacy_relation(predicate),
        "relation": to_legacy_relation(predicate),
        "target_name": str(decision.get("object_name") or claim["object_name"]).strip(),
        "target_type": object_type,
        "validation_reason": str(decision.get("reason") or "未提供核验说明"),
    }
