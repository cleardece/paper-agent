"""Structural Fact candidate score; embeddings are recall hints only."""

from __future__ import annotations


def score_fact_candidate(
    claim: dict, candidate: dict, embedding_similarity: float = 0.0
) -> float:
    subject_match = claim.get("subject_entity_id") == candidate.get("subject_entity_id")
    predicate_match = claim.get("predicate") == candidate.get("predicate")
    object_match = claim.get("object_entity_id") == candidate.get("object_entity_id")
    if subject_match and predicate_match and object_match:
        return 1.0
    score = (
        0.20 * float(subject_match)
        + 0.25 * float(predicate_match)
        + 0.50 * float(object_match)
        + 0.05 * max(0.0, min(1.0, float(embedding_similarity or 0.0)))
    )
    return round(score, 6)
