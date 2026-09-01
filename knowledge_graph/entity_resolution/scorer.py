"""Rule score for canonical entity candidates."""

from __future__ import annotations

from difflib import SequenceMatcher

from knowledge_graph.entity_resolution.normalizer import acronym, normalize_name


def _token_overlap(left: object, right: object) -> float:
    left_tokens = set(normalize_name(left).split())
    right_tokens = set(normalize_name(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def score_entity_candidate(
    mention: dict, candidate: dict, embedding_similarity: float = 0.0
) -> float:
    if mention.get("type") != candidate.get("type"):
        return 0.0
    mention_name = str(mention.get("name") or "")
    candidate_name = str(candidate.get("canonical_name") or candidate.get("name") or "")
    left = normalize_name(mention_name)
    right = normalize_name(candidate_name)
    if left and left == right:
        return 1.0
    left_acronym = acronym(mention_name)
    right_acronym = acronym(candidate_name)
    if left_acronym and left_acronym == right_acronym and (
        len(left.split()) == 1 or len(right.split()) == 1
    ):
        return 0.95
    aliases = {normalize_name(alias) for alias in candidate.get("aliases", [])}
    if left in aliases:
        return 1.0
    name_similarity = SequenceMatcher(None, left, right).ratio()
    context_similarity = _token_overlap(mention.get("context"), candidate.get("context"))
    score = (
        0.55 * name_similarity
        + 0.30 * max(0.0, min(1.0, float(embedding_similarity or 0.0)))
        + 0.15 * context_similarity
    )
    return round(score, 6)
