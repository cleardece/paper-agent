from knowledge_graph.fact_resolution.signature import build_fact_signature
from knowledge_graph.fact_resolution.scorer import score_fact_candidate


def test_fact_signature_uses_canonical_endpoints_not_qualifiers():
    left = build_fact_signature("E001", "SOLVES", "E009")
    right = build_fact_signature(
        "E001", "SOLVES", "E009", qualifiers={"reynolds_number": 10000}
    )
    assert left == right == "E001|SOLVES|E009"


def test_fact_score_is_structural_not_sentence_similarity():
    claim = {
        "subject_entity_id": "E001",
        "predicate": "IMPROVES",
        "object_entity_id": "accuracy",
        "qualifiers": {},
    }
    different_object = {
        "subject_entity_id": "E001",
        "predicate": "IMPROVES",
        "object_entity_id": "stability",
        "qualifiers": {},
    }
    assert score_fact_candidate(claim, different_object, 0.99) < 0.6
