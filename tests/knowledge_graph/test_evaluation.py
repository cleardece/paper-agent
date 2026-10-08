from knowledge_graph.evaluation import (
    duplicate_rate,
    incorrect_merge_rate,
    provenance_completeness,
    resolution_accuracy,
)


def test_resolution_and_duplicate_metrics_are_explicit():
    assert resolution_accuracy([
        {"predicted_id": "E1", "expected_id": "E1"},
        {"predicted_id": "E2", "expected_id": "E3"},
    ]) == 0.5
    assert duplicate_rate([
        {"signature": "A"}, {"signature": "A"}, {"signature": "B"},
    ], "signature") == 1 / 3
    assert incorrect_merge_rate([
        {"predicted_same": True, "expected_same": False},
        {"predicted_same": True, "expected_same": True},
    ]) == 0.5


def test_provenance_completeness_requires_source_and_evidence():
    assert provenance_completeness([
        {"paper_id": "p", "chunk_id": "p:1", "evidence": "x", "evidence_content_hash": "h"},
        {"paper_id": "p", "chunk_id": "", "evidence": "x", "evidence_content_hash": "h"},
    ]) == 0.5
