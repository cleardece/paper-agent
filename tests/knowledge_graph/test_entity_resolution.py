from knowledge_graph.entity_resolution.normalizer import acronym, normalize_name
from knowledge_graph.entity_resolution.scorer import score_entity_candidate


def test_name_normalization_collapses_pinn_variants():
    assert normalize_name("PINNs") == normalize_name("pinn")
    assert normalize_name("Physics-informed  neural networks") == (
        "physics informed neural network"
    )
    assert acronym("Physics-Informed Neural Networks") == "pinn"


def test_acronym_and_expanded_name_are_high_confidence_same_entity():
    score = score_entity_candidate(
        {"name": "PINN", "type": "method", "context": "fluid equations"},
        {
            "canonical_name": "Physics-Informed Neural Networks",
            "type": "method",
            "context": "fluid equations",
        },
        embedding_similarity=0.82,
    )
    assert score >= 0.9


def test_type_mismatch_can_never_merge():
    score = score_entity_candidate(
        {"name": "PINN", "type": "software"},
        {"canonical_name": "PINN", "type": "method"},
        embedding_similarity=1.0,
    )
    assert score == 0.0
