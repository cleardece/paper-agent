from knowledge_graph.schema.entity_types import ENTITY_TYPES, normalize_entity_type
from knowledge_graph.schema.predicates import (
    PREDICATES,
    normalize_predicate,
    to_legacy_relation,
)


def test_schema_is_closed_and_normalizes_known_values():
    assert normalize_entity_type("Physics Method") == "method"
    assert normalize_entity_type("datasets") == "dataset"
    assert normalize_entity_type("invented_type") is None
    assert normalize_predicate("was employed to address") == "SOLVES"
    assert normalize_predicate("made_up_relation") == "UNKNOWN"
    assert "UNKNOWN" in PREDICATES
    assert "method" in ENTITY_TYPES
    assert to_legacy_relation("USES_DATASET") == "uses"
