from knowledge_graph.vector_index import (
    ENTITY_COLLECTION,
    FACT_COLLECTION,
    KnowledgeGraphVectorIndex,
)


class FakeMilvusClient:
    def __init__(self):
        self.calls = []

    def delete(self, **kwargs):
        self.calls.append(("delete", kwargs))

    def insert(self, **kwargs):
        self.calls.append(("insert", kwargs))

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        field = "entity_id" if kwargs["collection_name"] == ENTITY_COLLECTION else "fact_id"
        return [[{"entity": {field: "X1"}, "distance": 0.91}]]


def test_kg_vector_operations_never_touch_rag_collections():
    client = FakeMilvusClient()
    index = KnowledgeGraphVectorIndex.__new__(KnowledgeGraphVectorIndex)
    index.client = client
    index.enabled = True

    index.upsert_entity("E1", "method", [0.1] * 1024)
    assert index.search_entities([0.1] * 1024, "method")[0]["entity_id"] == "X1"
    index.upsert_fact("F1", "SOLVES", [0.2] * 1024)
    assert index.search_facts([0.2] * 1024, "SOLVES")[0]["fact_id"] == "X1"

    collections = {kwargs["collection_name"] for _, kwargs in client.calls}
    assert collections == {ENTITY_COLLECTION, FACT_COLLECTION}
    assert "paper_chunks" not in collections
    assert "paper_embeddings" not in collections
