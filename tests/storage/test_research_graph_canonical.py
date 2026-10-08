import asyncio
from uuid import uuid4

import pytest
from pymongo import MongoClient

from knowledge_graph.graph.repository import CanonicalGraphRepository
from knowledge_graph.pipeline import KnowledgeGraphPipeline


class FakeEmbedder:
    def embed_texts(self, texts):
        return [[1.0] + [0.0] * 1023 for _ in texts]


class DisabledVectorIndex:
    enabled = False

    def search_entities(self, *_args, **_kwargs):
        return []

    def search_facts(self, *_args, **_kwargs):
        return []

    def upsert_entity(self, *_args, **_kwargs):
        return None

    def upsert_fact(self, *_args, **_kwargs):
        return None


@pytest.fixture
def canonical_repo():
    client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    database_name = f"pa_kg4_{uuid4().hex}"
    repo = CanonicalGraphRepository(client[database_name])
    yield repo
    client.drop_database(database_name)
    client.close()


def verified_claim(paper_id, stance, reynolds_number):
    evidence = f"PINNs solve the Navier-Stokes equations at Re = {reynolds_number}."
    return (
        {"arxiv_id": paper_id, "title": paper_id},
        [{
            "chunk_index": 0, "content": evidence,
            "metadata": {"section": "Results", "page": 3},
        }],
        [{
            "subject_name": "PINNs", "subject_type": "method",
            "predicate": "SOLVES", "predicate_raw": "solve",
            "object_name": "Navier-Stokes equations", "object_type": "equation",
            "qualifiers": {"reynolds_number": reynolds_number}, "stance": stance,
            "confidence": 0.95, "valid": True,
            "validation_verdict": "supported", "validation_reason": "direct evidence",
            "evidence_chunk_index": 0, "evidence": evidence,
            "relation": "studies", "target_name": "Navier-Stokes equations",
            "target_type": "equation",
        }],
    )


def test_incremental_claims_share_fact_and_preserve_conflict_provenance(canonical_repo):
    pipeline = KnowledgeGraphPipeline(
        canonical_repo, FakeEmbedder(), DisabledVectorIndex()
    )

    first = asyncio.run(pipeline.process(*verified_claim("p1", "support", 100)))
    second = asyncio.run(pipeline.process(*verified_claim("p2", "contradict", 10000)))

    assert canonical_repo.entities.count_documents({}) == 2
    assert canonical_repo.facts.count_documents({}) == 1
    assert canonical_repo.claims.count_documents({}) == 2
    fact = canonical_repo.facts.find_one({})
    assert fact["support_count"] == 1
    assert fact["contradict_count"] == 1
    claims = list(canonical_repo.claims.find().sort("paper_id", 1))
    assert claims[0]["qualifiers"] != claims[1]["qualifiers"]
    assert all(claim["paper_id"] and claim["chunk_id"] and claim["evidence"] for claim in claims)
    assert first["diagnostics"]["entity_resolution"]["llm"]["llm_called"] is False
    assert second["diagnostics"]["fact_resolution"]["route_counts"]["exact_signature"] == 1
    assert canonical_repo.evaluation()["provenance_completeness"] == 1.0
