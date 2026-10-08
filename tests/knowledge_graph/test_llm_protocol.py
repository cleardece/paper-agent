import json
from types import SimpleNamespace

import pytest

from knowledge_graph.entity_resolution.llm_resolver import resolve_entity_batch
from knowledge_graph.fact_resolution.llm_resolver import resolve_fact_batch


class FakeLLM:
    def __init__(self, content, **metadata):
        self.response = SimpleNamespace(content=content, response_metadata=metadata)

    def bind(self, **kwargs):
        return self

    def invoke(self, prompt):
        return self.response


@pytest.fixture(params=[
    (resolve_entity_batch, "mention_index", "entity_id"),
    (resolve_fact_batch, "fact_index", "fact_id"),
])
def resolver(request):
    return request.param


@pytest.mark.parametrize("content", ["", "not JSON", "[]", '[{"decision":"new"}',
                                     '[{"decision":"new"}]'])
def test_resolution_does_not_accept_malformed_or_missing_decisions(resolver, content):
    resolve, index_key, id_key = resolver
    items = [{index_key: 7, "candidates": [{id_key: "known"}]}]
    with pytest.raises(ValueError):
        resolve(FakeLLM(content), items)


def test_resolution_requires_unique_indexes_and_known_merge_target(resolver):
    resolve, index_key, id_key = resolver
    items = [{index_key: 7, "candidates": [{id_key: "known"}]}]
    for decisions in (
        [{index_key: 7, "decision": "merge", id_key: "invented"}],
        [{index_key: 8, "decision": "new"}],
        [{index_key: 7, "decision": "new"}] * 2,
    ):
        with pytest.raises(ValueError):
            resolve(FakeLLM(json.dumps(decisions)), items)


def test_resolution_preserves_global_indexes_and_supports_content_blocks(resolver):
    resolve, index_key, id_key = resolver
    items = [{index_key: 7, "candidates": [{id_key: "known"}]},
             {index_key: 12, "candidates": []}]
    decisions = [{index_key: 7, "decision": "merge", id_key: "known"},
                 {index_key: 12, "decision": "new"}]
    result = resolve(FakeLLM([{"type": "text", "text": json.dumps(decisions)}]), items)
    assert result["decisions"] == decisions


def test_resolution_rejects_length_finish_reason_for_valid_json(resolver):
    resolve, index_key, _ = resolver
    with pytest.raises(ValueError):
        resolve(FakeLLM(json.dumps([{index_key: 7, "decision": "new"}]),
                        finish_reason="length"), [{index_key: 7, "candidates": []}])
