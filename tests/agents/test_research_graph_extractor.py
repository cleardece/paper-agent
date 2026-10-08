import pytest

from agents.research_graph_extractor import ResearchGraphExtractor


class Response:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.prompt = ""

    def bind(self, **_kwargs):
        return self

    def invoke(self, prompt):
        self.prompt = prompt
        return Response(self.response)


class SequenceLLM(FakeLLM):
    def __init__(self, responses):
        super().__init__("")
        self.responses = iter(responses)

    def invoke(self, prompt):
        self.prompt = prompt
        return Response(next(self.responses))


PAPER = {"arxiv_id": "p1", "title": "A Paper"}
CHUNKS = [
    {"chunk_index": 0, "content": "A reference section should not be selected.", "metadata": {"heading": "References"}},
    {"chunk_index": 1, "content": "Our Method Alpha uses Dataset X for evaluation.", "metadata": {"heading": "Method"}},
]


def test_extractor_keeps_only_candidates_with_selected_evidence():
    llm = FakeLLM(
        '[{"relation":"uses","target_type":"dataset","target_name":"Dataset X",'
        '"evidence_chunk_index":1,"evidence":"Method Alpha uses Dataset X for evaluation",'
        '"confidence":0.91}]'
    )
    result = ResearchGraphExtractor(llm, max_chunks=1).extract(PAPER, CHUNKS)

    assert result[0]["target_name"] == "Dataset X"
    assert '"chunk_index": 1' in llm.prompt


def test_extractor_returns_empty_for_non_json_response():
    result = ResearchGraphExtractor(FakeLLM("I cannot determine this."), max_chunks=2).extract_with_diagnostics(PAPER, CHUNKS)

    assert result["candidates"] == []
    assert result["diagnostics"]["result_reason"] == "invalid_model_response"


def test_extractor_distinguishes_explicit_empty_array():
    result = ResearchGraphExtractor(FakeLLM("[]"), max_chunks=2).extract_with_diagnostics(PAPER, CHUNKS)

    assert result["diagnostics"]["result_reason"] == "model_returned_no_relations"


def test_response_text_extracts_provider_content_blocks():
    response = Response([
        {"type": "text", "text": '[{"subject_name":"A","predicate_raw":"uses",'
         '"object_name":"B","evidence_chunk_index":1,'
         '"evidence":"A uses B in the reported experiment."}]'},
    ])
    text = ResearchGraphExtractor._response_text(response)
    parsed, status = ResearchGraphExtractor._load_json(text)
    assert status == "parsed"
    assert parsed[0]["subject_name"] == "A"


def test_json_loader_accepts_single_object_but_rejects_truncated_items():
    single, single_status = ResearchGraphExtractor._load_json(
        '{"subject_name":"A","predicate_raw":"uses","object_name":"B"}'
    )
    assert single_status == "parsed_single_object"
    assert len(single) == 1
    with pytest.raises(ValueError):
        ResearchGraphExtractor._load_json(
            '[{"subject_name":"A","predicate_raw":"uses","object_name":"B"},'
            '{"subject_name":"incomplete"'
        )


def test_validator_rejects_missing_decision_instead_of_finalizing_batch():
    candidates = [
        {
            "relation": "uses", "target_type": "dataset", "target_name": "Dataset X",
            "evidence_chunk_index": 1,
            "evidence": "Method Alpha uses Dataset X for evaluation.",
        },
        {
            "relation": "uses", "target_type": "method", "target_name": "Method Beta",
            "evidence_chunk_index": 1,
            "evidence": "Method Alpha uses Method Beta during training.",
        },
    ]
    llm = FakeLLM(
        '[{"candidate_index":0,"verdict":"supported","relation":"uses",'
        '"reason":"direct evidence"}]'
    )

    with pytest.raises(ValueError):
        ResearchGraphExtractor(llm).validate_batch(PAPER, [{
            "chunk_index": 1,
            "content": "Method Alpha uses Dataset X for evaluation. Method Alpha uses Method Beta during training.",
        }], candidates)


def test_valid_json_with_length_finish_reason_is_not_complete():
    class LengthLimitedLLM(FakeLLM):
        def invoke(self, prompt):
            response = Response("[]")
            response.response_metadata = {"finish_reason": "length"}
            return response

    with pytest.raises(ValueError):
        ResearchGraphExtractor(LengthLimitedLLM("[]")).extract_batch(PAPER, CHUNKS[1:])


def test_validator_rejects_conflicting_duplicates():
    llm = FakeLLM('[{"candidate_index":0,"verdict":"supported"},'
                  '{"candidate_index":0,"verdict":"rejected"}]')
    with pytest.raises(ValueError):
        ResearchGraphExtractor(llm).validate_batch(PAPER, CHUNKS[1:], [{
            "relation": "uses", "target_type": "dataset", "target_name": "Dataset X",
            "evidence_chunk_index": 1, "evidence": CHUNKS[1]["content"],
        }])


def test_explicit_uncertain_decision_is_a_complete_response():
    llm = FakeLLM('[{"candidate_index":0,"verdict":"uncertain",'
                  '"reason":"ambiguous subject"}]')
    result = ResearchGraphExtractor(llm).validate_batch(PAPER, CHUNKS[1:], [{
        "relation": "uses", "target_type": "dataset", "target_name": "Dataset X",
        "evidence_chunk_index": 1, "evidence": CHUNKS[1]["content"],
    }])
    assert result["relations"][0]["validation_verdict"] == "uncertain"
    assert result["diagnostics"]["missing_or_invalid_decision_count"] == 0
