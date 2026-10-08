from core.evidence import validate_answer_evidence


CHUNKS = [
    {
        "paper_title": "Attention Is All You Need",
        "chunk_index": 3,
        "score": 0.91,
        "metadata": {},
    }
]


def test_accepts_a_retrieved_citation():
    report = validate_answer_evidence("结论来自 **[Attention Is All You Need]**。", CHUNKS)

    assert report["status"] == "pass"
    assert report["matched_citations"] == ["Attention Is All You Need"]


def test_rejects_a_missing_citation():
    report = validate_answer_evidence("结论来自 **[Unknown Paper]**。", CHUNKS)

    assert report["status"] == "retry"
    assert report["missing_citations"] == ["Unknown Paper"]


def test_allows_an_explicit_no_evidence_abstention():
    report = validate_answer_evidence("我无法根据现有论文证据回答。", [])

    assert report["status"] == "pass"
    assert report["reason"] == "no_evidence_abstention"
