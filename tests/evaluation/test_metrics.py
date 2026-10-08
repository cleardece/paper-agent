from evaluation.metrics import compute_recall_at_k, summarize_runs


def test_recall_at_k_counts_expected_papers():
    assert compute_recall_at_k(["a", "b"], ["b", "c", "a"], 2) == 0.5


def test_summary_reports_research_metrics():
    report = summarize_runs([
        {
            "expected_papers": ["a"],
            "retrieved_papers": ["a"],
            "evidence_status": "pass",
            "latency_ms": 120,
        },
        {
            "expected_papers": [],
            "retrieved_papers": [],
            "evidence_status": "pass",
            "latency_ms": 80,
            "should_abstain": True,
        },
    ])

    assert report == {
        "case_count": 2,
        "recall_at_5": 1.0,
        "citation_pass_rate": 1.0,
        "abstention_accuracy": 1.0,
        "latency_ms_p50": 100.0,
    }
