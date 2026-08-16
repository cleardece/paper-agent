"""Deterministic metrics used by the offline research evaluation runner."""

from __future__ import annotations

from statistics import median
from typing import Any


def compute_recall_at_k(expected: list[str], retrieved: list[str], k: int) -> float:
    """Return the fraction of unique expected paper IDs found in top-k results."""
    expected_set = set(expected)
    if not expected_set:
        return 1.0
    return len(expected_set & set(retrieved[:k])) / len(expected_set)


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, float | int]:
    """Aggregate retrieval, provenance, abstention, and latency indicators."""
    total = len(runs)
    recalls = [
        compute_recall_at_k(run["expected_papers"], run["retrieved_papers"], 5)
        for run in runs
    ]
    citation_passes = [run.get("evidence_status") == "pass" for run in runs]
    abstention_runs = [run for run in runs if run.get("should_abstain")]
    correct_abstentions = [
        not run.get("retrieved_papers") and run.get("evidence_status") == "pass"
        for run in abstention_runs
    ]
    latencies = [float(run["latency_ms"]) for run in runs]

    return {
        "case_count": total,
        "recall_at_5": sum(recalls) / total if total else 0.0,
        "citation_pass_rate": sum(citation_passes) / total if total else 0.0,
        "abstention_accuracy": (
            sum(correct_abstentions) / len(abstention_runs)
            if abstention_runs
            else 1.0
        ),
        "latency_ms_p50": median(latencies) if latencies else 0.0,
    }
