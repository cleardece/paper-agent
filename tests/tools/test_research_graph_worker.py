import asyncio
import sys
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from tools.research_graph_worker import (
    GraphProcessError,
    GraphExtractionTimeout,
    GraphSubprocessRunner,
    GraphRateLimitPause,
    ResearchGraphWorker,
)


def make_worker(pdf_pending: int):
    mongo = MagicMock()
    mongo.get_paper.return_value = {
        "arxiv_id": "p1", "title": "Paper", "status": "indexed",
    }
    mongo.get_chunks_by_paper.return_value = [
        {"chunk_index": 1, "content": "Method uses Dataset X.", "metadata": {}}
    ]
    container = MagicMock(mongodb=mongo)
    graph = MagicMock()
    graph.circuit_state.return_value = {"open": False}
    graph.finalize_nonretryable_retries.return_value = 0
    graph.revive_rate_limited_jobs.return_value = 0
    graph.get_extraction_checkpoint.return_value = None
    graph.effective_llm_tpm_limit.return_value = 8000
    graph.reserve_llm_capacity.return_value = {
        "wait_seconds": 0.0,
        "effective_tpm_limit": 8000,
        "estimated_tokens": 1,
    }
    queue = MagicMock()
    queue.count_pending.return_value = pdf_pending
    worker = ResearchGraphWorker(container, graph, queue, asyncio.Event())
    worker.runner = AsyncMock()
    return worker, graph, queue


def test_graph_worker_waits_for_pdf_queue_before_claiming():
    worker, graph, _queue = make_worker(pdf_pending=1)

    assert asyncio.run(worker.process_once()) is False
    graph.claim_next_job.assert_not_called()


def test_graph_worker_completes_with_diagnostics():
    worker, graph, _queue = make_worker(pdf_pending=0)
    graph.claim_next_job.return_value = {
        "paper_id": "p1", "attempt_count": 1, "max_attempts": 2,
        "graph_version": "evidence-graph-v2",
    }
    worker.runner.run.return_value = {
        "ok": True,
        "candidates": [{"relation": "uses"}],
        "diagnostics": {"model_candidate_count": 1},
    }
    graph.upsert_relations.return_value = [{"_id": "edge-1"}]

    assert asyncio.run(worker.process_once()) is True

    args = graph.complete_job.call_args.args
    assert args[0] == "p1" and args[2] == 1
    assert args[3]["evidence_rejected_count"] == 0


def test_graph_worker_records_timeout_and_releases_job():
    worker, graph, _queue = make_worker(pdf_pending=0)
    graph.claim_next_job.return_value = {
        "paper_id": "p1", "attempt_count": 1, "max_attempts": 2,
        "graph_version": "evidence-graph-v2",
    }
    worker.runner.run.side_effect = GraphExtractionTimeout("deadline")

    assert asyncio.run(worker.process_once()) is True

    assert graph.fail_attempt.call_args.args[3] == "timeout"
    graph.record_infrastructure_failure.assert_called_once()


def test_graph_worker_persists_batch_total_before_first_llm_failure():
    worker, graph, _queue = make_worker(pdf_pending=0)
    graph.claim_next_job.return_value = {
        "paper_id": "p1", "attempt_count": 1, "max_attempts": 2,
        "graph_version": "evidence-graph-v4",
    }
    worker.runner.run.side_effect = GraphProcessError("temporary failure")

    assert asyncio.run(worker.process_once()) is True

    graph.set_batch_total.assert_called_once()
    assert graph.set_batch_total.call_args.args[0] == "p1"
    assert graph.set_batch_total.call_args.args[2] > 0


def test_quota_exhaustion_is_terminal_and_opens_circuit_immediately():
    worker, graph, _queue = make_worker(pdf_pending=0)
    graph.claim_next_job.return_value = {
        "paper_id": "p1", "attempt_count": 1, "max_attempts": 2,
        "graph_version": "evidence-graph-v4",
    }
    worker.runner.run.side_effect = GraphProcessError(
        "Error code: 429 - {'error': {'message': 'Allocated quota exceeded, "
        "please increase your quota limit.', 'code': 'insufficient_quota'}}"
    )

    assert asyncio.run(worker.process_once()) is True

    assert graph.fail_attempt.call_args.args[3] == "llm_quota_exhausted"
    assert graph.fail_attempt.call_args.kwargs["retryable"] is False
    assert graph.record_infrastructure_failure.call_args.kwargs["threshold"] == 1


def test_invalid_extraction_json_adaptively_splits_only_the_failed_batch():
    worker, _graph, _queue = make_worker(pdf_pending=0)
    batch = [
        {"chunk_index": 1, "content": "A" * 100, "metadata": {}},
        {"chunk_index": 2, "content": "B" * 100, "metadata": {}},
    ]
    worker.runner.run.side_effect = [
        GraphProcessError("图谱抽取器返回的不是可恢复 JSON"),
        {"candidates": [], "diagnostics": {"parse_status": "empty_array"}},
        {"candidates": [], "diagnostics": {"parse_status": "empty_array"}},
    ]

    result = asyncio.run(worker._extract_and_validate_batch("p1", {"title": "P"}, batch))

    assert result["relations"] == []
    assert result["diagnostics"]["recovery"] == "adaptive_json_split"
    assert worker.runner.run.await_count == 3


def test_empty_llm_response_does_not_recursively_split_batch():
    worker, _graph, _queue = make_worker(pdf_pending=0)
    batch = [
        {"chunk_index": 1, "content": "A" * 100, "metadata": {}},
        {"chunk_index": 2, "content": "B" * 100, "metadata": {}},
    ]
    worker.runner.run.side_effect = GraphProcessError(
        "图谱抽取器返回的不是可恢复 JSON "
        "(response_length=0, response_digest=e3b0c44298fc)"
    )

    with pytest.raises(GraphProcessError):
        asyncio.run(worker._extract_and_validate_batch("p1", {"title": "P"}, batch))

    assert worker.runner.run.await_count == 1


def test_tpm_exhaustion_defers_without_consuming_business_attempt():
    worker, graph, _queue = make_worker(pdf_pending=0)
    graph.claim_next_job.return_value = {
        "paper_id": "p1", "attempt_count": 1, "max_attempts": 2,
        "graph_version": "evidence-graph-v4",
    }
    worker.runner.run.side_effect = GraphProcessError(
        "Error code: 429 - code: 429001 - inference tpm exhausted"
    )
    graph.record_rate_limit.return_value = {"retry_delay_seconds": 300}

    assert asyncio.run(worker.process_once()) is True

    graph.fail_attempt.assert_not_called()
    graph.record_rate_limit.assert_called_once()
    graph.defer_rate_limited.assert_called_once_with(
        "p1",
        worker.worker_id,
        "Error code: 429 - code: 429001 - inference tpm exhausted",
        retry_delay_seconds=300,
    )


def test_saved_extraction_checkpoint_skips_duplicate_extraction_call():
    worker, graph, _queue = make_worker(pdf_pending=0)
    graph.get_extraction_checkpoint.return_value = {
        "candidates": [{"candidate_id": 0, "relation": "uses"}],
        "diagnostics": {"model_candidate_count": 1},
    }
    worker.runner.run.return_value = {
        "relations": [{"relation": "uses"}],
        "diagnostics": {"validated_count": 1},
    }
    batch = [{"chunk_index": 1, "content": "Method uses X.", "metadata": {}}]

    result = asyncio.run(worker._extract_and_validate_batch(
        "p1", {"title": "P"}, batch, batch_index=0,
    ))

    assert result["relations"] == [{"relation": "uses"}]
    assert worker.runner.run.await_count == 1
    assert worker.runner.run.await_args.args[0]["mode"] == "validate"
    graph.save_extraction_checkpoint.assert_not_called()


def test_llm_capacity_wait_renews_job_lease():
    worker, graph, _queue = make_worker(pdf_pending=0)
    graph.reserve_llm_capacity.return_value = {
        "wait_seconds": 0.01,
        "effective_tpm_limit": 8000,
        "estimated_tokens": 2000,
    }
    worker.runner.run.return_value = {"candidates": []}

    asyncio.run(worker._run_graph_call(
        "p1", {"mode": "extract", "paper": {}, "batch": []}
    ))

    graph.heartbeat.assert_called()
    graph.record_llm_success.assert_called_once()


def test_oversized_extraction_is_split_before_subprocess_call():
    worker, graph, _queue = make_worker(pdf_pending=0)
    graph.effective_llm_tpm_limit.return_value = 4000
    worker.runner.run.return_value = {
        "candidates": [], "diagnostics": {"parse_status": "empty_array"},
    }
    batch = [{"chunk_index": 1, "content": "A" * 5000, "metadata": {}}]

    result = asyncio.run(worker._extract_batch("p1", {"title": "P"}, batch))

    assert result["diagnostics"]["recovery"] == "tpm_budget_split"
    assert worker.runner.run.await_count >= 2
    assert all(
        worker._estimated_tokens(call.args[0]) <= 4000
        for call in worker.runner.run.await_args_list
    )


def test_existing_global_pause_prevents_reserved_call_from_launching():
    worker, graph, _queue = make_worker(pdf_pending=0)
    graph.reserve_llm_capacity.return_value = {
        "wait_seconds": 10.0, "effective_tpm_limit": 4000,
    }
    graph.circuit_state.return_value = {
        "open": True, "retry_after_seconds": 123,
        "pause_kind": "llm_rate_limited",
    }

    with pytest.raises(GraphRateLimitPause) as captured:
        asyncio.run(worker._run_graph_call(
            "p1", {"mode": "extract", "paper": {}, "batch": []}
        ))

    assert captured.value.retry_after_seconds == 123
    worker.runner.run.assert_not_called()


def test_oversized_resolution_items_are_split_and_merged():
    worker, graph, _queue = make_worker(pdf_pending=0)
    graph.effective_llm_tpm_limit.return_value = 4000

    async def resolve(payload, _heartbeat):
        item = payload["items"][0]
        return {
            "decisions": [{"mention_index": item["mention_index"], "decision": "new"}],
            "diagnostics": {
                "llm_called": True, "requested_count": 1,
                "returned_count": 1, "accepted_count": 1,
            },
        }

    worker.runner.run.side_effect = resolve
    items = [
        {"mention_index": index, "context": "X" * 5000, "candidates": []}
        for index in range(2)
    ]

    result = asyncio.run(worker._run_resolution_call(
        "p1", "resolve_entities", {"items": items}
    ))

    assert [item["mention_index"] for item in result["decisions"]] == [0, 1]
    assert result["diagnostics"]["recovery"] == "tpm_budget_split"
    assert worker.runner.run.await_count == 2


def test_hard_timeout_kills_real_child_process():
    class SleepingRunner(GraphSubprocessRunner):
        @staticmethod
        def _command():
            return (sys.executable, "-c", "import time; time.sleep(20)")

    runner = SleepingRunner(timeout_seconds=1, heartbeat_seconds=0.1)
    started = time.monotonic()
    with pytest.raises(GraphExtractionTimeout):
        asyncio.run(runner.run({}, lambda: None))

    assert time.monotonic() - started < 4
    assert runner.current_process is None
