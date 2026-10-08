from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pymongo import MongoClient

from storage.research_graph import GRAPH_VERSION, ResearchGraphRepository


@pytest.fixture
def graph_repo():
    client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    database_name = f"paper_agent_graph_test_{uuid4().hex}"
    database = client[database_name]
    repo = ResearchGraphRepository(database)
    yield repo
    client.drop_database(database_name)
    client.close()


def insert_indexed(repo, paper_id):
    repo.papers.insert_one({
        "arxiv_id": paper_id, "title": paper_id, "status": "indexed",
        "created_at": datetime.now(timezone.utc),
    })


def test_reconcile_upgrades_old_job_and_does_not_duplicate_current_result(graph_repo):
    insert_indexed(graph_repo, "p1")
    graph_repo.jobs.insert_one({
        "job_id": "old", "paper_id": "p1", "status": "extracting",
        "attempt_count": 2, "max_attempts": 2,
        "created_at": datetime.now(timezone.utc),
    })

    first = graph_repo.reconcile_indexed_papers([{"arxiv_id": "p1"}])

    assert first == {"created": 0, "upgraded": 1, "unchanged": 0}
    job = graph_repo.get_job("p1")
    assert job["graph_version"] == GRAPH_VERSION
    assert job["status"] == "pending"
    assert job["attempt_count"] == 0

    second = graph_repo.reconcile_indexed_papers([{"arxiv_id": "p1"}])
    assert second["unchanged"] == 1
    assert graph_repo.jobs.count_documents({"paper_id": "p1"}) == 1


def test_expired_lease_is_recovered_and_empty_result_is_explicit(graph_repo):
    insert_indexed(graph_repo, "p1")
    graph_repo.enqueue("p1", source="backfill")
    job = graph_repo.claim_next_job("worker-a", lease_seconds=30)
    graph_repo.jobs.update_one(
        {"_id": job["_id"]},
        {"$set": {"lease_expires_at": datetime.now(timezone.utc) - timedelta(seconds=1)}},
    )

    recovered = graph_repo.recover_expired_leases()
    assert recovered == {"recovered": 1, "failed": 0}
    assert graph_repo.get_job("p1")["status"] == "retry_wait"

    claimed = graph_repo.claim_next_job("worker-b", lease_seconds=30)
    diagnostics = {"result_reason": "model_returned_no_relations"}
    assert graph_repo.complete_job("p1", "worker-b", 0, diagnostics)

    final = graph_repo.get_job("p1")
    assert final["status"] == "completed_empty"
    assert final["diagnostics"] == diagnostics
    assert graph_repo.papers.find_one({"arxiv_id": "p1"})["graph_status"] == "empty"


def test_new_paper_precedes_backfill_and_manual_retry_has_highest_priority(graph_repo):
    for paper_id in ("backfill", "new", "manual"):
        insert_indexed(graph_repo, paper_id)
    graph_repo.enqueue("backfill", source="backfill")
    graph_repo.enqueue("new", source="new")
    graph_repo.enqueue("manual", source="backfill")
    graph_repo.jobs.update_one({"paper_id": "manual"}, {"$set": {"status": "failed"}})
    graph_repo.manual_retry("manual")

    first = graph_repo.claim_next_job("worker", lease_seconds=30)
    assert first["paper_id"] == "manual"
    graph_repo.fail_attempt("manual", "worker", "x", "test", 60)

    second = graph_repo.claim_next_job("worker", lease_seconds=30)
    assert second["paper_id"] == "new"


def test_non_retryable_failure_stops_after_first_attempt(graph_repo):
    insert_indexed(graph_repo, "quota-paper")
    graph_repo.enqueue("quota-paper", source="backfill")
    graph_repo.claim_next_job("worker", lease_seconds=30)

    status = graph_repo.fail_attempt(
        "quota-paper",
        "worker",
        "Allocated quota exceeded",
        "llm_quota_exhausted",
        60,
        retryable=False,
    )

    job = graph_repo.get_job("quota-paper")
    assert status == "failed"
    assert job["attempt_count"] == 1
    assert job["next_attempt_at"] is None
    assert job["error_kind"] == "llm_quota_exhausted"


def test_existing_quota_retry_is_reclassified_without_another_attempt(graph_repo):
    insert_indexed(graph_repo, "old-quota-paper")
    graph_repo.enqueue("old-quota-paper", source="backfill")
    graph_repo.jobs.update_one(
        {"paper_id": "old-quota-paper"},
        {"$set": {
            "status": "retry_wait",
            "attempt_count": 1,
            "error_kind": "process_or_llm_error",
            "error": "Error code: 429 - code: insufficient_quota",
        }},
    )

    assert graph_repo.finalize_nonretryable_retries() == 1

    job = graph_repo.get_job("old-quota-paper")
    assert job["status"] == "failed"
    assert job["attempt_count"] == 1
    assert job["next_attempt_at"] is None
    assert job["error_kind"] == "llm_quota_exhausted"


def test_rate_limit_refunds_claim_and_opens_global_pause(graph_repo):
    insert_indexed(graph_repo, "rate-paper")
    graph_repo.enqueue("rate-paper", source="backfill")
    claimed = graph_repo.claim_next_job("worker-a", lease_seconds=30)
    assert claimed["attempt_count"] == 1

    congestion = graph_repo.record_rate_limit(
        "429001 inference tpm exhausted",
        base_delay_seconds=300,
        max_delay_seconds=3600,
        configured_tpm_limit=8000,
    )
    status = graph_repo.defer_rate_limited(
        "rate-paper",
        "worker-a",
        "429001 inference tpm exhausted",
        retry_delay_seconds=congestion["retry_delay_seconds"],
    )

    job = graph_repo.get_job("rate-paper")
    assert status == "retry_wait"
    assert job["status"] == "retry_wait"
    assert job["attempt_count"] == 0
    assert job["error_kind"] == "llm_rate_limited"
    assert graph_repo.circuit_state()["open"] is True


def test_rate_limit_checkpoint_survives_resume_and_clears_after_batch(graph_repo):
    insert_indexed(graph_repo, "checkpoint-paper")
    graph_repo.enqueue("checkpoint-paper", source="backfill")
    graph_repo.claim_next_job("worker-a", lease_seconds=30)
    candidates = [{"candidate_id": 0, "relation": "uses"}]
    assert graph_repo.save_extraction_checkpoint(
        "checkpoint-paper", "worker-a", 0, candidates, {"model_count": 1}
    )

    graph_repo.defer_rate_limited(
        "checkpoint-paper", "worker-a", "429001", retry_delay_seconds=300
    )
    graph_repo.jobs.update_one(
        {"paper_id": "checkpoint-paper"},
        {"$set": {"next_attempt_at": datetime.now(timezone.utc)}},
    )
    graph_repo.claim_next_job("worker-b", lease_seconds=30)

    checkpoint = graph_repo.get_extraction_checkpoint(
        "checkpoint-paper", "worker-b", 0
    )
    assert checkpoint["candidates"] == candidates
    assert graph_repo.save_batch_result(
        "checkpoint-paper", "worker-b", 0, 1, [], {"validated_count": 0}
    )
    assert graph_repo.get_extraction_checkpoint(
        "checkpoint-paper", "worker-b", 0
    ) is None


def test_legacy_rate_limit_failure_is_revived_once(graph_repo):
    insert_indexed(graph_repo, "legacy-rate-paper")
    graph_repo.enqueue("legacy-rate-paper", source="backfill")
    graph_repo.jobs.update_one(
        {"paper_id": "legacy-rate-paper"},
        {
            "$set": {
                "status": "failed",
                "attempt_count": 2,
                "error_kind": "process_or_llm_error",
                "error": "429001 inference tpm exhausted",
                "finished_at": datetime.now(timezone.utc),
                "attempt_history": [
                    {"run_number": 1, "error": "429001 inference tpm exhausted"},
                    {"run_number": 1, "error": "429001 inference tpm exhausted"},
                ],
            },
            "$unset": {"rate_limit_recovery_version": ""},
        },
    )

    assert graph_repo.revive_rate_limited_jobs() == 1
    job = graph_repo.get_job("legacy-rate-paper")
    assert job["status"] == "retry_wait"
    assert job["attempt_count"] == 0
    assert job["finished_at"] is None
    assert graph_repo.revive_rate_limited_jobs() == 0


def test_llm_capacity_reservation_is_persistent_across_repository_instances(graph_repo):
    first = graph_repo.reserve_llm_capacity(4000, configured_tpm_limit=8000)
    second_repo = ResearchGraphRepository(graph_repo.jobs.database)
    second = second_repo.reserve_llm_capacity(4000, configured_tpm_limit=8000)

    assert first["wait_seconds"] < 1
    assert second["wait_seconds"] >= 29
    assert second["effective_tpm_limit"] == 8000
