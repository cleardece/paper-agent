from copy import deepcopy
from datetime import datetime, timedelta, timezone

from storage.upload_queue import UploadQueueRepository


class FakeResult:
    def __init__(self, matched_count=0, modified_count=0, deleted_count=0):
        self.matched_count = matched_count
        self.modified_count = modified_count
        self.deleted_count = deleted_count


class FakeCursor(list):
    def sort(self, fields):
        for key, direction in reversed(fields):
            super().sort(key=lambda doc: doc.get(key), reverse=direction < 0)
        return self

    def limit(self, count):
        return FakeCursor(self[:count])


class FakeCollection:
    def __init__(self):
        self.docs = []

    def create_index(self, *_args, **_kwargs):
        pass

    def insert_one(self, doc):
        self.docs.append(deepcopy(doc))

    def insert_many(self, docs):
        self.docs.extend(deepcopy(docs))

    @staticmethod
    def _matches(doc, query):
        for key, expected in query.items():
            actual = doc.get(key)
            if isinstance(expected, dict):
                if "$in" in expected and actual not in expected["$in"]:
                    return False
                if "$lt" in expected and not actual < expected["$lt"]:
                    return False
                if "$gte" in expected and not actual >= expected["$gte"]:
                    return False
            elif actual != expected:
                return False
        return True

    def find_one(self, query, sort=None):
        matches = [doc for doc in self.docs if self._matches(doc, query)]
        if sort:
            for key, direction in reversed(sort):
                matches.sort(key=lambda doc: doc.get(key), reverse=direction < 0)
        return deepcopy(matches[0]) if matches else None

    def find(self, query):
        return FakeCursor(deepcopy([doc for doc in self.docs if self._matches(doc, query)]))

    def find_one_and_update(self, query, update, sort=None, return_document=None):
        match = self.find_one(query, sort=sort)
        if not match:
            return None
        for doc in self.docs:
            if doc["job_id"] == match["job_id"]:
                doc.update(deepcopy(update.get("$set", {})))
                for key, value in update.get("$inc", {}).items():
                    doc[key] = doc.get(key, 0) + value
                return deepcopy(doc)
        return None

    def update_one(self, query, update):
        for doc in self.docs:
            if self._matches(doc, query):
                doc.update(deepcopy(update.get("$set", {})))
                return FakeResult(1, 1)
        return FakeResult()

    def update_many(self, query, update):
        count = 0
        for doc in self.docs:
            if self._matches(doc, query):
                doc.update(deepcopy(update.get("$set", {})))
                count += 1
        return FakeResult(count, count)

    def count_documents(self, query):
        return sum(self._matches(doc, query) for doc in self.docs)


class FakeDatabase:
    def __init__(self):
        self.collections = {}

    def __getitem__(self, name):
        return self.collections.setdefault(name, FakeCollection())


def job(job_id, sequence, status="queued", attempt_count=0):
    return {
        "job_id": job_id,
        "batch_id": "batch-1",
        "sequence": sequence,
        "arxiv_id": job_id,
        "filename": f"{job_id}.pdf",
        "pdf_path": f"tmp_pdfs/{job_id}.pdf",
        "status": status,
        "attempt_count": attempt_count,
    }


def test_claim_next_job_uses_creation_then_sequence_order():
    repo = UploadQueueRepository(FakeDatabase())
    repo.create_batch("batch-1", 2)
    repo.create_jobs("batch-1", [job("job-2", 1), job("job-1", 0)])

    claimed = repo.claim_next_job()

    assert claimed["job_id"] == "job-1"
    assert claimed["status"] == "parsing"
    assert claimed["attempt_count"] == 1


def test_recover_processing_jobs_requeues_without_losing_attempt_count():
    repo = UploadQueueRepository(FakeDatabase())
    repo.create_batch("batch-1", 1)
    repo.create_jobs("batch-1", [job("job-1", 0, status="indexing", attempt_count=1)])

    assert repo.requeue_interrupted_jobs() == 1
    assert repo.get_job("job-1")["status"] == "queued"
    assert repo.get_job("job-1")["attempt_count"] == 1


def test_nonterminal_duplicate_is_detected_but_completed_job_is_not():
    repo = UploadQueueRepository(FakeDatabase())
    repo.create_batch("batch-1", 2)
    repo.create_jobs("batch-1", [job("job-1", 0, status="queued"), job("job-2", 1, status="completed")])

    assert repo.has_nonterminal_arxiv_id("job-1")
    assert not repo.has_nonterminal_arxiv_id("job-2")


def test_list_recent_batches_filters_window_and_orders_jobs(monkeypatch):
    repo = UploadQueueRepository(FakeDatabase())
    now = datetime(2026, 8, 22, tzinfo=timezone.utc)
    monkeypatch.setattr(repo, "_now", lambda: now)
    repo.create_batch("old", 1)
    repo.batches.docs[0]["created_at"] = now - timedelta(days=7, seconds=1)
    repo.create_batch("recent", 2)
    recent_jobs = [job("second", 1), job("first", 0)]
    for item in recent_jobs:
        item["batch_id"] = "recent"
    repo.create_jobs("recent", recent_jobs)

    batches = repo.list_recent_batches(days=7, limit=20)

    assert [batch["batch_id"] for batch in batches] == ["recent"]
    assert [item["job_id"] for item in batches[0]["jobs"]] == ["first", "second"]
