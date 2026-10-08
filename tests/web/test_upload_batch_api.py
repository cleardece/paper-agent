import asyncio
from io import BytesIO

from fastapi import UploadFile
import web.app as web_app


class FakeQueueRepository:
    def __init__(self, batches):
        self.batches = batches
        self.arguments = None

    def list_recent_batches(self, days, limit):
        self.arguments = (days, limit)
        return self.batches


def test_recent_upload_batches_are_capped_and_redacted(monkeypatch):
    repo = FakeQueueRepository([{
        "_id": "private-batch",
        "batch_id": "batch-1",
        "total_count": 1,
        "jobs": [{
            "_id": "private-job",
            "job_id": "job-1",
            "filename": "paper.pdf",
            "status": "completed",
            "pdf_path": "tmp_pdfs/private.pdf",
        }],
    }])
    monkeypatch.setattr(web_app, "get_upload_queue", lambda: repo)

    response = asyncio.run(web_app.list_recent_upload_batches(days=99, limit=99))

    assert repo.arguments == (7, 20)
    assert "_id" not in response["batches"][0]
    assert "_id" not in response["batches"][0]["jobs"][0]
    assert "pdf_path" not in response["batches"][0]["jobs"][0]


def test_rejected_only_submission_does_not_create_empty_batch(monkeypatch):
    class EmptyBatchRepository:
        def count_pending(self):
            return 0

        def create_batch(self, *_args):
            raise AssertionError("rejected-only submission must not create a batch")

    monkeypatch.setattr(web_app, "get_upload_queue", lambda: EmptyBatchRepository())

    result = asyncio.run(web_app._submit_upload_files([
        UploadFile(filename="not-a-paper.txt", file=BytesIO(b"not a PDF")),
    ]))

    assert result["batch_id"] is None
    assert result["accepted_count"] == 0
    assert result["rejected"] == [{"filename": "not-a-paper.txt", "reason": "只支持 PDF 文件"}]
