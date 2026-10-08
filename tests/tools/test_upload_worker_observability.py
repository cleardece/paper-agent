import asyncio
import logging

from tools.upload_worker import UploadQueueWorker


class FakeRepository:
    def update_job(self, *_args, **_kwargs):
        pass


class FakeParser:
    mineru_manager = None

    def parse(self, _pdf_path):
        return {"source": "mineru", "title": "A paper", "sections": [{"heading": "One"}]}

    def chunk(self, _sections):
        return [
            {"chunk_index": 0, "content": "first", "metadata": {}},
            {"chunk_index": 1, "content": "second", "metadata": {}},
        ]


class FakeMongo:
    def upsert_paper(self, _paper):
        pass

    def insert_chunks(self, _chunks):
        pass

    def update_paper_status(self, *_args, **_kwargs):
        pass


class FakeEmbedder:
    def embed_texts(self, texts):
        return [[0.0] for _ in texts]


class FakeMilvus:
    def insert(self, _records):
        pass

    def insert_paper_embedding(self, *_args):
        pass


def test_completed_job_logs_indexing_boundaries_in_order(caplog):
    container = type("Container", (), {
        "pdf_parser": FakeParser(),
        "mongodb": FakeMongo(),
        "embedder": FakeEmbedder(),
        "milvus": FakeMilvus(),
    })()
    worker = UploadQueueWorker(container, FakeRepository(), asyncio.Event())
    job = {
        "job_id": "job-1",
        "arxiv_id": "local-1",
        "filename": "paper.pdf",
        "pdf_path": "paper.pdf",
        "attempt_count": 1,
        "max_attempts": 2,
    }

    with caplog.at_level(logging.INFO, logger="paper-agent"):
        asyncio.run(worker.process_job(job))

    messages = [record.getMessage() for record in caplog.records]
    expected = [
        "开始生成 2 个向量",
        "向量生成完成，耗时",
        "开始写入 Milvus（2 条）",
        "Milvus 写入完成，耗时",
        "开始写入论文级向量",
        "论文级向量写入完成，耗时",
        "任务完成，总耗时",
    ]
    positions = [next(index for index, message in enumerate(messages) if phrase in message) for phrase in expected]
    assert positions == sorted(positions)
