import asyncio

from tools.pdf_parser import MinerUParseError
from web.app import _process_upload


def test_upload_marks_paper_as_parse_failed_when_accurate_parse_fails(tmp_path):
    updates = []
    container = type("Container", (), {})()
    container.pdf_parser = type(
        "Parser",
        (),
        {
            "parse": lambda *_: (_ for _ in ()).throw(
                MinerUParseError("MinerU 内存不足")
            )
        },
    )()
    container.mongodb = type(
        "Mongo", (), {"upsert_paper": lambda _, paper: updates.append(paper)}
    )()

    asyncio.run(
        _process_upload(container, str(tmp_path / "paper.pdf"), "paper.pdf", "local_test")
    )

    assert updates == [
        {
            "arxiv_id": "local_test",
            "title": "paper.pdf",
            "status": "parse_failed",
            "parse_error": "MinerU 内存不足",
            "parser_source": "mineru",
        }
    ]
