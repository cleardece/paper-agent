from storage.milvus import CHUNK_COLLECTION, MilvusClient, _truncate_utf8


class _FakeMilvus:
    def __init__(self):
        self.records = None

    def insert(self, *, collection_name, data):
        assert collection_name == CHUNK_COLLECTION
        self.records = data
        return {"insert_count": len(data)}


def test_insert_truncates_section_to_milvus_varchar_limit_without_mutating_source():
    fake = _FakeMilvus()
    client = object.__new__(MilvusClient)
    client.client = fake
    original_section = "A" * 134
    record = {"section": original_section, "heading": "Heading"}

    inserted = client.insert([record])

    assert inserted == 1
    assert record["section"] == original_section
    assert fake.records[0]["section"] == "A" * 128


def test_utf8_truncation_keeps_multibyte_text_decodable():
    value = _truncate_utf8("论文" * 100, 128)

    assert len(value.encode("utf-8")) <= 128
    assert value == "论文" * (128 // len("论文".encode("utf-8")))
