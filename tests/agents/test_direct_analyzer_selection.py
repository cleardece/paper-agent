from agents.direct_analyzer import DirectAnalyzerAgent


class FakeResponse:
    def __init__(self, content):
        self.content = content


class RecordingLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return FakeResponse(self.responses.pop(0))


class FakeMongo:
    def __init__(self, paper, chunks):
        self.paper = paper
        self.chunks = chunks
        self.status_updates = []

    def get_paper(self, arxiv_id):
        if self.paper and self.paper["arxiv_id"] == arxiv_id:
            return self.paper
        return None

    def list_papers(self, **_kwargs):
        return [self.paper] if self.paper else []

    def get_chunks_by_paper(self, arxiv_id):
        if self.paper and self.paper["arxiv_id"] == arxiv_id:
            return self.chunks
        return []

    def update_paper_status(self, arxiv_id, status, **_extra):
        self.status_updates.append((arxiv_id, status))


class FakeEmbedder:
    def __init__(self, fail=False):
        self.fail = fail

    def embed_texts(self, texts):
        if self.fail:
            raise RuntimeError("embedding unavailable")
        return [[float(index)] for index, _text in enumerate(texts)]


class FakeMilvus:
    def __init__(self, chunk_vector_count=0, paper_vector_count=0, fail_insert=False):
        self.chunk_vector_count = chunk_vector_count
        self.paper_vector_count = paper_vector_count
        self.fail_insert = fail_insert
        self.delete_calls = []
        self.inserted_records = []
        self.paper_embeddings = []

    def count(self, _arxiv_id):
        return self.chunk_vector_count

    def count_paper_embeddings(self, _arxiv_id):
        return self.paper_vector_count

    def delete_by_paper(self, arxiv_id):
        self.delete_calls.append(arxiv_id)

    def insert(self, records):
        if self.fail_insert:
            raise RuntimeError("Milvus unavailable")
        self.inserted_records.extend(records)

    def insert_paper_embedding(self, arxiv_id, title, embedding):
        self.paper_embeddings.append((arxiv_id, title, embedding))


class RecordingDirectAnalyzer(DirectAnalyzerAgent):
    def __init__(self, mongo, embedder=None, milvus=None):
        super().__init__(None, mongo, embedder, milvus, None)
        self.analyzed_text = None
        self.analysis_query = None

    def _analyze(self, _paper_info, core_text, _query=""):
        self.analyzed_text = core_text
        self.analysis_query = _query
        return {"answer": "analysis", "error": None}


def test_selected_id_uses_its_chunks_without_pdf_download():
    agent = RecordingDirectAnalyzer(
        FakeMongo(
            {"arxiv_id": "mmc", "title": "MMC", "status": "indexed"},
            [
                {
                    "chunk_index": 0,
                    "content": "method text",
                    "metadata": {"section": "Method"},
                }
            ],
        )
    )

    result = agent.invoke({"user_query": "分析", "target_paper_id": "mmc"})

    assert result["error"] is None
    assert result["resolved_paper_id"] == "mmc"
    assert result["primary_paper_id"] == "mmc"
    assert result["resolved_paper_ids"] == ["mmc"]
    assert agent._chunks_to_full_text(agent.mongo.chunks) == "# Method\nmethod text"
    assert "method text" in agent.analyzed_text


def test_unknown_selected_id_never_falls_back_to_another_paper():
    agent = RecordingDirectAnalyzer(FakeMongo(None, []))

    result = agent.invoke({"user_query": "分析", "target_paper_id": "missing"})

    assert result["error"] == "selected_paper_not_found"


def test_direct_analyzer_uses_contextual_retrieval_query():
    agent = RecordingDirectAnalyzer(
        FakeMongo(
            {"arxiv_id": "mmc", "title": "MMC", "status": "indexed"},
            [{
                "chunk_index": 0,
                "content": "result text",
                "metadata": {"section": "Experiment"},
            }],
        )
    )

    agent.invoke({
        "user_query": "实验呢？",
        "retrieval_query": "MMC 的实验设置、指标和结果是什么？",
        "target_paper_id": "mmc",
    })

    assert agent.analysis_query == "MMC 的实验设置、指标和结果是什么？"


def test_missing_resolved_target_returns_business_error_without_guessing():
    agent = RecordingDirectAnalyzer(FakeMongo(None, []))

    result = agent.invoke({
        "user_query": "分析它的实验方法",
        "turn_context": {
            "intent": "analyze",
            "primary_paper_id": None,
            "paper_ids": [],
            "allow_external_search": False,
        },
    })

    assert result["error"] == "NEED_PAPER_CONTEXT"
    assert result["primary_paper_id"] is None
    assert result["resolved_paper_ids"] == []


def test_chunked_paper_rebuilds_vectors_then_becomes_indexed():
    mongo = FakeMongo(
        {"arxiv_id": "mmc", "title": "MMC", "status": "chunked"},
        [{"chunk_index": 0, "content": "text", "metadata": {}}],
    )
    milvus = FakeMilvus()
    agent = RecordingDirectAnalyzer(mongo, FakeEmbedder(), milvus)

    agent.invoke({"user_query": "分析", "target_paper_id": "mmc"})

    assert milvus.delete_calls == ["mmc"]
    assert mongo.status_updates[-1] == ("mmc", "indexed")
    assert len(milvus.inserted_records) == 1
    assert len(milvus.paper_embeddings) == 1


def test_index_failure_keeps_chunks_and_records_embedding_failed():
    mongo = FakeMongo(
        {"arxiv_id": "mmc", "title": "MMC", "status": "chunked"},
        [{"chunk_index": 0, "content": "text", "metadata": {}}],
    )
    agent = RecordingDirectAnalyzer(mongo, FakeEmbedder(fail=True), FakeMilvus())

    agent.invoke({"user_query": "分析", "target_paper_id": "mmc"})

    assert mongo.status_updates[-1] == ("mmc", "embedding_failed")
    assert mongo.chunks[0]["content"] == "text"


def test_milvus_write_failure_keeps_chunks_and_records_milvus_failed():
    mongo = FakeMongo(
        {"arxiv_id": "mmc", "title": "MMC", "status": "chunked"},
        [{"chunk_index": 0, "content": "text", "metadata": {}}],
    )
    agent = RecordingDirectAnalyzer(mongo, FakeEmbedder(), FakeMilvus(fail_insert=True))

    agent.invoke({"user_query": "分析", "target_paper_id": "mmc"})

    assert mongo.status_updates[-1] == ("mmc", "milvus_failed")
    assert mongo.chunks[0]["content"] == "text"


def test_field_template_query_preserves_requested_fields_as_answer_contract():
    query = (
        "研究对象：\\\n"
        "Reynolds 数：\\\n"
        "流体从哪里进入、从哪里流出：\\\n"
        "圆柱后方会出现什么现象： &#x20;"
    )
    agent = DirectAnalyzerAgent(None, None, None, None, None)

    fields = agent._extract_requested_fields(query)
    requirements = agent._generate_analysis_requirements(query)

    assert fields == [
        "研究对象",
        "Reynolds 数",
        "流体从哪里进入、从哪里流出",
        "圆柱后方会出现什么现象",
    ]
    assert all(field in requirements for field in fields)
    assert "只回答用户列出的字段" in requirements
    assert "研究背景与动机" not in requirements


def test_flow_setup_question_includes_experiment_sections():
    agent = DirectAnalyzerAgent(None, None, None, None, None)

    sections = agent._classify_query(
        "Reynolds 数是多少？流体从哪里进入和流出？圆柱后方有什么现象？"
    )

    assert "experiment" in sections


def test_analysis_prompt_keeps_original_query_and_repairs_missing_fields():
    query = "研究对象：\nReynolds 数：\n入口和出口：\n尾流现象："
    llm = RecordingLLM([
        "这篇论文介绍了主动流动控制。",
        (
            "研究对象：二维圆柱绕流\n"
            "Reynolds 数：100\n"
            "入口和出口：左侧入口，右侧出口\n"
            "尾流现象：周期性涡脱落形成卡门涡街"
        ),
    ])
    agent = DirectAnalyzerAgent(llm, None, None, None, None)

    result = agent._analyze(
        {"title": "DRL flow control", "authors": ["Author"]},
        "Reynolds number is 100. Inlet is left and outlet is right.",
        query,
    )

    first_prompt = llm.calls[0][1].content
    repair_prompt = llm.calls[1][1].content
    assert query in first_prompt
    assert "用户原始问题" in first_prompt
    assert "研究对象" in repair_prompt
    assert "Reynolds 数" in repair_prompt
    assert len(llm.calls) == 2
    assert result["answer"].startswith("研究对象：")
